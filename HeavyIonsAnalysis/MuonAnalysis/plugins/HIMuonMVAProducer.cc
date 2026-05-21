#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "PhysicsTools/XGBoost/interface/XGBooster.h"
#include "fastjet/contrib/SoftKiller.hh"
#include "correction.h"
#include <fstream>


namespace pat {

  class HIMuonMVAProducer : public edm::global::EDProducer<> {
  public:
    explicit HIMuonMVAProducer(const edm::ParameterSet& iConfig)
        : muonToken_(consumes<pat::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
          pfCandidateToken_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("pfCandidates"))),
          centralityToken_(consumes<int>(iConfig.getParameter<edm::InputTag>("centrality"))),
          etaToken_(consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>("etaMap"))),
          rhoToken_(consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>("rhoMap"))),
          patMuonPutToken_(produces<pat::MuonCollection>()),
          pfMaxEta_(iConfig.getParameter<double>("pf_maxAbsEta")),
          skRadius_(iConfig.getParameter<double>("sk_radius")),
          muonMinPt_(iConfig.getParameter<double>("muon_minPt")),
          rVeto_(iConfig.getParameter<double>("iso_rVeto")),
          rCone_(iConfig.getParameter<double>("iso_rCone")),
          isoCorr_(getCorrection(iConfig)),
          isoModel_(getModel(iConfig)) {}
    ~HIMuonMVAProducer() override{};

    void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const edm::EDGetTokenT<pat::MuonCollection> muonToken_;
    const edm::EDGetTokenT<reco::CandidateView> pfCandidateToken_;
    const edm::EDGetTokenT<int> centralityToken_;
    const edm::EDGetTokenT<std::vector<double>> etaToken_;
    const edm::EDGetTokenT<std::vector<double>> rhoToken_;
    const edm::EDPutTokenT<pat::MuonCollection> patMuonPutToken_;
    const reco::PFCandidate convert_;
    const double pfMaxEta_, skRadius_, muonMinPt_, rVeto_, rCone_;
    const std::shared_ptr<const correction::Correction> isoCorr_;
    const std::unique_ptr<XGBooster> isoModel_;

    std::shared_ptr<const correction::Correction> getCorrection(const edm::ParameterSet& iConfig) {
      const auto& csetIsoRhoCorrections = correction::CorrectionSet::from_file(iConfig.getParameter<edm::FileInPath>("file_isoCorr").fullPath());
      return csetIsoRhoCorrections->at("iso_rho_correction");
    }

    XGBooster* getModel(const edm::ParameterSet& iConfig, const int& nfeat=8) {
      auto model = new XGBooster(iConfig.getParameter<edm::FileInPath>("file_isoModel").fullPath());
      for (int i=0; i<nfeat; i++) model->addFeature(std::to_string(i));
      return model;
    }

    enum WP { WP95, WP90, WP85, WP80 };
    bool passMVAIso(const double&, const double&, const WP& wp) const;
  };

}  // namespace pat


bool pat::HIMuonMVAProducer::passMVAIso(const double& mva, const double& cent, const WP& wp) const {
  double cut(10.);
  const auto cen = cent > 90. ? 90. : cent;
  const auto cen2 = cen*cen;
  const auto cen3 = cen*cen*cen;
  //Working point: WP95
  if (wp==WP95)
    cut = 7.978478076287510e-07*cen3 + -0.00010197402752356007*cen2 +  0.00073749187425983740*cent + 0.44973546555978620;
  //Working point: WP90
  else if (wp==WP90)
    cut = 5.023194760398722e-07*cen3 + -6.386564313645383e-05*cen2  + -0.00030034696427764694*cent + 0.26733467400525280;
  //Working point: WP85
  else if (wp==WP85)
    cut = 3.642678187960558e-07*cen3 + -4.4289339403249526e-05*cen2 + -0.00038178775816005510*cent + 0.17242030428600790;
  //Working point: WP80
  else if (wp==WP80)
    cut = 2.792961957599443e-07*cen3 + -3.314677611344172e-05*cen2  + -0.00028826679894283433*cent + 0.11887071187630002;
  return mva < cut;
}

void pat::HIMuonMVAProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // extract input information
  const auto& muons = iEvent.get(muonToken_);
  const auto& pfCandidates = iEvent.get(pfCandidateToken_);
  const auto& etaMap = iEvent.get(etaToken_);
  const auto& rhoMap = iEvent.get(rhoToken_);
  const double cent = iEvent.get(centralityToken_) / 2.0;

  // select PF candidates
  std::vector<std::tuple<double, double, double, int, int, double>> selPFCands;
  if (etaMap.size() > 1) {
    selPFCands.reserve(pfCandidates.size());
    std::vector<std::vector<fastjet::PseudoJet>> particlesForSK(etaMap.size()-1);
    for (const auto& pf : pfCandidates) {
      // determine eta category
      int ieta(-1);
      for (size_t i=1; i<etaMap.size(); i++)
        if (pf.eta() >= etaMap[i-1] && pf.eta() < etaMap[i]) {
          ieta = i-1;
          break;
        }
      if (ieta < 0)
        continue;
      // fill particles for soft killer
      particlesForSK[ieta].emplace_back(pf.px(), pf.py(), pf.pz(), pf.energy());
      // fill selected PF candidates
      const auto& id = convert_.translatePdgIdToType(pf.pdgId());
      if (id > 0 && id <= 5 && std::abs(pf.eta()) <= pfMaxEta_)
        selPFCands.emplace_back(pf.pt(), pf.eta(), pf.phi(), id, ieta, 0.0);
    }
  
    // compute soft killer thresholds
    std::vector<double> skThrs(etaMap.size()-1);
    for (size_t i=0; i<particlesForSK.size(); i++) {
	  const auto& particles = particlesForSK[i];
	  if (not particles.empty()) {
	    fastjet::contrib::SoftKiller soft_killer(etaMap[i], etaMap[i+1], skRadius_, skRadius_);
        std::vector<fastjet::PseudoJet> soft_killed_event;
        soft_killer.apply(particles, soft_killed_event, skThrs[i]);
      }
    }

    // add soft killer thresholds to selected PF candidates
    for (auto& cand : selPFCands)
      std::get<5>(cand) = skThrs[std::get<4>(cand)];
  }

  // initialize output muon collection
  pat::MuonCollection output(muons);

  // loop over output muons
  for (auto& muon : output) {
    if (muon.pt() < muonMinPt_)
      continue;

    // associate rho value
    double rho(-1.);
    for (size_t i=1; i<etaMap.size(); i++)
      if (muon.eta() >= etaMap[i-1] && muon.eta() < etaMap[i]) {
		rho = rhoMap[i-1];
		break;
	  }
    if (rho < 0)
      continue;

    // compute IP3D significance
    const auto ip3DSig = std::abs(muon.dB(pat::Muon::PV3D)) / muon.edB(pat::Muon::PV3D);

    // compute soft killer isolation
    double skPFChIso(0.), skPFNeuIso(0.), skPFPhoIso(0.);
    for (const auto& cand : selPFCands) {
	    const auto& [pt, eta, phi, id, ieta, skThr] = cand;
	    const auto dR2 = reco::deltaR2(muon.eta(), muon.phi(), eta, phi);
	    if (dR2 >= rVeto_ * rVeto_ && dR2 <= rCone_ * rCone_)
	      (id == 5 ? skPFNeuIso : (id == 4 ? skPFPhoIso : skPFChIso)) += pt * (pt > skThr);
    }
    const auto& skPFIso = skPFChIso + skPFNeuIso + skPFPhoIso;

    // extract PF isolation
    const auto& pfChIso = muon.pfIsolationR04().sumChargedHadronPt;
    const auto& pfNeuIso = muon.pfIsolationR04().sumNeutralHadronEt;
    const auto& pfPhoIso = muon.pfIsolationR04().sumPhotonEt;
    const auto& pfIso = pfChIso  + pfNeuIso + pfPhoIso;

    // correct the PF isolation variables
    const auto& pfRelIso = (pfIso - isoCorr_->evaluate({{"PFIso", "mu", rho}})) / muon.pt();
    const auto& pfChRelIso = (pfChIso - isoCorr_->evaluate({{"PFChIso", "mu", rho}})) / muon.pt();
    const auto& skPFRelIso = (skPFIso - isoCorr_->evaluate({{"skPFIso", "mu", rho}})) / muon.pt();
    const auto& skPFChRelIso = (skPFChIso - isoCorr_->evaluate({{"skPFChIso", "mu", rho}})) / muon.pt();

    // compute the isolation from MVA
    const std::vector<double> inputs({std::abs(muon.eta()), muon.phi(), rho, ip3DSig, pfRelIso, pfChRelIso, skPFRelIso, skPFChRelIso});
    const std::vector<float> features(inputs.begin(), inputs.end());
    const auto isoValue = 1. - isoModel_->predict(features);
    muon.addUserFloat("hiMVAIso", isoValue);
    muon.addUserInt("hiMVAIsoWP95", passMVAIso(isoValue, cent, WP95));
    muon.addUserInt("hiMVAIsoWP90", passMVAIso(isoValue, cent, WP90));
    muon.addUserInt("hiMVAIsoWP85", passMVAIso(isoValue, cent, WP85));
    muon.addUserInt("hiMVAIsoWP80", passMVAIso(isoValue, cent, WP80));
  }

  iEvent.emplace(patMuonPutToken_, std::move(output));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void pat::HIMuonMVAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muons", edm::InputTag("slimmedMuons"))->setComment("muon input collection");
  desc.add<edm::InputTag>("pfCandidates", edm::InputTag("packedPFCandidates"))->setComment("PF candidate input collection");
  desc.add<edm::InputTag>("centrality", edm::InputTag("centralityBin:HFtowers"))->setComment("centrality");
  desc.add<edm::InputTag>("etaMap", edm::InputTag("hiFJRhoProducerFinerBins:mapEtaEdges"))->setComment("eta ranges for rho and soft killer");
  desc.add<edm::InputTag>("rhoMap", edm::InputTag("hiFJRhoProducerFinerBins:mapToRho"))->setComment("rho");
  desc.add<double>("pf_maxAbsEta", 2.8)->setComment("Maximum absolute eta for PF candidates");
  desc.add<double>("sk_radius", 0.4)->setComment("Radius for soft killer threshold");
  desc.add<double>("muon_minPt", 0.0)->setComment("Muon minimum pt");
  desc.add<double>("iso_rVeto", 1.E-3)->setComment("Isolation veto radius");
  desc.add<double>("iso_rCone", 0.3)->setComment("Isolation cone radius");
  desc.add<edm::FileInPath>("file_isoModel", {})->setComment("Path to isolation model");
  desc.add<edm::FileInPath>("file_isoCorr", {})->setComment("Path to isolation rho correction");
  descriptions.add("hiMuons", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(HIMuonMVAProducer);

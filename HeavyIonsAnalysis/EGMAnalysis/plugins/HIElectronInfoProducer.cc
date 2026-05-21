#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "PhysicsTools/XGBoost/interface/XGBooster.h"
#include "fastjet/contrib/SoftKiller.hh"
#include "correction.h"


namespace pat {

  class HIElectronInfoProducer : public edm::global::EDProducer<> {
  public:
    explicit HIElectronInfoProducer(const edm::ParameterSet& iConfig)
        : electronToken_(consumes<pat::ElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons"))),
          pfCandidateToken_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("pfCandidates"))),
          centralityToken_(consumes<int>(iConfig.getParameter<edm::InputTag>("centrality"))),
          etaToken_(consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>("etaMap"))),
          rhoToken_(consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>("rhoMap"))),
          patElectronPutToken_(produces<pat::ElectronCollection>()),
          pfMaxEta_(iConfig.getParameter<double>("pf_maxAbsEta")),
          skRadius_(iConfig.getParameter<double>("sk_radius")),
          electronMinPt_(iConfig.getParameter<double>("electron_minPt")),
          rVeto_(iConfig.getParameter<double>("iso_rVeto")),
          rCone_(iConfig.getParameter<double>("iso_rCone")),
          isoCorr_(getCorrection(iConfig, "iso_rho_correction")),
          hoeCorr_(getCorrection(iConfig, "hoecorrector")),
          isoModel_(getModel(iConfig, "file_isoModel", 9)),
          idModel_(getModel(iConfig, "file_idModel", 11)) {}
    ~HIElectronInfoProducer() override{};

    void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const edm::EDGetTokenT<pat::ElectronCollection> electronToken_;
    const edm::EDGetTokenT<reco::CandidateView> pfCandidateToken_;
    const edm::EDGetTokenT<int> centralityToken_;
    const edm::EDGetTokenT<std::vector<double>> etaToken_;
    const edm::EDGetTokenT<std::vector<double>> rhoToken_;
    const edm::EDPutTokenT<pat::ElectronCollection> patElectronPutToken_;

    const reco::PFCandidate convert_;
    const double pfMaxEta_, skRadius_, electronMinPt_, rVeto_, rCone_;
    const std::shared_ptr<const correction::Correction> isoCorr_, hoeCorr_;
    const std::unique_ptr<XGBooster> isoModel_, idModel_;

    std::shared_ptr<const correction::Correction> getCorrection(const edm::ParameterSet& iConfig, const std::string& label) {
      const auto& csetIsoRhoCorrections = correction::CorrectionSet::from_file(iConfig.getParameter<edm::FileInPath>("file_corr").fullPath());
      return csetIsoRhoCorrections->at(label);
    }

    XGBooster* getModel(const edm::ParameterSet& iConfig, const std::string& f, const int& nfeat) {
      auto model = new XGBooster(iConfig.getParameter<edm::FileInPath>(f).fullPath());
      for (int i=0; i<nfeat; i++) model->addFeature(std::to_string(i));
      return model;
    }

    enum WP { WP95=0, WP90=1, WP85=4, WP80=2, WP70=3 };
    bool passMVAIso(const double&, const double&, const bool&, const WP& wp) const;
    bool passMVAId(const double&, const double&, const bool&, const WP& wp) const;
    bool passCutID(const double&, const double&, const double&, const double&, const double&, const double&, const double&, const double&, const double&, const bool&, const WP&) const;
  };

}  // namespace pat


bool pat::HIElectronInfoProducer::passMVAIso(const double& mva, const double& cent, const bool& isEB, const WP& wp) const {
  double cut(10.);
  const auto cen = cent > 90. ? 90. : cent;
  const auto cen2 = cen*cen;
  const auto cen3 = cen*cen*cen;
  //Working point: WP95
  if (wp==WP95) {
    if (isEB) cut =  1.1213742303052684e-06*cen3 + -0.00015171073540105780*cen2 + 0.00368025276158394370*cen + 0.69091803443919030;
    else      cut = -3.6813992861380774e-07*cen3 +  8.0837388929110730e-06*cen2 + 3.0096841471229364e-05*cen + 0.82161996960170210;
  }
  //Working point: WP90
  else if (wp==WP90) {
    if (isEB) cut =  1.4928752428837551e-06*cen3 + -0.00021314188364472087*cen2 + 0.00532921536800796700*cen + 0.50072628430984800;
    else      cut = -8.5265211094211280e-09*cen3 + -5.9102242593526630e-05*cen2 + 0.00232248702443364850*cen + 0.67891918907325090;
  }
  //Working point: WP85
  else if (wp==WP85) {
    if (isEB) cut =  1.4887049145962363e-06*cen3 + -0.00021245424491055573*cen2 + 0.00541199202152453040*cen + 0.35445907806427995;
    else      cut =  5.3677856468709490e-07*cen3 + -0.00013402600617040136*cen2 + 0.00483495827394446600*cen + 0.53750075876189360;
  }
  //Working point: WP80
  else if (wp==WP80) {
    if (isEB) cut =  1.3135127239015455e-06*cen3 + -0.00018734725305218886*cen2 + 0.00499209112243616000*cen + 0.24711431505380846;
    else      cut =  8.9750115827725710e-07*cen3 + -0.00017796358715325376*cen2 + 0.00642536444769568500*cen + 0.41419909838858876;
  }
  return mva < cut;
}

bool pat::HIElectronInfoProducer::passMVAId(const double& mva, const double& cen, const bool& isEB, const WP& wp) const {
  double cut(10.);
  const auto cen2 = cen*cen;
  const auto cen3 = cen*cen*cen;
  //Working point: WP95
  if (wp==WP95) {
    if (isEB) cut = -8.6895588483154620e-07*cen3 + 0.00013933876005428234*cen2 + -0.008619594547272720*cen + 0.45763429913293496;
    else      cut = -1.4970873446950470e-06*cen3 + 0.00024879582298806160*cen2 + -0.015436599736040429*cen + 0.72788786967830470;
  }
  //Working point: WP90
  else if (wp==WP90) {
    if (isEB) cut = -4.6614422355778817e-07*cen3 + 8.0092208959414990e-05*cen2 + -0.005222306985603681*cen + 0.23094224803139338;
    else      cut = -1.3010247333840360e-06*cen3 + 0.00021736111417101613*cen2 + -0.013308168801268280*cen + 0.51108687929103980;
  }
  //Working point: WP85
  else if (wp==WP85) {
    if (isEB) cut = -2.7396191468306490e-07*cen3 + 4.7113855045057714e-05*cen2 + -0.002967751640976904*cen + 0.12148874341679987;
    else      cut = -8.3321554760551400e-07*cen3 + 0.00014285719403057067*cen2 + -0.009134790058710965*cen + 0.34550384804492130;
  }
  //Working point: WP80
  else if (wp==WP80) {
    if (isEB) cut = -1.3961443236681390e-07*cen3 + 2.3527418966464617e-05*cen2 + -0.001472122698460129*cen + 0.06641013597502561;
    else      cut = -5.0247362141513420e-07*cen3 + 8.9424796689439320e-05*cen2 + -0.006059630793483640*cen + 0.23549473657009040;
  }
  return mva < cut;
}

bool pat::HIElectronInfoProducer::passCutID(const double& sInIn, const double& adEta, const double& adPhi, const double& hOverE, const double& eOverP, const double& aD0, const double& aDz, const double& missH, const double& cen, const bool& isEB, const WP& wp) const {
  std::array<double, 4> max_sInIn{{0.}}, max_adEta{{0.}}, max_adPhi{{0.}}, max_hOverE{{0.}}, max_eOverP{{0.}}, max_aD0{{0.}}, max_aDz{{0.}}, max_missH{{0}};
  if (isEB) {
    if (cen < 30.) {
      //            WP95     WP90     WP80     WP70 
      max_sInIn  = {{0.0131,  0.0125,  0.0106,  0.0106}};
      max_adEta  = {{0.00389, 0.00365, 0.00343, 0.00342}};
      max_adPhi  = {{0.0963,  0.0314,  0.0238,  0.0195}};
      max_hOverE = {{0.156,   0.155,   0.153,   0.124}};
      max_eOverP = {{0.421,   0.0547,  0.0285,  0.00837}};
      max_missH  = {{2,       1,       1,       1}};
      max_aD0    = {{0.05,    0.05,    0.05,    0.05}};
      max_aDz    = {{0.10,    0.10,    0.10,    0.10}};
    }
    else {
      max_sInIn  = {{0.0105,  0.0103,  0.0101,  0.0101}};
      max_adEta  = {{0.00457, 0.00377, 0.00322, 0.00305}};
      max_adPhi  = {{0.0634,  0.0554,  0.0262,  0.0185}};
      max_hOverE = {{0.107,   0.0762,  0.0555,  0.0401}};
      max_eOverP = {{0.137,   0.0513,  0.0477,  0.0327}};
      max_missH  = {{2,       1,       1,       1}};
      max_aD0    = {{0.05,    0.05,    0.05,    0.05}};
      max_aDz    = {{0.10,    0.10,    0.10,    0.10}};
    }
  }
  else {
    if (cen < 30.) {
      max_sInIn  = {{0.0382,  0.0329,  0.029,   0.0283}};
      max_adEta  = {{0.00881, 0.00682, 0.00576, 0.00489}};
      max_adPhi  = {{0.264,   0.19,    0.071,   0.024}};
      max_hOverE = {{0.178,   0.174,   0.172,   0.156}};
      max_eOverP = {{0.146,   0.133,   0.0585,  0.0174}};
      max_missH  = {{3,       1,       1,       1}};
      max_aD0    = {{0.10,    0.10,    0.10,    0.10}};
      max_aDz    = {{0.20,    0.20,    0.20,    0.20}};      
    }
    else {
      max_sInIn  = {{0.028,   0.0277,  0.0271,  0.0271}};
      max_adEta  = {{0.0074,  0.00731, 0.00629, 0.0059}};
      max_adPhi  = {{0.253,   0.0794,  0.0300,  0.0242}};
      max_hOverE = {{0.107,   0.075,   0.0639,  0.0338}};
      max_eOverP = {{0.142,   0.0462,  0.0144,  0.0122}};
      max_missH  = {{3,       1,       1,       1}};
      max_aD0    = {{0.10,    0.10,    0.10,    0.10}};
      max_aDz    = {{0.20,    0.20,    0.20,    0.20}};      
    }
  }
  return (sInIn < max_sInIn[wp]) && (adEta < max_adEta[wp]) && (adPhi < max_adPhi[wp]) && (hOverE < max_hOverE[wp]) && (eOverP < max_eOverP[wp]) && (missH <= max_missH[wp]) && (aD0 <max_aD0[wp]) && (aDz < max_aDz[wp]);
}

void pat::HIElectronInfoProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // extract input information
  const auto& electrons = iEvent.get(electronToken_);
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

  // initialize output electron collection
  pat::ElectronCollection output(electrons);

  // loop over output electrons
  for (auto& electron : output) {
    if (electron.pt() < electronMinPt_)
      continue;

    // associate rho value
    double rho(-1.);
    for (size_t i=1; i<etaMap.size(); i++)
      if (electron.eta() >= etaMap[i-1] && electron.eta() < etaMap[i]) {
		    rho = rhoMap[i-1];
		    break;
	    }
    if (rho < 0)
      continue;

    const auto absEta = std::abs(electron.eta());
    const bool isEB(absEta<1.45);

    // compute the identification from MVA
    const auto& sigmaIetaIeta = electron.sigmaIetaIeta();
    const auto& dEtaSeedAtVtx = electron.deltaEtaSeedClusterTrackAtVtx();
    const auto& dPhiAtVtx = electron.deltaPhiSuperClusterTrackAtVtx();
    const auto& track = electron.gsfTrack();
    const auto& d0 = electron.dB(pat::Electron::PV2D);
    const auto& dz = electron.dB(pat::Electron::PVDZ);
    const double& missHits = electron.gsfTrack()->numberOfLostHits();
    const auto& ecalEnergy = electron.hasUserFloat("rawEcalEnergy") ? electron.userFloat("rawEcalEnergy") : electron.ecalEnergy();
    const auto& eOverPInv = 1. / ecalEnergy - 1. / electron.trackMomentumAtVtx().R();
    const auto& corHoverEBc = electron.hcalOverEcalBc() - hoeCorr_->evaluate({{rho}});
    const std::vector<double> inputsID({absEta, electron.phi(), rho, sigmaIetaIeta, dEtaSeedAtVtx, dPhiAtVtx, d0, dz, missHits, eOverPInv, corHoverEBc});
    const std::vector<float> featuresID(inputsID.begin(), inputsID.end());
    const auto idValue = 1. - idModel_->predict(featuresID);
    electron.addUserFloat("hiMVAId", idValue);
    electron.addUserInt("hiMVAIdWP95", passMVAId(idValue, cent, isEB, WP95));
    electron.addUserInt("hiMVAIdWP90", passMVAId(idValue, cent, isEB, WP90));
    electron.addUserInt("hiMVAIdWP85", passMVAId(idValue, cent, isEB, WP85));
    electron.addUserInt("hiMVAIdWP80", passMVAId(idValue, cent, isEB, WP80));

    // compute IP3D significance
    const auto ip3DSig = std::abs(electron.dB(pat::Electron::PV3D)) / electron.edB(pat::Electron::PV3D);

    // compute PF and soft killer isolations
    double pfChIso(0.), pfNeuIso(0.), pfPhoIso(0.);
    double skPFChIso(0.), skPFNeuIso(0.), skPFPhoIso(0.);
    for (const auto& cand : selPFCands) {
	    const auto& [pt, eta, phi, id, ieta, skThr] = cand;
	    const auto dR2 = reco::deltaR2(electron.eta(), electron.phi(), eta, phi);
	    if (dR2 >= rVeto_ * rVeto_ && dR2 <= rCone_ * rCone_) {
        (id == 5 ? pfNeuIso : (id == 4 ? pfPhoIso : pfChIso)) += pt;
	    (id == 5 ? skPFNeuIso : (id == 4 ? skPFPhoIso : skPFChIso)) += pt * (pt > skThr);
      }
    }
    const auto& pfIso = pfChIso  + pfNeuIso + pfPhoIso;
    const auto& skPFIso = skPFChIso + skPFNeuIso + skPFPhoIso;

    // correct the PF isolation variables
    const auto& pfRelIso = (pfIso - isoCorr_->evaluate({{"PFIso", "ele", rho}})) / electron.pt();
    const auto& pfChRelIso = (pfChIso - isoCorr_->evaluate({{"PFChIso", "ele", rho}})) / electron.pt();
    const auto& skPFRelIso = (skPFIso - isoCorr_->evaluate({{"skPFIso", "ele", rho}})) / electron.pt();
    const auto& skPFChRelIso = (skPFChIso - isoCorr_->evaluate({{"skPFChIso", "ele", rho}})) / electron.pt();

    // compute the isolation from MVA
    const std::vector<double> inputsISO({absEta, electron.phi(), rho, ip3DSig, pfRelIso, pfChRelIso, skPFRelIso, skPFChRelIso, idValue});
    const std::vector<float> featuresISO(inputsISO.begin(), inputsISO.end());
    const auto isoValue = 1. - isoModel_->predict(featuresISO);
    electron.addUserFloat("hiMVAIso", isoValue);
    electron.addUserInt("hiMVAIsoWP95", passMVAIso(isoValue, cent, isEB, WP95));
    electron.addUserInt("hiMVAIsoWP90", passMVAIso(isoValue, cent, isEB, WP90));
    electron.addUserInt("hiMVAIsoWP85", passMVAIso(isoValue, cent, isEB, WP85));
    electron.addUserInt("hiMVAIsoWP80", passMVAIso(isoValue, cent, isEB, WP80));

    // compute the cut-based identification
    const auto& sInIn = electron.full5x5_sigmaIetaIeta();
    const auto& dEtaSeed = (electron.superCluster().isNonnull() && electron.superCluster()->seed().isNonnull()) ? (electron.deltaEtaSuperClusterTrackAtVtx() - electron.superCluster()->eta() + electron.superCluster()->seed()->eta()) : std::numeric_limits<float>::max();
    const auto adEta = std::abs(dEtaSeed);
    const auto adPhi = std::abs(dPhiAtVtx);
    const auto& hOverE = electron.full5x5_hcalOverEcalBc();
    const auto& ooEmooP = (1.0 - electron.eSuperClusterOverP()) / electron.ecalEnergy();
    const auto eOverP = (electron.ecalEnergy() > 0 && std::isfinite(electron.ecalEnergy())) ? std::abs(ooEmooP) : 1.e30;
    const auto aD0 = std::abs(d0);
    const auto aDz = std::abs(dz);
    electron.addUserInt("hiCutIdWP95", passCutID(sInIn, adEta, adPhi, hOverE, eOverP, aD0, aDz, missHits, cent, isEB, WP95));
    electron.addUserInt("hiCutIdWP90", passCutID(sInIn, adEta, adPhi, hOverE, eOverP, aD0, aDz, missHits, cent, isEB, WP90));
    electron.addUserInt("hiCutIdWP80", passCutID(sInIn, adEta, adPhi, hOverE, eOverP, aD0, aDz, missHits, cent, isEB, WP80));
    electron.addUserInt("hiCutIdWP70", passCutID(sInIn, adEta, adPhi, hOverE, eOverP, aD0, aDz, missHits, cent, isEB, WP70));
  }

  iEvent.emplace(patElectronPutToken_, std::move(output));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void pat::HIElectronInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("electrons", edm::InputTag("slimmedElectrons"))->setComment("electron input collection");
  desc.add<edm::InputTag>("pfCandidates", edm::InputTag("packedPFCandidates"))->setComment("PF candidate input collection");
  desc.add<edm::InputTag>("centrality", edm::InputTag("centralityBin:HFtowers"))->setComment("centrality");
  desc.add<edm::InputTag>("etaMap", edm::InputTag("hiFJRhoProducerFinerBins:mapEtaEdges"))->setComment("eta ranges for rho and soft killer");
  desc.add<edm::InputTag>("rhoMap", edm::InputTag("hiFJRhoProducerFinerBins:mapToRho"))->setComment("rho");
  desc.add<double>("pf_maxAbsEta", 2.8)->setComment("Maximum absolute eta for PF candidates");
  desc.add<double>("sk_radius", 0.4)->setComment("Radius for soft killer threshold");
  desc.add<double>("electron_minPt", 0.0)->setComment("Electron minimum pt");
  desc.add<double>("iso_rVeto", 0.026)->setComment("Isolation veto radius");
  desc.add<double>("iso_rCone", 0.3)->setComment("Isolation cone radius");
  desc.add<edm::FileInPath>("file_idModel", {})->setComment("Path to identification model");
  desc.add<edm::FileInPath>("file_isoModel", {})->setComment("Path to isolation model");
  desc.add<edm::FileInPath>("file_corr", {})->setComment("Path to rho correction");
  descriptions.add("hiElectrons", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(HIElectronInfoProducer);

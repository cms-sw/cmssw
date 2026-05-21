#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "fastjet/contrib/SoftKiller.hh"
#include "TTree.h"


class HiFJSoftKillerAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HiFJSoftKillerAnalyzer(const edm::ParameterSet& iConfig)
      : pfToken_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("source"))),
        etaMap_(iConfig.getParameter<std::vector<double>>("etaMap")) {};
  ~HiFJSoftKillerAnalyzer() override{};

  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<reco::CandidateView> pfToken_;
  const std::vector<double> etaMap_;

  //output
  TTree *tree_;
  edm::Service<TFileService> fs_;
  std::vector<float> radius_;
  std::vector<float> etaMin_;
  std::vector<float> etaMax_;
  std::vector<char> pfType_;
  std::vector<float> thr_;
  std::vector<unsigned short> nPar_;
};

void HiFJSoftKillerAnalyzer::beginJob()
{
  const auto jetTagTitle = "HiFJSoftKiller Jet background analysis tree";
  tree_ = fs_->make<TTree>("t", jetTagTitle);
  tree_->Branch("radius", &radius_);
  tree_->Branch("etaMin", &etaMin_);
  tree_->Branch("etaMax", &etaMax_);
  tree_->Branch("pfType", &pfType_);
  tree_->Branch("thr", &thr_);
  tree_->Branch("nPar", &nPar_);
}

void HiFJSoftKillerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //clear vectors
  radius_.clear();
  etaMin_.clear();
  etaMax_.clear();
  pfType_.clear();
  thr_.clear();
  nPar_.clear();

  //extract PF candidates
  static const reco::PFCandidate PF;
  std::map<std::pair<double, double>, std::map<char, std::vector<fastjet::PseudoJet>>> particles;
  for (const auto& pf : iEvent.get(pfToken_)) {
    for (size_t i=1; i<etaMap_.size(); i++) {
      if (pf.eta() >= etaMap_[i-1] && pf.eta() < etaMap_[i]) {
        auto& par = particles[{etaMap_[i-1], etaMap_[i]}];
        par[0].emplace_back(pf.px(), pf.py(), pf.pz(), pf.energy());
        if (pf.pdgId() == 211 || pf.pdgId() == -211)
          par[1].emplace_back(pf.px(), pf.py(), pf.pz(), pf.energy());
        else if (pf.pdgId() == 130 || pf.pdgId() == 1)
          par[2].emplace_back(pf.px(), pf.py(), pf.pz(), pf.energy());
        else if (pf.pdgId() == 22 || pf.pdgId() == 2)
          par[3].emplace_back(pf.px(), pf.py(), pf.pz(), pf.energy());
      }
    }
  }

  //extract soft killer threshold
  for (size_t i=1; i<etaMap_.size(); i++) {
    const auto& par = particles.find({etaMap_[i-1], etaMap_[i]});
    if (par == particles.end())
      continue;
    for (const auto& r : {0.2, 0.3, 0.4}) {
      fastjet::contrib::SoftKiller soft_killer(etaMap_[i-1], etaMap_[i], r, r);
      for (const auto& p : par->second) {
        double pt_threshold;
        std::vector<fastjet::PseudoJet> soft_killed_event;
        soft_killer.apply(p.second, soft_killed_event, pt_threshold);
        radius_.emplace_back(r);
        etaMin_.emplace_back(etaMap_[i-1]);
        etaMax_.emplace_back(etaMap_[i]);
        pfType_.emplace_back(p.first);
        thr_.emplace_back(pt_threshold);
        nPar_.emplace_back(p.second.size());
      }
    }
  }

  //fill tree
  tree_->Fill();
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HiFJSoftKillerAnalyzer);

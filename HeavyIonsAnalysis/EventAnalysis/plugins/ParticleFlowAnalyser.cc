#include "HeavyIonsAnalysis/EventAnalysis/plugins/ParticleFlowAnalyser.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// constructors and destructor
//
ParticleFlowAnalyser::ParticleFlowAnalyser(const edm::ParameterSet& iConfig)
    : pfCandidateToken_(
          consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandidateSrc"))),
      ptMin_(iConfig.getParameter<double>("ptMin")),
      absEtaMax_(iConfig.getParameter<double>("absEtaMax")) {}

ParticleFlowAnalyser::~ParticleFlowAnalyser() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void ParticleFlowAnalyser::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  clear();

  edm::Handle<pat::PackedCandidateCollection> pfCandidates;
  iEvent.getByToken(pfCandidateToken_, pfCandidates);

  for (const auto& pfcand : *pfCandidates) {
    float pt = pfcand.pt();
    if (pt < ptMin_)
      continue;

    float eta = pfcand.eta();
    if (std::abs(eta) > absEtaMax_)
      continue;

    /* dummy reco::PFCandidate used to convert pdgId */
    auto id = converter_.translatePdgIdToType(pfcand.pdgId());

    pfId_.push_back(id);
    pfPt_.push_back(pt);
    pfEta_.push_back(eta);
    pfPhi_.push_back(pfcand.phi());
    pfE_.push_back(pfcand.energy());
    pfM_.push_back(pfcand.mass());

    ++nPF_;
  }

  tree_->Fill();
}

void ParticleFlowAnalyser::beginJob() {
  converter_ = reco::PFCandidate();

  tree_ = fs_->make<TTree>("pftree", "packed candidates");

  tree_->Branch("nPF", &nPF_, "nPF/I");

  tree_->Branch("pfId", &pfId_);
  tree_->Branch("pfPt", &pfPt_);
  tree_->Branch("pfEta", &pfEta_);
  tree_->Branch("pfPhi", &pfPhi_);
  tree_->Branch("pfE", &pfE_);
  tree_->Branch("pfM", &pfM_);
}

void ParticleFlowAnalyser::endJob() {}

void ParticleFlowAnalyser::clear() {
  nPF_ = 0;

  pfId_.clear();
  pfPt_.clear();
  pfEta_.clear();
  pfPhi_.clear();
  pfE_.clear();
  pfM_.clear();
}

DEFINE_FWK_MODULE(ParticleFlowAnalyser);

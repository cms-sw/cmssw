// -*- C++ -*-
//
// Package:    Alignment/MillePedeAlignmentAlgorithm
// Class:      ZMuMuMassConstraintParameterFinder
//
/**\class ZMuMuMassConstraintParameterFinder ZMuMuMassConstraintParameterFinder.cc Alignment/MillePedeAlignmentAlgorithm/plugins/ZMuMuMassConstraintParameterFinder.cc

   Description: Determines the generator di-muon invariant-mass distribution

   Implementation:

   Generator muons are extracted from the genParticles collection and matched to
   Z-boson decays. The configured selection should match the selection applied
   on muon tracks in real data to obtain a "true" invariant mass distribution
   within the selection.

*/
//
// Original Author:  Gregor Mittag
//         Created:  Fri, 17 Jun 2016 09:42:56 GMT
//
//


// system include files
#include <cmath>
#include <cstdlib>
#include <vector>

// ROOT include files
#include "TTree.h"

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


//
// class declaration
//

class ZMuMuMassConstraintParameterFinder :
  public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit ZMuMuMassConstraintParameterFinder(const edm::ParameterSet&);
  ~ZMuMuMassConstraintParameterFinder() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  /// helper class containing information about a di-muon system
  class DiMuonInfo {
  public:
    DiMuonInfo(double, double);
    void setupTree(const std::string&, edm::Service<TFileService>&);
    void fill();
    std::vector<reco::GenParticle>& muons() { return muons_; }

  private:
    TTree* tree_{nullptr};
    std::vector<reco::GenParticle> muons_;
    double diMuonMass_{-1.0};
    int pdgMother_{0};
    bool passed_{false};

    const double minMassPair_;
    const double maxMassPair_;
  };

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;

  // particle IDs
  const int muonPdg_{13};
  const int zBosonPdg_{23};

  // muon cuts
  const double pMin_;
  const double ptMin_;
  const double etaMin_;
  const double etaMax_;
  const double phiMin_;
  const double phiMax_;

  // di-muon cuts
  const double minMassPair_;
  const double maxMassPair_;

  DiMuonInfo muonInfo_;
  DiMuonInfo muonInfoFromZ_;
};


//
// constructors and destructor
//
ZMuMuMassConstraintParameterFinder
::ZMuMuMassConstraintParameterFinder(const edm::ParameterSet& iConfig) :
  genParticlesToken_(consumes<reco::GenParticleCollection>(edm::InputTag{"genParticles"})),
  pMin_(iConfig.getParameter<double>("pMin")),
  ptMin_(iConfig.getParameter<double>("ptMin")),
  etaMin_(iConfig.getParameter<double>("etaMin")),
  etaMax_(iConfig.getParameter<double>("etaMax")),
  phiMin_(iConfig.getParameter<double>("phiMin")),
  phiMax_(iConfig.getParameter<double>("phiMax")),
  minMassPair_(iConfig.getParameter<double>("minMassPair")),
  maxMassPair_(iConfig.getParameter<double>("maxMassPair")),
  muonInfo_(minMassPair_, maxMassPair_),
  muonInfoFromZ_(minMassPair_, maxMassPair_)
{
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  muonInfo_.setupTree("di_muon", fs);
  muonInfoFromZ_.setupTree("di_muon_from_Z", fs);
}


ZMuMuMassConstraintParameterFinder
::~ZMuMuMassConstraintParameterFinder()
{
}


//
// member functions
//

// ------------ method called for each event  ------------
void
ZMuMuMassConstraintParameterFinder
::analyze(const edm::Event& iEvent, const edm::EventSetup&)
{
  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(genParticlesToken_, genParticles);

  for (const auto& particle: *(genParticles.product())) {
    if (std::abs(particle.pdgId()) != muonPdg_ || particle.status() != 1) continue;
    if (particle.p() < pMin_) continue;
    if (particle.pt() < ptMin_) continue;
    if (particle.eta() < etaMin_ || particle.eta() > etaMax_) continue;
    if (particle.phi() < phiMin_ || particle.phi() > phiMax_) continue;

    muonInfo_.muons().push_back(particle);
    if (particle.mother()->pdgId() == zBosonPdg_) {
      muonInfoFromZ_.muons().push_back(particle);
    }
  }

  muonInfo_.fill();
  muonInfoFromZ_.fill();
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ZMuMuMassConstraintParameterFinder
::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Extract information on 'Z -> mu mu' decays.");
  desc.add<double>("pMin", 3.0);
  desc.add<double>("ptMin", 15.0);
  desc.add<double>("etaMin", -3.0);
  desc.add<double>("etaMax", 3.0);
  desc.add<double>("phiMin", -3.1416);
  desc.add<double>("phiMax", 3.1416);
  desc.add<double>("minMassPair", 85.8);
  desc.add<double>("maxMassPair", 95.8);
  descriptions.add("zMuMuMassConstraintParameterFinder", desc);
}


// ------------ helper class definition ------------
ZMuMuMassConstraintParameterFinder
::DiMuonInfo
::DiMuonInfo(double minMass, double maxMass) :
  minMassPair_{minMass},
  maxMassPair_{maxMass}
{
}


void
ZMuMuMassConstraintParameterFinder
::DiMuonInfo
::setupTree(const std::string& name, edm::Service<TFileService>& fs)
{
  tree_ = fs->make<TTree>(name.c_str(), name.c_str());
  tree_->Branch("muons", &muons_);
  tree_->Branch("di_muon_mass", &diMuonMass_);
  tree_->Branch("pdg_mother", &pdgMother_);
  tree_->Branch("in_mass_window", &passed_);
}


void
ZMuMuMassConstraintParameterFinder
::DiMuonInfo
::fill()
{
  if (muons_.size() == 2) {
    diMuonMass_ = (muons_[0].p4() + muons_[1].p4()).M();
    pdgMother_ = muons_[0].mother()->pdgId();
    if (diMuonMass_ > minMassPair_ && diMuonMass_ < maxMassPair_) passed_ = true;
    tree_->Fill();
  }
  muons_.clear();
  diMuonMass_ = -1.0;
  pdgMother_ = 0;
  passed_ = false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZMuMuMassConstraintParameterFinder);

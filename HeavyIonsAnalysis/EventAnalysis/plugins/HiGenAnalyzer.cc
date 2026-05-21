// -*- C++ -*-
//
// Package:    HiGenAnalyzer
// Class:      HiGenAnalyzer
//
/**\class HiGenAnalyzer HiGenAnalyzer.cc

   Description: Analyzer that studies (HI) gen event info in miniAOD

   Implementation:
   This analyzer is copied from its AOD counterpart https://github.com/CmsHI/cmssw/blob/2c806f88506f7ef732b725142ae85750a31dc646/HeavyIonsAnalysis/EventAnalysis/src/HiEvtAnalyzer.cc and adapted for gen info in miniAOD
*/

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"

// root include file
#include "TFile.h"
#include "TTree.h"

using namespace std;

static const Int_t ETABINS = 3;  // Fix also in branch string

//
// class decleration
//

struct HydjetEvent {
  Int_t event;
  Float_t b;
  Float_t npart;
  Float_t ncoll;
  Float_t nhard;
  Float_t phi0;
  Float_t scale;

  Int_t n[ETABINS];
  Float_t ptav[ETABINS];

  Int_t mult;
  std::vector<Float_t> pt;
  std::vector<Float_t> eta;
  std::vector<Float_t> phi;
  std::vector<Int_t> pdg;
  std::vector<Int_t> chg;
  std::vector<Int_t> sube;
  std::vector<Int_t> sta;
  std::vector<Int_t> matchingID;
  std::vector<Int_t> nMothers;
  std::vector<std::vector<Int_t>> motherIndex;
  std::vector<Int_t> nDaughters;
  std::vector<std::vector<Int_t>> daughterIndex;

  Float_t vx;
  Float_t vy;
  Float_t vz;
  Float_t vr;
};

class HiGenAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HiGenAnalyzer(const edm::ParameterSet&);
  ~HiGenAnalyzer() override;

private:
  void beginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void endRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  vector<int> getMotherIdx(edm::Handle<std::vector<pat::PackedGenParticle>> parts, const pat::PackedGenParticle);
  vector<int> getDaughterIdx(edm::Handle<std::vector<pat::PackedGenParticle>> parts, const pat::PackedGenParticle);

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::SimVertexContainer> g4Label;

  TTree* hydjetTree_;
  HydjetEvent hev_;

  Bool_t doVertex_;
  Bool_t useHepMCProduct_;
  Bool_t doHI_;
  Bool_t doParticles_;
  std::vector<int> motherDaughterPDGsToSave_;

  Double_t etaMax_;
  Double_t ptMin_;
  Bool_t chargedOnly_;
  Bool_t stableOnly_;

  edm::EDGetTokenT<edm::HepMCProduct> src_;
  edm::EDGetTokenT<std::vector<pat::PackedGenParticle>> genParticleSrc_;
  edm::EDGetTokenT<edm::View<pat::PackedGenParticle>> signalPackedGenParticleSrc_;
  edm::EDGetTokenT<edm::GenHIEvent> genHIsrc_;
  edm::ESGetToken<HepPDT::ParticleDataTable, PDTRecord> tok_pdt_;
  edm::Service<TFileService> f;
};
//
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HiGenAnalyzer::HiGenAnalyzer(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  useHepMCProduct_ = iConfig.getUntrackedParameter<Bool_t>("useHepMCProduct", false);
  doHI_ = iConfig.getUntrackedParameter<Bool_t>("doHI", true);

  doVertex_ = iConfig.getUntrackedParameter<Bool_t>("doVertex", false);
  etaMax_ = iConfig.getUntrackedParameter<Double_t>("etaMax", 2);
  ptMin_ = iConfig.getUntrackedParameter<Double_t>("ptMin", 0);
  chargedOnly_ = iConfig.getUntrackedParameter<Bool_t>("chargedOnly", false);
  stableOnly_ = iConfig.getUntrackedParameter<Bool_t>("stableOnly", false);
  if (useHepMCProduct_) {
    src_ = consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("generator")));
  } else {
    genParticleSrc_ =
        consumes<std::vector<pat::PackedGenParticle>>(iConfig.getParameter<edm::InputTag>("genParticleSrc"));
    signalPackedGenParticleSrc_ =
        consumes<edm::View<pat::PackedGenParticle>>(iConfig.getParameter<edm::InputTag>("signalGenParticleSrc"));
  }
  if (doHI_) {
    genHIsrc_ =
        consumes<edm::GenHIEvent>(iConfig.getUntrackedParameter<edm::InputTag>("genHiSrc", edm::InputTag("heavyIon")));
  }
  tok_pdt_ = esConsumes<HepPDT::ParticleDataTable, PDTRecord>();
  doParticles_ = iConfig.getUntrackedParameter<Bool_t>("doParticles", true);
  vector<int> defaultPDGs;
  motherDaughterPDGsToSave_ = iConfig.getUntrackedParameter<std::vector<int>>("motherDaughterPDGsToSave", defaultPDGs);

  if (doVertex_) {
    g4Label = consumes<edm::SimVertexContainer>(iConfig.getUntrackedParameter<std::string>("ModuleLabel", "g4SimHits"));
  }
}

HiGenAnalyzer::~HiGenAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

vector<int> HiGenAnalyzer::getMotherIdx(edm::Handle<std::vector<pat::PackedGenParticle>> parts,
                                        const pat::PackedGenParticle pin) {
  vector<int> motherArr;
  if (!motherDaughterPDGsToSave_.empty()) {
    for (UInt_t i = 0; i < parts->size(); ++i) {
      const pat::PackedGenParticle& p = (*parts)[i];
      if (stableOnly_ && p.status() != 1)
        continue;
      if (p.pt() < ptMin_)
        continue;
      if (chargedOnly_ && p.charge() == 0)
        continue;
      bool saveFlag = false;
      for (unsigned int ipdg = 0; ipdg < motherDaughterPDGsToSave_.size(); ipdg++) {
        if (p.pdgId() == motherDaughterPDGsToSave_.at(ipdg))
          saveFlag = true;
      }
      if (!motherDaughterPDGsToSave_.empty() && saveFlag != true)
        continue;  //save all particles in vector unless vector is empty, then save all particles
      if (p.status() == 3)
        continue;  //don't match to the initial collision particles
      for (unsigned int idx = 0; idx < p.numberOfDaughters(); idx++) {
        //if (p.daughter(idx)->pt()*p.daughter(idx)->eta()*p.daughter(idx)->phi() == pin.pt()*pin.eta()*pin.phi()) motherArr.push_back(i);
        if (abs(p.daughter(idx)->pt() - pin.pt()) < 0.001 && abs(p.daughter(idx)->eta() - pin.eta()) < 0.001 &&
            abs(p.daughter(idx)->phi() - pin.phi()) < 0.001)
          motherArr.push_back(i);
      }
    }
  }
  if (motherArr.empty())
    motherArr.push_back(-999);
  return motherArr;
}

//----------------------------------------------------------

vector<int> HiGenAnalyzer::getDaughterIdx(edm::Handle<std::vector<pat::PackedGenParticle>> parts,
                                          const pat::PackedGenParticle pin) {
  vector<int> daughterArr;
  if (!motherDaughterPDGsToSave_.empty()) {
    for (UInt_t i = 0; i < parts->size(); ++i) {
      const pat::PackedGenParticle& p = (*parts)[i];
      if (stableOnly_ && p.status() != 1)
        continue;
      if (p.pt() < ptMin_)
        continue;
      if (chargedOnly_ && p.charge() == 0)
        continue;
      bool saveFlag = false;
      for (unsigned int ipdg = 0; ipdg < motherDaughterPDGsToSave_.size(); ipdg++) {
        if (p.pdgId() == motherDaughterPDGsToSave_.at(ipdg))
          saveFlag = true;
      }
      if (!motherDaughterPDGsToSave_.empty() && saveFlag != true)
        continue;  //save all particles in vector unless vector is empty, then save all particles
      if (p.status() == 3)
        continue;  //don't match to the initial collision particles
      for (unsigned int idx = 0; idx < p.numberOfMothers(); idx++) {
        //if (p.mother(idx)->pt()*p.mother(idx)->eta()*p.mother(idx)->phi() == pin.pt()*pin.eta()*pin.phi()) daughterArr.push_back(i);
        if (abs(p.mother(idx)->pt() - pin.pt()) < 0.001 && abs(p.mother(idx)->eta() - pin.eta()) < 0.001 &&
            abs(p.mother(idx)->phi() - pin.phi()) < 0.001)
          daughterArr.push_back(i);
      }
    }
  }
  if (daughterArr.empty())
    daughterArr.push_back(-999);
  return daughterArr;
}

// ------------ method called to for each event  ------------
void HiGenAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace HepMC;

  const HepPDT::ParticleDataTable* pdt = &iSetup.getData(tok_pdt_);

  hev_.pt.clear();
  hev_.eta.clear();
  hev_.phi.clear();
  hev_.pdg.clear();
  hev_.chg.clear();
  hev_.sube.clear();
  hev_.sta.clear();
  hev_.matchingID.clear();
  hev_.nMothers.clear();
  hev_.motherIndex.clear();
  hev_.nDaughters.clear();
  hev_.daughterIndex.clear();

  hev_.event = iEvent.id().event();
  for (Int_t ieta = 0; ieta < ETABINS; ++ieta) {
    hev_.n[ieta] = 0;
    hev_.ptav[ieta] = 0;
  }
  hev_.mult = 0;

  Double_t phi0 = 0;
  Double_t b = -1;
  Double_t scale = -1;
  Int_t npart = -1;
  Int_t ncoll = -1;
  Int_t nhard = -1;
  Double_t vx = -99;
  Double_t vy = -99;
  Double_t vz = -99;
  Double_t vr = -99;
  const GenEvent* evt;

  if (useHepMCProduct_) {
    Handle<edm::HepMCProduct> mc;
    iEvent.getByToken(src_, mc);
    evt = mc->GetEvent();
    scale = evt->event_scale();

    const HeavyIon* hi = evt->heavy_ion();
    if (hi) {
      b = hi->impact_parameter();
      npart = hi->Npart_proj() + hi->Npart_targ();
      ncoll = hi->Ncoll();
      nhard = hi->Ncoll_hard();
      phi0 = hi->event_plane_angle();
    }

    HepMC::GenEvent::particle_const_iterator begin = evt->particles_begin();
    HepMC::GenEvent::particle_const_iterator end = evt->particles_end();
    int nparticles = -1;
    for (HepMC::GenEvent::particle_const_iterator it = begin; it != end; ++it) {
      nparticles++;
      if ((*it)->momentum().perp() < ptMin_)
        continue;
      if (abs((*it)->momentum().eta()) > etaMax_)
        continue;
      Int_t pdg_id = (*it)->pdg_id();
      Float_t eta = (*it)->momentum().eta();
      Float_t phi = (*it)->momentum().phi();
      Float_t pt = (*it)->momentum().perp();
      const ParticleData* part = pdt->particle(pdg_id);
      Int_t charge = static_cast<Int_t>(part->charge());
      if (chargedOnly_ && charge == 0)
        continue;

      hev_.pt.push_back(pt);
      hev_.eta.push_back(eta);
      hev_.phi.push_back(phi);
      hev_.pdg.push_back(pdg_id);
      hev_.chg.push_back(charge);
      hev_.sta.push_back((*it)->status());
      hev_.matchingID.push_back(nparticles);

      eta = abs(eta);
      Int_t etabin = 0;
      if (eta > 0.5)
        etabin = 1;
      if (eta > 1.)
        etabin = 2;
      if (eta < 2.) {
        hev_.ptav[etabin] += pt;
        ++(hev_.n[etabin]);
      }
      ++(hev_.mult);
    }
  } else {
    edm::Handle<std::vector<pat::PackedGenParticle>> parts;
    iEvent.getByToken(genParticleSrc_, parts);

    edm::Handle<edm::View<pat::PackedGenParticle>> signalPackedGenParticles;
    bool hasSignalPackedGen = iEvent.getByToken(signalPackedGenParticleSrc_, signalPackedGenParticles);

    for (UInt_t i = 0; i < parts->size(); ++i) {
      //const reco::GenParticle& p = (*parts)[i];
      const pat::PackedGenParticle& p = (*parts)[i];
      if (stableOnly_ && p.status() != 1)
        continue;
      if (p.pt() < ptMin_)
        continue;
      if (abs(p.eta()) > etaMax_)
        continue;
      if (chargedOnly_ && p.charge() == 0)
        continue;
      hev_.pt.push_back(p.pt());
      hev_.eta.push_back(p.eta());
      hev_.phi.push_back(p.phi());
      hev_.pdg.push_back(p.pdgId());
      hev_.chg.push_back(p.charge());
      // collisionId_ is not kept in pat::PackedGenParticle, use "packedGenParticlesSignal" (added by https://github.com/cms-sw/cmssw/pull/32668/) to tag particles from signal process
      if (hasSignalPackedGen) {
        int tmpSube = 1;
        for (auto pSig = signalPackedGenParticles->begin(); pSig != signalPackedGenParticles->end(); ++pSig) {
          if (&(*pSig) == &(*parts)[i]) {
            tmpSube = 0;
            break;
          }
        }
        hev_.sube.push_back(tmpSube);
      } else {
        hev_.sube.push_back(-999);
      }
      hev_.sta.push_back(p.status());
      hev_.matchingID.push_back(i);
      hev_.nMothers.push_back(p.numberOfMothers());
      vector<int> tempMothers = getMotherIdx(parts, p);
      hev_.motherIndex.push_back(tempMothers);
      hev_.nDaughters.push_back(p.numberOfDaughters());
      vector<int> tempDaughters = getDaughterIdx(parts, p);
      hev_.daughterIndex.push_back(tempDaughters);
      Double_t eta = abs(p.eta());

      Int_t etabin = 0;
      if (eta > 0.5)
        etabin = 1;
      if (eta > 1.)
        etabin = 2;
      if (eta < 2.) {
        hev_.ptav[etabin] += p.pt();
        ++(hev_.n[etabin]);
      }
      ++(hev_.mult);
    }
    if (doHI_) {
      edm::Handle<edm::GenHIEvent> higen;
      iEvent.getByToken(genHIsrc_, higen);

      b = higen->b();
      npart = higen->Npart();
      ncoll = higen->Ncoll();
      nhard = higen->Nhard();
      phi0 = higen->evtPlane();
    }
  }

  if (doVertex_) {
    edm::Handle<edm::SimVertexContainer> simVertices;
    iEvent.getByToken(g4Label, simVertices);

    if (!simVertices.isValid())
      throw cms::Exception("FatalError") << "No vertices found\n";

    edm::SimVertexContainer::const_iterator it = simVertices->begin();
    if (it != simVertices->end()) {
      SimVertex vertex = (*it);
      vx = vertex.position().x();
      vy = vertex.position().y();
      vz = vertex.position().z();
      vr = vertex.position().rho();
    }
  }

  for (Int_t i = 0; i < 3; ++i) {
    hev_.ptav[i] = hev_.ptav[i] / hev_.n[i];
  }

  hev_.b = b;
  hev_.scale = scale;
  hev_.npart = npart;
  hev_.ncoll = ncoll;
  hev_.nhard = nhard;
  hev_.phi0 = phi0;
  hev_.vx = vx;
  hev_.vy = vy;
  hev_.vz = vz;
  hev_.vr = vr;

  hydjetTree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void HiGenAnalyzer::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {}

// ------------ method called once each job just after finishing event loop  ------------
void HiGenAnalyzer::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {}

void HiGenAnalyzer::beginJob() {
  hydjetTree_ = f->make<TTree>("hi", "Tree of Hi gen Event");
  hydjetTree_->Branch("event", &hev_.event, "event/I");
  if (doHI_) {
    hydjetTree_->Branch("b", &hev_.b, "b/F");
    hydjetTree_->Branch("npart", &hev_.npart, "npart/F");
    hydjetTree_->Branch("ncoll", &hev_.ncoll, "ncoll/F");
    hydjetTree_->Branch("nhard", &hev_.nhard, "nhard/F");
    hydjetTree_->Branch("phi0", &hev_.phi0, "phi0/F");
  }
  hydjetTree_->Branch("scale", &hev_.scale, "scale/F");

  hydjetTree_->Branch("n", hev_.n, "n[3]/I");
  hydjetTree_->Branch("ptav", hev_.ptav, "ptav[3]/F");

  if (doParticles_) {
    hydjetTree_->Branch("mult", &hev_.mult, "mult/I");
    hydjetTree_->Branch("pt", &hev_.pt);
    hydjetTree_->Branch("eta", &hev_.eta);
    hydjetTree_->Branch("phi", &hev_.phi);
    hydjetTree_->Branch("pdg", &hev_.pdg);
    hydjetTree_->Branch("chg", &hev_.chg);
    hydjetTree_->Branch("matchingID", &hev_.matchingID);
    hydjetTree_->Branch("nMothers", &hev_.nMothers);
    hydjetTree_->Branch("motherIdx", &hev_.motherIndex);
    hydjetTree_->Branch("nDaughters", &hev_.nDaughters);
    hydjetTree_->Branch("daughterIdx", &hev_.daughterIndex);
    if (!stableOnly_) {
      hydjetTree_->Branch("sta", &hev_.sta);
    }
    hydjetTree_->Branch("sube", &hev_.sube);

    if (doVertex_){
      hydjetTree_->Branch("vx", &hev_.vx, "vx/F");
      hydjetTree_->Branch("vy", &hev_.vy, "vy/F");
      hydjetTree_->Branch("vz", &hev_.vz, "vz/F");
      hydjetTree_->Branch("vr", &hev_.vr, "vr/F");
    }
    
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void HiGenAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(HiGenAnalyzer);

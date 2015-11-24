// -*- C++ -*-
//
// Package:    HiGenAnalyzer
// Class:      HiGenAnalyzer
//
/**\class HiGenAnalyzer HiGenAnalyzer.cc

   Description: Analyzer that studies (HI) gen event info

   Implementation:
   <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz, Frank Ma
//         Created:  Tue Dec 18 09:44:41 EST 2007
// $Id: HiGenAnalyzer.cc,v 1.9 2012/09/28 20:10:52 yilmaz Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <string>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// root include file
#include "TFile.h"
#include "TNtuple.h"
#include "TMath.h"
#include <vector>

using namespace std;

//static const Int_t MAXPARTICLES = 200000;
//static const Int_t MAXVTX = 1000;
static const Int_t ETABINS = 3; // Fix also in branch string

//
// class decleration
//

struct HydjetEvent{

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
  std::vector<std::vector<Int_t> > motherIndex;
  std::vector<Int_t> nDaughters;
  std::vector<std::vector<Int_t> > daughterIndex;

  Float_t vx;
  Float_t vy;
  Float_t vz;
  Float_t vr;

};

class HiGenAnalyzer : public edm::EDAnalyzer {
public:
  explicit HiGenAnalyzer(const edm::ParameterSet&);
  ~HiGenAnalyzer();


private:
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  vector<int> getMotherIdx(edm::Handle<reco::GenParticleCollection> parts, const reco::GenParticle);
  vector<int> getDaughterIdx(edm::Handle<reco::GenParticleCollection> parts, const reco::GenParticle);

  // ----------member data ---------------------------

  //std::string g4Label;
  edm::EDGetTokenT<edm::SimVertexContainer> g4Label;

  TTree* hydjetTree_;
  HydjetEvent hev_;

  TNtuple *nt;

  Bool_t doVertex_;
  Bool_t useHepMCProduct_;
  Bool_t doHI_;
  Bool_t doParticles_;
  std::vector<int> motherDaughterPDGsToSave_;

  Double_t etaMax_;
  Double_t ptMin_;
  Bool_t chargedOnly_;
  Bool_t stableOnly_;

  // edm::InputTag src_;
  // edm::InputTag genParticleSrc_;
  // edm::InputTag genHIsrc_;

  edm::EDGetTokenT<edm::HepMCProduct> src_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticleSrc_;
  edm::EDGetTokenT<edm::GenHIEvent> genHIsrc_;

  edm::ESHandle < ParticleDataTable > pdt;
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
HiGenAnalyzer::HiGenAnalyzer(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  useHepMCProduct_ = iConfig.getUntrackedParameter<Bool_t>("useHepMCProduct", false);
  doHI_ = iConfig.getUntrackedParameter<Bool_t>("doHI", true);

  doVertex_ = iConfig.getUntrackedParameter<Bool_t>("doVertex", false);
  etaMax_ = iConfig.getUntrackedParameter<Double_t>("etaMax", 2);
  ptMin_ = iConfig.getUntrackedParameter<Double_t>("ptMin", 0);
  chargedOnly_ = iConfig.getUntrackedParameter<Bool_t>("chargedOnly", false);
  stableOnly_ = iConfig.getUntrackedParameter<Bool_t>("stableOnly", false);
  if(useHepMCProduct_){
    src_ = consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter<edm::InputTag>("src",edm::InputTag("generator")));
  } else {
    genParticleSrc_ = consumes<reco::GenParticleCollection>(iConfig.getUntrackedParameter<edm::InputTag>("genParticleSrc",edm::InputTag("hiGenParticles")));
  }
  if(doHI_){
    genHIsrc_ = consumes<edm::GenHIEvent>(iConfig.getUntrackedParameter<edm::InputTag>("genHiSrc",edm::InputTag("heavyIon")));
  }
  doParticles_ = iConfig.getUntrackedParameter<Bool_t>("doParticles", true);
  vector<int> defaultPDGs;
  motherDaughterPDGsToSave_ = iConfig.getUntrackedParameter<std::vector<int> >("motherDaughterPDGsToSave",defaultPDGs);

  if(doVertex_){
    g4Label = consumes<edm::SimVertexContainer>(iConfig.getUntrackedParameter<std::string>("ModuleLabel","g4SimHits"));
  }

}


HiGenAnalyzer::~HiGenAnalyzer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

vector<int> HiGenAnalyzer::getMotherIdx(edm::Handle<reco::GenParticleCollection> parts, const reco::GenParticle pin){

  vector<int> motherArr;
  if(motherDaughterPDGsToSave_.size() != 0 ) {
    for(UInt_t i = 0; i < parts->size(); ++i){
      const reco::GenParticle& p = (*parts)[i];
      if (stableOnly_ && p.status()!=1) continue;
      if (p.pt()<ptMin_) continue;
      if (chargedOnly_&&p.charge()==0) continue;
      bool saveFlag=false;
      for(unsigned int ipdg=0; ipdg<motherDaughterPDGsToSave_.size(); ipdg++){
	if(p.pdgId() == motherDaughterPDGsToSave_.at(ipdg)) saveFlag=true;
      }
      if(motherDaughterPDGsToSave_.size()>0 && saveFlag!=true) continue; //save all particles in vector unless vector is empty, then save all particles
      if (p.status()==3) continue; //don't match to the initial collision particles
      for (unsigned int idx=0; idx<p.numberOfDaughters(); idx++){
	//if (p.daughter(idx)->pt()*p.daughter(idx)->eta()*p.daughter(idx)->phi() == pin.pt()*pin.eta()*pin.phi()) motherArr.push_back(i);
	if(fabs(p.daughter(idx)->pt()-pin.pt())<0.001 && fabs(p.daughter(idx)->eta()-pin.eta())<0.001 && fabs(p.daughter(idx)->phi()-pin.phi())<0.001) motherArr.push_back(i);
      }
    }
  }
  if(motherArr.size()==0) motherArr.push_back(-999);
  return motherArr;
}

//----------------------------------------------------------

vector<int> HiGenAnalyzer::getDaughterIdx(edm::Handle<reco::GenParticleCollection> parts, const reco::GenParticle pin){

  vector<int> daughterArr;
  if(motherDaughterPDGsToSave_.size() != 0 ) {
    for(UInt_t i = 0; i < parts->size(); ++i){
      const reco::GenParticle& p = (*parts)[i];
      if (stableOnly_ && p.status()!=1) continue;
      if (p.pt()<ptMin_) continue;
      if (chargedOnly_&&p.charge()==0) continue;
      bool saveFlag=false;
      for(unsigned int ipdg=0; ipdg<motherDaughterPDGsToSave_.size(); ipdg++){
	if(p.pdgId() == motherDaughterPDGsToSave_.at(ipdg)) saveFlag=true;
      }
      if(motherDaughterPDGsToSave_.size()>0 && saveFlag!=true) continue; //save all particles in vector unless vector is empty, then save all particles
      if (p.status()==3) continue; //don't match to the initial collision particles
      for(unsigned int idx=0; idx<p.numberOfMothers(); idx++){
	//if (p.mother(idx)->pt()*p.mother(idx)->eta()*p.mother(idx)->phi() == pin.pt()*pin.eta()*pin.phi()) daughterArr.push_back(i);
	if(fabs(p.mother(idx)->pt()-pin.pt())<0.001 && fabs(p.mother(idx)->eta()-pin.eta())<0.001 && fabs(p.mother(idx)->phi()-pin.phi())<0.001) daughterArr.push_back(i);
      }
    }
  }
  if(daughterArr.size()==0) daughterArr.push_back(-999);
  return daughterArr;
}

// ------------ method called to for each event  ------------
void
HiGenAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace HepMC;

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
  for(Int_t ieta = 0; ieta < ETABINS; ++ieta){
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

  Int_t nmix = -1;
  Int_t np = 0;
  Int_t sig = -1;
  Int_t src = -1;

  if(useHepMCProduct_){
    Handle<edm::HepMCProduct> mc;
    iEvent.getByToken(src_,mc);
    evt = mc->GetEvent();
    scale = evt->event_scale();

    const HeavyIon* hi = evt->heavy_ion();
    if(hi){
      b = hi->impact_parameter();
      npart = hi->Npart_proj()+hi->Npart_targ();
      ncoll = hi->Ncoll();
      nhard = hi->Ncoll_hard();
      phi0 = hi->event_plane_angle();
    }

    src = evt->particles_size();

    HepMC::GenEvent::particle_const_iterator begin = evt->particles_begin();
    HepMC::GenEvent::particle_const_iterator end = evt->particles_end();
    int nparticles=-1;
    for(HepMC::GenEvent::particle_const_iterator it = begin; it != end; ++it){
      nparticles++;
      if ((*it)->momentum().perp()<ptMin_) continue;
      //	if((*it)->status() == 1){
      Int_t pdg_id = (*it)->pdg_id();
      Float_t eta = (*it)->momentum().eta();
      Float_t phi = (*it)->momentum().phi();
      Float_t pt = (*it)->momentum().perp();
      const ParticleData * part = pdt->particle(pdg_id );
      Int_t charge = static_cast<Int_t>(part->charge());
      if (chargedOnly_&&charge==0) continue;

      hev_.pt.push_back( pt);
      hev_.eta.push_back( eta);
      hev_.phi.push_back( phi);
      hev_.pdg.push_back( pdg_id);
      hev_.chg.push_back( charge);
      hev_.sta.push_back( (*it)->status());
      hev_.matchingID.push_back( nparticles);

      eta = fabs(eta);
      Int_t etabin = 0;
      if(eta > 0.5) etabin = 1;
      if(eta > 1.) etabin = 2;
      if(eta < 2.){
	hev_.ptav[etabin] += pt;
	++(hev_.n[etabin]);
      }
      ++(hev_.mult);
      // if(hev_.mult >= MAXPARTICLES)
      // 	edm::LogError("Number of genparticles exceeds array bounds.");
    }
  }else{
    edm::Handle<reco::GenParticleCollection> parts;
    iEvent.getByToken(genParticleSrc_,parts);
    for(UInt_t i = 0; i < parts->size(); ++i){
      const reco::GenParticle& p = (*parts)[i];
      if (stableOnly_ && p.status()!=1) continue;
      if (p.pt()<ptMin_) continue;
      if (chargedOnly_&&p.charge()==0) continue;
      hev_.pt.push_back( p.pt());
      hev_.eta.push_back( p.eta());
      hev_.phi.push_back( p.phi());
      hev_.pdg.push_back( p.pdgId());
      hev_.chg.push_back( p.charge());
      hev_.sube.push_back( p.collisionId());
      hev_.sta.push_back( p.status());
      hev_.matchingID.push_back( i);
      hev_.nMothers.push_back( p.numberOfMothers());
      vector<int> tempMothers = getMotherIdx(parts, p);
      hev_.motherIndex.push_back(tempMothers);
      // for(unsigned int imother=0; imother<tempMothers.size(); imother++){
      // 	hev_.motherIndex[hev_.mult].push_back(tempMothers.at(imother));
      // }
      hev_.nDaughters.push_back( p.numberOfDaughters());
      vector<int> tempDaughters = getDaughterIdx(parts, p);
      hev_.daughterIndex.push_back(tempDaughters);
      // for(unsigned int idaughter=0; idaughter<tempDaughters.size(); idaughter++){
      // 	hev_.daughterIndex[hev_.mult].push_back(tempDaughters.at(idaughter));
      // }
      Double_t eta = fabs(p.eta());

      Int_t etabin = 0;
      if(eta > 0.5) etabin = 1;
      if(eta > 1.) etabin = 2;
      if(eta < 2.){
	hev_.ptav[etabin] += p.pt();
	++(hev_.n[etabin]);
      }
      ++(hev_.mult);
      // if(hev_.mult >= MAXPARTICLES)
      // 	edm::LogError("Number of genparticles exceeds array bounds.");
    }
    if(doHI_){
      edm::Handle<edm::GenHIEvent> higen;
      iEvent.getByToken(genHIsrc_,higen);

      b = higen->b();
      npart = higen->Npart();
      ncoll = higen->Ncoll();
      nhard = higen->Nhard();
      phi0 = higen->evtPlane();

    }
  }

  if(doVertex_){
    edm::Handle<edm::SimVertexContainer> simVertices;
    // iEvent.getByType<edm::SimVertexContainer>(simVertices);
    iEvent.getByToken(g4Label,simVertices);

    if (! simVertices.isValid() ) throw cms::Exception("FatalError") << "No vertices found\n";
    //Int_t inum = 0;

    edm::SimVertexContainer::const_iterator it=simVertices->begin();
    if(it != simVertices->end()){
      SimVertex vertex = (*it);
      //cout<<" Vertex position "<< inum <<" " << vertex.position().rho()<<" "<<vertex.position().z()<<endl;
      vx = vertex.position().x();
      vy = vertex.position().y();
      vz = vertex.position().z();
      vr = vertex.position().rho();
    }
  }

  for(Int_t i = 0; i<3; ++i){
    hev_.ptav[i] = hev_.ptav[i]/hev_.n[i];
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

  nt->Fill(nmix,np,src,sig);

  hydjetTree_->Fill();

}


// ------------ method called once each job just before starting event loop  ------------
void
HiGenAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& iSetup)
{
  iSetup.getData(pdt);
}

void
HiGenAnalyzer::beginJob()
{

  nt = f->make<TNtuple>("nt","Mixing Analysis","mix:np:src:sig");

  hydjetTree_ = f->make<TTree>("hi","Tree of Hi gen Event");
  hydjetTree_->Branch("event",&hev_.event,"event/I");
  hydjetTree_->Branch("b",&hev_.b,"b/F");
  hydjetTree_->Branch("npart",&hev_.npart,"npart/F");
  hydjetTree_->Branch("ncoll",&hev_.ncoll,"ncoll/F");
  hydjetTree_->Branch("nhard",&hev_.nhard,"nhard/F");
  hydjetTree_->Branch("phi0",&hev_.phi0,"phi0/F");
  hydjetTree_->Branch("scale",&hev_.scale,"scale/F");

  hydjetTree_->Branch("n",hev_.n,"n[3]/I");
  hydjetTree_->Branch("ptav",hev_.ptav,"ptav[3]/F");

  if(doParticles_){

    hydjetTree_->Branch("mult",&hev_.mult,"mult/I");
    hydjetTree_->Branch("pt",&hev_.pt);
    hydjetTree_->Branch("eta",&hev_.eta);
    hydjetTree_->Branch("phi",&hev_.phi);
    hydjetTree_->Branch("pdg",&hev_.pdg);
    hydjetTree_->Branch("chg",&hev_.chg);
    hydjetTree_->Branch("matchingID",&hev_.matchingID);
    hydjetTree_->Branch("nMothers",&hev_.nMothers);
    hydjetTree_->Branch("motherIdx",&hev_.motherIndex);
    hydjetTree_->Branch("nDaughters",&hev_.nDaughters);
    hydjetTree_->Branch("daughterIdx",&hev_.daughterIndex);
    if(!stableOnly_){
      hydjetTree_->Branch("sta",&hev_.sta);
    }
    hydjetTree_->Branch("sube",&hev_.sube);

    hydjetTree_->Branch("vx",&hev_.vx,"vx/F");
    hydjetTree_->Branch("vy",&hev_.vy,"vy/F");
    hydjetTree_->Branch("vz",&hev_.vz,"vz/F");
    hydjetTree_->Branch("vr",&hev_.vr,"vr/F");
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void
HiGenAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiGenAnalyzer);

// -*- C++ -*-
//
// Package:    AMPTAnalyzer
// Class:      AMPTAnalyzer
//

// system include files
#include <memory>
#include <iostream>
#include <string>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

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

#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// root include file
#include "TFile.h"
#include "TNtuple.h"

static const int MAXPARTICLES = 5000000;
static const int ETABINS = 3;  // Fix also in branch string

//
// class decleration
//

struct AMPTEvent {
  int event;
  float b;
  float npart;
  float ncoll;
  float nhard;
  float phi0;

  int n[ETABINS];
  float ptav[ETABINS];

  int mult;
  float pt[MAXPARTICLES];
  float eta[MAXPARTICLES];
  float phi[MAXPARTICLES];
  int pdg[MAXPARTICLES];
  int chg[MAXPARTICLES];

  float vx;
  float vy;
  float vz;
  float vr;
};

class AMPTAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit AMPTAnalyzer(const edm::ParameterSet&);
  ~AMPTAnalyzer() = default;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  // ----------member data ---------------------------

  std::ofstream out_b;
  std::string fBFileName;

  std::ofstream out_n;
  std::string fNFileName;

  std::ofstream out_m;
  std::string fMFileName;

  TTree* hydjetTree_;
  AMPTEvent hev_;

  TNtuple* nt;

  std::string output;  // Output filename

  bool doAnalysis_;
  bool printLists_;
  bool doCF_;
  bool doVertex_;
  double etaMax_;
  double ptMin_;
  edm::InputTag simVerticesTag_;

  const edm::ESGetToken<HepPDT::ParticleDataTable, PDTRecord> pdtToken_;
  const edm::EDGetTokenT<edm::HepMCProduct> mcToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> simVertToken_;
  edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct> > cfToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
AMPTAnalyzer::AMPTAnalyzer(const edm::ParameterSet& iConfig)
    : pdtToken_(esConsumes<HepPDT::ParticleDataTable, PDTRecord>()),
      mcToken_(consumes<edm::HepMCProduct>(
          iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("VtxSmeared")))) {
  usesResource(TFileService::kSharedResource);
  //now do what ever initialization is needed
  fBFileName = iConfig.getUntrackedParameter<std::string>("output_b", "b_values.txt");
  fNFileName = iConfig.getUntrackedParameter<std::string>("output_n", "n_values.txt");
  fMFileName = iConfig.getUntrackedParameter<std::string>("output_m", "m_values.txt");
  doAnalysis_ = iConfig.getUntrackedParameter<bool>("doAnalysis", true);
  printLists_ = iConfig.getUntrackedParameter<bool>("printLists", false);
  doCF_ = iConfig.getUntrackedParameter<bool>("doMixed", false);
  if (doCF_)
    cfToken_ = consumes<CrossingFrame<edm::HepMCProduct> >(edm::InputTag("mix", "source"));
  doVertex_ = iConfig.getUntrackedParameter<bool>("doVertex", false);
  if (doVertex_) {
    simVertToken_ = consumes<edm::SimVertexContainer>(iConfig.getParameter<edm::InputTag>("simVerticesTag"));
  }
  etaMax_ = iConfig.getUntrackedParameter<double>("etaMax", 2);
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin", 0);
}

//
// member functions
//

// ------------ method called to for each event  ------------
void AMPTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HepPDT::ParticleDataTable* pdt = &iSetup.getData(pdtToken_);

  hev_.event = iEvent.id().event();
  for (int ieta = 0; ieta < ETABINS; ++ieta) {
    hev_.n[ieta] = 0;
    hev_.ptav[ieta] = 0;
  }
  hev_.mult = 0;

  double phi0 = 0;
  double b = -1;
  int npart = -1;
  int ncoll = -1;
  int nhard = -1;
  double vx = -99;
  double vy = -99;
  double vz = -99;
  double vr = -99;
  const HepMC::GenEvent* evt;
  const HepMC::GenEvent* evt2;

  int nmix = -1;
  int np = 0;
  int sig = -1;
  int src = -1;

  if (doCF_) {
    const edm::Handle<CrossingFrame<edm::HepMCProduct> >& cf = iEvent.getHandle(cfToken_);

    MixCollection<edm::HepMCProduct> mix(cf.product());

    nmix = mix.size();

    edm::LogVerbatim("AMPTAnalysis") << "Mix Collection Size: " << mix;

    MixCollection<edm::HepMCProduct>::iterator mbegin = mix.begin();
    MixCollection<edm::HepMCProduct>::iterator mend = mix.end();

    for (MixCollection<edm::HepMCProduct>::iterator mixit = mbegin; mixit != mend; ++mixit) {
      const HepMC::GenEvent* subevt = (*mixit).GetEvent();
      int all = subevt->particles_size();
      np += all;

      /*
       HepMC::GenEvent::particle_const_iterator begin = subevt->particles_begin();
       HepMC::GenEvent::particle_const_iterator end = subevt->particles_end();
       for(HepMC::GenEvent::particle_const_iterator it = begin; it != end; ++it){
	 if((*it)->status() == 1){
           float pdg_id = (*it)->pdg_id();
           float eta = (*it)->momentum().eta();
           float pt = (*it)->momentum().perp();
           const ParticleData * part = pdt->particle(pdg_id );
           float charge = part->charge();
	   }
	 }
       }

       */
    }
  }

  const edm::Handle<edm::HepMCProduct>& mc = iEvent.getHandle(mcToken_);
  evt = mc->GetEvent();

  const edm::Handle<edm::HepMCProduct>& mc2 = iEvent.getHandle(mcToken_);
  evt2 = mc2->GetEvent();

  const HepMC::HeavyIon* hi = evt->heavy_ion();
  if (hi) {
    b = hi->impact_parameter();
    npart = hi->Npart_proj() + hi->Npart_targ();
    ncoll = hi->Ncoll();
    nhard = hi->Ncoll_hard();
    phi0 = hi->event_plane_angle();

    if (printLists_) {
      out_b << b << std::endl;
      out_n << npart << std::endl;
    }
  }

  src = evt->particles_size();
  sig = evt2->particles_size();

  HepMC::GenEvent::particle_const_iterator begin = evt->particles_begin();
  HepMC::GenEvent::particle_const_iterator end = evt->particles_end();
  for (HepMC::GenEvent::particle_const_iterator it = begin; it != end; ++it) {
    if ((*it)->status() == 1) {
      //if((*it)->status() != 1) edm::LogVerbatim("AMPTAnalysis") << (*it)->status();
      int pdg_id = (*it)->pdg_id();
      float eta = (*it)->momentum().eta();
      float phi = (*it)->momentum().phi();
      float pt = (*it)->momentum().perp();
      const ParticleData* part = pdt->particle(pdg_id);
      int charge = static_cast<int>(part->charge());

      hev_.pt[hev_.mult] = pt;
      hev_.eta[hev_.mult] = eta;
      hev_.phi[hev_.mult] = phi;
      hev_.pdg[hev_.mult] = pdg_id;
      hev_.chg[hev_.mult] = charge;

      eta = fabs(eta);
      int etabin = 0;
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
  }
  //   }

  if (doVertex_) {
    const edm::Handle<edm::SimVertexContainer>& simVertices = iEvent.getHandle(simVertToken_);

    if (!simVertices.isValid())
      throw cms::Exception("FatalError") << "No vertices found\n";
    int inum = 0;

    edm::SimVertexContainer::const_iterator it = simVertices->begin();
    SimVertex vertex = (*it);
    edm::LogVerbatim("AMPTAnalysis") << " Vertex position " << inum << " " << vertex.position().rho() << " "
                                     << vertex.position().z();
    vx = vertex.position().x();
    vy = vertex.position().y();
    vz = vertex.position().z();
    vr = vertex.position().rho();
  }

  for (int i = 0; i < 3; ++i) {
    hev_.ptav[i] = hev_.ptav[i] / hev_.n[i];
  }

  hev_.b = b;
  hev_.npart = npart;
  hev_.ncoll = ncoll;
  hev_.nhard = nhard;
  hev_.phi0 = phi0;
  hev_.vx = vx;
  hev_.vy = vy;
  hev_.vz = vz;
  hev_.vr = vr;

  nt->Fill(nmix, np, src, sig);

  hydjetTree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void AMPTAnalyzer::beginJob() {
  if (printLists_) {
    out_b.open(fBFileName.c_str());
    if (out_b.good() == false)
      throw cms::Exception("BadFile") << "Can\'t open file " << fBFileName;
    out_n.open(fNFileName.c_str());
    if (out_n.good() == false)
      throw cms::Exception("BadFile") << "Can\'t open file " << fNFileName;
    out_m.open(fMFileName.c_str());
    if (out_m.good() == false)
      throw cms::Exception("BadFile") << "Can\'t open file " << fMFileName;
  }

  if (doAnalysis_) {
    edm::Service<TFileService> f;
    nt = f->make<TNtuple>("nt", "Mixing Analysis", "mix:np:src:sig");

    hydjetTree_ = f->make<TTree>("hi", "Tree of AMPT Events");
    hydjetTree_->Branch("event", &hev_.event, "event/I");
    hydjetTree_->Branch("b", &hev_.b, "b/F");
    hydjetTree_->Branch("npart", &hev_.npart, "npart/F");
    hydjetTree_->Branch("ncoll", &hev_.ncoll, "ncoll/F");
    hydjetTree_->Branch("nhard", &hev_.nhard, "nhard/F");
    hydjetTree_->Branch("phi0", &hev_.phi0, "phi0/F");

    hydjetTree_->Branch("n", hev_.n, "n[3]/I");
    hydjetTree_->Branch("ptav", hev_.ptav, "ptav[3]/F");

    hydjetTree_->Branch("mult", &hev_.mult, "mult/I");
    hydjetTree_->Branch("pt", hev_.pt, "pt[mult]/F");
    hydjetTree_->Branch("eta", hev_.eta, "eta[mult]/F");
    hydjetTree_->Branch("phi", hev_.phi, "phi[mult]/F");
    hydjetTree_->Branch("pdg", hev_.pdg, "pdg[mult]/I");
    hydjetTree_->Branch("chg", hev_.chg, "chg[mult]/I");

    hydjetTree_->Branch("vx", &hev_.vx, "vx/F");
    hydjetTree_->Branch("vy", &hev_.vy, "vy/F");
    hydjetTree_->Branch("vz", &hev_.vz, "vz/F");
    hydjetTree_->Branch("vr", &hev_.vr, "vr/F");
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(AMPTAnalyzer);

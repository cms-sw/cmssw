// ============================================================================
// ExternalDecayInjector
//
// This EDProducer reads pre-generated decay information from a ROOT TTree
// (produced, for example, by the Pluto generator) and injects the
// corresponding daughter particles into an existing HepMC::GenEvent.
//
// Workflow:
// 1. Reads a random entry from the input ROOT file containing nDaughters
//    (pt, eta, phi, energy, PDG ID) for each decay.
// 2. Loops over all particles in the HepMC event and selects mothers
//    matching motherID_ and kinematic cuts (Pt, Eta).
// 3. Boosts the daughters according to the mother momentum and adds them
//    to a new decay vertex in the HepMC event.
// 4. Marks the mother as decayed (status = 2) and updates the event.
//
// ============================================================================
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepMC/GenEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <ctime>
#include "TRandom3.h"

#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"

#include <memory>
#include <chrono>
#include <vector>

#include <random>

UInt_t getRandomSeed() {
  std::random_device rd;
  return static_cast<UInt_t>(rd());
}

class ExternalDecayInjector : public edm::stream::EDProducer<> {
public:
  explicit ExternalDecayInjector(const edm::ParameterSet&);
  ~ExternalDecayInjector() override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<edm::HepMCProduct> token_;
  Int_t motherID_;
  double motherMinPt_;
  double motherMaxEta_;
  Int_t nDaughters_;
  Int_t maxMotherDecay_;
  std::string rootFileName_;
  std::string treeName_;

  TFile* f = nullptr;
  TTree* tree = nullptr;
  Long64_t nentries;

  TRandom3 rand;

  // dimensioni massime per stato finale
  static constexpr Int_t MAX_DAUGHTERS = 5;

  Double_t pt[MAX_DAUGHTERS], eta[MAX_DAUGHTERS], phi[MAX_DAUGHTERS], energy[MAX_DAUGHTERS];
  Int_t pdgid[MAX_DAUGHTERS];

  void loadRootFile();
};

ExternalDecayInjector::ExternalDecayInjector(const edm::ParameterSet& cfg)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(cfg.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      motherID_(cfg.getParameter<Int_t>("motherID")),
      motherMinPt_(cfg.getParameter<double>("motherMinPt")),
      motherMaxEta_(cfg.getParameter<double>("motherMaxEta")),
      nDaughters_(cfg.getParameter<Int_t>("nDaughters")),
      maxMotherDecay_(cfg.getParameter<Int_t>("NumDecayRequired")),
      rootFileName_(cfg.getParameter<std::string>("rootFile")),
      treeName_(cfg.getParameter<std::string>("treeName")) {
  //auto now = std::chrono::high_resolution_clock::now();
  //auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  //UInt_t seed = static_cast<UInt_t>(millis) + std::hash<std::string>{}(rootFileName_);
  UInt_t seed = getRandomSeed();
  //std::cout<<"Random seed: "<<seed<<std::endl;
  //UInt_t seed = static_cast<UInt_t>(std::time(nullptr)) + std::hash<std::string>{}(rootFileName_);
  rand.SetSeed(seed);

  produces<edm::HepMCProduct>();
  loadRootFile();
}

ExternalDecayInjector::~ExternalDecayInjector() {
  f->Close();
  delete f;
}

void ExternalDecayInjector::loadRootFile() {
  f = TFile::Open(rootFileName_.c_str(), "READ");
  if (f) {
    std::cout << "=== Contenuto fMuonFile ===" << std::endl;
    f->ls();
  }
  if (!f)
    throw cms::Exception("ExternalDecayInjector") << "Cannot open ROOT file";

  tree = dynamic_cast<TTree*>(f->Get(treeName_.c_str()));
  if (!tree)
    throw cms::Exception("ExternalDecayInjector") << "Missing TTrees";

  for (int i = 0; i < nDaughters_; i++) {
    tree->SetBranchAddress(Form("pt%d", i), &pt[i]);
    tree->SetBranchAddress(Form("eta%d", i), &eta[i]);
    tree->SetBranchAddress(Form("phi%d", i), &phi[i]);
    tree->SetBranchAddress(Form("energy%d", i), &energy[i]);
    tree->SetBranchAddress(Form("pdgid%d", i), &pdgid[i]);
  }

  nentries = tree->GetEntries();
}

void ExternalDecayInjector::produce(edm::Event& evt, const edm::EventSetup&) {
  //void ExternalDecayInjector::produce(edm::Event& evt, const edm::EventSetup& es, edm::StreamID streamID){
  edm::Handle<edm::HepMCProduct> h_evt;
  evt.getByToken(token_, h_evt);

  HepMC::GenEvent* evtPtr = const_cast<HepMC::GenEvent*>(h_evt->GetEvent());

  /*
    edm::Service<edm::RandomNumberGenerator> rng;
    if (!rng.isAvailable()) {
        throw cms::Exception("Configuration") << "RandomNumberGeneratorService not available";
    }
    CLHEP::HepRandomEngine& engine = rng->getEngine(evt.streamID());
    int RandomID = static_cast<int>(CLHEP::RandFlat::shoot(&engine, 0., nentries));
    */

  int RandomID = static_cast<int>(rand.Uniform(nentries));

  tree->GetEntry(RandomID);

  for (HepMC::GenEvent::particle_iterator part = evtPtr->particles_begin(); part != evtPtr->particles_end(); ++part) {
    HepMC::GenParticle* p = *part;

    int ndecays_ = 0;

    //Mother PDG ID filter
    if (p->pdg_id() == motherID_ && p->status() == 1) {
      std::cout << "Found Eta" << std::endl;

      double px = p->momentum().px();
      double py = p->momentum().py();
      double pz = p->momentum().pz();
      double E = p->momentum().e();

      TLorentzVector momVec;
      momVec.SetPxPyPzE(px, py, pz, E);

      //Mother Pt Eta Filter
      if (momVec.Pt() > motherMinPt_ && std::abs(momVec.Eta()) < motherMaxEta_) {
        TVector3 beta = momVec.BoostVector();

        auto vtx = p->production_vertex();
        if (!vtx) {
          vtx = new HepMC::GenVertex(HepMC::FourVector(0, 0, 0, 0));
          evtPtr->add_vertex(vtx);
          throw cms::Exception("ExternalDecayInjector") << "Missing Vertex";
        }

        double vx = vtx->position().x();
        double vy = vtx->position().y();
        double vz = vtx->position().z();
        double t = vtx->position().t();

        //vtx->remove_particle(p);
        p->set_status(2);

        HepMC::GenVertex* dv = new HepMC::GenVertex(HepMC::FourVector(vx, vy, vz, t));
        dv->add_particle_in(p);
        evtPtr->add_vertex(dv);

        // =====================================================
        // Loop on eta' Daughter form Pluto
        // =====================================================
        for (int i = 0; i < nDaughters_; i++) {
          TLorentzVector vec;
          vec.SetPtEtaPhiE(pt[i], eta[i], phi[i], energy[i]);
          Int_t particleID = pdgid[i];
          vec.Boost(beta);

          HepMC::FourVector fv(vec.Px(), vec.Py(), vec.Pz(), vec.E());
          HepMC::GenParticle* d = new HepMC::GenParticle(fv, particleID, 1);
          dv->add_particle_out(d);
        }
        ndecays_++;
      }
    }
    if (ndecays_ >= maxMotherDecay_)
      break;
  }

  //auto out = std::make_unique<edm::HepMCProduct>();
  //out->addHepMCData(evtPtr.release());
  //evt.put(std::move(out));
}

DEFINE_FWK_MODULE(ExternalDecayInjector);

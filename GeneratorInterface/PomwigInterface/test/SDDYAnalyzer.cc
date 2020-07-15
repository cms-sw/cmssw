////////// Header section /////////////////////////////////////////////
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TH1F.h"

class SDDYAnalyzer : public edm::EDAnalyzer {
public:
  /// Constructor
  SDDYAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  ~SDDYAnalyzer() override;

  // Operations

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  //virtual void beginJob(const edm::EventSetup& eventSetup) ;
  void beginJob() override;
  void endJob() override;

private:
  // Input from cfg file
  edm::InputTag genParticlesTag_;
  int particle1Id_;
  int particle2Id_;

  // Histograms
  TH1F* hPart1Pt;
  TH1F* hPart1Eta;
  TH1F* hPart1Phi;
  TH1F* hPart2Pt;
  TH1F* hPart2Eta;
  TH1F* hPart2Phi;
  TH1F* hBosonPt;
  TH1F* hBosonEta;
  TH1F* hBosonPhi;
  TH1F* hBosonM;
  TH1F* hEnergyvsEta;
  TH1F* hXiGen;
  TH1F* hProtonPt2;

  int nevents;
  bool debug;
  double Ebeam;
};

////////// Source code ////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

/// Constructor
SDDYAnalyzer::SDDYAnalyzer(const edm::ParameterSet& pset) {
  genParticlesTag_ = pset.getParameter<edm::InputTag>("GenParticleTag");
  particle1Id_ = pset.getParameter<int>("Particle1Id");
  particle2Id_ = pset.getParameter<int>("Particle2Id");

  debug = pset.getUntrackedParameter<bool>("debug", false);
  if (debug) {
    std::cout << ">>> First particle Id: " << particle1Id_ << std::endl;
    std::cout << ">>> Second particle Id: " << particle2Id_ << std::endl;
  }
}

/// Destructor
SDDYAnalyzer::~SDDYAnalyzer() {}

void SDDYAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  //TH1::SetDefaultSumw2(true);

  hPart1Pt = fs->make<TH1F>("hPart1Pt", "hPart1Pt", 100, 0., 100.);
  hPart1Eta = fs->make<TH1F>("hPart1Eta", "hPart1Eta", 100, -5., 5.);
  hPart1Phi = fs->make<TH1F>("hPart1Phi", "hPart1Phi", 100, -3.141592, 3.141592);
  hPart2Pt = fs->make<TH1F>("hPart2Pt", "hPart2Pt", 100, 0., 100.);
  hPart2Eta = fs->make<TH1F>("hPart2Eta", "hPart2Eta", 100, -5., 5.);
  hPart2Phi = fs->make<TH1F>("hPart2Phi", "hPart2Phi", 100, -3.141592, 3.141592);
  hBosonPt = fs->make<TH1F>("hBosonPt", "hBosonPt", 100, 0., 50.);
  hBosonEta = fs->make<TH1F>("hBosonEta", "hBosonEta", 100, -5., 5.);
  hBosonPhi = fs->make<TH1F>("hBosonPhi", "hBosonPhi", 100, -3.141592, 3.141592);
  hBosonM = fs->make<TH1F>("hBosonM", "hBosonM", 100, 40., 100.);

  hEnergyvsEta = fs->make<TH1F>("hEnergyvsEta", "hEnergyvsEta", 100, -15.0, 15.0);
  hXiGen = fs->make<TH1F>("hXiGen", "hXiGen", 100, 0., 0.21);
  hProtonPt2 = fs->make<TH1F>("hProtonPt2", "hProtonPt2", 100, 0., 3.0);

  nevents = 0;
  Ebeam = 5000.;  //Fix get the Ebeam from the event
}

void SDDYAnalyzer::endJob() { hEnergyvsEta->Scale(1 / (float)nevents); }

void SDDYAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup&) {
  nevents++;

  // Generator Information
  edm::Handle<reco::GenParticleCollection> genParticles;
  ev.getByLabel(genParticlesTag_, genParticles);
  double pz1max = 0.;
  double pz2min = 0.;
  reco::GenParticleCollection::const_iterator proton1 = genParticles->end();
  reco::GenParticleCollection::const_iterator proton2 = genParticles->end();
  reco::GenParticleCollection::const_iterator particle1 = genParticles->end();
  reco::GenParticleCollection::const_iterator particle2 = genParticles->end();
  for (reco::GenParticleCollection::const_iterator genpart = genParticles->begin(); genpart != genParticles->end();
       ++genpart) {
    //std::cout << ">>>>>>> pid,status,px,py,px,e= "  << genpart->pdgId() << " , " << genpart->status() << " , " << genpart->px() << " , " << genpart->py() << " , " << genpart->pz() << " , " << genpart->energy() << std::endl;
    if (genpart->status() != 1)
      continue;

    hEnergyvsEta->Fill(genpart->eta(), genpart->energy());

    double pz = genpart->pz();
    if ((genpart->pdgId() == 2212) && (pz > 0.75 * Ebeam)) {
      if (pz > pz1max) {
        proton1 = genpart;
        pz1max = pz;
      }
    } else if ((genpart->pdgId() == 2212) && (pz < -0.75 * Ebeam)) {
      if (pz < pz2min) {
        proton2 = genpart;
        pz2min = pz;
      }
    }

    //Fix add constraint on mother/daughter relation
    if ((particle1 == genParticles->end()) && (abs(genpart->pdgId()) == abs(particle1Id_))) {
      particle1 = genpart;
      continue;
    }
    if ((particle2 == genParticles->end()) && (abs(genpart->pdgId()) == abs(particle2Id_))) {
      particle2 = genpart;
      continue;
    }
  }

  if (proton1 != genParticles->end()) {
    if (debug)
      std::cout << "Proton 1: " << proton1->pt() << "  " << proton1->eta() << "  " << proton1->phi() << std::endl;
    double xigen1 = 1 - proton1->pz() / Ebeam;
    hXiGen->Fill(xigen1);
    hProtonPt2->Fill(proton1->pt() * proton1->pt());
  }

  if (proton2 != genParticles->end()) {
    if (debug)
      std::cout << "Proton 2: " << proton2->pt() << "  " << proton2->eta() << "  " << proton2->phi() << std::endl;
    double xigen2 = 1 + proton2->pz() / Ebeam;
    hXiGen->Fill(xigen2);
    hProtonPt2->Fill(proton2->pt() * proton2->pt());
  }

  if ((particle1 != genParticles->end()) && (particle2 != genParticles->end())) {
    if (debug)
      std::cout << ">>> particle 1 pt,eta: " << particle1->pt() << " , " << particle1->eta() << std::endl;
    hPart1Pt->Fill(particle1->pt());
    hPart1Eta->Fill(particle1->eta());
    hPart1Phi->Fill(particle1->phi());

    if (debug)
      std::cout << ">>> particle 2 pt,eta: " << particle2->pt() << " , " << particle2->eta() << std::endl;
    hPart2Pt->Fill(particle2->pt());
    hPart2Eta->Fill(particle2->eta());
    hPart2Phi->Fill(particle2->phi());

    math::XYZTLorentzVector myboson(particle1->px() + particle2->px(),
                                    particle1->py() + particle2->py(),
                                    particle1->pz() + particle2->pz(),
                                    particle1->energy() + particle2->energy());
    hBosonPt->Fill(myboson.pt());
    hBosonEta->Fill(myboson.eta());
    hBosonPhi->Fill(myboson.phi());
    hBosonM->Fill(myboson.M());
  }
}

DEFINE_FWK_MODULE(SDDYAnalyzer);

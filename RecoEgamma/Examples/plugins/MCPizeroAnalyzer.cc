#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/PizeroMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/PizeroMCTruthFinder.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"

#include <iostream>
#include <map>
#include <vector>

class MCPizeroAnalyzer : public edm::one::EDAnalyzer<> {
public:
  //
  explicit MCPizeroAnalyzer(const edm::ParameterSet&);
  ~MCPizeroAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  float etaTransformation(float a, float b);
  float phiNormalization(float& a);

  //
  PizeroMCTruthFinder* thePizeroMCTruthFinder_;

  std::string fOutputFileName_;
  TFile* fOutputFile_;

  int nEvt_;
  int nMatched_;

  /// global variable for the MC photon
  double mcPhi_;
  double mcEta_;

  std::string HepMCLabel;
  std::string SimTkLabel;
  std::string SimVtxLabel;
  std::string SimHitLabel;

  TH1F* h_MCPizE_;
  TH1F* h_MCPizEta_;
  TH1F* h_MCPizUnEta_;
  TH1F* h_MCPiz1ConEta_;
  TH1F* h_MCPiz2ConEta_;
  TH1F* h_MCPizPhi_;
  TH1F* h_MCPizMass1_;
  TH1F* h_MCPizMass2_;

  TH1F* h_MCEleE_;
  TH1F* h_MCEleEta_;
  TH1F* h_MCElePhi_;
  TH1F* h_BremFrac_;
  TH1F* h_BremEnergy_;

  TH2F* h_EleEvsPhoE_;

  TH1F* h_MCPhoE_;
  TH1F* h_MCPhoEta_;
  TH1F* h_MCPhoPhi_;
  TH1F* h_MCConvPhoE_;
  TH1F* h_MCConvPhoEta_;
  TH1F* h_MCConvPhoPhi_;
  TH1F* h_MCConvPhoR_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MCPizeroAnalyzer);

using namespace std;

MCPizeroAnalyzer::MCPizeroAnalyzer(const edm::ParameterSet& pset)
    : fOutputFileName_(pset.getUntrackedParameter<string>("HistOutFile", std::string("TestConversions.root"))),
      fOutputFile_(nullptr) {}

MCPizeroAnalyzer::~MCPizeroAnalyzer() { delete thePizeroMCTruthFinder_; }

void MCPizeroAnalyzer::beginJob() {
  nEvt_ = 0;

  thePizeroMCTruthFinder_ = new PizeroMCTruthFinder();

  fOutputFile_ = new TFile(fOutputFileName_.c_str(), "RECREATE");

  //// Pizeros
  h_MCPizE_ = new TH1F("MCPizE", "MC piz energy", 100, 0., 200.);
  h_MCPizPhi_ = new TH1F("MCPizPhi", "MC piz phi", 40, -3.14, 3.14);
  h_MCPizEta_ = new TH1F("MCPizEta", "MC piz eta", 40, -3., 3.);
  h_MCPizUnEta_ = new TH1F("MCPizUnEta", "MC un piz eta", 40, -3., 3.);
  h_MCPiz1ConEta_ = new TH1F("MCPiz1ConEta", "MC con piz eta: at least one converted photon", 40, -3., 3.);
  h_MCPiz2ConEta_ = new TH1F("MCPiz2ConEta", "MC con piz eta: two converted photons", 40, -3., 3.);
  h_MCPizMass1_ = new TH1F("MCPizMass1", "Piz mass unconverted ", 100, 0., 200);
  h_MCPizMass2_ = new TH1F("MCPizMass2", "Piz mass converted ", 100, 0., 200);

  // All Photons from Pizeros
  h_MCPhoE_ = new TH1F("MCPhoE", "MC photon energy", 100, 0., 200.);
  h_MCPhoPhi_ = new TH1F("MCPhoPhi", "MC photon phi", 40, -3.14, 3.14);
  h_MCPhoEta_ = new TH1F("MCPhoEta", "MC photon eta", 40, -3., 3.);

  // Converted photons
  h_MCConvPhoE_ = new TH1F("MCConvPhoE", "MC converted photon energy", 100, 0., 200.);
  h_MCConvPhoPhi_ = new TH1F("MCConvPhoPhi", "MC converted photon phi", 40, -3.14, 3.14);
  h_MCConvPhoEta_ = new TH1F("MCConvPhoEta", "MC converted photon eta", 40, -3., 3.);
  h_MCConvPhoR_ = new TH1F("MCConvPhoR", "MC converted photon R", 120, 0., 120.);
  // Electrons from converted photons
  h_MCEleE_ = new TH1F("MCEleE", "MC ele energy", 100, 0., 200.);
  h_MCElePhi_ = new TH1F("MCElePhi", "MC ele phi", 40, -3.14, 3.14);
  h_MCEleEta_ = new TH1F("MCEleEta", "MC ele eta", 40, -3., 3.);
  h_BremFrac_ = new TH1F("bremFrac", "brem frac ", 50, 0., 1.);
  h_BremEnergy_ = new TH1F("bremE", "Brem energy", 100, 0., 200.);

  h_EleEvsPhoE_ = new TH2F("eleEvsPhoE", "eleEvsPhoE", 100, 0., 200., 100, 0., 200.);

  return;
}

float MCPizeroAnalyzer::etaTransformation(float EtaParticle, float Zvertex) {
  //---Definitions
  const float PI = 3.1415927;
  //UNUSED const float TWOPI = 2.0*PI;

  //---Definitions for ECAL
  const float R_ECAL = 136.5;
  const float Z_Endcap = 328.0;
  const float etaBarrelEndcap = 1.479;

  //---ETA correction

  float Theta = 0.0;
  float ZEcal = R_ECAL * sinh(EtaParticle) + Zvertex;

  if (ZEcal != 0.0)
    Theta = atan(R_ECAL / ZEcal);
  if (Theta < 0.0)
    Theta = Theta + PI;
  float ETA = -log(tan(0.5 * Theta));

  if (fabs(ETA) > etaBarrelEndcap) {
    float Zend = Z_Endcap;
    if (EtaParticle < 0.0)
      Zend = -Zend;
    float Zlen = Zend - Zvertex;
    float RR = Zlen / sinh(EtaParticle);
    Theta = atan(RR / Zend);
    if (Theta < 0.0)
      Theta = Theta + PI;
    ETA = -log(tan(0.5 * Theta));
  }
  //---Return the result
  return ETA;
  //---end
}

float MCPizeroAnalyzer::phiNormalization(float& phi) {
  //---Definitions
  const float PI = 3.1415927;
  const float TWOPI = 2.0 * PI;

  if (phi > PI) {
    phi = phi - TWOPI;
  }
  if (phi < -PI) {
    phi = phi + TWOPI;
  }

  //  cout << " Float_t PHInormalization out " << PHI << endl;
  return phi;
}

void MCPizeroAnalyzer::analyze(const edm::Event& e, const edm::EventSetup&) {
  using namespace edm;
  //UNUSED const float etaPhiDistance=0.01;
  // Fiducial region
  //UNUSED const float TRK_BARL =0.9;
  //UNUSED const float BARL = 1.4442; // DAQ TDR p.290
  //UNUSED const float END_LO = 1.566;
  //UNUSED const float END_HI = 2.5;
  // Electron mass
  //UNUSED const Float_t mElec= 0.000511;

  nEvt_++;
  LogInfo("MCPizeroAnalyzer") << "MCPizeroAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_
                              << "\n";
  //  LogDebug("MCPizeroAnalyzer") << "MCPizeroAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
  std::cout << "MCPizeroAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ << "\n";

  //////////////////// Get the MC truth: SimTracks
  std::cout << " MCPizeroAnalyzer Looking for MC truth "
            << "\n";

  //get simtrack info
  std::vector<SimTrack> theSimTracks;
  std::vector<SimVertex> theSimVertices;

  edm::Handle<SimTrackContainer> SimTk;
  edm::Handle<SimVertexContainer> SimVtx;
  e.getByLabel("g4SimHits", SimTk);
  e.getByLabel("g4SimHits", SimVtx);

  theSimTracks.insert(theSimTracks.end(), SimTk->begin(), SimTk->end());
  theSimVertices.insert(theSimVertices.end(), SimVtx->begin(), SimVtx->end());
  std::cout << " MCPizeroAnalyzer This Event has " << theSimTracks.size() << " sim tracks " << std::endl;
  std::cout << " MCPizeroAnalyzer This Event has " << theSimVertices.size() << " sim vertices " << std::endl;
  if (theSimTracks.empty())
    std::cout << " Event number " << e.id() << " has NO sim tracks " << std::endl;

  std::vector<PizeroMCTruth> MCPizeroeros = thePizeroMCTruthFinder_->find(theSimTracks, theSimVertices);
  std::cout << " MCPizeroAnalyzer MCPizeroeros size " << MCPizeroeros.size() << std::endl;

  for (std::vector<PizeroMCTruth>::const_iterator iPiz = MCPizeroeros.begin(); iPiz != MCPizeroeros.end(); ++iPiz) {
    h_MCPizE_->Fill((*iPiz).fourMomentum().e());
    h_MCPizEta_->Fill((*iPiz).fourMomentum().pseudoRapidity());
    h_MCPizPhi_->Fill((*iPiz).fourMomentum().phi());

    std::vector<PhotonMCTruth> mcPhotons = (*iPiz).photons();
    std::cout << " MCPizeroAnalyzer mcPhotons size " << mcPhotons.size() << std::endl;

    float px = mcPhotons[0].fourMomentum().x() + mcPhotons[1].fourMomentum().x();
    float py = mcPhotons[0].fourMomentum().y() + mcPhotons[1].fourMomentum().y();
    float pz = mcPhotons[0].fourMomentum().z() + mcPhotons[1].fourMomentum().z();
    float e = mcPhotons[0].fourMomentum().e() + mcPhotons[1].fourMomentum().e();
    float invM = sqrt(e * e - px * px - py * py - pz * pz) * 1000;
    h_MCPizMass1_->Fill(invM);

    int converted = 0;
    for (std::vector<PhotonMCTruth>::const_iterator iPho = mcPhotons.begin(); iPho != mcPhotons.end(); ++iPho) {
      h_MCPhoE_->Fill((*iPho).fourMomentum().e());
      h_MCPhoEta_->Fill((*iPho).fourMomentum().pseudoRapidity());
      h_MCPhoPhi_->Fill((*iPho).fourMomentum().phi());
      if ((*iPho).isAConversion()) {
        converted++;

        h_MCConvPhoE_->Fill((*iPho).fourMomentum().e());
        h_MCConvPhoEta_->Fill((*iPho).fourMomentum().pseudoRapidity());
        h_MCConvPhoPhi_->Fill((*iPho).fourMomentum().phi());
        h_MCConvPhoR_->Fill((*iPho).vertex().perp());

        std::vector<ElectronMCTruth> mcElectrons = (*iPho).electrons();
        std::cout << " MCPizeroAnalyzer mcElectrons size " << mcElectrons.size() << std::endl;

        for (std::vector<ElectronMCTruth>::const_iterator iEl = mcElectrons.begin(); iEl != mcElectrons.end(); ++iEl) {
          if ((*iEl).fourMomentum().e() < 30)
            continue;
          h_MCEleE_->Fill((*iEl).fourMomentum().e());
          h_MCEleEta_->Fill((*iEl).fourMomentum().pseudoRapidity());
          h_MCElePhi_->Fill((*iEl).fourMomentum().phi());

          h_EleEvsPhoE_->Fill((*iPho).fourMomentum().e(), (*iEl).fourMomentum().e());

          float totBrem = 0;
          for (unsigned int iBrem = 0; iBrem < (*iEl).bremVertices().size(); ++iBrem)
            totBrem += (*iEl).bremMomentum()[iBrem].e();

          h_BremFrac_->Fill(totBrem / (*iEl).fourMomentum().e());
          h_BremEnergy_->Fill(totBrem);
        }
      }
    }

    if (converted > 0) {
      h_MCPiz1ConEta_->Fill((*iPiz).fourMomentum().pseudoRapidity());
      if (converted == 2)
        h_MCPiz2ConEta_->Fill((*iPiz).fourMomentum().pseudoRapidity());
    } else {
      h_MCPizUnEta_->Fill((*iPiz).fourMomentum().pseudoRapidity());
    }
  }
}

void MCPizeroAnalyzer::endJob() {
  fOutputFile_->Write();
  fOutputFile_->Close();

  edm::LogInfo("MCPizeroAnalyzer") << "Analyzed " << nEvt_ << "\n";
  std::cout << "MCPizeroAnalyzer::endJob Analyzed " << nEvt_ << " events "
            << "\n";

  return;
}

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
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruthFinder.h"
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

class MCElectronAnalyzer : public edm::one::EDAnalyzer<> {
public:
  //
  explicit MCElectronAnalyzer(const edm::ParameterSet&);
  ~MCElectronAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  float etaTransformation(float a, float b);
  float phiNormalization(float& a);

  //
  ElectronMCTruthFinder* theElectronMCTruthFinder_;

  const TrackerGeometry* trackerGeom;

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

  TH1F* h_MCEleE_;
  TH1F* h_MCEleEta_;
  TH1F* h_MCElePhi_;
  TH1F* h_BremFrac_;
  TH1F* h_BremEnergy_;

  TProfile* p_BremVsR_;
  TProfile* p_BremVsEta_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MCElectronAnalyzer);

using namespace std;

MCElectronAnalyzer::MCElectronAnalyzer(const edm::ParameterSet& pset)
    : fOutputFileName_(pset.getUntrackedParameter<string>("HistOutFile", std::string("TestConversions.root"))),
      fOutputFile_(nullptr) {}

MCElectronAnalyzer::~MCElectronAnalyzer() { delete theElectronMCTruthFinder_; }

void MCElectronAnalyzer::beginJob() {
  nEvt_ = 0;

  theElectronMCTruthFinder_ = new ElectronMCTruthFinder();

  fOutputFile_ = new TFile(fOutputFileName_.c_str(), "RECREATE");

  //// Primary electrons
  h_MCEleE_ = new TH1F("MCEleE", "MC ele energy", 100, 0., 200.);
  h_MCElePhi_ = new TH1F("MCElePhi", "MC ele phi", 40, -3.14, 3.14);
  h_MCEleEta_ = new TH1F("MCEleEta", "MC ele eta", 40, -3., 3.);
  h_BremFrac_ = new TH1F("bremFrac", "brem frac ", 100, 0., 1.);
  h_BremEnergy_ = new TH1F("BremE", "Brem energy", 100, 0., 200.);

  p_BremVsR_ = new TProfile("BremVsR", " Mean Brem energy vs R ", 48, 0., 120.);
  p_BremVsEta_ = new TProfile("BremVsEta", " Mean Brem energy vs Eta ", 50, -2.5, 2.5);

  return;
}

float MCElectronAnalyzer::etaTransformation(float EtaParticle, float Zvertex) {
  //---Definitions
  const float PI = 3.1415927;
  //const float TWOPI = 2.0*PI;

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

  if (std::abs(ETA) > etaBarrelEndcap) {
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

float MCElectronAnalyzer::phiNormalization(float& phi) {
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

void MCElectronAnalyzer::analyze(const edm::Event& e, const edm::EventSetup&) {
  using namespace edm;
  //const float etaPhiDistance=0.01;
  // Fiducial region
  //const float TRK_BARL =0.9;
  //const float BARL = 1.4442; // DAQ TDR p.290
  //const float END_LO = 1.566;
  //const float END_HI = 2.5;
  // Electron mass
  //const Float_t mElec= 0.000511;

  nEvt_++;
  LogInfo("MCElectronAnalyzer") << "MCElectronAnalyzer Analyzing event number: " << e.id() << " Global Counter "
                                << nEvt_ << "\n";
  //  LogDebug("MCElectronAnalyzer") << "MCElectronAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
  std::cout << "MCElectronAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ << "\n";

  //////////////////// Get the MC truth: SimTracks
  std::cout << " MCElectronAnalyzer Looking for MC truth "
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
  std::cout << " MCElectronAnalyzer This Event has " << theSimTracks.size() << " sim tracks " << std::endl;
  std::cout << " MCElectronAnalyzer This Event has " << theSimVertices.size() << " sim vertices " << std::endl;
  if (theSimTracks.empty())
    std::cout << " Event number " << e.id() << " has NO sim tracks " << std::endl;

  std::vector<ElectronMCTruth> MCElectronctrons = theElectronMCTruthFinder_->find(theSimTracks, theSimVertices);
  std::cout << " MCElectronAnalyzer MCElectronctrons size " << MCElectronctrons.size() << std::endl;

  for (std::vector<ElectronMCTruth>::const_iterator iEl = MCElectronctrons.begin(); iEl != MCElectronctrons.end();
       ++iEl) {
    h_MCEleE_->Fill((*iEl).fourMomentum().e());
    h_MCEleEta_->Fill((*iEl).fourMomentum().pseudoRapidity());
    h_MCElePhi_->Fill((*iEl).fourMomentum().phi());

    float totBrem = 0;
    unsigned int iBrem;
    for (iBrem = 0; iBrem < (*iEl).bremVertices().size(); ++iBrem) {
      float rBrem = (*iEl).bremVertices()[iBrem].perp();
      float etaBrem = (*iEl).bremVertices()[iBrem].eta();
      if (rBrem < 120) {
        totBrem += (*iEl).bremMomentum()[iBrem].e();
        p_BremVsR_->Fill(rBrem, (*iEl).bremMomentum()[iBrem].e());
        p_BremVsEta_->Fill(etaBrem, (*iEl).bremMomentum()[iBrem].e());
      }
    }

    h_BremFrac_->Fill(totBrem / (*iEl).fourMomentum().e());
    h_BremEnergy_->Fill(totBrem);
  }
}

void MCElectronAnalyzer::endJob() {
  fOutputFile_->Write();
  fOutputFile_->Close();

  edm::LogInfo("MCElectronAnalyzer") << "Analyzed " << nEvt_ << "\n";
  std::cout << "MCElectronAnalyzer::endJob Analyzed " << nEvt_ << " events "
            << "\n";

  return;
}

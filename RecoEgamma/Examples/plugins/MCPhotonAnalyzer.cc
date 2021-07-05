#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
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

class MCPhotonAnalyzer : public edm::one::EDAnalyzer<> {
public:
  //
  explicit MCPhotonAnalyzer(const edm::ParameterSet&);
  ~MCPhotonAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  float etaTransformation(float a, float b);
  float phiNormalization(float& a);

  //
  PhotonMCTruthFinder* thePhotonMCTruthFinder_;

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

  // all photons
  TH1F* h_MCPhoE_;
  TH1F* h_MCPhoEta_;
  TH1F* h_MCPhoEta1_;
  TH1F* h_MCPhoEta2_;
  TH1F* h_MCPhoEta3_;
  TH1F* h_MCPhoEta4_;
  TH1F* h_MCPhoPhi_;
  // Conversion
  TH1F* h_MCConvPhoE_;
  TH1F* h_MCConvPhoEta_;
  TH1F* h_MCConvPhoPhi_;
  TH1F* h_MCConvPhoR_;
  TH1F* h_MCConvPhoREta1_;
  TH1F* h_MCConvPhoREta2_;
  TH1F* h_MCConvPhoREta3_;
  TH1F* h_MCConvPhoREta4_;
  TH1F* h_convFracEta1_;
  TH1F* h_convFracEta2_;
  TH1F* h_convFracEta3_;
  TH1F* h_convFracEta4_;

  /// Conversions with two tracks
  TH1F* h_MCConvPhoTwoTracksE_;
  TH1F* h_MCConvPhoTwoTracksEta_;
  TH1F* h_MCConvPhoTwoTracksPhi_;
  TH1F* h_MCConvPhoTwoTracksR_;
  /// Conversions with one track
  TH1F* h_MCConvPhoOneTrackE_;
  TH1F* h_MCConvPhoOneTrackEta_;
  TH1F* h_MCConvPhoOneTrackPhi_;
  TH1F* h_MCConvPhoOneTrackR_;

  TH1F* h_MCEleE_;
  TH1F* h_MCEleEta_;
  TH1F* h_MCElePhi_;
  TH1F* h_BremFrac_;
  TH1F* h_BremEnergy_;
  TH2F* h_EleEvsPhoE_;
  TH2F* h_bremEvsEleE_;

  TProfile* p_BremVsR_;
  TProfile* p_BremVsEta_;

  TProfile* p_BremVsConvR_;
  TProfile* p_BremVsConvEta_;

  TH2F* h_bremFracVsConvR_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MCPhotonAnalyzer);

using namespace std;

MCPhotonAnalyzer::MCPhotonAnalyzer(const edm::ParameterSet& pset) {}

MCPhotonAnalyzer::~MCPhotonAnalyzer() { delete thePhotonMCTruthFinder_; }

void MCPhotonAnalyzer::beginJob() {
  nEvt_ = 0;

  thePhotonMCTruthFinder_ = new PhotonMCTruthFinder();

  edm::Service<TFileService> fs;

  //// All MC photons
  h_MCPhoE_ = fs->make<TH1F>("MCPhoE", "MC photon energy", 100, 0., 100.);
  h_MCPhoPhi_ = fs->make<TH1F>("MCPhoPhi", "MC photon phi", 40, -3.14, 3.14);
  h_MCPhoEta_ = fs->make<TH1F>("MCPhoEta", "MC photon eta", 25, 0., 2.5);
  h_MCPhoEta1_ = fs->make<TH1F>("MCPhoEta1", "MC photon eta", 40, -3., 3.);
  h_MCPhoEta2_ = fs->make<TH1F>("MCPhoEta2", "MC photon eta", 40, -3., 3.);
  h_MCPhoEta3_ = fs->make<TH1F>("MCPhoEta3", "MC photon eta", 40, -3., 3.);
  h_MCPhoEta4_ = fs->make<TH1F>("MCPhoEta4", "MC photon eta", 40, -3., 3.);
  /// conversions
  h_MCConvPhoE_ = fs->make<TH1F>("MCConvPhoE", "MC converted photon energy", 100, 0., 100.);
  h_MCConvPhoPhi_ = fs->make<TH1F>("MCConvPhoPhi", "MC converted photon phi", 40, -3.14, 3.14);
  h_MCConvPhoEta_ = fs->make<TH1F>("MCConvPhoEta", "MC converted photon eta", 25, 0., 2.5);
  h_MCConvPhoR_ = fs->make<TH1F>("MCConvPhoR", "MC converted photon R", 120, 0., 120.);

  h_MCConvPhoREta1_ = fs->make<TH1F>("MCConvPhoREta1", "MC converted photon R", 120, 0., 120.);
  h_MCConvPhoREta2_ = fs->make<TH1F>("MCConvPhoREta2", "MC converted photon R", 120, 0., 120.);
  h_MCConvPhoREta3_ = fs->make<TH1F>("MCConvPhoREta3", "MC converted photon R", 120, 0., 120.);
  h_MCConvPhoREta4_ = fs->make<TH1F>("MCConvPhoREta4", "MC converted photon R", 120, 0., 120.);

  h_convFracEta1_ = fs->make<TH1F>("convFracEta1", "Integrated(R) fraction of conversion |eta|=0.2", 120, 0., 120.);
  h_convFracEta2_ = fs->make<TH1F>("convFracEta2", "Integrated(R) fraction of conversion |eta|=0.9", 120, 0., 120.);
  h_convFracEta3_ = fs->make<TH1F>("convFracEta3", "Integrated(R) fraction of conversion |eta|=1.5", 120, 0., 120.);
  h_convFracEta4_ = fs->make<TH1F>("convFracEta4", "Integrated(R) fraction of conversion |eta|=2.0", 120, 0., 120.);
  /// conversions with two tracks
  h_MCConvPhoTwoTracksE_ =
      fs->make<TH1F>("MCConvPhoTwoTracksE", "MC converted photon with 2 tracks  energy", 100, 0., 100.);
  h_MCConvPhoTwoTracksPhi_ =
      fs->make<TH1F>("MCConvPhoTwoTracksPhi", "MC converted photon 2 tracks  phi", 40, -3.14, 3.14);
  h_MCConvPhoTwoTracksEta_ = fs->make<TH1F>("MCConvPhoTwoTracksEta", "MC converted photon 2 tracks eta", 40, -3., 3.);
  h_MCConvPhoTwoTracksR_ = fs->make<TH1F>("MCConvPhoTwoTracksR", "MC converted photon 2 tracks eta", 48, 0., 120.);
  // conversions with one track
  h_MCConvPhoOneTrackE_ =
      fs->make<TH1F>("MCConvPhoOneTrackE", "MC converted photon with 1 track  energy", 100, 0., 100.);
  h_MCConvPhoOneTrackPhi_ = fs->make<TH1F>("MCConvPhoOneTrackPhi", "MC converted photon 1 track  phi", 40, -3.14, 3.14);
  h_MCConvPhoOneTrackEta_ = fs->make<TH1F>("MCConvPhoOneTrackEta", "MC converted photon 1 track eta", 40, -3., 3.);
  h_MCConvPhoOneTrackR_ = fs->make<TH1F>("MCConvPhoOneTrackR", "MC converted photon 1 track eta", 48, 0., 120.);

  /// electrons from conversions
  h_MCEleE_ = fs->make<TH1F>("MCEleE", "MC ele energy", 100, 0., 200.);
  h_MCElePhi_ = fs->make<TH1F>("MCElePhi", "MC ele phi", 40, -3.14, 3.14);
  h_MCEleEta_ = fs->make<TH1F>("MCEleEta", "MC ele eta", 40, -3., 3.);
  h_BremFrac_ = fs->make<TH1F>("bremFrac", "brem frac ", 100, 0., 1.);
  h_BremEnergy_ = fs->make<TH1F>("BremE", "Brem energy", 100, 0., 200.);
  h_EleEvsPhoE_ = fs->make<TH2F>("eleEvsPhoE", "eleEvsPhoE", 100, 0., 200., 100, 0., 200.);
  h_bremEvsEleE_ = fs->make<TH2F>("bremEvsEleE", "bremEvsEleE", 100, 0., 200., 100, 0., 200.);

  p_BremVsR_ = fs->make<TProfile>("BremVsR", " Mean Brem Energy vs R ", 48, 0., 120.);
  p_BremVsEta_ = fs->make<TProfile>("BremVsEta", " Mean Brem Energy vs Eta ", 50, -2.5, 2.5);

  p_BremVsConvR_ = fs->make<TProfile>("BremVsConvR", " Mean Brem Fraction vs conversion R ", 48, 0., 120.);
  p_BremVsConvEta_ = fs->make<TProfile>("BremVsConvEta", " Mean Brem Fraction vs converion Eta ", 50, -2.5, 2.5);

  h_bremFracVsConvR_ = fs->make<TH2F>("bremFracVsConvR", "brem Fraction vs conversion R", 60, 0., 120., 100, 0., 1.);

  return;
}

float MCPhotonAnalyzer::etaTransformation(float EtaParticle, float Zvertex) {
  //---Definitions
  const float PI = 3.1415927;
  //	const float TWOPI = 2.0*PI;

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

float MCPhotonAnalyzer::phiNormalization(float& phi) {
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

void MCPhotonAnalyzer::analyze(const edm::Event& e, const edm::EventSetup&) {
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
  LogInfo("mcEleAnalyzer") << "MCPhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_
                           << "\n";
  //  LogDebug("MCPhotonAnalyzer") << "MCPhotonAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
  std::cout << "MCPhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ << "\n";

  //////////////////// Get the MC truth: SimTracks
  std::cout << " MCPhotonAnalyzer Looking for MC truth "
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
  std::cout << " MCPhotonAnalyzer This Event has " << theSimTracks.size() << " sim tracks " << std::endl;
  std::cout << " MCPhotonAnalyzer This Event has " << theSimVertices.size() << " sim vertices " << std::endl;
  if (theSimTracks.empty())
    std::cout << " Event number " << e.id() << " has NO sim tracks " << std::endl;

  std::vector<PhotonMCTruth> mcPhotons = thePhotonMCTruthFinder_->find(theSimTracks, theSimVertices);
  std::cout << " MCPhotonAnalyzer mcPhotons size " << mcPhotons.size() << std::endl;

  for (std::vector<PhotonMCTruth>::const_iterator iPho = mcPhotons.begin(); iPho != mcPhotons.end(); ++iPho) {
    if ((*iPho).fourMomentum().e() < 35)
      continue;

    h_MCPhoE_->Fill((*iPho).fourMomentum().e());
    //    float correta = etaTransformation( (*iPho).fourMomentum().pseudoRapidity(),  (*iPho).primaryVertex().z() );
    float Theta = (*iPho).fourMomentum().theta();
    float correta = -log(tan(0.5 * Theta));
    correta = etaTransformation(correta, (*iPho).primaryVertex().z());
    //h_MCPhoEta_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
    h_MCPhoEta_->Fill(fabs(correta) - 0.001);
    h_MCPhoPhi_->Fill((*iPho).fourMomentum().phi());

    /*
    if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 0.25 &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=0.15  )
      h_MCPhoEta1_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
    if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 0.95  &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=0.85  )
      h_MCPhoEta2_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
    if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 1.65  &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=1.55  )
      h_MCPhoEta3_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
    if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 2.05  &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=1.95  )
      h_MCPhoEta4_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
    */

    if (fabs(correta) <= 0.3 && fabs(correta) > 0.2)
      h_MCPhoEta1_->Fill(correta);
    if (fabs(correta) <= 1.00 && fabs(correta) > 0.9)
      h_MCPhoEta2_->Fill(correta);
    if (fabs(correta) <= 1.6 && fabs(correta) > 1.5)
      h_MCPhoEta3_->Fill(correta);
    if (fabs(correta) <= 2. && fabs(correta) > 1.9)
      h_MCPhoEta4_->Fill(correta);

    //    if ( (*iPho).isAConversion()  && (*iPho).vertex().perp()< 10 ) {
    if ((*iPho).isAConversion()) {
      h_MCConvPhoE_->Fill((*iPho).fourMomentum().e());
      //      h_MCConvPhoEta_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );

      h_MCConvPhoEta_->Fill(fabs(correta) - 0.001);
      h_MCConvPhoPhi_->Fill((*iPho).fourMomentum().phi());
      h_MCConvPhoR_->Fill((*iPho).vertex().perp());

      /*
      if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 0.25 &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=0.15  )
	h_MCConvPhoREta1_->Fill  ( (*iPho).vertex().perp() );
      if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 0.95  &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=0.85  )
	h_MCConvPhoREta2_->Fill  ( (*iPho).vertex().perp() );
      if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 1.65  &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=1.55  )
	h_MCConvPhoREta3_->Fill  ( (*iPho).vertex().perp() );
      if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 2.05  &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=1.95  )
	h_MCConvPhoREta4_->Fill  ( (*iPho).vertex().perp() );
      */

      if (fabs(correta) <= 0.3 && fabs(correta) > 0.2)
        h_MCConvPhoREta1_->Fill((*iPho).vertex().perp());
      if (fabs(correta) <= 1. && fabs(correta) > 0.9)
        h_MCConvPhoREta2_->Fill((*iPho).vertex().perp());
      if (fabs(correta) <= 1.6 && fabs(correta) > 1.5)
        h_MCConvPhoREta3_->Fill((*iPho).vertex().perp());
      if (fabs(correta) <= 2 && fabs(correta) > 1.9)
        h_MCConvPhoREta4_->Fill((*iPho).vertex().perp());

    }  // end conversions

  }  /// Loop over all MC photons in the event
}

void MCPhotonAnalyzer::endJob() {
  double s1 = 0;
  double s2 = 0;
  double s3 = 0;
  double s4 = 0;
  int e1 = 0;
  int e2 = 0;
  int e3 = 0;
  int e4 = 0;

  double nTotEta1 = h_MCPhoEta1_->GetEntries();
  double nTotEta2 = h_MCPhoEta2_->GetEntries();
  double nTotEta3 = h_MCPhoEta3_->GetEntries();
  double nTotEta4 = h_MCPhoEta4_->GetEntries();

  for (int i = 1; i <= 120; ++i) {
    e1 = (int)h_MCConvPhoREta1_->GetBinContent(i);
    e2 = (int)h_MCConvPhoREta2_->GetBinContent(i);
    e3 = (int)h_MCConvPhoREta3_->GetBinContent(i);
    e4 = (int)h_MCConvPhoREta4_->GetBinContent(i);
    s1 += e1;
    s2 += e2;
    s3 += e3;
    s4 += e4;
    h_convFracEta1_->SetBinContent(i, s1 * 100 / nTotEta1);
    h_convFracEta2_->SetBinContent(i, s2 * 100 / nTotEta2);
    h_convFracEta3_->SetBinContent(i, s3 * 100 / nTotEta3);
    h_convFracEta4_->SetBinContent(i, s4 * 100 / nTotEta4);
  }

  edm::LogInfo("MCPhotonAnalyzer") << "Analyzed " << nEvt_ << "\n";
  std::cout << "MCPhotonAnalyzer::endJob Analyzed " << nEvt_ << " events "
            << "\n";

  return;
}

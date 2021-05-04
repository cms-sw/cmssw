#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
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

class SimpleConvertedPhotonAnalyzer : public edm::one::EDAnalyzer<> {
public:
  //
  explicit SimpleConvertedPhotonAnalyzer(const edm::ParameterSet&);
  ~SimpleConvertedPhotonAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  float etaTransformation(float a, float b);

  //
  PhotonMCTruthFinder* thePhotonMCTruthFinder_;

  std::string fOutputFileName_;
  TFile* fOutputFile_;

  int nEvt_;
  int nMCPho_;
  int nMatched_;

  std::string HepMCLabel;
  std::string SimTkLabel;
  std::string SimVtxLabel;
  std::string SimHitLabel;

  std::string convertedPhotonCollectionProducer_;
  std::string convertedPhotonCollection_;

  TH1F* h_ErecoEMC_;
  TH1F* h_deltaPhi_;
  TH1F* h_deltaEta_;

  //// All MC photons
  TH1F* h_MCphoE_;
  TH1F* h_MCphoPhi_;
  TH1F* h_MCphoEta_;

  //// visible MC Converted photons
  TH1F* h_MCConvE_;
  TH1F* h_MCConvPt_;
  TH1F* h_MCConvEta_;

  // SC from reco photons
  TH1F* h_scE_;
  TH1F* h_scEta_;
  TH1F* h_scPhi_;
  //
  TH1F* h_phoE_;
  TH1F* h_phoEta_;
  TH1F* h_phoPhi_;
  //
  // All tracks from reco photons
  TH2F* h2_tk_nHitsVsR_;
  //
  TH2F* h2_tk_inPtVsR_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleConvertedPhotonAnalyzer);

using namespace std;

SimpleConvertedPhotonAnalyzer::SimpleConvertedPhotonAnalyzer(const edm::ParameterSet& pset)
    : fOutputFileName_(pset.getUntrackedParameter<string>("HistOutFile", std::string("TestConversions.root"))),
      fOutputFile_(nullptr) {
  convertedPhotonCollectionProducer_ = pset.getParameter<std::string>("phoProducer");
  convertedPhotonCollection_ = pset.getParameter<std::string>("convertedPhotonCollection");
  //
}

SimpleConvertedPhotonAnalyzer::~SimpleConvertedPhotonAnalyzer() { delete thePhotonMCTruthFinder_; }

void SimpleConvertedPhotonAnalyzer::beginJob() {
  nEvt_ = 0;
  nMCPho_ = 0;
  nMatched_ = 0;

  thePhotonMCTruthFinder_ = new PhotonMCTruthFinder();

  fOutputFile_ = new TFile(fOutputFileName_.c_str(), "RECREATE");

  /// Reco - MC
  h_ErecoEMC_ = new TH1F("deltaE", "    delta(reco-mc) energy", 100, 0., 2.);
  h_deltaPhi_ = new TH1F("deltaPhi", "  delta(reco-mc) phi", 100, -0.1, 0.1);
  h_deltaEta_ = new TH1F("deltaEta", "  delta(reco-mc) eta", 100, -0.05, 0.05);

  //// All MC photons
  h_MCphoE_ = new TH1F("MCphoE", "MC photon energy", 100, 0., 100.);
  h_MCphoPhi_ = new TH1F("MCphoPhi", "MC photon phi", 40, -3.14, 3.14);
  h_MCphoEta_ = new TH1F("MCphoEta", "MC photon eta", 40, -3., 3.);

  //// visible MC Converted photons
  h_MCConvE_ = new TH1F("MCConvE", "MC converted photon energy", 100, 0., 100.);
  h_MCConvPt_ = new TH1F("MCConvPt", "MC converted photon pt", 100, 0., 100.);
  h_MCConvEta_ = new TH1F("MCConvEta", "MC converted photon eta", 50, 0., 2.5);

  //// Reconstructed Converted photons
  h_scE_ = new TH1F("scE", "Uncorrected converted photons : SC Energy ", 100, 0., 200.);
  h_scEta_ = new TH1F("scEta", "Uncorrected converted photons:  SC Eta ", 40, -3., 3.);
  h_scPhi_ = new TH1F("scPhi", "Uncorrected converted photons: SC Phi ", 40, -3.14, 3.14);
  //
  h_phoE_ = new TH1F("phoE", "Uncorrected converted photons :  Energy ", 100, 0., 200.);
  h_phoEta_ = new TH1F("phoEta", "Uncorrected converted photons:   Eta ", 40, -3., 3.);
  h_phoPhi_ = new TH1F("phoPhi", "Uncorrected converted photons:  Phi ", 40, -3.14, 3.14);

  // Recontructed tracks from converted photon candidates
  h2_tk_nHitsVsR_ = new TH2F("tknHitsVsR", "Tracks Hits vs R  ", 12, 0., 120., 20, 0.5, 20.5);
  h2_tk_inPtVsR_ = new TH2F("tkInPtvsR", "Tracks inner Pt vs R  ", 12, 0., 120., 100, 0., 100.);

  return;
}

float SimpleConvertedPhotonAnalyzer::etaTransformation(float EtaParticle, float Zvertex) {
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

void SimpleConvertedPhotonAnalyzer::analyze(const edm::Event& e, const edm::EventSetup&) {
  using namespace edm;
  const float etaPhiDistance = 0.01;
  // Fiducial region
  //UNUSED const float TRK_BARL =0.9;
  const float BARL = 1.4442;  // DAQ TDR p.290
  const float END_LO = 1.566;
  const float END_HI = 2.5;
  // Electron mass
  //UNUSED const Float_t mElec= 0.000511;

  nEvt_++;
  LogInfo("ConvertedPhotonAnalyzer") << "ConvertedPhotonAnalyzer Analyzing event number: " << e.id()
                                     << " Global Counter " << nEvt_ << "\n";
  //  LogDebug("ConvertedPhotonAnalyzer") << "ConvertedPhotonAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
  std::cout << "ConvertedPhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ << "\n";

  ///// Get the recontructed  conversions
  Handle<reco::ConversionCollection> convertedPhotonHandle;
  e.getByLabel(convertedPhotonCollectionProducer_, convertedPhotonCollection_, convertedPhotonHandle);
  const reco::ConversionCollection phoCollection = *(convertedPhotonHandle.product());
  std::cout << "ConvertedPhotonAnalyzer  Converted photon collection size " << phoCollection.size() << "\n";

  //////////////////// Get the MC truth: SimTracks
  std::cout << " ConvertedPhotonAnalyzer Looking for MC truth "
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

  std::vector<PhotonMCTruth> mcPhotons = thePhotonMCTruthFinder_->find(theSimTracks, theSimVertices);
  std::cout << " ConvertedPhotonAnalyzer mcPhotons size " << mcPhotons.size() << std::endl;

  // Loop over simulated photons
  //UNUSED int iDet=0;
  //UNUSED int iRadius=-1;
  //UNUSED int indPho=0;

  for (std::vector<PhotonMCTruth>::const_iterator mcPho = mcPhotons.begin(); mcPho != mcPhotons.end(); mcPho++) {
    float mcPhi = (*mcPho).fourMomentum().phi();
    float mcEta = (*mcPho).fourMomentum().pseudoRapidity();
    mcEta = etaTransformation(mcEta, (*mcPho).primaryVertex().z());

    if (!(fabs(mcEta) <= BARL || (fabs(mcEta) >= END_LO && fabs(mcEta) <= END_HI))) {
      continue;
    }  // all ecal fiducial region

    std::cout << " ConvertedPhotonAnalyzer MC Photons before matching  " << std::endl;
    std::cout << " ConvertedPhotonAnalyzer Photons isAconversion " << (*mcPho).isAConversion()
              << " mcMatchingPhoton energy " << (*mcPho).fourMomentum().e() << " conversion vertex R "
              << (*mcPho).vertex().perp() << " Z " << (*mcPho).vertex().z() << " x " << (*mcPho).vertex().x() << " y "
              << (*mcPho).vertex().y() << " z " << (*mcPho).vertex().z() << std::endl;
    std::cout << " ConvertedPhotonAnalyzer mcEta " << mcEta << " mcPhi " << mcPhi << std::endl;

    h_MCphoE_->Fill((*mcPho).fourMomentum().e());
    h_MCphoEta_->Fill((*mcPho).fourMomentum().eta());
    h_MCphoPhi_->Fill((*mcPho).fourMomentum().phi());

    // keep only visible conversions
    if ((*mcPho).isAConversion() == 0)
      continue;

    nMCPho_++;

    h_MCConvEta_->Fill(fabs((*mcPho).fourMomentum().pseudoRapidity()) - 0.001);

    bool REJECTED;

    /// Loop over recontructed photons
    std::cout << " ConvertedPhotonAnalyzer  Starting loop over photon candidates "
              << "\n";
    for (reco::ConversionCollection::const_iterator iPho = phoCollection.begin(); iPho != phoCollection.end(); iPho++) {
      REJECTED = false;

      std::cout << " ConvertedPhotonAnalyzer Reco SC energy " << (*iPho).caloCluster()[0]->energy() << "\n";

      float phiClu = (*iPho).caloCluster()[0]->phi();
      float etaClu = (*iPho).caloCluster()[0]->eta();
      float deltaPhi = phiClu - mcPhi;
      float deltaEta = etaClu - mcEta;

      if (deltaPhi > Geom::pi())
        deltaPhi -= Geom::twoPi();
      if (deltaPhi < -Geom::pi())
        deltaPhi += Geom::twoPi();
      deltaPhi = std::pow(deltaPhi, 2);
      deltaEta = std::pow(deltaEta, 2);
      float delta = deltaPhi + deltaEta;
      if (delta >= etaPhiDistance)
        REJECTED = true;

      //      if ( ! (  fabs(etaClu) <= BARL || ( fabs(etaClu) >= END_LO && fabs(etaClu) <=END_HI ) ) ) REJECTED=true;

      if (REJECTED)
        continue;
      std::cout << " MATCHED " << std::endl;
      nMatched_++;

      std::cout << " ConvertedPhotonAnalyzer Matching candidate " << std::endl;

      std::cout << " ConvertedPhotonAnalyzer Photons isAconversion " << (*mcPho).isAConversion()
                << " mcMatchingPhoton energy " << (*mcPho).fourMomentum().e()
                << " ConvertedPhotonAnalyzer conversion vertex R " << (*mcPho).vertex().perp() << " Z "
                << (*mcPho).vertex().z() << std::endl;

      h_ErecoEMC_->Fill((*iPho).caloCluster()[0]->energy() / (*mcPho).fourMomentum().e());
      h_deltaPhi_->Fill((*iPho).caloCluster()[0]->position().phi() - mcPhi);
      h_deltaEta_->Fill((*iPho).caloCluster()[0]->position().eta() - mcEta);

      h_scE_->Fill((*iPho).caloCluster()[0]->energy());
      h_scEta_->Fill((*iPho).caloCluster()[0]->position().eta());
      h_scPhi_->Fill((*iPho).caloCluster()[0]->position().phi());

      for (unsigned int i = 0; i < (*iPho).tracks().size(); i++) {
        std::cout << " ConvertedPhotonAnalyzer Reco Track charge " << (*iPho).tracks()[i]->charge()
                  << "  Num of RecHits " << (*iPho).tracks()[i]->recHitsSize() << " inner momentum "
                  << sqrt((*iPho).tracks()[i]->innerMomentum().Mag2()) << "\n";

        h2_tk_nHitsVsR_->Fill((*mcPho).vertex().perp(), (*iPho).tracks()[i]->recHitsSize());
        h2_tk_inPtVsR_->Fill((*mcPho).vertex().perp(), sqrt((*iPho).tracks()[i]->innerMomentum().Mag2()));
      }

    }  /// End loop over Reco  particles

  }  /// End loop over MC particles
}

void SimpleConvertedPhotonAnalyzer::endJob() {
  fOutputFile_->Write();
  fOutputFile_->Close();

  edm::LogInfo("ConvertedPhotonAnalyzer") << "Analyzed " << nEvt_ << "\n";
  // std::cout  << "::endJob Analyzed " << nEvt_ << " events " << " with total " << nPho_ << " Photons " << "\n";
  std::cout << "ConvertedPhotonAnalyzer::endJob Analyzed " << nEvt_ << " events "
            << "\n";

  return;
}

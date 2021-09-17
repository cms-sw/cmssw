#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
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

class PhotonsWithConversionsAnalyzer : public edm::one::EDAnalyzer<> {
public:
  //
  explicit PhotonsWithConversionsAnalyzer(const edm::ParameterSet&);
  ~PhotonsWithConversionsAnalyzer() override;

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

  std::string photonCollectionProducer_;
  std::string photonCollection_;

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
  TH1F* h_scEt_;
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

using namespace std;

PhotonsWithConversionsAnalyzer::PhotonsWithConversionsAnalyzer(const edm::ParameterSet& pset) {
  photonCollectionProducer_ = pset.getParameter<std::string>("phoProducer");
  photonCollection_ = pset.getParameter<std::string>("photonCollection");
  //
}

PhotonsWithConversionsAnalyzer::~PhotonsWithConversionsAnalyzer() { delete thePhotonMCTruthFinder_; }

void PhotonsWithConversionsAnalyzer::beginJob() {
  nEvt_ = 0;
  nMCPho_ = 0;
  nMatched_ = 0;

  thePhotonMCTruthFinder_ = new PhotonMCTruthFinder();

  edm::Service<TFileService> fs;

  /// Reco - MC
  h_ErecoEMC_ = fs->make<TH1F>("deltaE", "    delta(reco-mc) energy", 100, 0., 2.);
  h_deltaPhi_ = fs->make<TH1F>("deltaPhi", "  delta(reco-mc) phi", 100, -0.1, 0.1);
  h_deltaEta_ = fs->make<TH1F>("deltaEta", "  delta(reco-mc) eta", 100, -0.05, 0.05);

  //// All MC photons
  h_MCphoE_ = fs->make<TH1F>("MCphoE", "MC photon energy", 100, 0., 100.);
  h_MCphoPhi_ = fs->make<TH1F>("MCphoPhi", "MC photon phi", 40, -3.14, 3.14);
  h_MCphoEta_ = fs->make<TH1F>("MCphoEta", "MC photon eta", 40, -3., 3.);

  //// MC Converted photons
  h_MCConvE_ = fs->make<TH1F>("MCConvE", "MC converted photon energy", 100, 0., 100.);
  h_MCConvPt_ = fs->make<TH1F>("MCConvPt", "MC converted photon pt", 100, 0., 100.);
  h_MCConvEta_ = fs->make<TH1F>("MCConvEta", "MC converted photon eta", 50, 0., 2.5);

  //// Reconstructed Converted photons
  h_scE_ = fs->make<TH1F>("scE", "SC Energy ", 100, 0., 200.);
  h_scEt_ = fs->make<TH1F>("scEt", "SC Et ", 100, 0., 200.);
  h_scEta_ = fs->make<TH1F>("scEta", "SC Eta ", 40, -3., 3.);
  h_scPhi_ = fs->make<TH1F>("scPhi", "SC Phi ", 40, -3.14, 3.14);
  //
  h_phoE_ = fs->make<TH1F>("phoE", "Photon Energy ", 100, 0., 200.);
  h_phoEta_ = fs->make<TH1F>("phoEta", "Photon Eta ", 40, -3., 3.);
  h_phoPhi_ = fs->make<TH1F>("phoPhi", "Photon  Phi ", 40, -3.14, 3.14);

  // Recontructed tracks from converted photon candidates
  h2_tk_nHitsVsR_ = fs->make<TH2F>("tknHitsVsR", "Tracks Hits vs R  ", 12, 0., 120., 20, 0.5, 20.5);
  h2_tk_inPtVsR_ = fs->make<TH2F>("tkInPtvsR", "Tracks inner Pt vs R  ", 12, 0., 120., 100, 0., 100.);

  return;
}

float PhotonsWithConversionsAnalyzer::etaTransformation(float EtaParticle, float Zvertex) {
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

void PhotonsWithConversionsAnalyzer::analyze(const edm::Event& e, const edm::EventSetup&) {
  using namespace edm;
  const float etaPhiDistance = 0.01;
  // Fiducial region
  //UNUSED const float TRK_BARL =0.9;
  //UNUSED const float BARL = 1.4442; // DAQ TDR p.290
  //UNUSED const float END_LO = 1.566;
  //UNUSED const float END_HI = 2.5;
  // Electron mass
  //UNUSED const Float_t mElec= 0.000511;

  nEvt_++;
  LogInfo("ConvertedPhotonAnalyzer") << "ConvertedPhotonAnalyzer Analyzing event number: " << e.id()
                                     << " Global Counter " << nEvt_ << "\n";
  //  LogDebug("ConvertedPhotonAnalyzer") << "ConvertedPhotonAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
  std::cout << "ConvertedPhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ << "\n";

  ///// Get the recontructed  conversions
  Handle<reco::PhotonCollection> photonHandle;
  e.getByLabel(photonCollectionProducer_, photonCollection_, photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());
  std::cout << "ConvertedPhotonAnalyzer  Photons with conversions collection size " << photonCollection.size() << "\n";

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

    if ((*mcPho).fourMomentum().et() < 20)
      continue;
    //    if ( ! (  fabs(mcEta) <= BARL || ( fabs(mcEta) >= END_LO && fabs(mcEta) <=END_HI ) ) ) {
    //     continue;
    //} // all ecal fiducial region

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
    //std::cout   << " ConvertedPhotonAnalyzer  Starting loop over photon candidates " << "\n";
    for (reco::PhotonCollection::const_iterator iPho = photonCollection.begin(); iPho != photonCollection.end();
         iPho++) {
      REJECTED = false;

      //      std::cout  << " ConvertedPhotonAnalyzer Reco SC energy " << (*iPho).superCluster()->energy() <<  "\n";

      float phiClu = (*iPho).superCluster()->phi();
      float etaClu = (*iPho).superCluster()->eta();
      float deltaPhi = phiClu - mcPhi;
      float deltaEta = etaClu - mcEta;

      if (deltaPhi > Geom::pi())
        deltaPhi -= Geom::twoPi();
      if (deltaPhi < -Geom::pi())
        deltaPhi += Geom::twoPi();
      deltaPhi = pow(deltaPhi, 2);
      deltaEta = pow(deltaEta, 2);
      float delta = deltaPhi + deltaEta;
      if (delta >= etaPhiDistance)
        REJECTED = true;

      //      if ( ! (  fabs(etaClu) <= BARL || ( fabs(etaClu) >= END_LO && fabs(etaClu) <=END_HI ) ) ) REJECTED=true;

      if (REJECTED)
        continue;
      std::cout << " MATCHED " << std::endl;
      nMatched_++;

      //      std::cout << " ConvertedPhotonAnalyzer Matching candidate " << std::endl;

      // std::cout << " ConvertedPhotonAnalyzer Photons isAconversion " << (*mcPho).isAConversion() << " mcMatchingPhoton energy " <<  (*mcPho).fourMomentum().e()  << " ConvertedPhotonAnalyzer conversion vertex R " <<  (*mcPho).vertex().perp() << " Z " <<  (*mcPho).vertex().z() <<  std::endl;

      h_ErecoEMC_->Fill((*iPho).superCluster()->energy() / (*mcPho).fourMomentum().e());
      h_deltaPhi_->Fill((*iPho).superCluster()->position().phi() - mcPhi);
      h_deltaEta_->Fill((*iPho).superCluster()->position().eta() - mcEta);

      h_scE_->Fill((*iPho).superCluster()->energy());
      h_scEt_->Fill((*iPho).superCluster()->energy() / cosh((*iPho).superCluster()->position().eta()));
      h_scEta_->Fill((*iPho).superCluster()->position().eta());
      h_scPhi_->Fill((*iPho).superCluster()->position().phi());

      h_phoE_->Fill((*iPho).energy());
      h_phoEta_->Fill((*iPho).eta());
      h_phoPhi_->Fill((*iPho).phi());

      if (!(*iPho).hasConversionTracks())
        continue;
      //   std::cout << " This photons has " << (*iPho).conversions().size() << " conversions candidates " << std::endl;
      reco::ConversionRefVector conversions = (*iPho).conversions();
      //std::vector<reco::ConversionRef> conversions = (*iPho).conversions();

      for (unsigned int i = 0; i < conversions.size(); i++) {
        //std::cout << " Conversion candidate Energy " << (*iPho).energy() << " number of tracks " << conversions[i]->nTracks() << std::endl;
        std::vector<edm::RefToBase<reco::Track> > tracks = conversions[i]->tracks();

        for (unsigned int i = 0; i < tracks.size(); i++) {
          //	  std::cout  << " ConvertedPhotonAnalyzer Reco Track charge " <<  tracks[i]->charge() << "  Num of RecHits " << tracks[i]->recHitsSize() << " inner momentum " <<  sqrt ( tracks[i]->innerMomentum().Mag2() )  <<  "\n";

          h2_tk_nHitsVsR_->Fill((*mcPho).vertex().perp(), tracks[i]->recHitsSize());
          h2_tk_inPtVsR_->Fill((*mcPho).vertex().perp(), sqrt(tracks[i]->innerMomentum().Mag2()));
        }
      }

    }  /// End loop over Reco  particles

  }  /// End loop over MC particles
}

void PhotonsWithConversionsAnalyzer::endJob() {
  //   fOutputFile_->Write() ;
  // fOutputFile_->Close() ;

  edm::LogInfo("ConvertedPhotonAnalyzer") << "Analyzed " << nEvt_ << "\n";
  // std::cout  << "::endJob Analyzed " << nEvt_ << " events " << " with total " << nPho_ << " Photons " << "\n";
  std::cout << "ConvertedPhotonAnalyzer::endJob Analyzed " << nEvt_ << " events "
            << "\n";

  return;
}

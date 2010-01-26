#ifndef IsolatedTracksTree_h
#define IsolatedTracksTree_h

// system include files
#include <memory>
#include <cmath>
#include <string>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <Math/GenVector/VectorUtil.h>
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Candidate/interface/Candidate.h"

// muons and tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
// track associator
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
// SimHit
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
//simtrack
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
// ecal / hcal
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
//L1 objects
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

//TFile Service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "Calibration/IsolatedParticles/interface/MatchingSimTrack.h"

#include "Calibration/IsolatedParticles/interface/CaloSimInfo.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TDirectory.h"

#include "TTree.h"

#include <cmath>

class IsolatedTracksTree : public edm::EDAnalyzer {
 public:
  explicit IsolatedTracksTree(const edm::ParameterSet&);
  ~IsolatedTracksTree();
  
 private:
  void   beginJob(const edm::EventSetup&) ;
  void   analyze(const edm::Event&, const edm::EventSetup&);
  void   endJob() ;

  void   printTrack(const reco::Track* pTrack);

  void   BookHistograms();

  double DeltaPhi(double v1, double v2);
  double DeltaR(double eta1, double phi1, double eta2, double phi2);

  int    chargeIsolation(CaloNavigator<DetId>& navigator,const DetId anyCell, int deta, int dphi);
  double chargeIsolation(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			 CaloNavigator<DetId>& navigator, 
			 reco::TrackCollection::const_iterator trkItr, 
			 edm::Handle<reco::TrackCollection> trkCollection, 
			 edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle, 
			 edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle, 
			 const CaloSubdetectorGeometry* gEB, const CaloSubdetectorGeometry* gEE, 
			 int ieta, int iphi);

  double chargeIsolationHcal(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			     reco::TrackCollection::const_iterator trkItr, 
			     edm::Handle<reco::TrackCollection> trkCollection,
			     const DetId ClosestCell, const HcalTopology* topology, 
			     const CaloSubdetectorGeometry* gHB,
			     int ieta, int iphi, bool debug=false);
  
 
  int    debugTrks_;
  bool   printTrkHitPattern_;
  int    myverbose_;
  bool   useJetTrigger_;
  double drLeadJetVeto_, ptMinLeadJet_;
  edm::InputTag L1extraTauJetSource_, L1extraCenJetSource_, L1extraFwdJetSource_;

  int    debugEcalSimInfo_;
  bool   applyEcalIsolation_;

  double minTrackP_, maxTrackEta_, maxNearTrackPT_;

  int    nEventProc;
  double genPartPBins[22], genPartEtaBins[5];

  // track associator to detector parameters 
  TrackAssociatorParameters parameters_;
  mutable TrackDetectorAssociator* trackAssociator_;

  TH1F *h_nTracks;

  TH1F *h_recPt_0,    *h_recP_0,    *h_recEta_0, *h_recPhi_0;
  TH2F *h_recEtaPt_0, *h_recEtaP_0;

  TH1F *h_recPt_1,    *h_recP_1,    *h_recEta_1, *h_recPhi_1;
  TH2F *h_recEtaPt_1, *h_recEtaP_1;

  TH1F *h_recPt_2,    *h_recP_2,    *h_recEta_2, *h_recPhi_2;
  TH2F *h_recEtaPt_2, *h_recEtaP_2;
 
  TTree* tree;

  int    t_nEvtProc;

  int    t_nTracks, t_nTracksAwayL1, t_nTracksIsoBy5GeV;

  int    t_infoHcal[200];
  double t_maxNearP[200];
  double t_maxNearP15x15[200], t_maxNearP21x21[200], t_maxNearP31x31[200];

  int    t_NLayersCrossed[200], t_trackNOuterHits[200];
  double t_trackP[200], t_trackPt[200], t_trackEta[200], t_trackPhi[200];
  
  double t_e3x3[200],    t_e5x5[200],    t_e7x7[200],    t_e9x9[200],    t_e11x11[200],    t_e13x13[200],    t_e15x15[200],    t_e25x25[200];
  double t_esim3x3[200], t_esim5x5[200], t_esim7x7[200], t_esim9x9[200], t_esim11x11[200], t_esim13x13[200], t_esim15x15[200], t_esim25x25[200];

  double t_trkEcalEne[200];

  double t_esim3x3PdgId[200], t_esim5x5PdgId[200], t_esim7x7PdgId[200], t_esim9x9PdgId[200], t_esim11x11PdgId[200], t_esim13x13PdgId[200], t_esim15x15PdgId[200];
  double t_esim3x3Matched[200], t_esim5x5Matched[200], t_esim7x7Matched[200], t_esim9x9Matched[200], t_esim11x11Matched[200], t_esim13x13Matched[200], t_esim15x15Matched[200];
  double t_esim3x3Rest[200], t_esim5x5Rest[200], t_esim7x7Rest[200], t_esim9x9Rest[200], t_esim11x11Rest[200], t_esim13x13Rest[200], t_esim15x15Rest[200];
  double t_esim3x3Photon[200], t_esim5x5Photon[200], t_esim7x7Photon[200], t_esim9x9Photon[200], t_esim11x11Photon[200], t_esim13x13Photon[200], t_esim15x15Photon[200];
  double t_esim3x3NeutHad[200], t_esim5x5NeutHad[200], t_esim7x7NeutHad[200], t_esim9x9NeutHad[200], t_esim11x11NeutHad[200], t_esim13x13NeutHad[200], t_esim15x15NeutHad[200];
  double t_esim3x3CharHad[200], t_esim5x5CharHad[200], t_esim7x7CharHad[200], t_esim9x9CharHad[200], t_esim11x11CharHad[200], t_esim13x13CharHad[200], t_esim15x15CharHad[200];

  double t_simTrackP[200];

  double t_trkHcalEne[200];
  double t_h3x3[200],    t_h5x5[200],    t_h7x7[200];
  double t_hsim3x3[200], t_hsim5x5[200], t_hsim7x7[200];

  double t_hsim3x3Matched[200], t_hsim5x5Matched[200], t_hsim7x7Matched[200];
  double t_hsim3x3Rest[200],    t_hsim5x5Rest[200],    t_hsim7x7Rest[200];
  double t_hsim3x3Photon[200],  t_hsim5x5Photon[200],  t_hsim7x7Photon[200];
  double t_hsim3x3NeutHad[200], t_hsim5x5NeutHad[200], t_hsim7x7NeutHad[200];
  double t_hsim3x3CharHad[200], t_hsim5x5CharHad[200], t_hsim7x7CharHad[200];

  double t_trkHcalEne_1[200];
  double t_h3x3_1[200],    t_h5x5_1[200],    t_h7x7_1[200];
  double t_hsim3x3_1[200], t_hsim5x5_1[200], t_hsim7x7_1[200];
  double t_maxNearHcalP3x3_1[200],  t_maxNearHcalP5x5_1[200],  t_maxNearHcalP7x7_1[200];

  double t_hsim3x3Matched_1[200], t_hsim5x5Matched_1[200], t_hsim7x7Matched_1[200];
  double t_hsim3x3Rest_1[200],    t_hsim5x5Rest_1[200],    t_hsim7x7Rest_1[200];
  double t_hsim3x3Photon_1[200],  t_hsim5x5Photon_1[200],  t_hsim7x7Photon_1[200];
  double t_hsim3x3NeutHad_1[200], t_hsim5x5NeutHad_1[200], t_hsim7x7NeutHad_1[200];
  double t_hsim3x3CharHad_1[200], t_hsim5x5CharHad_1[200], t_hsim7x7CharHad_1[200];

  edm::Service<TFileService> fs;

};

#endif

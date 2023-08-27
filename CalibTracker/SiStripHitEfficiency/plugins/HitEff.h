// system includes
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstdio>

// user includes
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/DetId/interface/DetIdVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

// ROOT includes
#include "TRandom2.h"
#include "TROOT.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"

class TrackerTopology;

class HitEff : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HitEff(const edm::ParameterSet& conf);
  ~HitEff() override = default;

private:
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<LumiScalersCollection> scalerToken_;
  const edm::EDGetTokenT<OnlineLuminosityRecord> metaDataToken_;
  const edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > commonModeToken_;

  SiStripClusterInfo siStripClusterInfo_;

  bool addLumi_;
  bool addCommonMode_;
  bool cutOnTracks_;
  unsigned int trackMultiplicityCut_;
  bool useFirstMeas_;
  bool useLastMeas_;
  bool useAllHitsFromTracksWithMissingHits_;
  bool doMissingHitsRecovery_;

  const edm::EDGetTokenT<reco::TrackCollection> combinatorialTracks_token_;
  const edm::EDGetTokenT<std::vector<Trajectory> > trajectories_token_;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackAsso_token_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusters_token_;
  const edm::EDGetTokenT<DetIdCollection> digisCol_token_;
  const edm::EDGetTokenT<DetIdVector> digisVec_token_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> trackerEvent_token_;

  // ES tokens

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<StripClusterParameterEstimator, TkStripCPERecord> cpeToken_;
  const edm::ESGetToken<SiStripQuality, SiStripQualityRcd> siStripQualityToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<MeasurementTracker, CkfComponentsRecord> measurementTkToken_;
  const edm::ESGetToken<Chi2MeasurementEstimatorBase, TrackingComponentsRecord> chi2MeasurementEstimatorToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;

  edm::ParameterSet conf_;

  TTree* traj;
  int events, EventTrackCKF;

  int compSettings;
  unsigned int layers;
  bool DEBUG;
  unsigned int whatlayer;

  std::vector<unsigned int> hitRecoveryCounters;
  std::vector<unsigned int> hitTotalCounters;
// Tree declarations
// Trajectory positions for modules included in the study
#ifdef ExtendedCALIBTree
  float timeDT, timeDTErr;
  int timeDTDOF;
  float timeECAL, dedx;
  int dedxNOM;
  int nLostHits;
  float p, chi2;
#endif
  int totalNbHits;
  std::vector<int> missHitPerLayer;
  float TrajGlbX, TrajGlbY, TrajGlbZ;
  float TrajLocX, TrajLocY, TrajLocAngleX, TrajLocAngleY;
  float TrajLocErrX, TrajLocErrY;
  float ClusterLocX, ClusterLocY, ClusterLocErrX, ClusterLocErrY, ClusterStoN;
  float ResX, ResXSig;
  unsigned int ModIsBad;
  unsigned int Id;
  unsigned int SiStripQualBad;
  bool withinAcceptance;
  bool highPurity;
  int nHits;
  float pT;
  unsigned int trajHitValid, run, event, bunchx;
  int tquality;
  float instLumi, PU;
  float commonMode;
};

//#endif

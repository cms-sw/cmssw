#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"

#include "TROOT.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include <vector>
#include "TTree.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "Riostream.h"
#include "TRandom2.h"

class TrackerTopology;

class HitEff : public edm::EDAnalyzer {
public:
  explicit HitEff(const edm::ParameterSet& conf);
  double checkConsistency(const StripClusterParameterEstimator::LocalValues& parameters, double xx, double xerr);
  bool isDoubleSided(unsigned int iidd, const TrackerTopology* tTopo) const;
  bool check2DPartner(unsigned int iidd, const std::vector<TrajectoryMeasurement>& traj);
  ~HitEff() override;
  unsigned int checkLayer(unsigned int iidd, const TrackerTopology* tTopo);

private:
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<LumiScalersCollection> scalerToken_;
  const edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > commonModeToken_;

  SiStripClusterInfo siStripClusterInfo_;

  bool addLumi_;
  bool addCommonMode_;
  bool cutOnTracks_;
  unsigned int trackMultiplicityCut_;
  bool useFirstMeas_;
  bool useLastMeas_;
  bool useAllHitsFromTracksWithMissingHits_;

  const edm::EDGetTokenT<reco::TrackCollection> combinatorialTracks_token_;
  const edm::EDGetTokenT<std::vector<Trajectory> > trajectories_token_;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> trajTrackAsso_token_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusters_token_;
  const edm::EDGetTokenT<DetIdCollection> digis_token_;
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

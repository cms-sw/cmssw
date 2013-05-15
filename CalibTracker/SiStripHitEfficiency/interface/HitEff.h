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
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h" 
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"


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
  virtual ~HitEff();
  unsigned int checkLayer(unsigned int iidd, const TrackerTopology* tTopo);

 private:
  virtual void beginJob();
  virtual void endJob(); 
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

        // ----------member data ---------------------------

  edm::ParameterSet conf_;
  
  TTree* traj;
  int events,EventTrackCKF;
  
  unsigned int layers;
  bool DEBUG;
  unsigned int whatlayer;
  
  // Tree declarations
  // Trajectory positions for modules included in the study
  float TrajGlbX, TrajGlbY, TrajGlbZ;
  float TrajLocX, TrajLocY, TrajLocErrX, TrajLocErrY, TrajLocAngleX, TrajLocAngleY;
  float ClusterLocX, ClusterLocY, ClusterLocErrX, ClusterLocErrY, ClusterStoN;
  float ResX, ResXSig;
  unsigned int ModIsBad; unsigned int Id; unsigned int SiStripQualBad; bool withinAcceptance;
  int nHits, nLostHits; 
  float p, pT, chi2;
  unsigned int trajHitValid, run, event, bunchx;
  float timeDT, timeDTErr;
  int timeDTDOF;
  float timeECAL, dedx;
  int dedxNOM;
  int tquality;
  int istep;
};


//#endif

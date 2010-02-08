#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/EDProduct.h"
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

#include "TROOT.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include <vector>
#include "TTree.h"
#include <iostream>
#include "stdlib.h"
#include "stdio.h"
#include "Riostream.h"
#include "TRandom2.h"


class HitEff : public edm::EDAnalyzer {
 public:  
  explicit HitEff(const edm::ParameterSet& conf);
  double checkConsistency(StripClusterParameterEstimator::LocalValues parameters, double xx, double xerr);
  double checkConsistency(const SiStripRecHit2D* rechit, double xx, double xerr);
  bool isDoubleSided(uint iidd) const;
  bool check2DPartner(uint iidd, std::vector<TrajectoryMeasurement> traj);
  virtual ~HitEff();

 private:
  virtual void beginJob(const edm::EventSetup& c);
  virtual void endJob(); 
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

        // ----------member data ---------------------------

  edm::ParameterSet conf_;
  
  TTree* traj;
  int events,EventTrackCKF;
  
  uint layers;
  bool DEBUG;
  uint whatlayer;
  
  // Tree declarations
  // Trajectory positions for modules included in the study
  float TrajGlbX, TrajGlbY, TrajGlbZ;
  float TrajLocX, TrajLocY, TrajLocErrX, TrajLocErrY, TrajLocAngleX, TrajLocAngleY;
  float ClusterLocX, ClusterLocY, ClusterLocErrX, ClusterLocErrY, ClusterStoN;
  float ResX, ResXSig;
  uint ModIsBad; uint Id; uint SiStripQualBad; bool withinAcceptance;
  uint run; uint event;
  float timeDT, timeDTErr;
  int timeDTDOF;
  float timeECAL, dedx;
  int dedxNOM;
};


//#endif

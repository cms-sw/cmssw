#ifndef CalibTracker_SiPixelLorentzAngle_SiPixelLorentzAngle_h
#define CalibTracker_SiPixelLorentzAngle_SiPixelLorentzAngle_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// #include "CalibTracker/SiPixelLorentzAngle/interface/TrackLocalAngle.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h>
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TF1.h>
#include "TROOT.h"


/**
* 
* Class to determine the lorentz angle in the barrel pixel detector 
* returns a txt file with the fit for every of the 8 rings in the 3 layers
* 
*/

// ggiurgiu@fnal.gov : remove namespace 12/27/09
//namespace
//{
  static const int maxpix = 1000;
  struct Pixinfo
  {
    int npix;
    float row[maxpix];
    float col[maxpix];
    float adc[maxpix];
    float x[maxpix];
    float y[maxpix];
  };
  struct Hit
  {
    float x;
    float y;
    double alpha;
    double beta;
    double gamma;
  };
  struct Clust 
  {
    float x;
    float y;
    float charge;
    int size_x;
    int size_y;
    int maxPixelCol;
    int maxPixelRow;
    int minPixelCol;
    int minPixelRow;
  };
  struct Rechit 
  {
    float x;
    float y;
  };
//}

//SiPixelLorentzAngle is already the name of a data product
namespace analyzer {
class SiPixelLorentzAngle : public edm::EDAnalyzer
{
 public:
  
  explicit SiPixelLorentzAngle(const edm::ParameterSet& conf);
  
  virtual ~SiPixelLorentzAngle();
  //virtual void beginJob(const edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob(); 
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  
 private:
  
  void fillPix(const SiPixelCluster & LocPix, const PixelTopology * topol, Pixinfo& pixinfo);


  void findMean(int i, int i_ring);
  
  TFile* hFile_;
  TTree* SiPixelLorentzAngleTree_;
  TTree* SiPixelLorentzAngleTreeForward_;
  
  // tree branches barrel
  int run_;
  int event_;
  int module_;
  int ladder_;
  int layer_;
  int isflipped_;
  float pt_;
  float eta_;
  float phi_;
  double chi2_;
  double ndof_;
  Pixinfo pixinfo_;
  Hit simhit_, trackhit_;
  Clust clust_;
  Rechit rechit_;
  
  // tree branches forward
  int runF_;
  int eventF_;  
  int sideF_;
  int diskF_;
  int bladeF_;
  int panelF_;
  int moduleF_;
  float ptF_;
  float etaF_;
  float phiF_;
  double chi2F_;
  double ndofF_;
  Pixinfo pixinfoF_;
  Hit simhitF_, trackhitF_;
  Clust clustF_;
  Rechit rechitF_;
  
  // parameters from config file
  edm::ParameterSet conf_;
  std::string filename_;
  std::string filenameFit_;
  double ptmin_;
  bool simData_;
  double normChi2Max_;
  int clustSizeYMin_;
  double residualMax_;
  double clustChargeMax_;
  int hist_depth_;
  int hist_drift_;
  
  // histogram etc
  int hist_x_;
  int hist_y_;
  double min_x_;
  double max_x_;
  double min_y_;
  double max_y_;
  double width_;
  double min_depth_;
  double max_depth_;
  double min_drift_;
  double max_drift_;
  
  std::map<int, TH2F*> _h_drift_depth_adc_;
  std::map<int, TH2F*> _h_drift_depth_adc2_;
  std::map<int, TH2F*> _h_drift_depth_noadc_;
  std::map<int, TH2F*> _h_drift_depth_;
  TH1F* h_drift_depth_adc_slice_;
  std::map<int, TH1F*> _h_mean_;
  TH2F *h_cluster_shape_adc_;
  TH2F *h_cluster_shape_noadc_;
  TH2F *h_cluster_shape_;
  TH2F *h_cluster_shape_adc_rot_;
  TH2F *h_cluster_shape_noadc_rot_;
  TH2F *h_cluster_shape_rot_;
  TH1F *h_tracks_;
  
  
  int event_counter_, trackEventsCounter_,pixelTracksCounter_, hitCounter_, usedHitCounter_;
  
  // CMSSW classes needed
  PropagatorWithMaterial  *thePropagator;
  PropagatorWithMaterial  *thePropagatorOp;
  KFUpdator *theUpdator;
  Chi2MeasurementEstimator *theEstimator;
  const TransientTrackingRecHitBuilder *RHBuilder;
  const KFTrajectorySmoother * theSmoother;
  const KFTrajectoryFitter * theFitter;
  const TrackerGeometry * tracker;
  const MagneticField * magfield;
  TrajectoryStateTransform tsTransform;
  edm::EDGetTokenT<TrajTrackAssociationCollection> t_trajTrack;
  
};
}

#endif

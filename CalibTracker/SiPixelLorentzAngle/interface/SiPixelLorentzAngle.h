#ifndef CalibTracker_SiPixelLorentzAngle_SiPixelLorentzAngle_h
#define CalibTracker_SiPixelLorentzAngle_SiPixelLorentzAngle_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/Common/interface/EDProduct.h"
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
#include "CalibTracker/SiPixelLorentzAngle/interface/TrackLocalAngle.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h>
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TF1.h>
#include <TProfile.h>

class SiPixelLorentzAngle : public edm::EDAnalyzer
{
 public:
  
  explicit SiPixelLorentzAngle(const edm::ParameterSet& conf);
  
  virtual ~SiPixelLorentzAngle();
  virtual void beginJob(const edm::EventSetup& c);
  virtual void endJob(); 
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  void findtrackangle(const TrajectorySeed& seed,
				       const TrackingRecHitCollection &hits,
				       const edm::Event& e, 
					const edm::EventSetup& es);
  TrajectoryStateOnSurface startingTSOS(const TrajectorySeed& seed)const;
  
 private:
 
  edm::ParameterSet conf_;
  std::string filename_;
  
  TrackLocalAngle *anglefinder_;
	
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
  int eventcounter_, eventnumber_, trackcounter_;
	
	TrackLocalAngle::Trackhit trackhit_;
	
	static const int maxpix = 100;
	
	struct Pixinfo
	{
		int npix;
		float row[maxpix];
		float col[maxpix];
		float adc[maxpix];
  	float x[maxpix];
		float y[maxpix];
	} pixinfo_;
	
	struct Simhit{
		float x;
		float y;
		double alpha;
		double beta;
		double gamma;
	} simhit_;
	
	struct Clust {
		float x;
		float y;
		float charge;
		int size_x;
		int size_y;
		int maxPixelCol;
		int maxPixelRow;
		int minPixelCol;
		int minPixelRow;
	} clust_;
	
	struct Rechit {
		float x;
		float y;
	} rechit_;
	
	void fillPix(const SiPixelCluster & LocPix, const RectangularPixelTopology * topol);

  TFile* hFile_;
  TTree* SiPixelLorentzAngleTree_;
	
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
	int hist_depth_;
	int hist_drift_;
	
	TH2F *h_cluster_shape_adc_;
	TH2F *h_cluster_shape_noadc_;
	TH2F *h_cluster_shape_;
	TH2F *h_cluster_shape_adc_rot_;
	TH2F *h_cluster_shape_noadc_rot_;
	TH2F *h_cluster_shape_rot_;
	TH2F *h_drift_depth_adc_;
	TH2F *h_drift_depth_noadc_;
	TH2F *h_drift_depth_;
	TH2F *h_drift_depth_int_;
		
	bool seed_plus_;
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
  
};


#endif

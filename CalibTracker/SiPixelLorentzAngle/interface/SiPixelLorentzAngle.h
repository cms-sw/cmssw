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
#include <TH3F.h>
#include <TF1.h>
#include <TProfile.h>
#include <TMatrixFSym.h>
#include <TVectorF.h>
#include <TMinuit.h>
#include "TROOT.h"

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
// 	void fit_chi2_fcn(int &npar, double *gin, double &chi2, double* par, int iflag);
	
 private:
	 
// 	void fit_chi2_fcn(int &npar, double *gin, double &chi2, double* par, int iflag);
	void fillPix(const SiPixelCluster & LocPix, const RectangularPixelTopology * topol);
	void fillHistograms();
	void fitLinear(int i);
	
  edm::ParameterSet conf_;
  std::string filename_;
	int event_counter_;
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
	
// 	int lower_bin_;
	int upper_bin_;
	
	// trackhit histograms
	TH2F *h_cluster_shape_adc_;
	TH2F *h_cluster_shape_noadc_;
	TH2F *h_cluster_shape_;
	TH2F *h_cluster_shape_adc_rot_;
	TH2F *h_cluster_shape_noadc_rot_;
	TH2F *h_cluster_shape_rot_;
	TH2F *h_drift_depth_adc_;
	TH2F *h_drift_depth_adc2_;
	TH2F *h_drift_depth_adc_error2_;
	TH2F *h_drift_depth_noadc_;
	TH2F *h_drift_depth_;
	TH2F *h_drift_depth_error2_;
	TH2F *h_drift_depth_int_;
	TH3F *h_drift_depth_int_error_matrix_;
	TH1F *h_drift_depth_int_slice_;
	TH1F *h_drift_depth_adc_slice_;
// 	TH1F h_drift_depth_int_slice2_;
	TH2F *h_drift_depth_int_slice_error_matrix_;
	TH1F *h_fit_middle_;
	TH1F *h_fit_width_;
	TH1F *h_mean_;
	std::map<int, TH1F*> _h_drift_depth_int_slice_;
	std::map<int, TF1*> _f_drift_depth_int_slice_;
	std::map<int, TH1F*> _h_drift_depth_adc_slice_;
	
	// simhit histograms
	TH2F *h_cluster_shape_adc_sim_;
	TH2F *h_cluster_shape_noadc_sim_;
	TH2F *h_cluster_shape_sim_;
	TH2F *h_cluster_shape_adc_rot_sim_;
	TH2F *h_cluster_shape_noadc_rot_sim_;
	TH2F *h_cluster_shape_rot_sim_;
	TH2F *h_drift_depth_adc_sim_;
	TH2F *h_drift_depth_adc2_sim_;
	TH2F *h_drift_depth_adc_error2_sim_;
	TH2F *h_drift_depth_noadc_sim_;
	TH2F *h_drift_depth_sim_;
	TH2F *h_drift_depth_error2_sim_;
	TH2F *h_drift_depth_int_sim_;
	TH3F *h_drift_depth_int_error_matrix_sim_;
	TH1F *h_drift_depth_int_slice_sim_;
	TH1F *h_drift_depth_adc_slice_sim_;
// 	TH1F h_drift_depth_int_slice2_;
	TH2F *h_drift_depth_int_slice_error_matrix_sim_;
	TH1F *h_fit_middle_sim_;
	TH1F *h_fit_width_sim_;
	TH1F *h_mean_sim_;
	
	std::map<int, TH1F*> _h_drift_depth_int_slice_sim_;
	std::map<int, TF1*> _f_drift_depth_int_slice_sim_;
	std::map<int, TH1F*> _h_drift_depth_adc_slice_sim_;
	
	TMatrixFSym *m_covariance_;
// 	TMatrixFSym m_covariance_inv_;
// 	TVectorF *v_chi2_;
	float data_covariance_[10000];
// 	float* data_covariance_;
// 	float * delta_x_;
	TMinuit* minuit;	
	
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

#ifndef AnalysisExamples_SiStripDetectorPerformance_TIFNtupleMaker_h
#define AnalysisExamples_SiStripDetectorPerformance_TIFNtupleMaker_h

#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
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
#include "DataFormats/DetId/interface/DetId.h"

//#include "CommonTools/SiStripZeroSuppression/interface/SiStripNoiseService.h"

#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackLocalAngleTIF.h"

#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TH1F.h>
#include <TF1.h>
#include <TProfile.h>
#include <TFolder.h>
#include <TDirectory.h>
#include <TAxis.h>
#include <TMath.h>


class TIFNtupleMaker : public edm::EDAnalyzer
{
 public:
  
  explicit TIFNtupleMaker(const edm::ParameterSet& conf);
  
  virtual ~TIFNtupleMaker();
  
  virtual void beginJob(const edm::EventSetup& c);
  
  virtual void endJob(); 
  
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  void findtrackangle(const TrajectorySeed& seed,
		      const TrackingRecHitCollection &hits,
		      const edm::Event& e, 
		      const edm::EventSetup& es);
  
  TrajectoryStateOnSurface startingTSOS(const TrajectorySeed& seed)const;
  
  const char* makename(DetId detid);
  
  const char* makedescription(DetId detid);
  
 private:

//  const TrackerGeometry::DetIdContainer& Id;
  
  typedef std::map <int, TProfile*> histomap;
  typedef StripSubdetector::SubDetector SiSubDet;
  typedef std::map<SiSubDet, std::vector<TH1D *> > DetPlots;
  
  histomap	      histos;
  DetPlots	      oDetPlots;
  std::vector<TH1D *> oGlobalPlots;
  
  typedef struct {double chi2; int ndf; double p0; double p1; double p2; double errp0; double errp1; double errp2; double min;} histofit ;
  typedef std::map <int, histofit*> fitmap;
  fitmap fits;
  
  typedef std::vector<std::pair<const TrackingRecHit *,float> > hitanglevector;  
  typedef std::map <const reco::Track *, hitanglevector> trackhitmap;
  typedef std::map <const reco::Track *, TrackLocalAngleTIF::HitLclDirAssociation> trklcldirmap;
  typedef std::map <const reco::Track *, TrackLocalAngleTIF::HitGlbDirAssociation> trkglbdirmap;
  typedef std::vector<SiStripDigi> DigisVector;

  // New provate methods
  // De Mattia - 24/1/2007
  void _summaryHistos();
  void _directoryHierarchy();
  // -------------------

  double getClusterEta( const std::vector<uint16_t> &roSTRIP_AMPLITUDES,
			const int		    &rnFIRST_STRIP,
			const DigisVector	    &roDIGIS) const;
  double getClusterCrossTalk( const std::vector<uint16_t> &roSTRIP_AMPLITUDES,
			      const int		          &rnFIRST_STRIP,
			      const DigisVector	          &roDIGIS) const;
  // First argument is double to simplify calculations otherwise convertion
  // to double should be performed each time division is used
  double calculateClusterCrossTalk( const double &rdADC_STRIPL,
                                    const int    &rnADC_STRIP,
				    const int    &rnADC_STRIPR) const;
    
  edm::ParameterSet conf_;
  std::string filename_;
  std::string oSiStripDigisLabel_;
  std::string oSiStripDigisProdInstName_;
  bool bUseLTCDigis_;
  const double dCROSS_TALK_ERR;
  //SiStripNoiseService m_oSiStripNoiseService; 
  
  std::vector<DetId> Detvector;
  
  TrackLocalAngleTIF *anglefinder_;
  
  int tiftibcorr, tiftobcorr;
  
  int monodscounter;
  int monosscounter;
  int stereocounter;
  
  int run;
  int event;
  int size;
  int module;
  int string;
  int rod;
  int extint;
  int bwfw;
  int wheel;
  int type;
  int layer;
  int sign;
  int charge;
  int hitspertrack;

  float clusterpos;
  float clustereta;
  float clusterchg;
  float clusterchgl;
  float clusterchgr;
  float clusternoise;
  float clustermaxchg;
  float clusterbarycenter;
  float clusterseednoise;
  float clustercrosstalk;

  float angle;
  float tk_phi;
  float tk_theta;
  float stereocorrection;
  float localmagfield;
  int monostereo;
  float momentum, pt;
  float eta;
  float phi;
  float normchi2;
  float chi2;
  float ndof;
  bool bTrack;
  bool bTriggerDT;
  bool bTriggerCSC;
  bool bTriggerRBC1;
  bool bTriggerRBC2;
  bool bTriggerRPC;
  float dLclX;
  float dLclY;
  float dLclZ;
  float dGlbX;
  float dGlbY;
  float dGlbZ;

  // For tracks number
  int numberoftracks;
  int numberofclusters;
  int numberoftkclusters;
  int numberofnontkclusters;
  // -----------------

  int eventcounter, trackcounter, hitcounter;
  LocalVector localmagdir;
    
  TF1* fitfunc;
  TFile* hFile;
  TTree* TIFNtupleMakerTree;
  TTree	*poTrackTree;
  TTree	*poTrackNum;
   
  TH1F  *poClusterChargeTH1F;
  TH1F  *hphi, *hnhit;
  TH1F  *hEventTrackNumber;
  TH1F  *htaTIBL1mono;
  TH1F  *htaTIBL1stereo;
  TH1F  *htaTIBL2mono;
  TH1F  *htaTIBL2stereo;
  TH1F  *htaTIBL3;
  TH1F  *htaTIBL4;
  TH1F  *htaTOBL1mono;
  TH1F  *htaTOBL1stereo;
  TH1F  *htaTOBL2mono;
  TH1F  *htaTOBL2stereo;
  TH1F  *htaTOBL3;
  TH1F  *htaTOBL4;
  TH1F  *htaTOBL5;
  TH1F  *htaTOBL6;
  TProfile *hwvst;
      
  bool seed_plus;
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
    
  //Directory hierarchy  
  
  TDirectory *histograms;
  TDirectory *summary;  
  
  //TIB-TID-TOB-TEC    
  
  TDirectory *TIB;
  TDirectory *TOB;
  TDirectory *TID;
  TDirectory *TEC;
  
  //Forward-Backward
  
  TDirectory *TIBfw;
  TDirectory *TIDfw;
  TDirectory *TOBfw;
  TDirectory *TECfw;
  
  TDirectory *TIBbw;
  TDirectory *TIDbw;
  TDirectory *TOBbw;
  TDirectory *TECbw; 
  
  //TIB directories
  
  TDirectory *TIBfw1;
  TDirectory *TIBfw2;
  TDirectory *TIBfw3;
  TDirectory *TIBfw4;
  
  TDirectory *TIBbw1;
  TDirectory *TIBbw2;
  TDirectory *TIBbw3;
  TDirectory *TIBbw4;
  
  //TID directories
  
  TDirectory *TIDfw1;
  TDirectory *TIDfw2;
  TDirectory *TIDfw3;
  
  TDirectory *TIDbw1;
  TDirectory *TIDbw2;
  TDirectory *TIDbw3; 
  
  //TOB directories
  
  TDirectory *TOBfw1;
  TDirectory *TOBfw2;
  TDirectory *TOBfw3;
  TDirectory *TOBfw4;
  TDirectory *TOBfw5;
  TDirectory *TOBfw6;
  
  TDirectory *TOBbw1;
  TDirectory *TOBbw2;
  TDirectory *TOBbw3;
  TDirectory *TOBbw4;
  TDirectory *TOBbw5;
  TDirectory *TOBbw6;
  
  //TEC directories
  
  TDirectory *TECfw1;
  TDirectory *TECfw2;
  TDirectory *TECfw3;
  TDirectory *TECfw4;
  TDirectory *TECfw5;
  TDirectory *TECfw6;
  TDirectory *TECfw7;
  TDirectory *TECfw8;
  TDirectory *TECfw9;
  
  TDirectory *TECbw1;
  TDirectory *TECbw2;
  TDirectory *TECbw3;
  TDirectory *TECbw4;
  TDirectory *TECbw5;
  TDirectory *TECbw6;
  TDirectory *TECbw7;
  TDirectory *TECbw8;
  TDirectory *TECbw9;
  
};


#endif // AnalysisExamples_SiStripDetectorPerformance_TIFNtupleMaker_h

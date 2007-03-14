#ifndef AnalysisExamples_SiStripDetectorPerformance_TIFNtupleMaker_h
#define AnalysisExamples_SiStripDetectorPerformance_TIFNtupleMaker_h

#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
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
  
 private:

//  const TrackerGeometry::DetIdContainer& Id;
  
  typedef StripSubdetector::SubDetector                                            SiSubDet;
  typedef std::vector<std::pair<const TrackingRecHit *,float> >                    hitanglevector;  
  typedef std::map <const reco::Track *, hitanglevector>                           trackhitmap;
  typedef std::map <const reco::Track *, TrackLocalAngleTIF::HitLclDirAssociation> trklcldirmap;
  typedef std::map <const reco::Track *, TrackLocalAngleTIF::HitGlbDirAssociation> trkglbdirmap;
  typedef std::vector<SiStripDigi> DigisVector;

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
  
  TrackLocalAngleTIF *Anglefinder;
  
  int tiftibcorr, tiftobcorr;
  

  // TIFNtupleMaker::TIFNtupleMaker
  // ------------------------------
  bool bTriggerDT;
  bool bTriggerCSC;
  bool bTriggerRBC1;
  bool bTriggerRBC2;
  bool bTriggerRPC;


  TFile* hFile;
  TTree* TIFNtupleMakerTree;
  TTree* poTrackTree;
  TTree* poTrackNum;

  // Main tree on hits variables
  // ---------------------------  
  int   run;                     
  int   event;
  int   eventcounter;
  int   module;
  int   type;
  int   layer;
  int   string;
  int   rod;
  int   extint;
  int   size;
  float angle;
  float tk_phi;
  float tk_theta;
  int   tk_id;
  bool  shared;
  int   sign;
  int   bwfw;
  int   wheel;
  int   monostereo;
  float stereocorrection;
  float localmagfield;
  float momentum;
  float pt;
  int   charge;
  float eta;
  float phi;
  int   hitspertrack;
  float normchi2;
  float chi2;
  float ndof;
  int   numberoftracks;
  int   trackcounter;
  float clusterpos;
  float clustereta;
  float clusterchg;
  float clusterchgl;
  float clusterchgr;
  float clusternoise;
  float clusterbarycenter;
  float clustermaxchg;
  float clusterseednoise;
  float clustercrosstalk;
  float dLclX;
  float dLclY;
  float dLclZ;
  float dGlbX;
  float dGlbY;
  float dGlbZ;
  int   numberofclusters;
  int   TESTnumberofclusters;
  int   numberoftkclusters;
  int   numberofnontkclusters; 

  // Track tree variables
  // --------------------

  // already declared as Main tree on hits variables
  //  int   run;
  //  int   event;
  //  int   eventcounter;
  //  float momentum;
  //  float pt;
  //  int   charge;
  //  float eta;
  //  float phi;
  //  int   hitspertrack;
  //  float normchi2;
  //  float chi2;
  //  float ndof;
  //  int   numberoftracks;
  //  int   trackcounter;
  //  int   numberofclusters;
  //  int   numberoftkclusters;
  //  int   numberofnontkclusters;


  // New tree for number of tracks variables
  // ---------------------------------------

  // already declared as Main tree on hits variables
  //  int   numberoftracks;
  // already declared as Track tree variables
  //  int numberofclusters;
  //  int numberoftkclusters;
  //  int numberofnontkclusters;

  const TrackerGeometry * tracker;
  const MagneticField * magfield;
    

  int nTrackClusters;

  LocalVector localmagdir;

  int monodscounter;
  int monosscounter;
  int stereocounter;

};


#endif // AnalysisExamples_SiStripDetectorPerformance_TIFNtupleMaker_h

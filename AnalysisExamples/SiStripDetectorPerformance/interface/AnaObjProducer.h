//
// Package:     AnalysisExample/SiStripDetectorPerformance
// Class:       AnaObjProducer
//
//
// Description: Produce AnalyzedTrack and AnalyzedCluster from
//              track, trackinfo and cluster, clusterinfo.
//              They are not interfaces, they are standalone.
//
// Original     Authors: M. De Mattia, M. Tosi
// Created:     15/3/2007
//

#ifndef AnalysisExamples_SiStripDetectorPerformance_AnaObjProducer_h
#define AnalysisExamples_SiStripDetectorPerformance_AnaObjProducer_h

#include <map>

#include "FWCore/Framework/interface/EDProducer.h"
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
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

//#include "CommonTools/SiStripZeroSuppression/interface/SiStripNoiseService.h"

#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackLocalAngleTIF.h"
#include "AnalysisExamples/AnalysisObjects/interface/AnalyzedCluster.h"
#include "AnalysisExamples/AnalysisObjects/interface/AnalyzedTrack.h"

#include <TMath.h>


class AnaObjProducer : public edm::EDProducer
{
 public:
  
  explicit AnaObjProducer(const edm::ParameterSet& conf);
  
  virtual ~AnaObjProducer();
  
  virtual void beginJob(const edm::EventSetup& c);
  
  virtual void endJob(); 
  
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
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
  // Map to store the pair needed to identify the cluster with the AnalyzedCluster                         
  // and the index of the AnalyzedCluster stored in the vector (to have access to it later)  
  typedef std::map<std::pair <uint32_t, int>, int> clustermap;
  // Map to store the pair needed to identify the track with the AnalyzedTrack
  // and the index of the AnalyzedTrack stored in the vector (to have access to it later)
  typedef std::map<std::pair <uint32_t, int>, int> trackmap;

  // Ref from cluster to track
  typedef edm::Ref<anaobj::AnalyzedTrackCollection> AnalyzedTrackRef;
  // Ref from track to cluster
  typedef edm::Ref<anaobj::AnalyzedClusterCollection> AnalyzedClusterRef;

  // ProdRef: this is needed since the Cluster is not yet written in the Event when the Ref is created
  // From cluster to tracks
  typedef edm::RefProd<anaobj::AnalyzedTrackCollection> AnalyzedTrackRefProd;
  // From track to clusters
  typedef edm::RefProd<anaobj::AnalyzedClusterCollection> AnalyzedClusterRefProd;

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

  void GetSubDetInfo(StripSubdetector oStripSubdet);

//  double moduleThickness( const TrackingRecHit* hit );
  double moduleThickness( const unsigned int detid );

  edm::ParameterSet conf_;
  std::string filename_;
  std::string oSiStripDigisLabel_;
  std::string oSiStripDigisProdInstName_;
  std::string analyzedtrack_;
  std::string analyzedcluster_;
//  bool bUseLTCDigis_;
  const double dCROSS_TALK_ERR;
  //SiStripNoiseService m_oSiStripNoiseService; 
  
  std::vector<DetId> Detvector;
  
  TrackLocalAngleTIF *Anglefinder;
  
  int tiftibcorr, tiftobcorr;
  
  edm::InputTag Cluster_src_;
  std::vector<uint32_t> ModulesToBeExcluded_;
  
  // Main tree on hits variables
  // ---------------------------  
  int      run;                     
  int      event;
  int      eventcounter;
  int      module;
  int      type;
  int      layer;
  int      string;
  int      rod;
  int      extint;
  int      size;
  float    angle;
  float    tk_phi;
  float    tk_theta;
  int      tk_id;
  bool     shared;
  int      sign;
  int      bwfw;
  int      wheel;
  int      monostereo;
  float    stereocorrection;
  float    localmagfield;
  float    momentum;
  float    pt;
  int      charge;
  float    eta;
  float    phi;
  int      hitspertrack;
  float    normchi2;
  float    chi2;
  float    ndof;
  int      numberoftracks;
  int      trackcounter;
  float    clusterpos;
  float    clustereta;
  float    clusterchg;
  float    clusterchgl;
  float    clusterchgr;
  float    clusternoise;
  float    clusterbarycenter;
  float    clustermaxchg;
  float    clusterseednoise;
  std::vector<float>    clusterstripnoises;
  float    clustercrosstalk;
  uint32_t geoId;
  uint16_t firstStrip;
  float    LclPos_X;
  float    LclPos_Y;
  float    LclPos_Z;
  float    GlbPos_X;
  float    GlbPos_Y;
  float    GlbPos_Z;
  int      numberofclusters;
  int      TESTnumberofclusters;
  int      numberoftkclusters;
  int      numberofnontkclusters; 

  float d0;
  float vx;
  float vy;
  float vz;
  float outerPt;

  const TrackerGeometry * tracker;
  const MagneticField * magfield;
    

  int nTrackClusters;

  int nTotClustersTIB;
  int nTotClustersTID;
  int nTotClustersTOB;
  int nTotClustersTEC;


  LocalVector localmagdir;

  int monodscounter;
  int monosscounter;
  int stereocounter;


  int countOn;
  int countOff;
  int countAll;
  int countAllinJob;

  int clustercounter;
  int clusterTKcounter;
  int clusterNNTKcounter;

};

#endif // AnalysisExamples_SiStripDetectorPerformance_AnaObjProducer_h

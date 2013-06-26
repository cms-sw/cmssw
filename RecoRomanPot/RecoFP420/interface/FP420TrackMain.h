#ifndef FP420TrackMain_h
#define FP420TrackMain_h
   
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/FP420Cluster/interface/ClusterCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/TrackCollectionFP420.h"
#include "DataFormats/FP420Cluster/interface/TrackFP420.h"

class TrackProducerFP420;

class FP420TrackMain 
{
 public:
  
  FP420TrackMain(const edm::ParameterSet& conf);
  ~FP420TrackMain();

  /// Runs the algorithm
  void run(edm::Handle<ClusterCollectionFP420> &input,
	   std::auto_ptr<TrackCollectionFP420> &toutput
	   );

 private:


  edm::ParameterSet conf_;
  TrackProducerFP420 *finderParameters_;
  std::string trackMode_;


  bool validTrackerizer_;

  int verbosity;
 // Number of Detectors:
 int dn0;
 // Number of Stations:
 int sn0_;
 // Number of planes:
 int pn0_;
 // Number of semsors:
 int rn0_;
 // Number of planes types:
 int xytype_;

	bool UseHalfPitchShiftInX_;
	bool UseHalfPitchShiftInY_;

	bool UseHalfPitchShiftInXW_;
	bool UseHalfPitchShiftInYW_;

	//double zUnit_; 
	double z420_; 
	double zD2_; 
	double zD3_; 
	double pitchX_;
	double pitchY_;
	double pitchXW_;
	double pitchYW_;
        double ZGapLDet_;
	//	double ZBoundDet_;
	double ZSiStep_;
	double ZSiPlane_;
	double ZSiDet_;
	double zBlade_;
	double gapBlade_;

	double dXX_;
	double dYY_;
	double chiCutX_;
	double chiCutY_;

	double zinibeg_;

	double XsensorSize_;
	double YsensorSize_;

};

#endif

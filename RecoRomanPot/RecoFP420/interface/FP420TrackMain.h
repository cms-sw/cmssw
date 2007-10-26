#ifndef FP420TrackMain_h
#define FP420TrackMain_h
   
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoRomanPot/RecoFP420/interface/ClusterCollectionFP420.h"
#include "RecoRomanPot/RecoFP420/interface/TrackCollectionFP420.h"
#include "RecoRomanPot/RecoFP420/interface/TrackFP420.h"

class TrackProducerFP420;

class FP420TrackMain 
{
 public:
  
  FP420TrackMain(const edm::ParameterSet& conf);
  ~FP420TrackMain();

  /// Runs the algorithm
  void run(const ClusterCollectionFP420 &input,
	   TrackCollectionFP420 &toutput
	   );

 private:


  edm::ParameterSet conf_;
  TrackProducerFP420 *finderParameters_;
  std::string trackMode_;


  bool validTrackerizer_;

  int verbosity;
 // Number of Stations:
 int sn0_;
 // Number of planes:
 int pn0_;
 // Number of planes types:
 int zn0_;

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
	double ZSiDetL_;
	double ZSiDetR_;

	double dXX_;
	double dYY_;
	double chiCutX_;
	double chiCutY_;

	double zinibeg_;

};

#endif

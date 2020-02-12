#ifndef __STUBWINDOWSUGGEST_H__
#define __STUBWINDOWSUGGEST_H__

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"

#include <vector>

using namespace std;

class TrackerTopology;

namespace TMTT {

class Stub;

/** 
 * ========================================================================================================
 *  This provides recommendations to CMS for the stub window sizes to be used in the FE electronics.
 *  It prints the output as a python configuration file in the form of
 *  L1Trigger/TrackTrigger/python/TTStubAlgorithmRegister_cfi.py .
 *
 *  The recommendations are based on the TMTT method of using the stub bend. Whilst they give 
 *  high efficiency, they do not take into account the requirement to limit the FE electronics band-width,
 *  so tighter cuts may be needed in reality.
 * ========================================================================================================
 */

class StubWindowSuggest {

public:
  
  // Initialize (for use with TMTT).
  StubWindowSuggest(const Settings* settings, const TrackerTopology*  trackerTopo) :
    settings_(settings), ptMin_(settings->houghMinPt()), theTrackerTopo_(trackerTopo) {} 

  // Initialize (for use with HYBRID)
  StubWindowSuggest(const Settings* settings) : settings_(settings), ptMin_(settings->houghMinPt()) {}

  ~StubWindowSuggest() {}

  // Analyse stub window required for this stub.
  void process(const Stub* stub);

  // Print results (should be done in endJob();
  static void printResults();

private:

  // Update stored stub window size with this stub.
  void updateStoredWindow(const Stub* stub, double bendWind);

private:

  // Configuration parameters.
  const Settings* settings_;
  const float ptMin_;  

  const TrackerTopology*  theTrackerTopo_;

  // Stub window sizes as encoded in L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h
  static std::vector< double >                barrelCut_;
  static std::vector< std::vector< double > > ringCut_;
  static std::vector< std::vector< double > > tiltedCut_;
  static std::vector< double >                barrelNTilt_;
};

}

#endif


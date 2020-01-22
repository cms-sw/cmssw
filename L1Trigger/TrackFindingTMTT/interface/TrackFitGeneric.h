///=== This is the base class for all the track fit algorithms

///=== Written by: Alexander D. Morton and Sioni Summers

#ifndef __TrackFitGeneric_H__
#define __TrackFitGeneric_H__

// Don't fit track candidates if they have more than this number of stubs.
#define __MAX_STUBS_PER_TRK__   30

#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"

#include <vector>
#include <utility>

using namespace std;

namespace TMTT {

class Settings;

class TrackFitGeneric {

public:

  // Set configuration parameters.
  TrackFitGeneric( const Settings* settings, const string &fitterName="" );

  virtual ~TrackFitGeneric() {}

  // Static method to produce a fitter based on a string
//  static std::auto_ptr<TrackFitGeneric> create(std::string, const Settings* settings);
  static TrackFitGeneric* create(std::string, const Settings* settings);
  virtual void bookHists(){}

  virtual void initRun() {}
  // Fit a track candidate obtained from the Hough Transform.
  virtual L1fittedTrack fit( const L1track3D& l1track3D );

  // Optional debug printout at end of job.
  virtual void endJob() {}

  const Settings* getSettings()const{return settings_;}
  unsigned nDupStubs()const{ return nDupStubs_; }

protected:

  // Configuration parameters
  const Settings* settings_;
  const string    fitterName_;
  unsigned nDupStubs_;
};

}

#endif


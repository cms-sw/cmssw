#ifndef BEAMSPOTONLINEOBJECTS_H
#define BEAMSPOTONLINEOBJECTS_H
/** \class BeamSpotOnlineObjects
 *
 * Class inheriting from BeamSpotObjects. 
 * New members of the class:
 *  - lastAnalyzedLumi : last lumisection analyzed
 *  - lastAnalyzedRun  : run of the last analyzed lumisection
 *  - lastAnalyzedFill : fill of the last analyzed lumisection
 *
 * \author Francisco Brivio, Milano-Bicocca (francesco.brivio@cern.ch)
 *
 */

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <cmath>
#include <sstream>
#include <cstring>
#include <vector>
#include <string>

class BeamSpotOnlineObjects : public BeamSpotObjects {
public:
  /// default constructor
  BeamSpotOnlineObjects() {
    lastAnalyzedLumi_ = 0;
    lastAnalyzedRun_ = 0;
    lastAnalyzedFill_ = 0;
    intParams_.resize(ISIZE, std::vector<int>(1, 0));
    timeParams_.resize(TSIZE, std::vector<unsigned long long>(1, 0ULL));
  }

  ~BeamSpotOnlineObjects() override {}

  /// Enums
  enum IntParamIndex { NUM_TRACKS = 0, NUM_PVS = 1, ISIZE = 2 };
  enum TimeParamIndex { CREATE_TIME = 0, TSIZE = 1 };

  /// Setters Methods
  // set lastAnalyzedLumi_, last analyzed lumisection
  void SetLastAnalyzedLumi(int val) { lastAnalyzedLumi_ = val; }

  // set lastAnalyzedRun_, run of the last analyzed lumisection
  void SetLastAnalyzedRun(int val) { lastAnalyzedRun_ = val; }

  // set lastAnalyzedFill_, fill of the last analyzed lumisection
  void SetLastAnalyzedFill(int val) { lastAnalyzedFill_ = val; }

  // set number of tracks used in the BeamSpot fit
  void SetNumTracks(int val);

  // set number of Primary Vertices used in the BeamSpot fit
  void SetNumPVs(int val);

  // set creation time of the payload
  void SetCreationTime(cond::Time_t val);

  /// Getters Methods
  // get lastAnalyzedLumi_, last analyzed lumisection
  int GetLastAnalyzedLumi() const { return lastAnalyzedLumi_; }

  // get lastAnalyzedRun_, run of the last analyzed lumisection
  int GetLastAnalyzedRun() const { return lastAnalyzedRun_; }

  // get lastAnalyzedFill_, fill of the last analyzed lumisection
  int GetLastAnalyzedFill() const { return lastAnalyzedFill_; }

  // get number of tracks used in the BeamSpot fit
  int GetNumTracks() const;

  // get number of Primary Vertices used in the BeamSpot fit
  int GetNumPVs() const;

  // get creation time of the payload
  cond::Time_t GetCreationTime() const;

  /// Print BeamSpotOnline parameters
  void print(std::stringstream& ss) const;

private:
  int lastAnalyzedLumi_;
  int lastAnalyzedRun_;
  int lastAnalyzedFill_;
  std::vector<std::vector<int> > intParams_;
  std::vector<std::vector<float> > floatParams_;
  std::vector<std::vector<std::string> > stringParams_;
  std::vector<std::vector<unsigned long long> > timeParams_;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, BeamSpotOnlineObjects beam);

#endif
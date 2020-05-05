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

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <cmath>
#include <sstream>
#include <cstring>

class BeamSpotOnlineObjects : public BeamSpotObjects {
public:
  /// default constructor
  BeamSpotOnlineObjects() {
    lastAnalyzedLumi_ = 0;
    lastAnalyzedRun_ = 0;
    lastAnalyzedFill_ = 0;
  }

  ~BeamSpotOnlineObjects() override {}


  /// Setters Methods
  // set lastAnalyzedLumi_, last analyzed lumisection
  void SetLastAnalyzedLumi(int val) { lastAnalyzedLumi_ = val; }

  // set lastAnalyzedRun_, run of the last analyzed lumisection
  void SetLastAnalyzedRun(int val) { lastAnalyzedRun_ = val; }

  // set lastAnalyzedFill_, fill of the last analyzed lumisection
  void SetLastAnalyzedFill(int val) { lastAnalyzedFill_ = val; }


  /// Getters Methods
  // get lastAnalyzedLumi_, last analyzed lumisection
  int GetLastAnalyzedLumi() const { return lastAnalyzedLumi_; }

  // get lastAnalyzedRun_, run of the last analyzed lumisection
  int GetLastAnalyzedRun() const { return lastAnalyzedRun_; }

  // get lastAnalyzedFill_, fill of the last analyzed lumisection
  int GetLastAnalyzedFill() const { return lastAnalyzedFill_; }


  /// Print BeamSpotOnline parameters
  void print(std::stringstream& ss) const;


private:
  int lastAnalyzedLumi_;
  int lastAnalyzedRun_;
  int lastAnalyzedFill_;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, BeamSpotOnlineObjects beam);

#endif
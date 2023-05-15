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
    floatParams_.resize(FSIZE, std::vector<float>(1, 0.0));
    stringParams_.resize(SSIZE, std::vector<std::string>(1, ""));
    timeParams_.resize(TSIZE, std::vector<unsigned long long>(1, 0ULL));
  }

  ~BeamSpotOnlineObjects() override {}

  /// Enums
  enum IntParamIndex { NUM_TRACKS = 0, NUM_PVS = 1, USED_EVENTS = 2, MAX_PVS = 3, ISIZE = 4 };
  enum FloatParamIndex { MEAN_PV = 0, ERR_MEAN_PV = 1, RMS_PV = 2, ERR_RMS_PV = 3, FSIZE = 4 };
  enum StringParamIndex { START_TIME = 0, END_TIME = 1, LUMI_RANGE = 2, SSIZE = 3 };
  enum TimeParamIndex { CREATE_TIME = 0, START_TIMESTAMP = 1, END_TIMESTAMP = 2, TSIZE = 3 };

  /// Setters Methods
  // copy all copiable members from BeamSpotObjects
  void copyFromBeamSpotObject(const BeamSpotObjects& bs);

  // set lastAnalyzedLumi_, last analyzed lumisection
  void setLastAnalyzedLumi(int val) { lastAnalyzedLumi_ = val; }

  // set lastAnalyzedRun_, run of the last analyzed lumisection
  void setLastAnalyzedRun(int val) { lastAnalyzedRun_ = val; }

  // set lastAnalyzedFill_, fill of the last analyzed lumisection
  void setLastAnalyzedFill(int val) { lastAnalyzedFill_ = val; }

  // set number of tracks used in the BeamSpot fit
  void setNumTracks(int val);

  // set number of Primary Vertices used in the BeamSpot fit
  void setNumPVs(int val);

  // set number of Events used in the BeamSpot fit (for DIP)
  void setUsedEvents(int val);

  // set max number of Primary Vertices used in the BeamSpot fit (for DIP)
  void setMaxPVs(int val);

  // set mean number of PVs (for DIP)
  void setMeanPV(float val);

  // set error on mean number of PVs (for DIP)
  void setMeanErrorPV(float val);

  // set rms of number of PVs (for DIP)
  void setRmsPV(float val);

  // set error on rm of number of PVs (for DIP)
  void setRmsErrorPV(float val);

  // set start time of the firs LS as string (for DIP)
  void setStartTime(std::string val);

  // set end time of the last LS as string (for DIP)
  void setEndTime(std::string val);

  // set lumi range as string (for DIP)
  void setLumiRange(std::string val);

  // set creation time of the payload
  void setCreationTime(cond::Time_t val);

  // set timestamp of the first LS (for DIP)
  void setStartTimeStamp(cond::Time_t val);

  // set timestamp of the last LS (for DIP)
  void setEndTimeStamp(cond::Time_t val);

  /// Getters Methods
  // get lastAnalyzedLumi_, last analyzed lumisection
  int lastAnalyzedLumi() const { return lastAnalyzedLumi_; }

  // get lastAnalyzedRun_, run of the last analyzed lumisection
  int lastAnalyzedRun() const { return lastAnalyzedRun_; }

  // get lastAnalyzedFill_, fill of the last analyzed lumisection
  int lastAnalyzedFill() const { return lastAnalyzedFill_; }

  // get number of tracks used in the BeamSpot fit
  int numTracks() const;

  // get number of Primary Vertices used in the BeamSpot fit
  int numPVs() const;

  // get number of Events used in the BeamSpot fit (for DIP)
  int usedEvents() const;

  // get max number of Primary Vertices used in the BeamSpot fit (for DIP)
  int maxPVs() const;

  // get mean number of PVs (for DIP)
  float meanPV() const;

  // get error on mean number of PVs (for DIP)
  float meanErrorPV() const;

  // get rms of number of PVs (for DIP)
  float rmsPV() const;

  // get error on rm of number of PVs (for DIP)
  float rmsErrorPV() const;

  // get start time of the firs LS as string (for DIP)
  std::string startTime() const;

  // get end time of the last LS as string (for DIP)
  std::string endTime() const;

  // get lumi range as string (for DIP)
  std::string lumiRange() const;

  // get creation time of the payload
  cond::Time_t creationTime() const;

  // get timestamp of the first LS (for DIP)
  cond::Time_t startTimeStamp() const;

  // get timestamp of the last LS (for DIP)
  cond::Time_t endTimeStamp() const;

  /// Print BeamSpotOnline parameters
  void print(std::stringstream& ss) const;

private:
  int lastAnalyzedLumi_;
  int lastAnalyzedRun_;
  int lastAnalyzedFill_;
  std::vector<std::vector<int> > intParams_;
  std::vector<std::vector<float> > floatParams_;
  std::vector<std::vector<std::string> > stringParams_;
  std::vector<std::vector<unsigned long long> > timeParams_;  // unsigned long long is equal to cond::Time_t

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, BeamSpotOnlineObjects beam);

#endif

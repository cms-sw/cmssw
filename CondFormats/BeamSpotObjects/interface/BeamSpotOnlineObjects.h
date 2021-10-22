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

  // set number of Events used in the BeamSpot fit (for DIP)
  void SetUsedEvents(int val);

  // set max number of Primary Vertices used in the BeamSpot fit (for DIP)
  void SetMaxPVs(int val);

  // set mean number of PVs (for DIP)
  void SetMeanPV(float val);

  // set error on mean number of PVs (for DIP)
  void SetMeanErrorPV(float val);

  // set rms of number of PVs (for DIP)
  void SetRmsPV(float val);

  // set error on rm of number of PVs (for DIP)
  void SetRmsErrorPV(float val);

  // set start time of the firs LS as string (for DIP)
  void SetStartTime(std::string val);

  // set end time of the last LS as string (for DIP)
  void SetEndTime(std::string val);

  // set lumi range as string (for DIP)
  void SetLumiRange(std::string val);

  // set creation time of the payload
  void SetCreationTime(cond::Time_t val);

  // set timestamp of the first LS (for DIP)
  void SetStartTimeStamp(cond::Time_t val);

  // set timestamp of the last LS (for DIP)
  void SetEndTimeStamp(cond::Time_t val);

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

  // get number of Events used in the BeamSpot fit (for DIP)
  int GetUsedEvents() const;

  // get max number of Primary Vertices used in the BeamSpot fit (for DIP)
  int GetMaxPVs() const;

  // get mean number of PVs (for DIP)
  float GetMeanPV() const;

  // get error on mean number of PVs (for DIP)
  float GetMeanErrorPV() const;

  // get rms of number of PVs (for DIP)
  float GetRmsPV() const;

  // get error on rm of number of PVs (for DIP)
  float GetRmsErrorPV() const;

  // get start time of the firs LS as string (for DIP)
  std::string GetStartTime() const;

  // get end time of the last LS as string (for DIP)
  std::string GetEndTime() const;

  // get lumi range as string (for DIP)
  std::string GetLumiRange() const;

  // get creation time of the payload
  cond::Time_t GetCreationTime() const;

  // get timestamp of the first LS (for DIP)
  cond::Time_t GetStartTimeStamp() const;

  // get timestamp of the last LS (for DIP)
  cond::Time_t GetEndTimeStamp() const;

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
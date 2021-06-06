#ifndef SISTRIPDETVOFF_SRC_BUILDER_H
#define SISTRIPDETVOFF_SRC_BUILDER_H
#define USING_NEW_CORAL

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "CalibTracker/SiStripDCS/interface/SiStripCoralIface.h"
#include "CalibTracker/SiStripDCS/interface/SiStripPsuDetIdMap.h"
// #include "CalibTracker/SiStripDCS/interface/SiStripPsuDetIdMapFromFile.h"

#include "CoralBase/TimeStamp.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"
//#include "DataFormats/Provenance/interface/Timestamp.h"

#include <fstream>
#include <iostream>
#include <memory>

/**	
   \class SiStripDetVOffBuilder
   \brief Builds the SiStripDetVOff object for transfer by O2O
   \author J.Cole modified by Marco De Mattia
*/

// Unit test class for SiStripDetVOffBuilder
class TestSiStripDetVOffBuilder;

class SiStripDetVOffBuilder {
  friend class TestSiStripDetVOffBuilder;

public:
  /** Destructor. */
  ~SiStripDetVOffBuilder();
  /** Default constructor. */
  SiStripDetVOffBuilder(const edm::ParameterSet&, const edm::ActivityRegistry&);
  /** Build the SiStripDetVOff object for transfer. */
  void BuildDetVOffObj();
  /** Return modules Off vector of objects. */
  std::vector<std::pair<SiStripDetVOff*, cond::Time_t> > getModulesVOff() {
    reduction(deltaTmin_, maxIOVlength_);
    return modulesOff;
  }
  /** Return statistics about payloads transferred for storage in logDB. */
  std::vector<std::vector<uint32_t> > getPayloadStats() { return payloadStats; }
  /** Store the last payload transferred to DB as starting point for creation of new object list.
      ONLY WORKS FOR STATUSCHANGE OPTION. */
  void setLastSiStripDetVOff(SiStripDetVOff* lastPayload, cond::Time_t lastTimeStamp);

  /// Operates the reduction of the fast sequences of ramping up and down of the voltages
  void reduce(std::vector<std::pair<SiStripDetVOff*, cond::Time_t> >::iterator& it,
              std::vector<std::pair<SiStripDetVOff*, cond::Time_t> >::iterator& initialIt,
              std::vector<std::pair<SiStripDetVOff*, cond::Time_t> >& resultVec,
              const bool last = false);

  void reduction(const uint32_t deltaTmin, const uint32_t maxIOVlength);

  /// Removes IOVs as dictated by reduction
  void discardIOVs(std::vector<std::pair<SiStripDetVOff*, cond::Time_t> >::iterator& it,
                   std::vector<std::pair<SiStripDetVOff*, cond::Time_t> >::iterator& initialIt,
                   std::vector<std::pair<SiStripDetVOff*, cond::Time_t> >& resultVec,
                   const bool last,
                   const unsigned int first);
  bool FileExists(std::string filename);

private:
  // typedefs
  typedef std::vector<std::pair<std::vector<uint32_t>, coral::TimeStamp> > DetIdTimeStampVector;

  bool whichQuery;

  void printPar(std::stringstream& ss, const std::vector<int>& par);

  std::string timeToStream(const coral::TimeStamp& coralTime, const string& comment = "");
  std::string timeToStream(const cond::Time_t& condTime, const string& comment = "");

  /** Returns the PSU channel setting, based on date.  Works from DP ID. */
  int findSetting(uint32_t id,
                  const coral::TimeStamp& changeDate,
                  const std::vector<uint32_t>& settingID,
                  const std::vector<coral::TimeStamp>& settingDate);

  /** Returns the PSU channel setting, based on date.  Works from PSU channel name. Overloaded. */
  int findSetting(std::string dpname,
                  const coral::TimeStamp& changeDate,
                  const std::vector<std::string>& settingDpname,
                  const std::vector<coral::TimeStamp>& settingDate);

  /** Extract the lastValue values from file rather than from the PVSS cond DB. */
  void readLastValueFromFile(std::vector<uint32_t>& dpIDs,
                             std::vector<float>& vmonValues,
                             std::vector<coral::TimeStamp>& dateChange);

  /** Utility code to convert a coral timestamp to the correct time format for O2O timestamp. */
  cond::Time_t getCondTime(const coral::TimeStamp& coralTime);

  /** Utility code to convert an O2O timestamp into a coral timestamp. */
  coral::TimeStamp getCoralTime(cond::Time_t iovTime);

  /** Utility code to remove all the duplicates from a vector of uint32_t. */
  void removeDuplicates(std::vector<uint32_t>& vec);
  /** */
  cond::Time_t findMostRecentTimeStamp(const std::vector<coral::TimeStamp>& coralDate);

  // member data
  std::vector<std::vector<uint32_t> > payloadStats;
  std::vector<std::pair<SiStripDetVOff*, cond::Time_t> > modulesOff;
  std::pair<SiStripDetVOff*, cond::Time_t> lastStoredCondObj;

  // configurable parameters
  std::string onlineDbConnectionString;
  std::string authenticationPath;
  std::string whichTable;
  std::string lastValueFileName;
  bool fromFile;
  std::string psuDetIdMapFile_;
  bool debug_;
  coral::TimeStamp tmax, tmin, tsetmin;
  std::vector<int> tDefault, tmax_par, tmin_par, tset_par;
  uint32_t deltaTmin_, maxIOVlength_;

  std::string detIdListFile_;
  std::string excludedDetIdListFile_;
  // Threshold to consider a high voltage channel on
  double highVoltageOnThreshold_;

  // Structure used to store variables needed when building the database objects
  struct TimesAndValues {
    TimesAndValues() : latestTime(0) {}
    std::vector<coral::TimeStamp> changeDate;  // used by both
    std::vector<std::string> dpname;           // only used by DB access, not file access
    std::vector<float> actualValue;            // only used by DB access, not file access
    std::vector<uint32_t> dpid;                // only used by file access
    std::vector<int> actualStatus;             // filled using actualValue info
    cond::Time_t latestTime;                   // used for timestamp when using lastValue from file
  };

  struct DetIdListTimeAndStatus {
    DetIdListTimeAndStatus() : notMatched(0) {}
    DetIdTimeStampVector detidV;
    std::vector<bool> StatusGood;
    unsigned int notMatched;
    std::vector<std::string> psuName;
    std::vector<unsigned int> isHV;
  };

  void statusChange(cond::Time_t& lastTime, TimesAndValues& tStruct);
  void lastValue(TimesAndValues& tStruct);
  void lastValueFromFile(TimesAndValues& tStruct);

  void buildPSUdetIdMap(TimesAndValues& tStruct, DetIdListTimeAndStatus& dStruct);

  void setPayloadStats(const uint32_t afterV, const uint32_t numAdded, const uint32_t numRemoved);
  std::pair<int, int> extractDetIdVector(const unsigned int i,
                                         SiStripDetVOff* modV,
                                         DetIdListTimeAndStatus& detIdStruct);

  std::unique_ptr<SiStripCoralIface> coralInterface;
};
#endif

#ifndef SISTRIPMODULEHV_SRC_BUILDER_H
#define SISTRIPMODULEHV_SRC_BUILDER_H

#define USING_NEW_CORAL

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CoralBase/TimeStamp.h"
//#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Common/interface/Time.h"
#include "boost/cstdint.hpp"

/**	
   \class SiStripModuleHVBuilder
   \brief Builds the SiStripModuleHV object for transfer by O2O
   \author J.Cole
*/

class SiStripModuleHVBuilder
{
 public:
  /** Destructor. */
  ~SiStripModuleHVBuilder();
  /** Default constructor. */
  SiStripModuleHVBuilder(const edm::ParameterSet&,const edm::ActivityRegistry&);
  /** Build the SiStripModuleHV object for transfer. */
  void BuildModuleHVObj();
  /** Return modules Off vector of objects. */
  std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > getModulesVOff() {return modulesOff;}
  /** Return statistics about payloads transferred for storage in logDB. */
  std::vector< std::vector<uint32_t> > getPayloadStats() {return payloadStats;}
  /** Store the last payload transferred to DB as starting point for creation of new object list.
      ONLY WORKS FOR STATUSCHANGE OPTION. */
  void retrieveLastSiStripDetVOff( SiStripDetVOff * lastPayload, cond::Time_t lastTimeStamp );
  
 private:
  // typedefs
  typedef std::vector< std::pair< std::vector<uint32_t>,coral::TimeStamp> > DetIdTimeStampVector ;
  
  /** Returns the PSU channel setting, based on date.  Works from DP ID. */
  int findSetting(uint32_t id, coral::TimeStamp changeDate, std::vector<uint32_t> settingID, std::vector<coral::TimeStamp> settingDate);
  /** Returns the PSU channel setting, based on date.  Works from PSU channel name. Overloaded. */
  int findSetting(std::string dpname, coral::TimeStamp changeDate, std::vector<std::string> settingDpname, std::vector<coral::TimeStamp> settingDate);
  /** Extract the lastValue values from file rather than from the PVSS cond DB. */
  void readLastValueFromFile(std::vector<uint32_t> &dpIDs, std::vector<float> &vmonValues, std::vector<coral::TimeStamp> &dateChange);
  /** Utility code to convert a coral timestamp to the correct time format for O2O timestamp. */
  cond::Time_t getIOVTime(coral::TimeStamp coralTime);
  /** Utility code to convert an O2O timestamp into a coral timestamp. */
  coral::TimeStamp getCoralTime(cond::Time_t iovTime);
  /** Utility code to remove all the duplicates from a vector of uint32_t. */
  void removeDuplicates( std::vector<uint32_t> & vec );
  /** */
  cond::Time_t findMostRecentTimeStamp( std::vector<coral::TimeStamp> coralDate );
  
  // member data
  std::vector< std::vector<uint32_t> > payloadStats;
  std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > modulesOff;
  std::pair<SiStripDetVOff *, cond::Time_t> lastStoredCondObj;
  
  // configurable parameters
  std::string onlineDbConnectionString;
  std::string authenticationPath;
  std::string whichTable;
  std::string lastValueFileName;
  bool fromFile;
  bool debug_;
  coral::TimeStamp tmax, tmin, tsetmin;
  std::vector<int> tmax_par, tmin_par, tset_par, tDefault;
};
#endif


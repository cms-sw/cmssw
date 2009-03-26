#ifndef SISTRIPMODULEHV_SRC_BUILDER_H
#define SISTRIPMODULEHV_SRC_BUILDER_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/SiStripObjects/interface/SiStripModuleHV.h"
#include "CondFormats/DataRecord/interface/SiStripModuleHVRcd.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CoralBase/TimeStamp.h"
#include "CondCore/DBCommon/interface/Time.h"
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
  /** Return the HV channels object. */
  //  SiStripModuleHV* getSiStripModuleHV() {return SiStripModuleHV_;}
  std::vector< std::pair<SiStripModuleHV*,cond::Time_t> > getSiStripModuleHV() {return resultHV;}
  /** Return the LV channels object. */
  //  SiStripModuleHV* getSiStripModuleLV() {return SiStripModuleLV_;}
  std::vector< std::pair<SiStripModuleHV*,cond::Time_t> > getSiStripModuleLV() {return resultLV;}
  //
  std::vector< std::vector<uint32_t> > getPayloadStats( std::string powerType );
  
 private:
  // typedefs
  typedef std::vector< std::pair< std::vector<uint32_t>,coral::TimeStamp> > DetIdTimeStampVector ;
  typedef std::vector< std::pair< std::vector<uint32_t>, cond::Time_t> > DetIdCondTimeVector ;
  
  /** Returns the PSU channel setting, based on date.  Works from DP ID. */
  int findSetting(uint32_t id, coral::TimeStamp changeDate, std::vector<uint32_t> settingID, std::vector<coral::TimeStamp> settingDate);
  /** Returns the PSU channel setting, based on date.  Works from PSU channel name. Overloaded. */
  int findSetting(std::string dpname, coral::TimeStamp changeDate, std::vector<std::string> settingDpname, std::vector<coral::TimeStamp> settingDate);
  /** Extract the lastValue values from file rather than from the PVSS cond DB. */
  void readLastValueFromFile(std::vector<uint32_t> &dpIDs, std::vector<float> &vmonValues, std::vector<coral::TimeStamp> &dateChange);
  cond::Time_t getIOVTime(coral::TimeStamp coralTime);
  bool compareCoralTime(coral::TimeStamp timeA, coral::TimeStamp timeB);
  DetIdCondTimeVector mergeVectors(DetIdTimeStampVector inputVector, std::vector<bool> inputStatus, std::vector<bool> & outputStatus);
  std::vector< std::pair<SiStripModuleHV*,cond::Time_t> > buildObjectVector(DetIdCondTimeVector inputVector, std::vector<bool> statusVector, std::vector< std::vector<unsigned int> > & statsVector);
  
  // member data
  std::vector< std::pair<SiStripModuleHV*,cond::Time_t> > resultHV, resultLV;
  std::vector< std::vector<uint32_t> > payloadStatsHV, payloadStatsLV;
  
  std::string onlineDbConnectionString;
  std::string authenticationPath;
  std::string whichTable;
  std::string lastValueFileName;
  bool fromFile;
  coral::TimeStamp tmax, tmin;
  std::vector<int> tmax_par, tmin_par, tDefault;
};
#endif


#ifndef SISTRIPMODULEHV_SRC_BUILDER_H
#define SISTRIPMODULEHV_SRC_BUILDER_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/SiStripObjects/interface/SiStripModuleHV.h"
#include "CondFormats/DataRecord/interface/SiStripModuleHVRcd.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CoralBase/TimeStamp.h"

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
  SiStripModuleHVBuilder(const edm::ParameterSet& pset);
  /** Build the SiStripModuleHV object for transfer. */
  void BuildModuleHVObj();
  /** Return the HV channels object. */
  SiStripModuleHV* getSiStripModuleHV() {return SiStripModuleHV_;}
  /** Return the LV channels object. */
  SiStripModuleHV* getSiStripModuleLV() {return SiStripModuleLV_;}
  
 private:
  /** Returns the PSU channel setting, based on date.  Works from DP ID. */
  int findSetting(uint32_t id, coral::TimeStamp changeDate, std::vector<uint32_t> settingID, std::vector<coral::TimeStamp> settingDate);
  /** Returns the PSU channel setting, based on date.  Works from PSU channel name. Overloaded. */
  int findSetting(std::string dpname, coral::TimeStamp changeDate, std::vector<std::string> settingDpname, std::vector<coral::TimeStamp> settingDate);
  /** Extract the lastValue values from file rather than from the PVSS cond DB. */
  void readLastValueFromFile(std::vector<uint32_t> &dpIDs, std::vector<float> &vmonValues, std::vector<coral::TimeStamp> &dateChange);
  
  // member data
  SiStripModuleHV *SiStripModuleHV_, *SiStripModuleLV_;
  std::string onlineDbConnectionString;
  std::string authenticationPath;
  std::string whichTable;
  std::string lastValueFileName;
  bool fromFile;
  coral::TimeStamp tmax, tmin;
  std::vector<int> tmax_par, tmin_par, tDefault;
};
#endif


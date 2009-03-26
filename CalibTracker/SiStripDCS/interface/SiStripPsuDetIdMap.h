#ifndef SISTRIPPSUDETIDMAP_H
#define SISTRIPPSUDETIDMAP_H

#include <memory>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/MapOfVectors.h"

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "DeviceFactory.h"

#include "boost/cstdint.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>

class SiStripConfigDb;

/**	
   \class SiStripPsuDetIdMap
   \brief Extension to SiStripConfigDb to map PSU channels to DetIDs using DCU-PSU map and DCU-DetID map
   \author J.Cole
*/

class SiStripPsuDetIdMap
{
 public:
  /** Constructor */
  SiStripPsuDetIdMap();
  /** Destructor */
  ~SiStripPsuDetIdMap();
  /** Returns the DetIDs associate to the specified PSU channel. */
  std::vector<uint32_t> getDetID(std::string pvss);
  /** Returns the PSU channel name for the specified Det ID. */
  std::string getPSUName(uint32_t detid);
  /** Returns the detector location for the specified Det ID. */
  std::string getDetectorLocation(uint32_t detid);
  /** Returns the detector location for the specified PSU channel. */
  std::string getDetectorLocation(std::string pvss);
  /** Returns the DCU ID for the specified PSU channel. */
  uint32_t getDcuId(std::string pvss);
  uint32_t getDcuId(uint32_t detid);
  /** Produces a formatted printout of the PSU-DETID map. */

  /** Return the PSU-DETID map as a vector. */
  std::vector< std::pair<uint32_t, std::string> > getPsuDetIdMap();
  /** Return the detector locations as a vector - one-to-one correspondance with the contents of the PSU-DetID map vector. */
  std::vector<std::string> getDetectorLocations();
  std::vector<uint32_t> getDcuIds();

  void printMap();
  /** Main routine that accesses the DB and builds the PSU-DETID map. */
  void BuildMap();
  /** Returns the DCU-PSU map as a vector. */
  std::vector< std::pair<uint32_t, std::string> > getDcuPsuMap();
  /** Returns 1 if the specified PSU channel is a HV channel, 0 if it is a LV channel.  -1 means error. */
  int IsHVChannel(std::string pvss);
  
 private:
  // typedefs
  typedef std::vector<TkDcuPsuMap *> DcuPsuVector ;
  typedef std::vector< std::pair<uint32_t, std::string> > PsuDetIdMap;
  typedef edm::MapOfVectors<std::string,TkDcuPsuMap*> DcuPsus;
  typedef DcuPsus::range DcuPsusRange;
  /** Extracts the DCU-PSU map from the DB. */
  void getDcuPsuMap(DcuPsusRange &pRange, std::string partition = "");
  /** Utility to clone a DCU-PSU map. */
  void clone(DcuPsuVector &input, DcuPsuVector &output);
  /** Produces a detailed debug of the input values. */
  // for debugging
  void checkMapInputValues(SiStripConfigDb::DcuDetIdsV dcuDetIds_, DcuPsuVector dcuPsus_);
  
  // member variables
  edm::Service<SiStripConfigDb> db_;
  PsuDetIdMap pgMap;
  std::vector<std::string> detectorLocations;
  std::vector<uint32_t> dcuIds;
  DcuPsus DcuPsuMapPG_;
};
#endif


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

  std::vector<uint32_t> getLvDetID(std::string pvss);
  std::vector<uint32_t> getHvDetID(std::string pvss);

  /** Returns the DetIDs associate to the specified PSU channel. */
  std::vector<uint32_t> getDetID(std::string pvss);
  /** Returns the PSU channel name for the specified Det ID, for power groups only. */
  std::string getPSUName(uint32_t detid);
  /** Returns the PSU channel name for the specified Det ID. */
  std::string getPSUName(uint32_t detid, std::string group);
  /** Returns the detector location for the specified Det ID, for power groups only. */
  std::string getDetectorLocation(uint32_t detid);
  /** Returns the detector location for the specified PSU channel. */
  std::string getDetectorLocation(std::string pvss);
  /** Returns the detector location for the specified Det ID and specified group type (PG or CG). */
  std::string getDetectorLocation(uint32_t detid, std::string group);
  /** Returns the DCU ID for the specified PSU channel - checks power and control groups. */
  uint32_t getDcuId(std::string pvss);
  /** Returns the DCU ID associated to the specified Det ID.  NB.  This checks power groups only, by definition. */
  uint32_t getDcuId(uint32_t detid);
  
  /** Return the PG PSU-DETID map as a vector. */
  std::vector< std::pair<uint32_t, std::string> > getPsuDetIdMap() {return pgMap;}
  /** Return the PG detector locations as a vector - one-to-one correspondance with the contents of the PSU-DetID map vector. */
  std::vector<std::string> getDetectorLocations() {return detectorLocations;}
  /** Return the DCU IDs associated to the PG map. */
  std::vector<uint32_t> getDcuIds() {return dcuIds;}
  /** Return the CG PSU-DETID map as a vector. */
  std::vector< std::pair<uint32_t, std::string> > getControlPsuDetIdMap() {return cgMap;}
  /** Return the CG detector locations as a vector - one-to-one correspondance with the contents of the PSU-DetID map vector. */
  std::vector<std::string> getControlDetectorLocations() {return controlLocations;}
  /** Return the module DCU IDs associated to the CG map. */
  std::vector<uint32_t> getCgDcuIds() {return cgDcuIds;}
  /** Return the CCU DCU IDs associated to the CG map. */
  std::vector<uint32_t> getCcuDcuIds() {return ccuDcuIds;}
  
  /** Produces a formatted printout of the PSU-DETID map. */
  void printMap();
  /** Produces a formatted printout of the control PSU-DETID map. */
  void printControlMap();
  /** Main routine that accesses the DB and builds the PSU-DETID map. */
  void BuildMap();

  /**
   * Build the map from given file.
   * ATTENTION: this will only build the pgMap, not the cgMap.
   */
  void BuildMap( const std::string & mapFile );
  /// Overloaded method that does the buidling
  void BuildMap( const std::string & mapFile, std::vector< std::pair<uint32_t, std::string> > & map );

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
  void getDcuPsuMap(DcuPsusRange &pRange, DcuPsusRange &cRange, std::string partition);
  /** Extracts the DCU device descriptions and stores them for further use. Only used for control groups. */
  //  std::vector< std::pair<uint32_t, SiStripConfigDb::DeviceAddress> >  retrieveDcuDeviceAddresses(std::string partition);
  std::vector< std::pair< std::vector<uint16_t> , std::vector<uint32_t> > > retrieveDcuDeviceAddresses(std::string partition);
  /** Searches the DCU device descriptions for the specified DCU ID. Needed for control groups. */
  std::vector<uint32_t>  findDcuIdFromDeviceAddress(uint32_t dcuid_);
  /** Utility to clone a DCU-PSU map. */
  void clone(DcuPsuVector &input, DcuPsuVector &output);
  /** Produces a detailed debug of the input values. */
  // for debugging
  void checkMapInputValues(SiStripConfigDb::DcuDetIdsV dcuDetIds_, DcuPsuVector dcuPsus_);
  
  // member variables
  edm::Service<SiStripConfigDb> db_;
  PsuDetIdMap pgMap, cgMap;
  std::vector<std::string> detectorLocations, controlLocations;
  std::vector<uint32_t> dcuIds, cgDcuIds, ccuDcuIds;
  DcuPsus DcuPsuMapPG_, DcuPsuMapCG_;
  //  std::vector< std::pair<uint32_t, SiStripConfigDb::DeviceAddress> > dcu_device_addr_vector;
  std::vector< std::pair< std::vector<uint16_t> , std::vector<uint32_t> > > dcu_device_addr_vector;
};
#endif

#ifndef SISTRIPPSUDETIDMAP_H
#define SISTRIPPSUDETIDMAP_H

#include <memory>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/MapOfVectors.h"

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "DeviceFactory.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>
#include <cstdint>

class SiStripConfigDb;

/**	
   \class SiStripPsuDetIdMap
   \brief Extension to SiStripConfigDb to map PSU channels to DetIDs using DCU-PSU map and DCU-DetID map
   \author J.Cole
*/

class SiStripPsuDetIdMap {
public:
  /** Constructor */
  SiStripPsuDetIdMap();
  /** Destructor */
  ~SiStripPsuDetIdMap();

  std::vector<uint32_t> getLvDetID(std::string psu);
  void getHvDetID(std::string psuchannel,
                  std::vector<uint32_t> &ids,
                  std::vector<uint32_t> &unmapped_ids,
                  std::vector<uint32_t> &crosstalking_ids);

  //Produces 3 list of detIDs:
  //1-detids (positively matching the PSUChannel for HV case, positively matching the PSU for the LV case)
  //2-unmapped_detids (matching the PSUChannel000 for the HV case, empty for LV case)
  //3-crosstalking_detids (matching the PSUChannel999 for the HV case, empty for the LV case)
  void getDetID(std::string pvss,
                bool,
                std::vector<uint32_t> &detids,
                std::vector<uint32_t> &unmapped_detids,
                std::vector<uint32_t> &crosstalking_detids);
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

  //Return the HVUnmapped PSU channels as a map initialized to all channels (002/003) OFF (false):
  std::map<std::string, std::vector<uint32_t> > getHVUnmappedMap() { return HVUnmapped_Map; }
  //Return the HVCrosstalking PSU channels as a map initialized to all channels (002/003) OFF (false):
  std::map<std::string, std::vector<uint32_t> > getHVCrosstalkingMap() { return HVCrosstalking_Map; }
  //PsuDetIdMap getHVUnmappedDetIdMap() {return HVUnmapped_Map}
  //Return the HVUnmapped PSUchannel to (HV status) map initialized to all OFF:
  //PsuDetIdMap getHVUnmappedChannelMap() {return HVUnmapped_ChanStatus}
  //PsuDetIdMap getHVCrosstalkingChannelMap() {return HVCrossTalking_ChanStatus}
  /** Return the PG PSU-DETID map as a vector. */
  std::vector<std::pair<uint32_t, std::string> > getPsuDetIdMap() { return pgMap; }
  /** Return the PG detector locations as a vector - one-to-one correspondance with the contents of the PSU-DetID map vector. */
  std::vector<std::string> getDetectorLocations() { return detectorLocations; }
  /** Return the DCU IDs associated to the PG map. */
  std::vector<uint32_t> getDcuIds() { return dcuIds; }
  /** Return the CG PSU-DETID map as a vector. */
  std::vector<std::pair<uint32_t, std::string> > getControlPsuDetIdMap() { return cgMap; }
  /** Return the CG detector locations as a vector - one-to-one correspondance with the contents of the PSU-DetID map vector. */
  std::vector<std::string> getControlDetectorLocations() { return controlLocations; }
  /** Return the module DCU IDs associated to the CG map. */
  std::vector<uint32_t> getCgDcuIds() { return cgDcuIds; }
  /** Return the CCU DCU IDs associated to the CG map. */
  std::vector<uint32_t> getCcuDcuIds() { return ccuDcuIds; }

  /** Produces a formatted printout of the PSU-DETID map. */
  void printMap();
  /** Produces a formatted printout of the control PSU-DETID map. */
  void printControlMap();
  /** Main routine that accesses the DB and builds the PSU-DETID map. */
  //void BuildMap();
  /**
   * Build the map from given file.
   * ATTENTION: this will only build the pgMap, not the cgMap.
   */
  void BuildMap(const std::string &mapFile, const bool debug);
  //Old "rawmap" (vector of pairs) method to be used by excludeddetids:
  void BuildMap(const std::string &mapFile, std::vector<std::pair<uint32_t, std::string> > &rawmap);
  /// Overloaded method that does the buidling
  void BuildMap(const std::string &mapFile,
                const bool debug,
                std::map<std::string, std::vector<uint32_t> > &LVmap,
                std::map<std::string, std::vector<uint32_t> > &HVmap,
                std::map<std::string, std::vector<uint32_t> > &HVUnmappedmap,
                std::map<std::string, std::vector<uint32_t> > &HVCrosstalkingmap);

  //Service function to remove duplicated from vectors of detids:
  void RemoveDuplicateDetIDs(std::vector<uint32_t> &detids);

  /** Returns the DCU-PSU map as a vector. */
  std::vector<std::pair<uint32_t, std::string> > getDcuPsuMap();
  /** Returns 1 if the specified PSU channel is a HV channel, 0 if it is a LV channel.  -1 means error. */
  int IsHVChannel(std::string pvss);

private:
  // typedefs
  typedef std::vector<TkDcuPsuMap *> DcuPsuVector;
  typedef std::map<std::string, std::vector<uint32_t> > PsuDetIdMap;
  typedef edm::MapOfVectors<std::string, TkDcuPsuMap *> DcuPsus;
  typedef DcuPsus::range DcuPsusRange;
  /** Extracts the DCU-PSU map from the DB. */
  void getDcuPsuMap(DcuPsusRange &pRange, DcuPsusRange &cRange, std::string partition);
  /** Extracts the DCU device descriptions and stores them for further use. Only used for control groups. */
  //  std::vector< std::pair<uint32_t, SiStripConfigDb::DeviceAddress> >  retrieveDcuDeviceAddresses(std::string partition);
  std::vector<std::pair<std::vector<uint16_t>, std::vector<uint32_t> > > retrieveDcuDeviceAddresses(
      std::string partition);
  /** Searches the DCU device descriptions for the specified DCU ID. Needed for control groups. */
  std::vector<uint32_t> findDcuIdFromDeviceAddress(uint32_t dcuid_);
  /** Utility to clone a DCU-PSU map. */
  void clone(DcuPsuVector &input, DcuPsuVector &output);
  /** Produces a detailed debug of the input values. */
  // for debugging
  void checkMapInputValues(const SiStripConfigDb::DcuDetIdsV &dcuDetIds_, const DcuPsuVector &dcuPsus_);

  // member variables
  edm::Service<SiStripConfigDb> db_;
  PsuDetIdMap LVMap, HVMap, HVUnmapped_Map, HVCrosstalking_Map;
  std::vector<std::pair<uint32_t, std::string> > pgMap, cgMap;
  std::vector<std::string> detectorLocations, controlLocations;
  std::vector<uint32_t> dcuIds, cgDcuIds, ccuDcuIds;
  DcuPsus DcuPsuMapPG_, DcuPsuMapCG_;
  //  std::vector< std::pair<uint32_t, SiStripConfigDb::DeviceAddress> > dcu_device_addr_vector;
  std::vector<std::pair<std::vector<uint16_t>, std::vector<uint32_t> > > dcu_device_addr_vector;
};
#endif

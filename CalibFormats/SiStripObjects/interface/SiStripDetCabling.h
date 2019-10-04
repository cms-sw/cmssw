#ifndef SiStripObjects_SiStripDetCabling_h
#define SiStripObjects_SiStripDetCabling_h
// -*- C++ -*-
//
// Package:     CalibFormats/SiStripObjects
// Class  :     SiStripDetCabling
/**\class SiStripDetCabling SiStripDetCabling.h
 CalibFormats/SiStripObjects/interface/SiStripDetCabling.h

 Description: give detector view of the cabling of the silicon strip tracker
*/
// Original Author:  dkcira
//         Created:  Wed Mar 22 12:24:20 CET 2006
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include <map>
#include <string>
#include <vector>
#include <cstdint>
class TrackerTopology;
class SiStripDetCabling {
public:
  SiStripDetCabling(const TrackerTopology *const topology);
  virtual ~SiStripDetCabling();
  SiStripDetCabling(const SiStripFedCabling &, const TrackerTopology *const topology);

  SiStripDetCabling(const SiStripDetCabling &) = delete;
  const SiStripDetCabling &operator=(const SiStripDetCabling &) = delete;

  void addDevices(const FedChannelConnection &, std::map<uint32_t, std::vector<const FedChannelConnection *>> &);
  void addDevices(const FedChannelConnection &);  // special case of above addDevices
  // getters
  inline const std::map<uint32_t, std::vector<const FedChannelConnection *>> &getDetCabling() const {
    return fullcabling_;
  }
  // for DQM use: all detectors that have at least one connected APV
  void addActiveDetectorsRawIds(
      std::vector<uint32_t> &) const;  // add to vector Ids of connected modules (active == connected)
  void addAllDetectorsRawIds(
      std::vector<uint32_t> &vector_to_fill_with_detids) const;  // add to vector Ids of all modules
  void getAllDetectorsContiguousIds(
      std::map<uint32_t, unsigned int> &) const;  // map of all connected, detected, undetected to contiguous Ids -
                                                  // map is reset first!
  void getActiveDetectorsContiguousIds(
      std::map<uint32_t, unsigned int> &) const;  // map of all connected to contiguous Ids - map is reset first!
  // for RECO use
  void addConnected(std::map<uint32_t, std::vector<int>> &)
      const;  // map of detector to list of APVs for APVs seen from FECs and FEDs
  void addDetected(
      std::map<uint32_t, std::vector<int>> &) const;  // map of detector to list of APVs for APVs seen from FECs but not
                                                      // from FEDs
  void addUnDetected(
      std::map<uint32_t, std::vector<int>> &) const;  // map of detector to list of APVs for APVs seen neither from FECS
                                                      // or FEDs
  void addNotConnectedAPVs(
      std::map<uint32_t, std::vector<int>> &) const;  // map of detector to list of APVs that are not connected -
                                                      // combination of addDetected and addUnDetected
  // other
  const std::vector<const FedChannelConnection *> &getConnections(uint32_t det_id) const;
  const FedChannelConnection &getConnection(uint32_t det_id, unsigned short apv_pair) const;
  const unsigned int getDcuId(uint32_t det_id) const;
  const uint16_t nApvPairs(uint32_t det_id) const;  // maximal nr. of apvpairs a detector can have (2 or 3)
  bool IsConnected(const uint32_t &det_id) const;
  bool IsDetected(const uint32_t &det_id) const;
  bool IsUndetected(const uint32_t &det_id) const;

  /** Added missing print method. */
  void print(std::stringstream &) const;

  /** The printSummary method outputs the number of
   * connected/detected/undetected modules for each layer of each subdetector.*/
  void printSummary(std::stringstream &ss, const TrackerTopology *trackerTopo) const;
  /** The printDebug method returns all the connected/detected/undetected
   * modules.*/
  void printDebug(std::stringstream &ss, const TrackerTopology *trackerTopo) const;

  // Methods to get the number of connected, detected and undetected modules for
  // each layer of each subdetector.
  uint32_t connectedNumber(const std::string &subDet, const uint16_t layer) const {
    return detNumber(subDet, layer, 0);
  }
  uint32_t detectedNumber(const std::string &subDet, const uint16_t layer) const { return detNumber(subDet, layer, 1); }
  uint32_t undetectedNumber(const std::string &subDet, const uint16_t layer) const {
    return detNumber(subDet, layer, 2);
  }
  inline const SiStripFedCabling *fedCabling() const { return fedCabling_; }
  inline const TrackerTopology *const trackerTopology() const { return tTopo; }

  std::map<uint32_t, std::vector<int>> const &connected() const { return connected_; }

private:
  void addFromSpecificConnection(std::map<uint32_t, std::vector<int>> &,
                                 const std::map<uint32_t, std::vector<int>> &,
                                 std::map<int16_t, uint32_t> *connectionsToFill = nullptr) const;
  bool IsInMap(const uint32_t &det_id, const std::map<uint32_t, std::vector<int>> &) const;
  int16_t layerSearch(const uint32_t detId) const;
  uint32_t detNumber(const std::string &subDet, const uint16_t layer, const int connectionType) const;

  // ---------- member data --------------------------------
  // map of KEY=detid DATA=vector<FedChannelConnection>
  std::map<uint32_t, std::vector<const FedChannelConnection *>> fullcabling_;
  // map of KEY=detid DATA=vector of apvs, maximum 6 APVs per detector module :
  // 0,1,2,3,4,5
  std::map<uint32_t, std::vector<int>> connected_;   // seen from FECs and FEDs
  std::map<uint32_t, std::vector<int>> detected_;    // seen from FECs but not from FEDs
  std::map<uint32_t, std::vector<int>> undetected_;  // seen from neither FECs or FEDs, DetIds inferred from
                                                     // static Look-Up-Table in the configuration database

  // Map containing the number of detectors for each connectionType
  // 0 = connected
  // 1 = detected
  // 2 = undetected
  std::map<int16_t, uint32_t> connectionCount[3];
  const SiStripFedCabling *fedCabling_;
  const TrackerTopology *const tTopo;
};
#endif

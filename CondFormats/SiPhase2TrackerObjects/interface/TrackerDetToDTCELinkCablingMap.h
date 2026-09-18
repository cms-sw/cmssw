#ifndef CondFormats_Phase2TrackerDTC_TrackerDetToDTCELinkCablingMap_h
#define CondFormats_Phase2TrackerDTC_TrackerDetToDTCELinkCablingMap_h

// -*- C++ -*-
//
// Package:    CondFormats/Phase2TrackerDTC
// Class:      TrackerDetToDTCELinkCablingMap
//
/**\class TrackerDetToDTCELinkCablingMap TrackerDetToDTCELinkCablingMap.cc CondFormats/Phase2TrackerDTC/src/TrackerDetToDTCELinkCablingMap.cc

Description: Map associating DTCELinkId of Phase2 tracker DTCs to DetId of the sensors connected to each of them.

Implementation:
		[Notes on implementation]
*/
//
// Original Author :  Luigi Calligaris, SPRACE, Sao Paulo, BR
// Created         :  Wed, 27 Feb 2019 21:41:13 GMT
// Updated         :  Si Hyun Jeon, Boston University, Thu, 21 May 2026
// Updated Content :  Detailed module info field for inner trackers
//
//

#include <map>
#include <vector>
#include <unordered_map>
#include <cstdint>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"

class TrackerDetToDTCELinkCablingMap {
public:
  /// Section : uint8_t convention used by ModuleInfo::section.
  enum class Section : uint8_t { Unknown = 0, TBPX = 1, TFPX = 2, TEPX = 3 };

  /// Per-module Aurora/cabling info (in addition to the per-elink mapping).
  /// Default-constructed = "unknown / not filled in this DB version".
  struct ModuleInfo {
    uint8_t nChips  = 0;  // chips on the module
    uint8_t nElinks = 0;  // elinks on the module
    uint8_t section = 0;  // see Section enum
    uint8_t layer   = 0;  // layer/disk indexes
    uint8_t ring    = 0;  // ring indexes
    uint8_t subtype = 0;  // subtype of the module

    COND_SERIALIZABLE;
  };

  TrackerDetToDTCELinkCablingMap();
  virtual ~TrackerDetToDTCELinkCablingMap();

  /// Resolves the raw DetId of the detector connected to the eLink identified by a DTCELinkId
  std::unordered_map<DTCELinkId, uint32_t>::const_iterator dtcELinkIdToDetId(DTCELinkId const&) const;

  /// Resolves one or more DTCELinkId of eLinks which are connected to the detector identified by the given raw DetId
  std::pair<std::unordered_multimap<uint32_t, DTCELinkId>::const_iterator,
            std::unordered_multimap<uint32_t, DTCELinkId>::const_iterator>
  detIdToDTCELinkId(uint32_t const) const;

  /// Returns true if the cabling map has a record corresponding to a detector identified by the given raw DetId
  bool knowsDTCELinkId(DTCELinkId const&) const;

  /// Returns true if the cabling map has a record corresponding to an eLink identified by the given DTCELinkId
  bool knowsDetId(uint32_t) const;

  /// Return all DetIds associated with a given DTCId
  std::vector<uint32_t> getAllDetIdsForDTCId(unsigned int dtcId) const;

  // IMPORTANT: The following information is not stored, to preserve space in memory.
  // As these vectors are generated each time the functions are called, you are encouraged to
  // either cache the results or avoid calling them in hot loops.
  // NOTE: This vectors are unsorted

  /// Returns a vector containing all elink DTCELinkId known to the map
  std::vector<DTCELinkId> getKnownDTCELinkIds() const;

  /// Returns a vector containing all DTCIds (unique) known to the map
  std::vector<unsigned int> getKnownDTCIds() const;
  std::vector<std::pair<unsigned int, unsigned int>> getKnownDTCIdsWithIndex() const;

  /// Returns a vector containing all detector DetId known to the map
  std::vector<uint32_t> getKnownDetIds() const;

  /// Inserts in the cabling map a record corresponding to the connection of an eLink identified by the given DTCELinkId to a detector identified by the given raw DetId
  void insert(DTCELinkId const&, uint32_t const);

  // Set / get per-module info (one entry per DetId)
  void setModuleInfo(uint32_t detId, ModuleInfo const& info);
  bool hasModuleInfo(uint32_t detId) const;
  ModuleInfo const& getModuleInfo(uint32_t detId) const;

  /// Clears the map
  void clear();

private:
  std::unordered_multimap<uint32_t, DTCELinkId> cablingMapDetIdToDTCELinkId_;
  std::unordered_map<DTCELinkId, uint32_t> cablingMapDTCELinkIdToDetId_;
  std::map<uint32_t, ModuleInfo> moduleInfoByDetId_;

  COND_SERIALIZABLE;
};

#endif  // end CondFormats_Phase2TrackerDTC_TrackerDetToDTCELinkCablingMap_h

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
// Original Author:  Luigi Calligaris, SPRACE, Sao Paulo, BR
// Created        :  Wed, 27 Feb 2019 21:41:13 GMT
//
//

#include <vector>
#include <unordered_map>
#include <cstdint>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"

class TrackerDetToDTCELinkCablingMap {
public:
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

  // IMPORTANT: The following information is not stored, to preserve space in memory.
  // As these vectors are generated each time the functions are called, you are encouraged to
  // either cache the results or avoid calling them in hot loops.
  // NOTE: This vectors are unsorted

  /// Returns a vector containing all elink DTCELinkId nown to the map
  std::vector<DTCELinkId> getKnownDTCELinkIds() const;

  /// Returns a vector containing all detector DetId known to the map
  std::vector<uint32_t> getKnownDetIds() const;

  /// Inserts in the cabling map a record corresponding to the connection of an eLink identified by the given DTCELinkId to a detector identified by the given raw DetId
  void insert(DTCELinkId const&, uint32_t const);

  /// Clears the map
  void clear();

private:
  std::unordered_multimap<uint32_t, DTCELinkId> cablingMapDetIdToDTCELinkId_;
  std::unordered_map<DTCELinkId, uint32_t> cablingMapDTCELinkIdToDetId_;

  COND_SERIALIZABLE;
};

#endif  // end CondFormats_Phase2TrackerDTC_TrackerDetToDTCELinkCablingMap_h

#ifndef L1TMuonEndCap_Common_h
#define L1TMuonEndCap_Common_h

#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFRoad.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"

#include "L1Trigger/L1TMuon/interface/GeometryTranslator.h"
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitiveFwd.h"

#include "L1Trigger/L1TMuonEndCap/interface/EMTFSubsystemTag.h"


// Resolve namespaces

typedef l1t::EMTFHit             EMTFHit;
typedef l1t::EMTFHitCollection   EMTFHitCollection;
typedef l1t::EMTFRoad            EMTFRoad;
typedef l1t::EMTFRoadCollection  EMTFRoadCollection;
typedef l1t::EMTFTrack           EMTFTrack;
typedef l1t::EMTFTrackCollection EMTFTrackCollection;
typedef l1t::EMTFPtLUT           EMTFPtLUT;

typedef L1TMuon::GeometryTranslator         GeometryTranslator;
typedef L1TMuon::TriggerPrimitive           TriggerPrimitive;
typedef L1TMuon::TriggerPrimitiveCollection TriggerPrimitiveCollection;

typedef TriggerPrimitive::CSCData CSCData;
typedef TriggerPrimitive::RPCData RPCData;
typedef TriggerPrimitive::GEMData GEMData;

typedef emtf::CSCTag CSCTag;
typedef emtf::RPCTag RPCTag;
typedef emtf::GEMTag GEMTag;

// Constants

// Phase 2 Geometry a.k.a. HL-LHC
#define PHASE_TWO_GEOMETRY 0

// from DataFormats/MuonDetId/interface/CSCDetId.h
#define MIN_ENDCAP 1
#define MAX_ENDCAP 2

// from DataFormats/MuonDetId/interface/CSCTriggerNumbering.h
#define MIN_TRIGSECTOR 1
#define MAX_TRIGSECTOR 6
#define NUM_SECTORS 12

// Zones
#define NUM_ZONES 4
#define NUM_ZONE_HITS 160

// Stations
#define NUM_STATIONS 4
#define NUM_STATION_PAIRS 6

// Fixed-size arrays
#include <array>
template<typename T>
using sector_array = std::array<T, NUM_SECTORS>;
template<typename T>
using zone_array = std::array<T, NUM_ZONES>;

#endif


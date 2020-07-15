#ifndef DataFormats_MuonDetId_GEMDetId_h
#define DataFormats_MuonDetId_GEMDetId_h

/** \class GEMDetId
 * 
 *  DetUnit identifier for GEMs
 *
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iosfwd>
#include <iostream>

class GEMDetId : public DetId {
public:
  static constexpr int32_t minRegionId = -1;
  static constexpr int32_t maxRegionId = 1;
  static constexpr int32_t minRingId = 1;
  static constexpr int32_t maxRingId = 3;
  static constexpr int32_t minStationId0 = 0;
  static constexpr int32_t minStationId = 1;
  // in the detId there is space to go up to 5 stations. Only 3 implemented now (0,1,2)
  static constexpr int32_t maxStationId = 2;
  static constexpr int32_t minChamberId = 0;
  static constexpr int32_t maxChamberId = 36;
  static constexpr int32_t minLayerId = 0;  // LayerId = 0 is superChamber
  static constexpr int32_t maxLayerId0 = 6;
  static constexpr int32_t maxLayerId = 2;  // GE1/GE2 has 2 layers
  static constexpr int32_t minRollId = 0;
  static constexpr int32_t maxRollId = 16;

private:
  static constexpr uint32_t RegionNumBits = 2;
  static constexpr uint32_t RegionStartBit = 0;
  static constexpr uint32_t RegionMask = 0x3;
  static constexpr uint32_t RingNumBits = 3;
  static constexpr uint32_t RingStartBit = RegionStartBit + RegionNumBits;
  static constexpr uint32_t RingMask = 0x7;
  static constexpr uint32_t StationNumBits = 3;
  static constexpr uint32_t StationStartBit = RingStartBit + RingNumBits;
  static constexpr uint32_t StationMask = 0x7;
  static constexpr uint32_t ChamberNumBits = 6;
  static constexpr uint32_t ChamberStartBit = StationStartBit + StationNumBits;
  static constexpr uint32_t ChamberStartBitM = RegionStartBit + RegionNumBits;
  static constexpr uint32_t ChamberMask = 0x3F;
  static constexpr uint32_t LayerNumBits = 5;
  static constexpr uint32_t LayerNumBitsP = 2;
  static constexpr uint32_t LayerStartBit = ChamberStartBit + ChamberNumBits;
  static constexpr uint32_t LayerStartBitM = ChamberStartBitM + ChamberNumBits;
  static constexpr uint32_t LayerMask = 0x1F;
  static constexpr uint32_t LayerMaskP = 0x3;
  static constexpr uint32_t RollNumBits = 5;
  static constexpr uint32_t RollStartBit = LayerStartBit + LayerNumBits;
  static constexpr uint32_t RollStartBitP = LayerStartBit + LayerNumBitsP;
  static constexpr uint32_t RollStartBitM = LayerStartBitM + LayerNumBits;
  static constexpr uint32_t RollMask = 0x1F;
  static constexpr uint32_t FormatNumBits = 1;
  static constexpr uint32_t FormatStartBit = RollStartBit + RollNumBits;
  static constexpr uint32_t FormatMask = 0x1;
  static constexpr uint32_t kGEMIdFormat = 0x1000000;
  static constexpr uint32_t kMuonIdMask = 0xF0000000;
  static constexpr uint32_t chamberIdMask = ~(RollMask << RollStartBit);
  static constexpr uint32_t superChamberIdMask = chamberIdMask + ~(LayerMask << LayerStartBit);

public:
  /** Create a null detId */
  constexpr GEMDetId() : DetId(DetId::Muon, MuonSubdetId::GEM) {}
  /** Construct from a packed id. It is required that the Detector part of
      id is Muon and the SubDet part is GEM, otherwise an exception is thrown*/
  constexpr GEMDetId(uint32_t id) : DetId(id) {
    if (det() != DetId::Muon || (subdetId() != MuonSubdetId::GEM && subdetId() != MuonSubdetId::ME0))
      throw cms::Exception("InvalidDetId")
          << "GEMDetId ctor: det: " << det() << " subdet: " << subdetId() << " is not a valid GEM id\n";

    if (v11Format())
      id_ = v12Form(id);
  }
  /** Construct from a generic cell ID */
  constexpr GEMDetId(DetId id) : DetId(id) {
    if (det() != DetId::Muon || (subdetId() != MuonSubdetId::GEM && subdetId() != MuonSubdetId::ME0))
      throw cms::Exception("InvalidDetId")
          << "GEMDetId ctor: det: " << det() << " subdet: " << subdetId() << " is not a valid GEM id\n";
    if (v11Format())
      id_ = v12Form(id.rawId());
  }
  /// Construct from fully qualified identifier.
  constexpr GEMDetId(int region, int ring, int station, int layer, int chamber, int roll)
      : DetId(DetId::Muon, MuonSubdetId::GEM) {
    if (region < minRegionId || region > maxRegionId || ring < minRingId || ring > maxRingId ||
        station < minStationId0 || station > maxStationId || layer < minLayerId || layer > maxLayerId0 ||
        chamber < minChamberId || chamber > maxChamberId || roll < minRollId || roll > maxRollId)
      throw cms::Exception("InvalidDetId")
          << "GEMDetId ctor: Invalid parameters:  region " << region << " ring " << ring << " station " << station
          << " layer " << layer << " chamber " << chamber << " roll " << roll << std::endl;

    int regionInBits = region - minRegionId;
    int ringInBits = ring - minRingId;
    int stationInBits = station - minStationId0;
    int layerInBits = layer - minLayerId;
    int chamberInBits = chamber - (minChamberId + 1);
    int rollInBits = roll;

    id_ |= ((regionInBits & RegionMask) << RegionStartBit | (ringInBits & RingMask) << RingStartBit |
            (stationInBits & StationMask) << StationStartBit | (layerInBits & LayerMask) << LayerStartBit |
            (chamberInBits & ChamberMask) << ChamberStartBit | (rollInBits & RollMask) << RollStartBit | kGEMIdFormat);
  }

  /** Assignment from a generic cell id */
  constexpr GEMDetId& operator=(const DetId& gen) {
    if (!gen.null()) {
      int subdet = gen.subdetId();
      if (gen.det() != Muon || (subdet != MuonSubdetId::GEM && subdet != MuonSubdetId::ME0))
        throw cms::Exception("InvalidDetId")
            << "GEMDetId ctor: Cannot assign GEMDetID from  " << std::hex << gen.rawId() << std::dec;
      if (v11Format())
        id_ = v12Form(gen.rawId());
      else
        id_ = gen.rawId();
    } else {
      id_ = gen.rawId();
    }
    return (*this);
  }

  /** Comparison operator */
  constexpr bool operator==(const GEMDetId& gen) const {
    uint32_t rawid = gen.rawId();
    if (rawid == id_)
      return true;
    int reg(0), ri(0), stn(-1), lay(0), chamb(0), rol(0);
    unpackId(rawid, reg, ri, stn, lay, chamb, rol);
    return (((id_ & kMuonIdMask) == (rawid & kMuonIdMask)) && (reg == region()) && (ri == ring()) &&
            (stn == station()) && (lay == layer()) && (chamb == chamber()) && (rol == roll()));
  }
  constexpr bool operator!=(const GEMDetId& gen) const {
    uint32_t rawid = gen.rawId();
    if (rawid == id_)
      return false;
    int reg(0), ri(0), stn(-1), lay(0), chamb(0), rol(0);
    unpackId(rawid, reg, ri, stn, lay, chamb, rol);
    return (((id_ & kMuonIdMask) != (rawid & kMuonIdMask)) || (reg != region()) || (ri != ring()) ||
            (stn != station()) || (lay != layer()) || (chamb != chamber()) || (rol != roll()));
  }

  /** Sort Operator based on the raw detector id */
  constexpr bool operator<(const GEMDetId& r) const {
    if (r.station() == this->station()) {
      if (this->layer() == r.layer()) {
        return this->rawId() < r.rawId();
      } else {
        return (this->layer() < r.layer());
      }
    } else {
      return this->station() < r.station();
    }
  }

  /** Check the format */
  constexpr bool v11Format() const { return ((id_ & kGEMIdFormat) == 0); }

  /** Region id: 0 for Barrel Not in use, +/-1 For +/- Endcap */
  constexpr int region() const { return (static_cast<int>((id_ >> RegionStartBit) & RegionMask) + minRegionId); }

  /** Ring id: GEM are installed only on ring 1
      the ring is the group of chambers with same r (distance of beam axis) 
      and increasing phi */
  constexpr int ring() const { return (static_cast<int>((id_ >> RingStartBit) & RingMask) + minRingId); }

  /** Station id : the station is the set of chambers at same disk */
  constexpr int station() const { return (static_cast<int>((id_ >> StationStartBit) & StationMask) + minStationId0); }

  /** Chamber id: it identifies a chamber in a ring it goes from 1 to 36 
      for GE1 and GE2 and 1 to 18 for ME0 */
  constexpr int chamber() const {
    return (static_cast<int>((id_ >> ChamberStartBit) & ChamberMask) + (minChamberId + 1));
  }

  /** Layer id: each station have two layers of chambers for GE1 and GE2: 
      layer 1 is the inner chamber and layer 2 is the outer chamber 
      For ME0 there are 6 layers of chambers */
  constexpr int layer() const { return (static_cast<int>((id_ >> LayerStartBit) & LayerMask) + minLayerId); }

  /** Roll id  (also known as eta partition): each chamber is divided along 
      the strip direction in  several parts  (rolls) GEM up to 12 */
  constexpr int roll() const {
    return (static_cast<int>((id_ >> RollStartBit) & RollMask));  // value 0 is used as wild card
  }

  /** Return the corresponding ChamberId */
  constexpr GEMDetId chamberId() const { return GEMDetId(id_ & chamberIdMask); }

  /** Return the corresponding superChamberId */
  constexpr GEMDetId superChamberId() const { return GEMDetId(id_ & superChamberIdMask); }

  /** Return the corresponding LayerId (mask eta partition) */
  constexpr GEMDetId layerId() const { return GEMDetId(id_ & chamberIdMask); }

  /** Return total # of layers for this type of detector */
  constexpr int nlayers() const {
    return ((station() == 0) ? maxLayerId0 : ((station() > maxStationId) ? 0 : maxLayerId));
  }

  constexpr uint32_t v12Form() const { return v12Form(id_); }

  constexpr static uint32_t v12Form(const uint32_t& inpid) {
    uint32_t rawid(inpid);
    if ((rawid & kGEMIdFormat) == 0) {
      int region(0), ring(0), station(-1), layer(0), chamber(0), roll(0);
      unpackId(rawid, region, ring, station, layer, chamber, roll);
      int regionInBits = region - minRegionId;
      int ringInBits = ring - minRingId;
      int stationInBits = station - minStationId0;
      int layerInBits = layer - minLayerId;
      int chamberInBits = chamber - (minChamberId + 1);
      int rollInBits = roll;
      rawid = (((DetId::Muon & DetId::kDetMask) << DetId::kDetOffset) |
               ((MuonSubdetId::GEM & DetId::kSubdetMask) << DetId::kSubdetOffset) |
               ((regionInBits & RegionMask) << RegionStartBit) | ((ringInBits & RingMask) << RingStartBit) |
               ((stationInBits & StationMask) << StationStartBit) | ((layerInBits & LayerMask) << LayerStartBit) |
               ((chamberInBits & ChamberMask) << ChamberStartBit) | ((rollInBits & RollMask) << RollStartBit) |
               kGEMIdFormat);
    }
    return rawid;
  }

private:
  constexpr void v12FromV11(const uint32_t& rawid) { id_ = v12Form(rawid); }

  constexpr static void unpackId(
      const uint32_t& rawid, int& region, int& ring, int& station, int& layer, int& chamber, int& roll) {
    if (((rawid >> DetId::kDetOffset) & DetId::kDetMask) == DetId::Muon) {
      int subdet = ((rawid >> DetId::kSubdetOffset) & DetId::kSubdetMask);
      if (subdet == MuonSubdetId::GEM) {
        region = static_cast<int>(((rawid >> RegionStartBit) & RegionMask) + minRegionId);
        ring = (static_cast<int>((rawid >> RingStartBit) & RingMask) + minRingId);
        chamber = (static_cast<int>((rawid >> ChamberStartBit) & ChamberMask) + (minChamberId + 1));
        if ((rawid & kGEMIdFormat) == 0) {
          station = (static_cast<int>((rawid >> StationStartBit) & StationMask) + minStationId);
          layer = (static_cast<int>((rawid >> LayerStartBit) & LayerMaskP) + minLayerId);
          roll = (static_cast<int>((rawid >> RollStartBitP) & RollMask));
        } else {
          station = (static_cast<int>((rawid >> StationStartBit) & StationMask) + minStationId0);
          layer = (static_cast<int>((rawid >> LayerStartBit) & LayerMask) + minLayerId);
          roll = (static_cast<int>((rawid >> RollStartBit) & RollMask));
        }
      } else if (subdet == MuonSubdetId::ME0) {
        region = static_cast<int>(((rawid >> RegionStartBit) & RegionMask) + minRegionId);
        ring = 1;
        station = 0;
        chamber = (static_cast<int>((rawid >> ChamberStartBitM) & ChamberMask) + (minChamberId));
        layer = (static_cast<int>((rawid >> LayerStartBitM) & LayerMask) + minLayerId);
        roll = (static_cast<int>((rawid >> RollStartBitM) & RollMask));
      }
    }
  }

};  // GEMDetId

std::ostream& operator<<(std::ostream& os, const GEMDetId& id);

#endif

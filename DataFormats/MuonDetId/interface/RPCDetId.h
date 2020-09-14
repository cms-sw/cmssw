#ifndef MuonDetId_RPCDetId_h
#define MuonDetId_RPCDetId_h

/** \class RPCDetId
 * 
 *  DetUnit identifier for RPCs
 *
 *  \author Ilaria Segoni
 */

#include <DataFormats/DetId/interface/DetId.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <iosfwd>

class RPCDetId : public DetId {
public:
  RPCDetId();

  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is RPC, otherwise an exception is thrown.
  RPCDetId(uint32_t id);
  RPCDetId(DetId id);

  /// Construct from fully qualified identifier.
  RPCDetId(int region, int ring, int station, int sector, int layer, int subsector, int roll);

  /// Sort Operator based on the raw detector id
  bool operator<(const RPCDetId& r) const {
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

  void buildfromDB(int region,
                   int ring,
                   int layer,
                   int sector,
                   const std::string& subsector,
                   const std::string& roll,
                   const std::string& dbname);

  /// Built from the trigger det Index
  void buildfromTrIndex(int trIndex);

  /// Region id: 0 for Barrel, +/-1 For +/- Endcap
  int region() const { return int((id_ >> RegionStartBit_) & RegionMask_) + minRegionId; }

  /// Ring id: Wheel number in Barrel (from -2 to +2) Ring Number in Endcap (from 1 to 3)
  /// Ring has a different meaning in Barrel and Endcap! In Barrel it is wheel, in Endcap
  /// it is the physical ring located on a disk (a disk contains three rings). In Endcap
  /// the ring is the group of chambers with same r (distance of beam axis) and increasing phi
  int ring() const {
    int ring_ = (id_ >> RingStartBit_) & RingMask_;

    if (ring_ < RingBarrelOffSet) {
      if (this->region() == 0) {
        throw cms::Exception("InvalidDetId") << "RPCDetId ctor:"
                                             << " Ring - Region Inconsistency, "
                                             << " region " << this->region() << " ring " << ring_ << std::endl;
      }

      return int(ring_ + minRingForwardId);

    } else {  // if(ring_ >= RingBarrelOffSet)
      return int(ring_ - RingBarrelOffSet + minRingBarrelId);
    }
  }

  /// Station id : For Barrel: the four groups of chambers at same r (distance from beam axis) and increasing phi
  ///              For Endcap: the three groups of chambers at same z (distance from interaction point), i.e. the disk
  int station() const { return int((id_ >> StationStartBit_) & StationMask_) + minStationId; }

  /// Sector id: the group of chambers at same phi (and increasing r)
  int sector() const { return int((id_ >> SectorStartBit_) & SectorMask_) + (minSectorId + 1); }

  /// Layer id: each station can have two layers of chambers: layer 1 is the inner chamber and layer 2 is the outer chamber (when present)
  /// Only in Barrel: RB1 and RB2.
  int layer() const { return int((id_ >> LayerStartBit_) & LayerMask_) + minLayerId; }

  /// SubSector id : some sectors are divided along the phi direction in subsectors (from 1 to 4 in Barrel, from 1 to 6 in Endcap)
  int subsector() const { return int((id_ >> SubSectorStartBit_) & SubSectorMask_) + (minSubSectorId + 1); }

  /// Roll id  (also known as eta partition): each chamber is divided along the strip direction in
  /// two or three parts (rolls) for Barrel and two, three or four parts for endcap
  int roll() const {
    return int((id_ >> RollStartBit_) & RollMask_);  // value 0 is used as wild card
  }

  int trIndex() const { return trind; }

  /// Return the corresponding ChamberId
  RPCDetId chamberId() const { return RPCDetId(id_ & chamberIdMask_); }

  static constexpr int minRegionId = -1;
  static constexpr int maxRegionId = 1;

  static constexpr int minRingForwardId = 1;
  static constexpr int maxRingForwardId = 3;
  static constexpr int minRingBarrelId = -2;
  static constexpr int maxRingBarrelId = 2;
  static constexpr int RingBarrelOffSet = 3;

  static constexpr int minStationId = 1;
  static constexpr int maxStationId = 4;

  static constexpr int minSectorId = 0;
  static constexpr int maxSectorId = 12;
  static constexpr int minSectorBarrelId = 1;
  static constexpr int maxSectorBarrelId = 12;
  static constexpr int minSectorForwardId = 1;
  static constexpr int maxSectorForwardId = 6;

  static constexpr int minLayerId = 1;
  static constexpr int maxLayerId = 2;

  static constexpr int minSubSectorId = 0;
  static constexpr int maxSubSectorId = 6;
  static constexpr int minSubSectorBarrelId = 1;
  static constexpr int maxSubSectorBarrelId = 4;
  static constexpr int minSubSectorForwardId = 1;
  static constexpr int maxSubSectorForwardId = 6;

  static constexpr int minRollId = 0;
  static constexpr int maxRollId = 5;  // used to be 4 ... need 5 for upgrade
                                       // if we decide to divide the upgrade RE3/1 and RE4/1 up to eta=2.4
                                       // and we want basically to have 0.10 eta / roll ...

private:
  static constexpr int RegionNumBits_ = 2;
  static constexpr int RegionStartBit_ = 0;
  static constexpr int RegionMask_ = 0X3;

  static constexpr int RingNumBits_ = 3;
  static constexpr int RingStartBit_ = RegionStartBit_ + RegionNumBits_;
  static constexpr unsigned int RingMask_ = 0X7;

  static constexpr int StationNumBits_ = 2;
  static constexpr int StationStartBit_ = RingStartBit_ + RingNumBits_;
  static constexpr unsigned int StationMask_ = 0X3;

  static constexpr int SectorNumBits_ = 4;
  static constexpr int SectorStartBit_ = StationStartBit_ + StationNumBits_;
  static constexpr unsigned int SectorMask_ = 0XF;

  static constexpr int LayerNumBits_ = 1;
  static constexpr int LayerStartBit_ = SectorStartBit_ + SectorNumBits_;
  static constexpr unsigned int LayerMask_ = 0X1;

  static constexpr int SubSectorNumBits_ = 3;
  static constexpr int SubSectorStartBit_ = LayerStartBit_ + LayerNumBits_;
  static constexpr unsigned int SubSectorMask_ = 0X7;

  static constexpr int RollNumBits_ = 3;
  static constexpr int RollStartBit_ = SubSectorStartBit_ + SubSectorNumBits_;
  static constexpr unsigned int RollMask_ = 0X7;

public:
  static constexpr uint32_t chamberIdMask_ = ~(RollMask_ << RollStartBit_);

private:
  void init(int region, int ring, int station, int sector, int layer, int subsector, int roll);

  int trind;
};  // RPCDetId

std::ostream& operator<<(std::ostream& os, const RPCDetId& id);

#endif

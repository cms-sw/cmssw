#ifndef DataFormats_MuonDetId_ME0DetId_h
#define DataFormats_MuonDetId_ME0DetId_h

/** \class ME0DetId
 * 
 *  DetUnit identifier for ME0s
 *
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iosfwd>
#include <iostream>

class ME0DetId : public DetId {
public:
  ME0DetId();

  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is ME0, otherwise an exception is thrown.
  ME0DetId(uint32_t id);
  ME0DetId(DetId id);

  /// Construct from fully qualified identifier.
  ME0DetId(int region, int layer, int chamber, int roll);

  /// Sort Operator based on the raw detector id
  bool operator<(const ME0DetId& r) const {
    if (this->layer() == r.layer()) {
      return this->rawId() < r.rawId();
    } else {
      return (this->layer() > r.layer());
    }
  }

  /// Region id: 0 for Barrel Not in use, +/-1 For +/- Endcap
  int region() const { return int((id_ >> RegionStartBit_) & RegionMask_) + minRegionId; }

  /// Chamber id: it identifies a chamber in a ring it goes from 1 to 36
  int chamber() const { return int((id_ >> ChamberStartBit_) & ChamberMask_) + minChamberId; }

  /// Layer id: each chamber has six layers of chambers: layer 1 is the inner layer and layer 6 is the outer layer
  int layer() const { return int((id_ >> LayerStartBit_) & LayerMask_) + minLayerId; }

  /// Roll id  (also known as eta partition): each chamber is divided along the strip direction in
  /// several parts  (rolls) ME0 up to 10
  int roll() const {
    return int((id_ >> RollStartBit_) & RollMask_) + minRollId;  // value 0 is used as wild card
  }

  /// Return the corresponding ChamberId (mask layers)
  ME0DetId chamberId() const { return ME0DetId(id_ & chamberIdMask_); }
  /// Return the corresponding LayerId (mask eta partition)
  ME0DetId layerId() const { return ME0DetId(id_ & layerIdMask_); }

  //Return the stationId, always 1 for now
  int station() const { return 1; }

  /// For future modifications (implement more layers)
  int nlayers() const { return int(maxLayerId); }

  static constexpr int minRegionId = -1;
  static constexpr int maxRegionId = 1;

  static constexpr int minChamberId = 0;
  static constexpr int maxChamberId = 18;  // ME0 ring consists of 18 chambers spanning 20 degrees

  static constexpr int minLayerId = 0;
  static constexpr int maxLayerId = 6;

  static constexpr int minRollId = 0;
  static constexpr int maxRollId = 10;  // ME0 layer consists of 10 etapartitions

private:
  static constexpr int RegionNumBits_ = 2;
  static constexpr int RegionStartBit_ = 0;
  static constexpr int RegionMask_ = 0X3;

  static constexpr int ChamberNumBits_ = 6;
  static constexpr int ChamberStartBit_ = RegionStartBit_ + RegionNumBits_;
  static constexpr unsigned int ChamberMask_ = 0X3F;

  static constexpr int LayerNumBits_ = 5;
  static constexpr int LayerStartBit_ = ChamberStartBit_ + ChamberNumBits_;
  static constexpr unsigned int LayerMask_ = 0X1F;

  static constexpr int RollNumBits_ = 5;
  static constexpr int RollStartBit_ = LayerStartBit_ + LayerNumBits_;
  static constexpr unsigned int RollMask_ = 0X1F;

public:
  static constexpr uint32_t chamberIdMask_ = ~((LayerMask_ << LayerStartBit_) | (RollMask_ << RollStartBit_));
  static constexpr uint32_t layerIdMask_ = ~(RollMask_ << RollStartBit_);

private:
  void init(int region, int layer, int chamber, int roll);

  int trind;
};  // ME0DetId

std::ostream& operator<<(std::ostream& os, const ME0DetId& id);

#endif

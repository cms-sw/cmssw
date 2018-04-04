#ifndef L1TMuonEndCap_EMTFGEMDetId_h
#define L1TMuonEndCap_EMTFGEMDetId_h

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

#include <cstdint>
#include <iosfwd>


class GEMDetId;
class ME0DetId;

class EMTFGEMDetId {
public:
  explicit EMTFGEMDetId(const GEMDetId& id);
  explicit EMTFGEMDetId(const ME0DetId& id);

  /// Sort Operator based on the raw detector id
  bool operator < (const EMTFGEMDetId& r) const;

  /// The identifiers
  int region() const;
  int ring() const;  // NOTE: use ME0 --> ring 4 convention
  int station() const;  // NOTE: use ME0 --> station 1 convention
  int layer() const;
  int chamber() const;
  int roll() const;

  bool isME0() const { return isME0_; }

  GEMDetId getGEMDetId() const { return gemDetId_; }

  ME0DetId getME0DetId() const { return me0DetId_; }

private:
  GEMDetId gemDetId_;
  ME0DetId me0DetId_;
  bool isME0_;
};

std::ostream& operator<<( std::ostream& os, const EMTFGEMDetId& id );

#endif

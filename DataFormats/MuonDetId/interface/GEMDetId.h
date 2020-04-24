#ifndef DataFormats_MuonDetId_GEMDetId_h
#define DataFormats_MuonDetId_GEMDetId_h

/** \class GEMDetId
 * 
 *  DetUnit identifier for GEMs
 *
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iosfwd>
#include <iostream>

class GEMDetId :public DetId {
  
 public:
      
  GEMDetId();

  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is GEM, otherwise an exception is thrown.
  GEMDetId(uint32_t id);
  GEMDetId(DetId id);


  /// Construct from fully qualified identifier.
  GEMDetId(int region, 
	   int ring,
	   int station, 
	   int layer,
	   int chamber,
	   int roll);
	   

  /// Sort Operator based on the raw detector id
  bool operator < (const GEMDetId& r) const{
    if (r.station() == this->station()  ){
      if (this->layer() ==  r.layer()  ){
	return this->rawId()<r.rawId();
      }
      else{
	return (this->layer() < r.layer());
      }
    }
    else {
      return this->station() < r.station();
    }
  }

  /// Region id: 0 for Barrel Not in use, +/-1 For +/- Endcap
  int region() const{
    return int((id_>>RegionStartBit_) & RegionMask_) + minRegionId;
  }

  /// Ring id: GEM are installed only on ring 1
  /// the ring is the group of chambers with same r (distance of beam axis) and increasing phi
  int ring() const{
      return int((id_>>RingStartBit_) & RingMask_) + minRingId;
  }

  /// Station id : the station is the pair of chambers at same disk
  int station() const{
    return int((id_>>StationStartBit_) & StationMask_) + minStationId;
  }

  /// Layer id: each station have two layers of chambers: layer 1 is the inner chamber and layer 2 is the outer chamber (when present)  
  int layer() const{
    return int((id_>>LayerStartBit_) & LayerMask_) + minLayerId;
  }

  /// Chamber id: it identifies a chamber in a ring it goes from 1 to 36 
  int chamber() const{
    return int((id_>>ChamberStartBit_) & ChamberMask_) + (minChamberId+1);
  }

 /// Roll id  (also known as eta partition): each chamber is divided along the strip direction in  
 /// several parts  (rolls) GEM up to 10
  int roll() const{
    return int((id_>>RollStartBit_) & RollMask_); // value 0 is used as wild card
  }

  /// Return the corresponding ChamberId
  GEMDetId chamberId() const {
    return GEMDetId(id_ & chamberIdMask_);
  }

  /// Return the corresponding superChamberId
  GEMDetId superChamberId() const {
    return GEMDetId(id_ & superChamberIdMask_);
  }

  static const int minRegionId=     -1;
  static const int maxRegionId=      1;
 
  static const int minRingId=   1;
  static const int maxRingId=   3;

  static const int minStationId=     1;
  static const int maxStationId=     2;  // in the detId there is space to go up to 4 stations. Only 2 implemented now

  static const int minChamberId=     0;
  static const int maxChamberId=     36;

  // LayerId = 0 is superChamber
  static const int minLayerId=     0;
  static const int maxLayerId=     2;

  static const int minRollId=	  0;
  static const int maxRollId=	 15;

 private:
  static const int RegionNumBits_  =  2;
  static const int RegionStartBit_ =  0;  
  static const int RegionMask_     =  0X3;

  static const int RingNumBits_  =  3;
  static const int RingStartBit_ =  RegionStartBit_+RegionNumBits_;  
  static const unsigned int RingMask_     =  0X7;

  static const int StationNumBits_  =  3;
  static const int StationStartBit_ =  RingStartBit_+RingNumBits_;  
  static const unsigned int StationMask_     =  0X7;


  static const int ChamberNumBits_  =  6;
  static const int ChamberStartBit_ =  StationStartBit_+StationNumBits_;  
  static const unsigned int ChamberMask_     =  0X3F;

  static const int LayerNumBits_  =  2;
  static const int LayerStartBit_ =  ChamberStartBit_+ChamberNumBits_;  
  static const unsigned int LayerMask_     =  0X3;

  static const int RollNumBits_  =  5;
  static const int RollStartBit_ =  LayerStartBit_+LayerNumBits_;  
  static const unsigned int RollMask_     =  0X1F;
 
  static const uint32_t chamberIdMask_ = ~(RollMask_<<RollStartBit_);

  static const uint32_t superChamberIdMask_ = chamberIdMask_ + ~(LayerMask_<<LayerStartBit_);
  
 private:
  void init(int region, 
	    int ring,
	    int station, 
	    int layer,
	    int chamber,
	    int roll);
  
  int trind;
}; // GEMDetId

std::ostream& operator<<( std::ostream& os, const GEMDetId& id );

#endif

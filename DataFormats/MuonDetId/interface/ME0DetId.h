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

class ME0DetId :public DetId {
  
 public:
      
  ME0DetId();

  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is ME0, otherwise an exception is thrown.
  ME0DetId(uint32_t id);
  ME0DetId(DetId id);


  /// Construct from fully qualified identifier.
  ME0DetId(int region, 
	   int layer,
	   int chamber,
	   int roll);
	   

  /// Sort Operator based on the raw detector id
  bool operator < (const ME0DetId& r) const{
    if (this->layer() ==  r.layer()  ){
      return this->rawId()<r.rawId();
    }
    else{
      return (this->layer() >  r.layer());
    }
  }

  /// Region id: 0 for Barrel Not in use, +/-1 For +/- Endcap
  int region() const{
    return int((id_>>RegionStartBit_) & RegionMask_) + minRegionId;
  }

  /// Chamber id: it identifies a chamber in a ring it goes from 1 to 36 
  int chamber() const{
    return int((id_>>ChamberStartBit_) & ChamberMask_) + minChamberId;
  }

  /// Layer id: each chamber has six layers of chambers: layer 1 is the inner layer and layer 6 is the outer layer 
  int layer() const{
    return int((id_>>LayerStartBit_) & LayerMask_) + minLayerId;
  }

  /// Roll id  (also known as eta partition): each chamber is divided along the strip direction in  
  /// several parts  (rolls) ME0 up to 10
  int roll() const{
    return int((id_>>RollStartBit_) & RollMask_) + minRollId; // value 0 is used as wild card
  }

  /// Return the corresponding ChamberId (mask layers)
  ME0DetId chamberId() const {
    return ME0DetId(id_ & chamberIdMask_ & layerIdMask_);
  }
  /// Return the corresponding LayerId (mask eta partition)
  ME0DetId layerId() const {
    return ME0DetId(id_ & layerIdMask_);
  }

  /// For future modifications (implement more layers)
  int nlayers() const{
    return int(maxLayerId);
  }

  static const int minRegionId=     -1;
  static const int maxRegionId=      1;
 
  static const int minChamberId=     0;
  static const int maxChamberId=     36; // ME0 ring consists of 36 chambers spanning 10 degrees

  static const int minLayerId=     0;
  static const int maxLayerId=     6; // ME0 chamber consists of 6 layers for now, change here when changing ME0 Geometry

  static const int minRollId=	  0;
  static const int maxRollId=	  1; // ME0 layer consits of 1 etapartition for now, change here when changing ME0 Geometry

 private:
  static const int RegionNumBits_  =  2;
  static const int RegionStartBit_ =  0;  
  static const int RegionMask_     =  0X3;

  static const int ChamberNumBits_  =  6;
  static const int ChamberStartBit_ =  RegionStartBit_+RegionNumBits_;  
  static const unsigned int ChamberMask_     =  0X3F;

  static const int LayerNumBits_  =  5;
  static const int LayerStartBit_ =  ChamberStartBit_+ChamberNumBits_;  
  static const unsigned int LayerMask_     =  0X1F;

  static const int RollNumBits_  =  5;
  static const int RollStartBit_ =  LayerStartBit_+LayerNumBits_;  
  static const unsigned int RollMask_     =  0X1F;
 
  static const uint32_t chamberIdMask_ = ~( (LayerMask_<<LayerStartBit_) | (RollMask_<<RollStartBit_));
  static const uint32_t layerIdMask_ = ~(RollMask_<<RollStartBit_);

 private:
  void init(int region, 
	    int layer,
	    int chamber,
	    int roll);
  
  int trind;
}; // ME0DetId

std::ostream& operator<<( std::ostream& os, const ME0DetId& id );

#endif

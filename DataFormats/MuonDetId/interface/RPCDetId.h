#ifndef MuonDetId_RPCDetId_h
#define MuonDetId_RPCDetId_h

/** \class RPCDetId
* 
*  DetUnit identifier for DT chambers
 *
 *  $Date: 2005/10/25 14:01:42 $
 *  \version $Id: RPCDetId.cc,v 1.1 2005/10/19 13:42:37 segoni Exp $
 *  $Revision: 1.2 $
 *  \author Stefano ARGIRO
 */

#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>
#include <DataFormats/DetId/interface/DetId.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <iosfwd>

class RPCDetId :public DetId {
  
 public:
      
  RPCDetId();

  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is RPC, otherwise an exception is thrown.
  explicit RPCDetId(uint32_t id);


  /// Construct from fully qualified identifier. Wire is optional since it
  /// is not reqired to identify a DetUnit, however it is part of the interface
  /// since it is required for the numbering schema.
  RPCDetId(int region, 
	   int ring,
	   int station, 
	   int sector,
	   int layer,
	   int subsector,
	   int roll);
	   
  /// Region id
  int region() const{
    return int((id_>>RegionStartBit_) & RegionMask_) + minRegionId;
  }


  /// Ring id
  int ring() const{
  
    int ring_= (id_>>RingStartBit_) & RingMask_;
        
    if(ring_ <  RingBarrelOffSet) return int(ring_ + minRingForwardId);
    if(ring_ >= RingBarrelOffSet) return int(ring_ - RingBarrelOffSet + minRingBarrelId);
   
  // check consistency
 
  }

  /// Station id
  int station() const{
    return int((id_>>StationStartBit_) & StationMask_) + minStationId;
  }


  /// Sector id
  int sector() const{
    return int((id_>>SectorStartBit_) & SectorMask_) + minSectorId;
  }

  /// Layer id
  int layer() const{
    return int((id_>>LayerStartBit_) & LayerMask_) + minLayerId;
  }


  /// SubSector id
  int subsector() const{
    return int((id_>>SubSectorStartBit_) & SubSectorMask_) + minSubSectorId;
  }

  /// Roll id
  int roll() const{
    return int((id_>>RollStartBit_) & RollMask_); // value 0 is used as wild card
  }



  static const int minRegionId=     -1;
  static const int maxRegionId=      1;
 
  static const int minRingId=	1;
  static const int maxRingId=	7;  
  static const int minRingForwardId=   1;
  static const int maxRingForwardId=   3;
  static const int minRingBarrelId=   -2;
  static const int maxRingBarrelId=    2;
  static const int RingBarrelOffSet=   3;
 
  static const int minStationId=     1;
  static const int maxStationId=     4;

  static const int minSectorId=     1;
  static const int maxSectorId=     4;

  static const int minLayerId=     1;
  static const int maxLayerId=     2;

  static const int minSubSectorId=	 1;
  static const int maxSubSectorId=	 4;

  static const int minRollId=	  1;
  static const int maxRollId=	  3;


 private:
  static const int RegionNumBits_  =  2;
  static const int RegionStartBit_ =  0;  
  static const int RegionMask_     =  0X3;

  static const int RingNumBits_  =  3;
  static const int RingStartBit_ =  RegionStartBit_+RegionNumBits_;  
  static const unsigned int RingMask_     =  0X7;

  static const int StationNumBits_  =  2;
  static const int StationStartBit_ =  RingStartBit_+RingNumBits_;  
  static const unsigned int StationMask_     =  0X2;


  static const int SectorNumBits_  =  4;
  static const int SectorStartBit_ =  StationStartBit_+StationNumBits_;  
  static const unsigned int SectorMask_     =  0XF;

  static const int LayerNumBits_  =  1;
  static const int LayerStartBit_ =  SectorStartBit_+SectorNumBits_;  
  static const unsigned int LayerMask_     =  0X1;

  static const int SubSectorNumBits_  =  2;
  static const int SubSectorStartBit_ =  LayerStartBit_+LayerNumBits_;  
  static const unsigned int SubSectorMask_     =  0X2;
  
  static const int RollNumBits_  =  2;
  static const int RollStartBit_ =  SubSectorStartBit_+SubSectorNumBits_;  
  static const unsigned int RollMask_     =  0X2;


//<<<<<<< RPCDetId.h
//=======

//>>>>>>> 1.2
  

 

}; // RPCDetId

std::ostream& operator<<( std::ostream& os, const RPCDetId& id );

#endif

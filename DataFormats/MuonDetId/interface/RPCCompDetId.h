#ifndef MuonDetId_RPCCompDetId_h
#define MuonDetId_RPCCompDetId_h
// -*- C++ -*-
//
// Package:     MuonDetId
// Class  :     RPCCompDetId
// 
/**\class RPCCompDetId RPCCompDetId.h DataFormats/MuonDetId/interface/RPCCompDetId.h

 Description: DetId for composite RPC objects

*/
//
// Author:      Marcello Maggi
// Created:     Wed Nov  2 12:09:10 CET 2011
// $Id: RPCCompDetId.h,v 1.1 2011/11/05 10:39:53 mmaggi Exp $
//
#include <DataFormats/DetId/interface/DetId.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <string>

class RPCCompDetId :public DetId {
  
 public:
      
  RPCCompDetId();

  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is RPC, otherwise an exception is thrown.
  RPCCompDetId(uint32_t id);
  RPCCompDetId(DetId id);


  /// Construct from fully qualified identifier.
  RPCCompDetId(int region, 
	       int ring,
	       int station, 
	       int sector,
	       int layer,
	       int subsector,
	       int type);

  /// Construct from name stored in DB
  RPCCompDetId(const std::string& dbname, int type);

  /// Sort Operator based on the name
  bool operator < (const RPCCompDetId& r) const;

  int region() const;
  int ring() const;
  int wheel() const;
  int station() const;
  int disk() const;
  int sector() const;
  int layer() const;
  int subsector() const;
  int type() const;
  std::string dbname() const; 

  static const int minRegionId=     -1;
  static const int maxRegionId=      1;
  static const int allRegionId=minRegionId-1;
 
  static const int minRingForwardId=   1;
  static const int maxRingForwardId=   3;
  static const int minRingBarrelId=   -2;
  static const int maxRingBarrelId=    2;
  static const int RingBarrelOffSet=   3;
  static const int allRingId=minRingBarrelId-1; 
 
  static const int minStationId=     1;
  static const int maxStationId=     4;
  static const int allStationId=minStationId-1; 

  static const int minSectorId=     1;
  static const int maxSectorId=     36;
  static const int minSectorBarrelId=     1;
  static const int maxSectorBarrelId=     12;
  static const int minSectorForwardId=     1;
  static const int maxSectorForwardId=    36;
  static const int allSectorId=minSectorId-1; 

  static const int minLayerId=     1;
  static const int maxLayerId=     2;
  static const int allLayerId=minLayerId-1; 
  

  static const int minSubSectorId= 1;
  static const int maxSubSectorId= 2;
  static const int allSubSectorId=minSubSectorId-1; 

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

  static const int SectorNumBits_  =  6;
  static const int SectorStartBit_ =  StationStartBit_+StationNumBits_;  
  static const unsigned int SectorMask_     =  0X3F;

  static const int LayerNumBits_  =  2;
  static const int LayerStartBit_ =  SectorStartBit_+SectorNumBits_;  
  static const unsigned int LayerMask_     =  0X3;

  static const int SubSectorNumBits_  =  2;
  static const int SubSectorStartBit_ =  LayerStartBit_+LayerNumBits_;  
  static const unsigned int SubSectorMask_     =  0X3;


 private:
  void init(int region, 
	    int ring,
	    int station, 
	    int sector,
	    int layer,
	    int subsector);

  void init();
  void initGas();
  std::string gasDBname() const;
 private:
  std::string _dbname;
  int _type;

}; // RPCCompDetId

std::ostream& operator<<( std::ostream& os, const RPCCompDetId& id );

#endif

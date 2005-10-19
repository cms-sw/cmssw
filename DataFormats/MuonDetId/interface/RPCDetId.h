#ifndef MuonDetId_RPCDetId_h
#define MuonDetId_RPCDetId_h

/** \class RPCDetId
* 
*  DetUnit identifier for DT chambers
 *
 *  $Date: 2005/10/18 17:57:47 $
 *  $Revision: 1.5 $
 *  \author Stefano ARGIRO
 */

#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>
#include <DataFormats/DetId/interface/DetId.h>

#include <iosfwd>

class RPCDetId :public DetId {
  
 public:
      
  RPCDetId();

  /// Construct from fully qualified identifier. Wire is optional since it
  /// is not reqired to identify a DetUnit, however it is part of the interface
  /// since it is required for the numbering schema.
  RPCDetId(int roll, 
	  unsigned int copy, 
	  unsigned int sector,
	  unsigned int plane,
	  unsigned int eta) :
    DetId(DetId::Muon, MuonSubdetId::RPC ){ 
      id_ |= (roll& rollMask_)  << rollStartBit_     |
	     (copy & copyMask_)  << copyStartBit_   |
	     (sector  &sectorMask_ )   << sectorStartBit_    |
	     (plane & planeMask_)    << planeStartBit_    |
	     (eta & etaMask_)          << etaStartBit_ ;
    }
  
  /// wheel id
  int roll() const{
    return int((id_>>rollStartBit_) & rollMask_);
  }

  /// copy id
  unsigned int copy() const
  { return ((id_>>copyStartBit_) & copyMask_) ;}

  /// sector id
  unsigned int sector() const 
  { return ((id_>>sectorStartBit_)& sectorMask_) ;}

  /// plane id
  unsigned int plane() const 
  {return ((id_>>planeStartBit_)&planeMask_) ;}

  /// eta id
  unsigned int eta() const 
  { return ((id_>>etaStartBit_)&etaMask_) ;}



  /// lowest roll number
  static const int minRollId=              0;
  /// highest roll number
  static const int maxRollId=              0;
  /// lowest copy id
  static const unsigned int minCopyId=     0;
  /// highest copy id
  static const unsigned int maxCopyId=     0;
  /// lowest sector id
  static const unsigned int minSectorId=   0;
  /// highest sector id
  static const unsigned int maxSectorId=   0;
  /// loweset plane id
  static const unsigned int minPlaneId=    0;
  /// highest plane id
  static const unsigned int maxPlaneId=    0;
  /// lowest eta id
  static const unsigned int minEtaId=      0;
  /// highest eta id
  static const unsigned int maxEtaId=      0;
 

 private:
  static const unsigned int rollNumBits_   =  0;  
  static const unsigned int rollStartBit_  =  0;
  static const unsigned int copyNumBits_   =  0;
  static const unsigned int copyStartBit_  =  rollStartBit_ + rollNumBits_;
  static const unsigned int sectorNumBits_ =  0;
  static const unsigned int sectorStartBit_=  copyStartBit_+ copyNumBits_;
  static const unsigned int planeNumBits_  =  0;
  static const unsigned int planeStartBit_ =  sectorStartBit_+sectorNumBits_;
  static const unsigned int etaNumBits_    =  0;
  static const unsigned int etaStartBit_   =  planeBit_+planeNumBits_;

  

  static const unsigned int rollMask_  =  0x0;
  static const unsigned int copyMask_  =  0x0;
  static const unsigned int sectorMask_=  0x0;
  static const unsigned int planeMask_ =  0x0;
  static const unsigned int etaMask_   =  0x0;
 

}; // RPCDetId

std::ostream& operator<<( std::ostream& os, const RPCDetId& id );

#endif

#ifndef ECALDETID_ECALTRIGTOWERDETID_H
#define ECALDETID_ECALTRIGTOWERDETID_H

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EcalTrigTowerDetId
    
   DetId for an Ecal Trigger tower

   $Id: EcalTrigTowerDetId.h,v 1.9 2009/01/16 23:51:34 heltsley Exp $
*/


class EcalTrigTowerDetId : public DetId {
 public:
  /** Constructor of a null id */
  EcalTrigTowerDetId();
  /** Constructor from a raw value */
  EcalTrigTowerDetId(uint32_t rawid);  
  /** \brief Constructor from signed ieta, iphi
   */
  EcalTrigTowerDetId(int zside, EcalSubdetector subdet, int i, int j, int mode=SUBDETIJMODE);
  
  /** Constructor from a generic cell id */
  EcalTrigTowerDetId(const DetId& id);
  /** Assignment from a generic cell id */
  EcalTrigTowerDetId& operator=(const DetId& id);
  

  /// get the z-side of the tower (1/-1)
  int zside() const { return (id_&0x8000)?(1):(-1); }

  /// get the subDetector associated to the Trigger Tower
  EcalSubdetector subDet() const { return (id_&0x4000) ? EcalBarrel:EcalEndcap; }

  /// get the absolute value of the tower ieta 
  int ietaAbs() const 
   { 
     /*       if ( subDet() == EcalBarrel) */
     return (id_>>7)&0x7f; 
     /*       else */
     /* 	throw(std::runtime_error("EcalTrigTowerDetId: ietaAbs not applicable for this subDetector.")); */
   }  

  /// get the tower ieta 
  int ieta() const 
    { 
      /*       if ( subDet() == EcalBarrel) */
      return zside()*ietaAbs(); 
      /*       else */
      /* 	throw(std::runtime_error("EcalTrigTowerDetId: ieta not applicable for this subDetector.")); */
    } 
  
  /// get the tower iphi 
  int iphi() const 
    { 
      /*       if ( subDet() == EcalBarrel) */
      return id_&0x7F; 
      /*       else */
      /* 	throw(std::runtime_error("EcalTrigTowerDetId: iphi not applicable for this subDetector.")); */
      
    } 

  int iquadrant() const ; 


/*   /// get the tower ix (Endcap case) */
/*   int ix() const  */
/*     {  */
/*       if ( subDet() == EcalEndcap) */
/* 	return (id_>>7)&0x7f;  */
/*       else */
/* 	throw(std::runtime_error("EcalTrigTowerDetId: ix not applicable for this subDetector.")); */
/*     }  */
  
/*   /// get the tower iy (Endcap case) */
/*   int iy() const throw(std::runtime_error) */
/*     {  */
/*       if ( subDet() == EcalEndcap) */
/* 	return id_&0x7F;  */
/*       else */
/* 	throw(std::runtime_error("EcalTrigTowerDetId: ix not applicable for this subDetector.")); */
/*     }  */
  

  /// get a compact index for arrays [TODO: NEEDS WORK]
  int hashedIndex() const;

  uint32_t denseIndex() const { return hashedIndex() ; }

  static bool validDenseIndex( uint32_t din ) { return ( din < kSizeForDenseIndexing ) ; }

  static EcalTrigTowerDetId detIdFromDenseIndex( uint32_t di ) ;

  /// check if a valid index combination
  static bool validDetId( int iz, 
			  EcalSubdetector sd, 
			  int i, int j ) ;

  /// get the ECAL DCC id - in the  barrrel ism == iDCC
  int iDCC() const ; 

  /// sequential index within one DCC
  int iTT() const ; 



  static const int MIN_I = 1;
  static const int MIN_J = 1;
  static const int MAX_I = 127;
  static const int MAX_J = 127;

  static const int kEBTowersInPhi = 4; // per SM (in the Barrel)
  static const int kEBTowersPerSM = 68; // per SM (in the Barrel)
  static const int kEBTowersInEta = 17; // per SM (in the Barrel)
//  static const int kEETowersInEta = 11; // Endcap
  static const int kEETowersInPhiPerQuadrant = 18; // per Quadrant (in the Endcap)

  // function modes for (int, int) constructor
  static const int SUBDETIJMODE = 0;
  static const int SUBDETDCCTTMODE = 1;

      enum { kEETowersInPhiPerEndcap = 4* kEETowersInPhiPerQuadrant ,
	     kEEOuterEta             = 18 ,
	     kEEInnerEta             = 28 ,
	     kEETowersInEta          = ( kEEInnerEta - kEEOuterEta + 1 ) ,
	     kEBHalfTowers           = kEBTowersPerSM*18 ,
	     kEBTotalTowers          = kEBHalfTowers*2 ,
	     kEETowersPerEndcap      = kEETowersInEta*kEETowersInPhiPerEndcap - 72,
	     kEETotalTowers          = kEETowersPerEndcap*2,
	     kSizeForDenseIndexing   = kEBTotalTowers + kEETotalTowers } ;
};

std::ostream& operator<<(std::ostream&,const EcalTrigTowerDetId& id);

#endif

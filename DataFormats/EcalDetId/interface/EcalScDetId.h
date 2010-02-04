// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
//$Id: EcalScDetId.h,v 1.4 2009/12/18 13:41:51 fra Exp $

#ifndef EcalDetId_EcalScDetId_h
#define EcalDetId_EcalScDetId_h

#include <ostream>
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EcalScDetId
 *  Supercrystal identifier class for the ECAL endcap.
 *  <P>Note: internal representation of ScDetId:
 *  <CODE>
 *  31              .               15              .              0
 *  |-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-|-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-| 
 *  |  det  |sudet|         0       |1|z|     ix      |     iy      |
 *  +-------+-----+-----------------+-+-+-------------+-------------+
 *  </CODE>
 */

class EcalScDetId : public DetId {
 public:
  /** Constructor of a null id */
  EcalScDetId();
  /** Constructor from a raw value */
  EcalScDetId(uint32_t rawid);
  /** Constructor from crystal ix,iy,iz (iz=+1/-1) 
   or from sc,cr,iz */
  EcalScDetId(int ix, int iy, int iz);  
  /** Constructor from a generic cell id */
  EcalScDetId(const DetId& id);
  /// assignment operator
  EcalScDetId& operator=(const DetId& id);

  /// Gets the subdetector
  EcalSubdetector subdet() const { return EcalSubdetector(subdetId()); }
  /// Gets the z-side of the crystal (1/-1)
  int zside() const { return (id_&0x4000)?(1):(-1); }
  /// Gets the crystal ix
  int ix() const { return (id_>>7)&0x7F; }
  /// Gets the crystal iy
  int iy() const { return id_&0x7F; }  
  /// Gets the SuperCrystal number
  int isc() const ; // within endcap, from 1-316, #70,149,228,307 are EMPTY
  /// Gets the quadrant of the DetId
  int iquadrant() const ;
  /// Gets a compact index for arrays
  int hashedIndex() const { return isc() + ( zside() + 1 )*ISC_MAX/2 - 1 ; } // from 0-631

  uint32_t denseIndex() const { return hashedIndex() ; }

  static bool validDenseIndex( uint32_t din ) { return ( MIN_HASH<=(int)din && MAX_HASH>=(int)din ) ; }
  static bool validHashIndex( int hi ) { return validDenseIndex( hi ) ; }

  static const int IX_MIN=1;
  static const int IY_MIN=1;
  static const int IX_MAX=20;
  static const int IY_MAX=20;
  static const int ISC_MIN=EEDetId::ISC_MIN ;
  static const int ISC_MAX=EEDetId::ISC_MAX ;
  static const int MIN_HASH=0;
  static const int MAX_HASH=2*ISC_MAX - 1;

  /// check if a valid index combination
  static bool validDetId(int i, int j, int iz) ;

  enum { kSizeForDenseIndexing = 2*ISC_MAX } ;

// private:
};


std::ostream& operator<<(std::ostream& s,const EcalScDetId& id);


#endif //EcalDetId_EcalScDetId_h not defined

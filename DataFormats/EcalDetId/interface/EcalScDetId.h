// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
//$Id: EcalScDetId.h,v 1.2 2007/02/12 17:07:29 meridian Exp $

#ifndef EcalDetId_EcalScDetId_h
#define EcalDetId_EcalScDetId_h

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
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
  int isc() const;
  /// Gets the quadrant of the DetId
  int iquadrant() const ;
  /// Gets a compact index for arrays
  int hashedIndex() const;

  static const int IX_MIN=1;
  static const int IY_MIN=1;
  static const int IX_MAX=20;
  static const int IY_MAX=20;
  static const int ISC_MIN=1;
  static const int ISC_MAX=316;

  /// check if a valid index combination
  static bool validDetId(int i, int j, int iz) ;

 private:

  //Functions adapted from similar B. Kennedy's code of EEDetId class to
  //retrieve ix and iy from SC and Crystal number

  static const int nCols = 10;
  static const int nCrys = 5; /* Number of crystals per row in SC */
  static const int QuadColLimits[nCols+1];
  static const int iYoffset[nCols+1];
  int ix(int iSC,int iCrys) const;
  int iy(int iSC,int iCrys) const;
  int ixQuadrantOne() const;
  int iyQuadrantOne() const;

};


std::ostream& operator<<(std::ostream& s,const EcalScDetId& id);


#endif //EcalDetId_EcalScDetId_h not defined

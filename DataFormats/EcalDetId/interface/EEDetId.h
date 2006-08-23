#ifndef ECALDETID_EEDETID_H
#define ECALDETID_EEDETID_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EEDetId
 *  Crystal/cell identifier class for the ECAL endcap
 *
 *
 *  $Id: EEDetId.h,v 1.6 2006/05/25 15:29:36 meridian Exp $
 */


class EEDetId : public DetId {
 public:
  /** Constructor of a null id */
  EEDetId();
  /** Constructor from a raw value */
  EEDetId(uint32_t rawid);
  /** Constructor from crystal ix,iy,iz (iz=+1/-1) 
   or from sc,cr,iz */
  EEDetId(int i, int j, int iz, int mode = XYMODE);  
  /** Constructor from a generic cell id */
  EEDetId(const DetId& id);
  /// assignment operator
  EEDetId& operator=(const DetId& id);

  /// get the subdetector
  EcalSubdetector subdet() const { return EcalSubdetector(subdetId()); }
  /// get the z-side of the crystal (1/-1)
  int zside() const { return (id_&0x4000)?(1):(-1); }
  /// get the crystal ix
  int ix() const { return (id_>>7)&0x7F; }
  /// get the crystal iy
  int iy() const { return id_&0x7F; }  
  /// get the SuperCrystal number
  int isc() const;
  /// get crystal number inside SuperCrystal
  int ic() const;
  /// get the quadrant of the DetId
  int iquadrant() const ;
  /// get a compact index for arrays
  int hashedIndex() const;

  static const int IX_MIN=1;
  static const int IY_MIN=1;
  static const int IX_MAX=100;
  static const int IY_MAX=100;
  static const int ISC_MIN=1;
  static const int ICR_MIN=1;
  static const int ISC_MAX=316;
  static const int ICR_MAX=25;

  // function modes for (int, int) constructor
  static const int XYMODE = 0;
  static const int SCCRYSTALMODE = 1;

  
 private:

  //Functions from B. Kennedy to retrieve ix and iy from SC and Crystal number

  static const int nCols = 10;
  static const int QuadColLimits[nCols+1];
  static const int iYoffset[nCols+1];
  int ix(int iSC,int iCrys) const;
  int iy(int iSC,int iCrys) const;
  /// check if a valid index combination
  bool validDetId(int i, int j, int iz) const;
};


std::ostream& operator<<(std::ostream& s,const EEDetId& id);


#endif

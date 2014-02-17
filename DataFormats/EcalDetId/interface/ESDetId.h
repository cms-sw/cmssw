#ifndef ECALDETID_ESDETID_H
#define ECALDETID_ESDETID_H

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

/** \class ESDetId

   Det id for a preshower (endcap) strip
    
   $Id: ESDetId.h,v 1.8 2012/11/02 13:07:52 innocent Exp $
*/

class ESDetId : public DetId {
 public:

  enum { Subdet = EcalPreshower } ;
  /** Constructor of a null id */
  ESDetId();
  /** Constructor from a raw value */
  ESDetId(uint32_t rawid);  
  /// constructor from strip, ix, iy, plane, and iz
  ESDetId(int strip, int ixs, int iys, int plane, int iz);
  /** constructor from a generic DetId */
  ESDetId(const DetId& id);
  /** assignment from a generic DetId */
  ESDetId& operator=(const DetId& id);

  /// get the subdetector
  EcalSubdetector subdet() const { return EcalSubdetector(subdetId()); }
  /** get the zside */
  int zside() const { return (id_&0x80000)?(1):(-1); }
  /** get the plane */
  int plane() const { return ((id_>>18)&0x1)+1; }
  /** get the sensor ix */
  int six() const { return (id_>>6)&0x3F; }
  /** get the sensor iy */
  int siy() const { return (id_>>12)&0x3F; }
  /** get the strip */
  int strip() const { return (id_&0x3F); }
  /// get a compact index for arrays [TODO: NEEDS WORK]
  int hashedIndex() const ;

  uint32_t denseIndex() const { return hashedIndex() ; }

  static bool validDenseIndex( uint32_t din ) { return validHashIndex( din ) ; }

  static ESDetId detIdFromDenseIndex( uint32_t din ) { return unhashIndex( din ) ; }

  /// get a DetId from a compact index for arrays
  static ESDetId unhashIndex(    int hi ) ;
  static bool    validHashIndex( int hi ) { return ( hi < kSizeForDenseIndexing ) ; }
  /// check if a valid index combination
  static bool validDetId(int istrip, int ixs, int iys, int iplane, int iz) ;

  static const int IX_MIN=1;
  static const int IY_MIN=1;
  static const int IX_MAX=40;
  static const int IY_MAX=40;
  static const int ISTRIP_MIN=1;
  static const int ISTRIP_MAX=32;
  static const int PLANE_MIN=1;
  static const int PLANE_MAX=2;
  static const int IZ_NUM=2;

   private :

      enum { kXYMAX=1072,
	     kXYMIN=   1,
	     kXMAX =  40,
	     kYMAX =  40,
	     kXMIN =   1,
	     kYMIN =   1,
	     // now normalize to A-D notation for ease of use
	     kNa   =IZ_NUM,
	     kNb   =PLANE_MAX - PLANE_MIN + 1,
	     kNc   =kXYMAX - kXYMIN + 1,
	     kNd   =ISTRIP_MAX - ISTRIP_MIN + 1,
	     kLd   =kNd,
	     kLc   =kLd*kNc,
	     kLb   =kLc*kNb,
	     kLa   =kLb*kNa } ;
      
      static const unsigned short hxy1[ kXMAX ][ kYMAX ] ;
      static const unsigned short hx1[ kXYMAX ] ;
      static const unsigned short hy1[ kXYMAX ] ;
      
      static const unsigned short hxy2[ kXMAX ][ kYMAX ] ;
      static const unsigned short hx2[ kXYMAX ] ;
      static const unsigned short hy2[ kXYMAX ] ;

   public:

      enum { kSizeForDenseIndexing = kLa } ;

};

std::ostream& operator<<(std::ostream&,const ESDetId& id);


#endif

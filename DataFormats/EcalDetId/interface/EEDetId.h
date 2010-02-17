#ifndef ECALDETID_EEDETID_H
#define ECALDETID_EEDETID_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EEDetId
 *  Crystal/cell identifier class for the ECAL endcap
 *
 *
 *  $Id: EEDetId.h,v 1.20 2008/09/30 17:39:09 heltsley Exp $
 */


class EEDetId : public DetId {
   public: 
      enum { Subdet=EcalEndcap};
      /** Constructor of a null id */
      EEDetId() {}
      /** Constructor from a raw value */
      EEDetId(uint32_t rawid) : DetId(rawid) {}
      /** Constructor from crystal ix,iy,iz (iz=+1/-1) 
	  or from sc,cr,iz */
      EEDetId(int i, int j, int iz, int mode = XYMODE);  
      /** Constructor from a generic cell id */
      EEDetId(const DetId& id);
      /// assignment operator
      EEDetId& operator=(const DetId& id);

      /// get the subdetector
      //  EcalSubdetector subdet() const { return EcalSubdetector(subdetId()); }
      static EcalSubdetector subdet() { return EcalEndcap;}

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
      // is z positive?
      bool positiveZ() const { return id_&0x4000;}

      int iPhiOuterRing() const ; // 1-360 else==0 if not on outer ring!

      static EEDetId idOuterRing( int iPhi , int zEnd ) ;

      /// get a compact index for arrays
      int hashedIndex() const 
      {
	 const uint32_t jx ( ix() ) ;
	 const uint32_t jd ( 2*( iy() - 1 ) + ( jx - 1 )/50 ) ;
	 return (  ( zside()<0 ? 0 : kEEhalf ) + kdi[jd] + jx - kxf[jd] ) ;
      }

      uint32_t denseIndex() const { return hashedIndex() ; }

      static bool validDenseIndex( uint32_t din ) { return validHashIndex( din ) ; }

      static EEDetId detIdFromDenseIndex( uint32_t din ) { return unhashIndex( din ) ; }

      static bool isNextToBoundary(     EEDetId id ) ;

      static bool isNextToDBoundary(    EEDetId id ) ;

      static bool isNextToRingBoundary( EEDetId id ) ;

      /// get a DetId from a compact index for arrays
      static EEDetId unhashIndex( int hi ) ;

      /// check if a valid hash index
      static bool validHashIndex( int i ) { return ( i < kSizeForDenseIndexing ) ; }

      /// check if a valid index combination
      static bool validDetId( int i, int j, int iz ) ;

      //return the distance in x units between two EEDetId
      static int distanceX(const EEDetId& a,const EEDetId& b); 
      //return the distance in y units between two EEDetId
      static int distanceY(const EEDetId& a,const EEDetId& b); 
      
      static const int IX_MIN =1;
      static const int IY_MIN =1;
      static const int IX_MAX =100;
      static const int IY_MAX =100;
      static const int ISC_MIN=1;
      static const int ICR_MIN=1;
      static const int ISC_MAX=316;
      static const int ICR_MAX=25;


      enum { kEEhalf = 7324 ,
	     kSizeForDenseIndexing = 2*kEEhalf } ;

      // function modes for (int, int) constructor
      static const int XYMODE        = 0;
      static const int SCCRYSTALMODE = 1;

   private:

      bool        isOuterRing() const ;

      static bool isOuterRingXY( int ax, int ay ) ;

      //Functions from B. Kennedy to retrieve ix and iy from SC and Crystal number

      static const int nCols = 10;
      static const int nCrys = 5; /* Number of crystals per row in SC */
      static const int QuadColLimits[nCols+1];
      static const int iYoffset[nCols+1];

      static const unsigned short kxf[2*IY_MAX] ;
      static const unsigned short kdi[2*IY_MAX] ;
  
      int ix( int iSC, int iCrys ) const;
      int iy( int iSC, int iCrys ) const;
      int ixQuadrantOne() const;
      int iyQuadrantOne() const;
};


std::ostream& operator<<(std::ostream& s,const EEDetId& id);

#endif

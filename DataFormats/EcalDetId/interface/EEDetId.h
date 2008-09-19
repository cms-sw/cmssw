#ifndef ECALDETID_EEDETID_H
#define ECALDETID_EEDETID_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EEDetId
 *  Crystal/cell identifier class for the ECAL endcap
 *
 *
 *  $Id: EEDetId.h,v 1.18 2008/07/03 00:11:06 heltsley Exp $
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

      /// get a compact index for arrays. warning this one has gaps in the indexing
      int hashedIndex() const //this one for historical purposes where DB indexing is used
      {
	 return ( iy() - 
		  nBegin[ ix() - 1 ] + 
		  nIntegral[ ix() - 1 ]  + 
		  ( positiveZ() ? ICR_FEE : 0 ) ) ;     
      }

      uint32_t denseIndex() const //this one has no gaps. NOT THE SAME AS HASHEDINDEX!!
      {
	 const uint32_t jx ( ix() ) ;
	 const uint32_t jd ( 2*( iy() - 1 ) + ( jx - 1 )/50 ) ;
	 return (  ( zside()<0 ? 0 : kEEhalf ) + kdi[jd] + jx - kxf[jd] ) ;
      }

      static bool validDenseIndex( uint32_t din ) { return ( din < kSizeForDenseIndexing ) ; }

      static EEDetId detIdFromDenseIndex( uint32_t din ) ; // backwards from denseIndex

      static bool isNextToBoundary(     EEDetId id ) ;

      static bool isNextToDBoundary(    EEDetId id ) ;

      static bool isNextToRingBoundary( EEDetId id ) ;

      /// get a DetId from a compact index for arrays
      static EEDetId unhashIndex( int hi ) ;

      /// check if a valid hash index
      static bool validHashIndex( int i ) ;

      /// check if a valid index combination
      static bool validDetId( int i, int j, int iz ) ;

      static const int IX_MIN =1;
      static const int IY_MIN =1;
      static const int IX_MAX =100;
      static const int IY_MAX =100;
      static const int ISC_MIN=1;
      static const int ICR_MIN=1;
      static const int ISC_MAX=316;
      static const int ICR_MAX=25;

      // to speed up hashedIndex()

      static const int ICR_FD   =3870;
      static const int ICR_FEE  =7740;
      static const int SIZE_HASH=2*ICR_FEE;
      static const int MIN_HASH =  0; // always 0 ...
      static const int MAX_HASH =  2*ICR_FEE-1;

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

      static const int nBegin[IX_MAX];
      static const int nIntegral[IX_MAX];

      static const unsigned short kxf[2*IY_MAX] ;
      static const unsigned short kdi[2*IY_MAX] ;
  
      int ix( int iSC, int iCrys ) const;
      int iy( int iSC, int iCrys ) const;
      int ixQuadrantOne() const;
      int iyQuadrantOne() const;
};


std::ostream& operator<<(std::ostream& s,const EEDetId& id);

#endif

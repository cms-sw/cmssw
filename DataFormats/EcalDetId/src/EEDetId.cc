#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
    
#include <algorithm>
   
const int EEDetId::QuadColLimits[EEDetId::nCols+1] = { 0, 8,17,27,36,45,54,62,70,76,79 };

const int EEDetId::iYoffset[EEDetId::nCols+1]      = { 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

const unsigned short EEDetId::kxf[] = {
  41,  51,  41,  51,  41,  51,  36,  51,  36,  51,
  26,  51,  26,  51,  26,  51,  21,  51,  21,  51,
  21,  51,  21,  51,  21,  51,  16,  51,  16,  51,
  14,  51,  14,  51,  14,  51,  14,  51,  14,  51,
   9,  51,   9,  51,   9,  51,   9,  51,   9,  51,
   6,  51,   6,  51,   6,  51,   6,  51,   6,  51,
   6,  51,   6,  51,   6,  51,   6,  51,   6,  51,
   4,  51,   4,  51,   4,  51,   4,  51,   4,  56,
   1,  58,   1,  59,   1,  60,   1,  61,   1,  61,
   1,  62,   1,  62,   1,  62,   1,  62,   1,  62,
   1,  62,   1,  62,   1,  62,   1,  62,   1,  62,
   1,  61,   1,  61,   1,  60,   1,  59,   1,  58,
   4,  56,   4,  51,   4,  51,   4,  51,   4,  51,
   6,  51,   6,  51,   6,  51,   6,  51,   6,  51,
   6,  51,   6,  51,   6,  51,   6,  51,   6,  51,
   9,  51,   9,  51,   9,  51,   9,  51,   9,  51,
  14,  51,  14,  51,  14,  51,  14,  51,  14,  51,
  16,  51,  16,  51,  21,  51,  21,  51,  21,  51,
  21,  51,  21,  51,  26,  51,  26,  51,  26,  51,
  36,  51,  36,  51,  41,  51,  41,  51,  41,  51
} ;

const unsigned short EEDetId::kdi[] = {
    0,   10,   20,   30,   40,   50,   60,   75,   90,  105,
  120,  145,  170,  195,  220,  245,  270,  300,  330,  360,
  390,  420,  450,  480,  510,  540,  570,  605,  640,  675,
  710,  747,  784,  821,  858,  895,  932,  969, 1006, 1043,
 1080, 1122, 1164, 1206, 1248, 1290, 1332, 1374, 1416, 1458,
 1500, 1545, 1590, 1635, 1680, 1725, 1770, 1815, 1860, 1905,
 1950, 1995, 2040, 2085, 2130, 2175, 2220, 2265, 2310, 2355,
 2400, 2447, 2494, 2541, 2588, 2635, 2682, 2729, 2776, 2818,
 2860, 2903, 2946, 2988, 3030, 3071, 3112, 3152, 3192, 3232,
 3272, 3311, 3350, 3389, 3428, 3467, 3506, 3545, 3584, 3623,
 3662, 3701, 3740, 3779, 3818, 3857, 3896, 3935, 3974, 4013,
 4052, 4092, 4132, 4172, 4212, 4253, 4294, 4336, 4378, 4421,
 4464, 4506, 4548, 4595, 4642, 4689, 4736, 4783, 4830, 4877,
 4924, 4969, 5014, 5059, 5104, 5149, 5194, 5239, 5284, 5329,
 5374, 5419, 5464, 5509, 5554, 5599, 5644, 5689, 5734, 5779,
 5824, 5866, 5908, 5950, 5992, 6034, 6076, 6118, 6160, 6202,
 6244, 6281, 6318, 6355, 6392, 6429, 6466, 6503, 6540, 6577,
 6614, 6649, 6684, 6719, 6754, 6784, 6814, 6844, 6874, 6904,
 6934, 6964, 6994, 7024, 7054, 7079, 7104, 7129, 7154, 7179,
 7204, 7219, 7234, 7249, 7264, 7274, 7284, 7294, 7304, 7314
} ;

EEDetId::EEDetId( int index1, int index2, int iz, int mode ) : DetId( Ecal, EcalEndcap ) 
{
   int crystal_ix=0;
   int crystal_iy=0;
   if (mode == XYMODE) 
   {
      crystal_ix = index1;
      crystal_iy = index2;  
   } 
   else if (mode == SCCRYSTALMODE) 
   {
      int SC = index1;
      int crystal = index2;
      //      std::cout << "iz " << iz << " SC " << index1 << "crystal " << index2  << std::endl;
      
      crystal_ix=iz*ix(SC,crystal);
      if (crystal_ix<0)
	crystal_ix++;
      crystal_ix+=50;
      crystal_iy=iy(SC,crystal);
      if (crystal_iy<0)
	 crystal_iy++;
      crystal_iy+=50;

   } 
   else 
   {
      throw cms::Exception("InvalidDetId") << "EEDetId:  Cannot create object.  Unknown mode for (int, int, int) constructor.";
   }
  
   if (!validDetId(crystal_ix,crystal_iy,iz))
   {
      throw cms::Exception("InvalidDetId") << "EEDetId:  Cannot create object.  Indexes out of bounds \n"
                                           << "x = " << crystal_ix << " y = " << crystal_iy << " z = " << iz;
   }
  
   id_|=(crystal_iy&0x7f)|((crystal_ix&0x7f)<<7)|((iz>0)?(0x4000):(0));
}
  

  

EEDetId 
EEDetId::unhashIndex( int hi )
{
   if( validHashIndex( hi ) )
   {
      const int iz ( hi<kEEhalf ? -1 : 1 ) ;
      const uint32_t di ( hi%kEEhalf ) ;
      const int ii ( ( std::upper_bound( kdi, kdi+(2*IY_MAX), di ) - kdi ) - 1 ) ;
      const int iy ( 1 + ii/2 ) ;
      const int ix ( kxf[ii] + di - kdi[ii] ) ;
      return EEDetId( ix, iy, iz ) ;
   }
   else
   {
      return EEDetId() ;
   }
}

int 
EEDetId::ix( int iSC, int iCrys ) const 
{
  /*
   *  ix() return individual crystal x-coordinate
   *
   *  Author    : B W Kennedy
   *  Version   : 1.00
   *  Created   : 21 December 2005
   *  Last Mod  : 31 January 2006
   *
   *  Input     : iSC, iCrys - Supercrystal and crystal ids
   */
  
  
   int nSCinQuadrant = QuadColLimits[nCols];

   if (iSC > 4*nSCinQuadrant || iSC < 1) 
   {
      throw new std::exception();
   }
  
   //  Map SC number into (x>0,y>0) quadrant.
   int iSCmap, iqx,iq;
   if (iSC > 3*nSCinQuadrant) 
   {
      iSCmap = iSC - 3*nSCinQuadrant;
      iqx =  1;
      iq=4;
   } 
   else if (iSC > 2*nSCinQuadrant) 
   {
      iSCmap = iSC - 2*nSCinQuadrant;
      iqx = -1;
      iq=3;
   } 
   else if (iSC > nSCinQuadrant) 
   {
      iSCmap = iSC - nSCinQuadrant;
      iqx = -1;
      iq=2;
   } 
   else 
   {
      iSCmap = iSC;
      iqx = 1;
      iq=1;
   }

   // Decide which column the SC is in
   int iCol = 0 ;
   while (iSCmap > QuadColLimits[iCol++]) ;
   iCol-- ;

   int ixCrys=-1;
   if (iq == 1 || iq == 3) 
      ixCrys = iqx*(5*(iCol-1) + (int)(iCrys+4)/5);
   else   if (iq == 2 || iq == 4) 
      ixCrys = iqx*(5*(iCol-1) + (iCrys-1)%5 + 1);
   
   // returning a value from 1 to 100  

   return ixCrys;
}

int EEDetId::iy( int iSC, int iCrys ) const 
{
  /*
   *  iy() return individual crystal y-coordinate
   *
   *  Author    : B W Kennedy
   *  Version   : 1.00
   *  Created   : 21 December 2005
   *  Last Mod  : 31 January 2006
   *
   *  Input     : iSC, iCrys - Supercrystal and crystal ids
   */

   int nSCinQuadrant = QuadColLimits[nCols];
   if (iSC > 4*nSCinQuadrant || iSC < 1) 
   {
      throw new std::exception();
   }

   //  Map SC number into (x>0,y>0) quadrant
   int iSCmap, iqy,iq;
   if (iSC > 3*nSCinQuadrant) 
   {
      iSCmap = iSC - 3*nSCinQuadrant;
      iqy = -1;
      iq=4;
   } 
   else if (iSC > 2*nSCinQuadrant) 
   {
      iSCmap = iSC - 2*nSCinQuadrant;
      iqy = -1;
      iq=3;
   } 
   else if (iSC > nSCinQuadrant) 
   {
      iSCmap = iSC - nSCinQuadrant;
      iqy = 1;
      iq=2;
   } else 
   {
      iSCmap = iSC;
      iqy = 1;
      iq=1;
   }

   // Decide which column the SC is in
   int iCol = 0;
   while (iSCmap > QuadColLimits[iCol++]) ;
   iCol--;

   int iSCy = iSCmap - QuadColLimits[iCol-1] + iYoffset[iCol];
  
   int iyCrys=-1;
   if (iq == 1 || iq == 3)
      iyCrys = iqy*(5*(iSCy-1) + (iCrys-1)%5 + 1);
   else if (iq == 2 || iq == 4)
      iyCrys = iqy*(5*(iSCy-1) + (int)(iCrys+4)/5 );
   return iyCrys;
}

int EEDetId::ixQuadrantOne() const
{ 
   int iQuadrant = iquadrant();
   if ( iQuadrant == 1 || iQuadrant == 4)
      return (ix() - 50);
   else if ( iQuadrant == 2 || iQuadrant == 3)
      return (51 - ix());
   //Should never be reached
   return -1;
}

int EEDetId::iyQuadrantOne() const
{ 
   int iQuadrant = iquadrant();
   if ( iQuadrant == 1 || iQuadrant == 2)
      return (iy() - 50);
   else if ( iQuadrant == 3 || iQuadrant == 4)
      return 51 - iy();
   //Should never be reached
   return -1;
}

int 
EEDetId::iquadrant() const 
{
   if (ix()>50)
   {
      if(iy()>50)
	 return 1;
      else
	 return 4;
   }
   else
   {
      if(iy()>50)
	 return 2;
      else
	 return 3;
   }
   //Should never be reached
   return -1;
}  

int 
EEDetId::isc() const
{ 
   return isc( 1 + ( ix() - 1 )/nCrys,
	       1 + ( iy() - 1 )/nCrys ) ; 
}

int 
EEDetId::isc( int jx, int jy ) 
{
   if( 0  < jx &&
       21 > jx &&
       0  < jy &&
       21 > jy    )
   {
      const int iquad (  ( 10<jx && 10<jy ? 1 :
			   ( 11>jx && 10<jy ? 2 :
			     ( 11>jx && 11>jy ? 3 : 4 ) ) ) ) ;
  
      const int iCol = ( 1 == iquad || 4 == iquad ? jx - 10 : 11 - jx ) ;
      const int iRow = ( 1 == iquad || 2 == iquad ? jy - 10 : 11 - jy ) ;

      static int nSCinQuadrant = ISC_MAX/4;

      const int yOff ( iYoffset[iCol] ) ;

      const int qOff ( nSCinQuadrant*( iquad - 1 ) ) ;

      const int iscOne ( QuadColLimits[iCol-1] + iRow - yOff ) ;

      return ( yOff                >= iRow   ? -1 : 
	       ( QuadColLimits[iCol] <  iscOne ? -2 :
		 iscOne + qOff ) ) ;
   }
   else
   {
      return -3 ; // bad inputs
   }
}  

int EEDetId::ic() const 
{
   /*
    *  Return crystal number from (x,y) coordinates.
    *
    *  Author    : B W Kennedy
    *  Version   : 1.00
    *  Created   : 5 May 2006
    *  Last Mod  :
    *
    *  Input     : ix, iy - (x,y) position of crystal
    */
   
   /*  Useful constants . */
   int iQuadrant = iquadrant();
   int icrCol=-1;
   int icrRow=-1;
   
   if (iQuadrant == 1 || iQuadrant == 3)
   {
      icrCol=(ixQuadrantOne()-1) % nCrys;
      icrRow=(iyQuadrantOne()-1) % nCrys;
   }
  
   else if (iQuadrant == 2 || iQuadrant == 4)
   {
      icrRow=(ixQuadrantOne()-1) % nCrys;
      icrCol=(iyQuadrantOne()-1) % nCrys;
   } 

   int icrys = 5*icrCol + icrRow + 1;
  
   return icrys;
}  


bool 
EEDetId::isNextToBoundary( EEDetId id ) 
{
   return isNextToDBoundary( id ) || isNextToRingBoundary( id ) ;
}

bool 
EEDetId::isNextToDBoundary( EEDetId id ) 
{
   // hardcoded values for D boundary
   return id.ix() == 50 || id.ix() == 51 ;
}


bool 
EEDetId::isNextToRingBoundary(EEDetId id) 
{
   for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
	 if ( ! validDetId( id.ix() + i, id.iy() + j, id.zside() ) ) {
	    return true;
	 }
      }
   }
   return false;
}

int
EEDetId::iPhiOuterRing() const
{
   int returnValue ( 0 ) ;
   if( isOuterRing() )
   {
      const int ax ( abs( ix() - IX_MAX/2 ) ) ;
      const int ay ( abs( iy() - IY_MAX/2 ) ) ;
      returnValue = ax + 50 - ay ;
      if( ay <= 47 ) --returnValue ;
      if( ay <= 45 ) --returnValue ;
      if( ay <= 42 ) --returnValue ;
      if( ay <= 37 ) --returnValue ;
      if( ay <= 35 ) --returnValue ;
      if( ay <= 30 ) --returnValue ;
      if( ay <= 25 ) --returnValue ;
      if( ay <= 15 ) --returnValue ;
      if( ay <= 10 ) --returnValue ;
      const int iq ( iquadrant() ) ;
      if( 1==iq )
      {
	 returnValue = 91 - returnValue ;
      }
      else
      {
	 if( 2==iq )
	 {
	    returnValue += 90 ;
	 }
	 else
	 {
	    if( 3==iq )
	    {
	       returnValue = 271 - returnValue ;
	    }
	    else
	    {
	       returnValue += 270 ;
	    }
	 }
      }
      returnValue = 1 + ( 360 + returnValue - 10 -1 )%360 ;
   }
//   if( positiveZ() ) returnValue += 360 ;
   return returnValue ;
}

EEDetId 
EEDetId::idOuterRing( int iPhi , int zEnd )
{
   iPhi -= 10 ; // phi=1 in barrel is at -10deg
   while( iPhi <   1 ) iPhi+=360 ;
   while( iPhi > 360 ) iPhi-=360 ;

   const int index1 ( iPhi - 1 ) ;
   const int quad   ( index1/90 ) ;
   int       indexq (  index1 - quad*90 + 1 ) ;
   if( 0==quad || 2==quad ) indexq = 91 - indexq ;
   const int indexh ( indexq > 45 ? 91 - indexq : indexq ) ;
   const int axh    ( indexh<=10 ? indexh :
		      ( indexh<=12 ? 10 :
			( indexh<=17 ? indexh - 2 :
			  ( indexh<=18 ? 15 :
			    ( indexh<=28 ? indexh - 3 :
			      ( indexh<=30 ? 25 :
				( indexh<=35 ? indexh - 5 :
				  ( indexh<=39 ? 30 :
				    ( indexh<=44 ? indexh - 9 : 35 ))))))))) ;
   const int ayh    ( indexh<=10 ? 50 :
		      ( indexh<=12 ? 60 - indexh :
			( indexh<=17 ? 47 :
			  ( indexh<=18 ? 64 - indexh : 
			    ( indexh<=28 ? 45 :
			      ( indexh<=30 ? 73 - indexh :
				( indexh<=35 ? 42 :
				  ( indexh<=39 ? 77 - indexh :
				    ( indexh<=44 ? 37 : 36 ))))))))) ;
   const int bxh ( indexq>45 ? ayh : axh ) ;
   const int byh ( indexq>45 ? axh : ayh ) ;
   const int cx  ( ( quad==0 || quad==3 ? bxh : -bxh+1 ) + IX_MAX/2 ) ;
   const int cy  ( ( quad==0 || quad==1 ? byh : -byh+1 ) + IY_MAX/2 ) ;

   return EEDetId( cx, cy, ( zEnd > 0 ? 1 : -1 ) ) ;
}


EEDetId 
EEDetId::offsetBy(int nrStepsX, int nrStepsY ) const
{
        int newX = ix() + nrStepsX;
        int newY = iy() + nrStepsY;

        if( validDetId( newX, newY, zside() ) ) {
                return EEDetId( newX, newY, zside() );
        } else {
                return EEDetId(0);
        }
}

EEDetId
EEDetId::switchZSide() const
{
        int newZSide = -1 * zside();
        if( validDetId(ix(), iy(), newZSide ) ) {
                return EEDetId( ix(), iy(), newZSide );
        } else {
                return EEDetId(0);
        }
}

DetId 
EEDetId::offsetBy( const DetId startId, int nrStepsX, int nrStepsY )
{
        if( startId.det() == DetId::Ecal && startId.subdetId() == EcalEndcap ) {
                EEDetId eeStartId( startId );
                return eeStartId.offsetBy( nrStepsX, nrStepsY ).rawId();
        } else {
                return DetId(0);
        }
}

DetId 
EEDetId::switchZSide( const DetId startId )
{
        if( startId.det() == DetId::Ecal && startId.subdetId() == EcalEndcap ) {
                EEDetId eeStartId(startId);
                return eeStartId.switchZSide().rawId();
        } else {
                return DetId(0);
        }
}

bool 
EEDetId::isOuterRing() const
{
   const int kx ( ix() ) ;
   const int ky ( iy() ) ;
   const int ax ( kx>IX_MAX/2 ? kx-IX_MAX/2 : IX_MAX/2 + 1 - kx ) ;
   const int ay ( ky>IY_MAX/2 ? ky-IY_MAX/2 : IY_MAX/2 + 1 - ky ) ;
   return ( isOuterRingXY( ax, ay ) ||
	    isOuterRingXY( ay, ax )    ) ;
}

bool 
EEDetId::isOuterRingXY( int ax, int ay )
{
   return ( ( ax<=10 &&           ay==50 ) ||
	    ( ax==10 &&           ay>=48 ) ||
	    ( ax<=15 && ax>=11 && ay==47 ) ||
	    ( ax==15 &&           ay==46 ) ||
	    ( ax<=25 && ax>=16 && ay==45 ) ||
	    ( ax==25 &&           ay<=44 && ay>=43 ) ||
	    ( ax<=30 && ax>=26 && ay==42 ) ||
	    ( ax==30 &&           ay<=41 && ay>=38 ) ||
	    ( ax<=35 && ax>=31 && ay==37 ) ||
	    ( ax==35 &&           ay==36 )              ) ;
}

bool 
EEDetId::slowValidDetId(int crystal_ix, int crystal_iy) 
{
  return // negative logic!
    !( 
      (crystal_ix >= 1 && crystal_ix <= 3 && (crystal_iy <= 40 || crystal_iy > 60) ) ||
      (crystal_ix >= 4 && crystal_ix <= 5 && (crystal_iy <= 35 || crystal_iy > 65) ) || 
      (crystal_ix >= 6 && crystal_ix <= 8 && (crystal_iy <= 25 || crystal_iy > 75) ) || 
      (crystal_ix >= 9 && crystal_ix <= 13 && (crystal_iy <= 20 || crystal_iy > 80) ) || 
      (crystal_ix >= 14 && crystal_ix <= 15 && (crystal_iy <= 15 || crystal_iy > 85) ) || 
      (crystal_ix >= 16 && crystal_ix <= 20 && (crystal_iy <= 13 || crystal_iy > 87) ) || 
      (crystal_ix >= 21 && crystal_ix <= 25 && (crystal_iy <= 8 || crystal_iy > 92) ) || 
      (crystal_ix >= 26 && crystal_ix <= 35 && (crystal_iy <= 5 || crystal_iy > 95) ) || 
      (crystal_ix >= 36 && crystal_ix <= 39 && (crystal_iy <= 3 || crystal_iy > 97) ) || 
      (crystal_ix >= 98 && crystal_ix <= 100 && (crystal_iy <= 40 || crystal_iy > 60) ) ||
      (crystal_ix >= 96 && crystal_ix <= 97 && (crystal_iy <= 35 || crystal_iy > 65) ) || 
      (crystal_ix >= 93 && crystal_ix <= 95 && (crystal_iy <= 25 || crystal_iy > 75) ) || 
      (crystal_ix >= 88 && crystal_ix <= 92 && (crystal_iy <= 20 || crystal_iy > 80) ) || 
      (crystal_ix >= 86 && crystal_ix <= 87 && (crystal_iy <= 15 || crystal_iy > 85) ) || 
      (crystal_ix >= 81 && crystal_ix <= 85 && (crystal_iy <= 13 || crystal_iy > 87) ) || 
      (crystal_ix >= 76 && crystal_ix <= 80 && (crystal_iy <= 8 || crystal_iy > 92) ) || 
      (crystal_ix >= 66 && crystal_ix <= 75 && (crystal_iy <= 5 || crystal_iy > 95) ) || 
      (crystal_ix >= 62 && crystal_ix <= 65 && (crystal_iy <= 3 || crystal_iy > 97) ) ||
      ( (crystal_ix == 40 || crystal_ix == 61) && ( (crystal_iy >= 46 && crystal_iy <= 55 ) || crystal_iy <= 3 || crystal_iy > 97 )) ||
      ( (crystal_ix == 41 || crystal_ix == 60) && crystal_iy >= 44 && crystal_iy <= 57 ) ||
      ( (crystal_ix == 42 || crystal_ix == 59) && crystal_iy >= 43 && crystal_iy <= 58 ) ||
      ( (crystal_ix == 43 || crystal_ix == 58) && crystal_iy >= 42 && crystal_iy <= 59 ) ||
      ( (crystal_ix == 44 || crystal_ix == 45 || crystal_ix == 57 || crystal_ix == 56) && crystal_iy >= 41 && crystal_iy <= 60 ) ||
      ( crystal_ix >= 46 && crystal_ix <= 55 && crystal_iy >= 40 && crystal_iy <= 61 ) 
       );
}

int EEDetId::distanceX(const EEDetId& a,const EEDetId& b)
{
    return abs(a.ix()-b.ix());
}

int EEDetId::distanceY(const EEDetId& a,const EEDetId& b)
{
  return abs(a.iy() - b.iy()); 
}

#include <ostream>
std::ostream& operator<<(std::ostream& s,const EEDetId& id) 
{
   return s << "(EE iz " << ((id.zside()>0)?("+ "):("- ")) 
	    << " ix " << id.ix() << " , iy " << id.iy() << ')';
}

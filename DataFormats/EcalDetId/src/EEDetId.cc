#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
const int EEDetId::QuadColLimits[EEDetId::nCols+1] = { 0, 8,17,27,36,45,54,62,70,76,79 };
const int EEDetId::iYoffset[EEDetId::nCols+1]      = { 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

const int EEDetId::nBegin[IX_MAX] = { 41, 41, 41, 36, 36, 26, 26, 26, 21, 21, 21, 21, 21, 16, 16, 14, 14, 14, 14, 14, 9, 9, 9, 9, 9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9, 14, 14, 14, 14, 14, 16, 16, 21, 21, 21, 21, 21, 26, 26, 26, 36, 36, 41, 41, 41 };
const int EEDetId::nIntegral[IX_MAX] = { 0, 20, 40, 60, 90, 120, 170, 220, 270, 330, 390, 450, 510, 570, 640, 710, 784, 858, 932, 1006, 1080, 1164, 1248, 1332, 1416, 1500, 1590, 1680, 1770, 1860, 1950, 2040, 2130, 2220, 2310, 2400, 2494, 2588, 2682, 2776, 2870, 2970, 3070, 3170, 3270, 3370, 3470, 3570, 3670, 3770, 3870, 3970, 4070, 4170, 4270, 4370, 4470, 4570, 4670, 4770, 4870, 4964, 5058, 5152, 5246, 5340, 5430, 5520, 5610, 5700, 5790, 5880, 5970, 6060, 6150, 6240, 6324, 6408, 6492, 6576, 6660, 6734, 6808, 6882, 6956, 7030, 7100, 7170, 7230, 7290, 7350, 7410, 7470, 7520, 7570, 7620, 7650, 7680, 7700, 7720 };

EEDetId::EEDetId(int index1, int index2, int iz, int mode) : DetId(Ecal,EcalEndcap) 
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
  
EEDetId::EEDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Ecal || gen.subdetId()!=EcalEndcap)) {
    throw cms::Exception("InvalidDetId"); 
  }
  id_=gen.rawId();
}
  
EEDetId& EEDetId::operator=(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalEndcap )) {
    throw cms::Exception("InvalidDetId"); 
  }
  id_=gen.rawId();
  return *this;
}

EEDetId EEDetId::unhashIndex( int hi )
{
	int zside = hi / ICR_FEE * 2 - 1;
	int ix = binarySearch(  hi % ICR_FEE, 0, IX_MAX ) + 1;
	int iy = hi % ICR_FEE - nIntegral[ ix - 1 ] + nBegin[ ix - 1 ];
	return EEDetId(ix, iy, zside, XYMODE);
}

int EEDetId::binarySearch(int key, int start, int end)
{
	int middle = start + (end-start) / 2;
	if (end<=start) {
		return start;
	}
	if (key < nIntegral[middle]) {
		if (middle == start + 1 ) {
			return start;
		}
		return binarySearch( key, start, middle );
	} else if (key > nIntegral[middle]) {
		if (middle == end - 1 ) {
			return middle;
		}
		return binarySearch( key, middle, end );
	} else {
		return middle;
	}
	return 0;
}


int EEDetId::ix(int iSC, int iCrys) const 
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

  if (iSC > 4*nSCinQuadrant || iSC < 1) {
    throw new std::exception();
  }
  
  //  Map SC number into (x>0,y>0) quadrant.
  int iSCmap, iqx,iq;
  if (iSC > 3*nSCinQuadrant) {
    iSCmap = iSC - 3*nSCinQuadrant;
    iqx =  1;
    iq=4;
  } else if (iSC > 2*nSCinQuadrant) {
    iSCmap = iSC - 2*nSCinQuadrant;
    iqx = -1;
    iq=3;
  } else if (iSC > nSCinQuadrant) {
    iSCmap = iSC - nSCinQuadrant;
    iqx = -1;
    iq=2;
  } else {
    iSCmap = iSC;
    iqx = 1;
    iq=1;
  }

  // Decide which column the SC is in
  int iCol = 0;
  while (iSCmap > QuadColLimits[iCol++]) ;
  iCol--;

  int ixCrys=-1;
  if (iq == 1 || iq == 3) 
    ixCrys = iqx*(5*(iCol-1) + (int)(iCrys+4)/5);
  else   if (iq == 2 || iq == 4) 
    ixCrys = iqx*(5*(iCol-1) + (iCrys-1)%5 + 1);

  // returning a value from 1 to 100  

  return ixCrys;
}

int EEDetId::iy(int iSC, int iCrys) const 
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
  if (iSC > 4*nSCinQuadrant || iSC < 1) {
     throw new std::exception();
  }

  //  Map SC number into (x>0,y>0) quadrant
  int iSCmap, iqy,iq;
  if (iSC > 3*nSCinQuadrant) {
    iSCmap = iSC - 3*nSCinQuadrant;
    iqy = -1;
    iq=4;
  } else if (iSC > 2*nSCinQuadrant) {
    iSCmap = iSC - 2*nSCinQuadrant;
    iqy = -1;
    iq=3;
  } else if (iSC > nSCinQuadrant) {
    iSCmap = iSC - nSCinQuadrant;
    iqy = 1;
    iq=2;
  } else {
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

int EEDetId::iquadrant() const {
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

int EEDetId::isc() const 
{
  /*
   *  Return SC number from (x,y) coordinates.
   *
   *  Author    : B W Kennedy
   *  Version   : 1.00
   *  Created   : 5 May 2006
   *  Last Mod  :
   *
   *  Input     : ix, iy - (x,y) position of crystal
   */
  
  int iCol = int((ixQuadrantOne() - 1)/nCrys) + 1;
  int iRow = int((iyQuadrantOne() - 1)/nCrys) + 1;
  int nSCinQuadrant = QuadColLimits[nCols];
  int iSC;
  
  if (iRow <= iYoffset[iCol]) 
    return -1;
  else 
    iSC = QuadColLimits[iCol-1] + iRow - iYoffset[iCol];

  if (iSC > QuadColLimits[iCol]) 
    return -2;
  
  if (iSC>0) 
      iSC += nSCinQuadrant*(iquadrant()-1);
  
  return iSC;
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

bool EEDetId::validDetId(int crystal_ix, int crystal_iy, int iz) {

  bool valid = false;
  if (crystal_ix < IX_MIN ||  crystal_ix > IX_MAX ||
      crystal_iy < IY_MIN || crystal_iy > IY_MAX || abs(iz) != 1 ) 
    { return valid; }
  if ( (crystal_ix >= 1 && crystal_ix <= 3 && (crystal_iy <= 40 || crystal_iy > 60) ) ||
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
       )
    { return valid; }
  valid = true;
  return valid;
}

std::ostream& operator<<(std::ostream& s,const EEDetId& id) {
  return s << "(EE iz " << ((id.zside()>0)?("+ "):("- ")) << " ix " << id.ix() << " , iy " << id.iy() << ')';
}

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <iostream>
const int EEDetId::QuadColLimits[EEDetId::nCols+1] = { 0, 8,17,27,36,45,54,62,70,76,79 };
const int EEDetId::iYoffset[EEDetId::nCols+1]      = { 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

EEDetId::EEDetId() : DetId() {
}
EEDetId::EEDetId(uint32_t rawid) : DetId(rawid) {
}
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
      throw(std::runtime_error("EEDetId:  Cannot create object.  Unknown mode for (int, int, int) constructor."));
    }
  
  if (crystal_ix < IX_MIN ||  crystal_ix > IX_MAX ||
      crystal_iy < IY_MIN || crystal_iy > IY_MAX || abs(iz) != 1 ) 
    {
      std::cout << "EEDetId:: ERROR in constructor" << std::endl;  
      std::cout << "Construction Mode " << mode << " index1 " << index1 << " index2 " << index2 << " iz " << iz << std::endl;
      std::cout << "crystal_x " << crystal_ix << " crystal_iy " << crystal_iy << std::endl;
      throw(std::runtime_error("EEDetId:  Cannot create object.  Indexes out of bounds."));
    }
  id_|=(crystal_iy&0x7f)|((crystal_ix&0x7f)<<7)|((iz>0)?(0x4000):(0));
}
  
EEDetId::EEDetId(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalEndcap) {
    throw new std::exception();
  }
  id_=gen.rawId();
}
  
EEDetId& EEDetId::operator=(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalEndcap) {
    throw new std::exception();
  }
  id_=gen.rawId();
  return *this;
}
  
int EEDetId::hashedIndex() const {
  return ((zside()>0)?(IX_MAX*IY_MAX):(0))+(iy()-1)*IX_MAX+(ix()-1);
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


int EEDetId::isc() const {
  throw(std::runtime_error("EEDetId: Method not yet implemented"));
}  

int EEDetId::ic() const {
  throw(std::runtime_error("EEDetId: Method not yet implemented"));
}  

std::ostream& operator<<(std::ostream& s,const EEDetId& id) {
  return s << "(EE iz " << ((id.zside()>0)?("+ "):("- ")) << " ix " << id.ix() << " , iy " << id.iy() << ')';
}

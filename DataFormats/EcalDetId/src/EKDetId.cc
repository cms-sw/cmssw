#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
    
#include <iostream>
#include <algorithm>
   
const int EKDetId::QuadColLimits[EKDetId::nCols+1] = { 0,13,27,40,54,70,87,104,120,136,151,166,180,193,205,215,225,232,234 };

const int EKDetId::iYoffset[EKDetId::nCols+1]      = { 0, 5, 4, 4, 3, 1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 };

EKDetId::EKDetId(int module_ix, int module_iy, int fiber, int ro, 
		 int iz) : DetId( Ecal, EcalShashlik) {
  id_ |= (module_iy&0xff) | ((module_ix&0xff)<<8) |
    ((fiber&0x7)<<16) | ((ro&0x3)<<19) | ((iz>0)?(0x200000):(0));
}

EKDetId::EKDetId(int index1, int index2, int fiber, int ro, int iz, 
		 int mode) : DetId(Ecal, EcalShashlik) {

  int module_ix(0), module_iy(0);
  if (mode == XYMODE) {
    module_ix = index1;
    module_iy = index2;  
   } else if (mode == SCMODULEMODE) {
    int supermodule = index1;
    int module      = index2;
    //      std::cout << "iz " << iz << " SM " << index1 << " module " << index2  << std::endl;
    module_ix = ix(supermodule, module);
    module_iy = iy(supermodule, module);
  } else {
    throw cms::Exception("InvalidDetId") << "EKDetId:  Cannot create object.  Unknown mode for (int, int, int, int, int, int) constructor.";
  }
  
  if (!slowValidDetId(module_ix,module_iy,fiber,ro,iz)) {
    throw cms::Exception("InvalidDetId") 
      << "EKDetId:  Cannot create object.  Indexes out of bounds \n"
      << "x = " << module_ix << " y = " << module_iy << " fiber|ro "
      << fiber << "|" << ro << " z = " << iz;
   }
  id_ |= (module_iy&0xff) | ((module_ix&0xff)<<8) |
    ((fiber&0x7)<<16) | ((ro&0x3)<<19) | ((iz>0)?(0x200000):(0));
}

void EKDetId::setFiber(int fib, int ro) {
  uint32_t idc = (id_ & 0xffe0ffff);
  id_ = (idc) | ((fib&0x7)<<16) | ((ro&0x3)<<19);
}

int EKDetId::ism() const { 
  return ism((1 + (ix() - 1)/nMods), (1 + (iy() - 1)/nMods)); 
}

int EKDetId::imod() const {
  return imod(ix(), iy());
}

int EKDetId::iquadrant() const {
  if (ix()>nRows) {
    if (iy()>nRows) return 1;
    else            return 4;
  } else {
    if (iy()>nRows) return 2;
    else            return 3;
  }
}

int EKDetId::hashedIndex() const {
  int iSM  = ism();
  int iMod = imod();
  int iFib = fiber();
  int iRO  = readout();
  return ((positiveZ() ? kEKhalf : 0) +
	  ((((iSM-1)*IMOD_MAX+iMod-1)*FIB_MAX+iFib)*RO_MAX+iRO));
}
  
EKDetId EKDetId::unhashIndex(int hi) {

  if (validHashIndex(hi)) {
    const int iz (hi<kEKhalf ? -1 : 1);
    const uint32_t di (hi%kEKhalf);
    const uint32_t ro (di%RO_MAX);
    const uint32_t fib (((di-ro)/RO_MAX)%FIB_MAX);
    const uint32_t iMD (((((di-ro)/RO_MAX)-fib)/FIB_MAX)%IMOD_MAX+1);
    const uint32_t iSM (((((di-ro)/RO_MAX)-fib)/FIB_MAX-iMD+1)/IMOD_MAX+1);
    return EKDetId(iSM, iMD, fib, ro, iz, SCMODULEMODE);
  } else {
    return EKDetId() ;
  }
}
  
bool EKDetId::validDetId(int iSM, int iMD, int fib, int ro, int iz) {

  return (iSM >= ISM_MIN && iSM <= ISM_MAX && iMD >= IMOD_MIN &&
	  iMD <= IMOD_MAX && fib >= 0 && fib < FIB_MAX &&
	  ro >= 0 && ro < RO_MAX && abs(iz) == 1);
}
  
bool EKDetId::slowValidDetId(int jx, int jy, int fib, int ro, int iz) const {
  int iSM = ism((1 + (jx - 1)/nMods), (1 + (jy - 1)/nMods));
  int iMD = imod(jx,jy);
  return validDetId(iSM, iMD, fib, ro, iz);
}

bool EKDetId::isNextToBoundary(EKDetId id) {
  return isNextToDBoundary(id)  || isNextToRingBoundary(id) ;
}

bool EKDetId::isNextToDBoundary(EKDetId id) {
  // hardcoded values for D boundary
  return id.ix() == nRows || id.ix() == nRows+1 ;
}

bool EKDetId::isNextToRingBoundary(EKDetId id) {
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      const int iSM(ism(id.ix()+i, id.iy()+j));
      const int iMD(imod(id.ix()+i, id.iy()+j));
      if ( !validDetId(iSM, iMD, id.fiber(), id.readout(), id.zside())) {
	return true;
      }
    }
  }
  return false;
}

EKDetId EKDetId::offsetBy(int nrStepsX, int nrStepsY ) const {
  int newX = ix() + nrStepsX;
  int newY = iy() + nrStepsY;

  if (slowValidDetId(newX, newY, fiber(), readout(), zside())) {
    return EKDetId(newX, newY, fiber(), readout(), zside());
  } else {
    return EKDetId(0);
  }
}

EKDetId EKDetId::switchZSide() const {
  int newZSide = -1 * zside();
  if(slowValidDetId(ix(), iy(), fiber(), readout(), newZSide)) {
    return EKDetId( ix(), iy(), fiber(), readout(), newZSide);
  } else {
    return EKDetId(0);
  }
}

DetId EKDetId::offsetBy(const DetId startId, int nrStepsX, int nrStepsY ) {
  if (startId.det() == DetId::Ecal && startId.subdetId() == EcalShashlik) {
    EKDetId eeStartId(startId);
    return eeStartId.offsetBy(nrStepsX, nrStepsY).rawId();
  } else {
    return DetId(0);
  }
}

DetId EKDetId::switchZSide(const DetId startId) {
  if (startId.det() == DetId::Ecal && startId.subdetId() == EcalShashlik) {
    EKDetId eeStartId(startId);
    return eeStartId.switchZSide().rawId();
  } else {
    return DetId(0);
  }
}

int EKDetId::distanceX(const EKDetId& a,const EKDetId& b) {
  return abs(a.ix()-b.ix());
}

int EKDetId::distanceY(const EKDetId& a,const EKDetId& b) {
  return abs(a.iy() - b.iy()); 
}

int EKDetId::ism(int ismCol, int ismRow) {

  if (0  < ismCol && 2*nCols >= ismCol &&  0  < ismRow && 2*nCols >= ismRow) {
    const int iquad ((nCols<ismCol && nCols<ismRow ? 1 :
		      (nCols>=ismCol && nCols<ismRow ? 2 :
		       (nCols>=ismCol && nCols>=ismRow ? 3 : 4))));
    
    const int iCol = (1 == iquad || 4 == iquad ? ismCol - nCols : nCols+1 - ismCol);
    const int iRow = (1 == iquad || 2 == iquad ? ismRow - nCols : nCols+1 - ismRow ) ;

    static int nSMinQuadrant = ISM_MAX/4;
    const int yOff (iYoffset[iCol]);
    const int qOff (nSMinQuadrant*(iquad - 1));

    const int ismOne (QuadColLimits[iCol-1] + iRow - yOff);
    return (yOff                >= iRow   ? -1 : 
	    (QuadColLimits[iCol] <  ismOne ? -2 : ismOne + qOff)) ;
  } else {
    return -3 ; // bad inputs
  }
}  

int EKDetId::imod(int jx, int jy) {
  /*
   *  Return module number from (x,y) coordinates.
   *
   *  Input     : ix, iy - (x,y) position of module
   */
  const int iquad ((jx>nRows) ? ((jy>nRows) ? 1 : 4) : ((jy>nRows) ? 2 : 3));
  const int imodCol(((iquad == 1 || iquad == 4)) ? (jx-nRows-1)%nMods : 
		    (jx-1)%nMods);
  const int imodRow(((iquad == 1 || iquad == 2)) ? (jy-nRows-1)%nMods :
		    (jy-1)%nMods);
  return (5*imodCol + imodRow + 1);
}  

int EKDetId::ix(int iSM, int iMod) const {
  /*
   *  ix() return individual module x-coordinate
   *
   *  Input     : iSM, iMod - SuperModule and module ids
   */
  
   int nSMinQuadrant = QuadColLimits[nCols];
   if (iSM > 4*nSMinQuadrant || iSM < 1) {
     throw cms::Exception("InvalidDetId") << "EKDetId::ix() called with wrong arguments SM/Module " << iSM << ":" << iMod;
   }
  
   //  Map SC number into (x>0,y>0) quadrant.
   int iSMmap, iqx,iq;
   if (iSM > 3*nSMinQuadrant) {
     iSMmap = iSM - 3*nSMinQuadrant;
     iqx    = 1;
     iq     = 4;
   } else if (iSM > 2*nSMinQuadrant) {
     iSMmap = iSM - 2*nSMinQuadrant;
     iqx    =-1;
     iq     = 3;
   } else if (iSM > nSMinQuadrant) {
     iSMmap = iSM - nSMinQuadrant;
     iqx    =-1;
      iq    = 2;
   } else {
     iSMmap = iSM;
     iqx    = 1;
     iq     = 1;
   }

   // Decide which column the SC is in
   int iCol = 0 ;
   while (iSMmap > QuadColLimits[iCol++]) ;
   iCol-- ;

   int ixMod = IX_MAX/2;
   if (iq == 1 || iq == 4) { 
     ixMod += iqx*(5*(iCol-1)) + (int)(iMod+4)/5;
   } else {
     ixMod += iqx*(5*iCol) + (int)(iMod+4)/5;
   } 
   // returning a value from 1 to 180  
   return ixMod;
}

int EKDetId::iy(int iSM, int iMod) const {
  /*
   *  iy() return individual module y-coordinate
   *
   *  Input     : iSM, iMod - Supermodule and module ids
   */

   int nSMinQuadrant = QuadColLimits[nCols];
   if (iSM > 4*nSMinQuadrant || iSM < 1) {
     throw cms::Exception("InvalidDetId") << "EKDetId::iy() called with wrong arguments SM/Module " << iSM << ":" << iMod;
   }

   //  Map SC number into (x>0,y>0) quadrant
   int iSMmap, iqy, iq;
   if (iSM > 3*nSMinQuadrant) {
     iSMmap = iSM - 3*nSMinQuadrant;
     iqy    =-1;
     iq     = 4;
   } else if (iSM > 2*nSMinQuadrant) {
     iSMmap = iSM - 2*nSMinQuadrant;
     iqy    =-1;
     iq     = 3;
   } else if (iSM > nSMinQuadrant) {
     iSMmap = iSM - nSMinQuadrant;
     iqy    = 1;
     iq     = 2;
   } else {
     iSMmap = iSM;
     iqy    = 1;
     iq     = 1;
   }

   // Decide which column the SM is in
   int iCol = 0;
   while (iSMmap > QuadColLimits[iCol++]) ;
   iCol--;

   int iSMy  = iSMmap - QuadColLimits[iCol-1] + iYoffset[iCol];
  
   int iyMod = IY_MAX/2;
   if (iq == 3 || iq == 4) {
     iyMod += iqy*(5*iSMy) + (iMod-1)%5 + 1;
   } else {
     iyMod += iqy*(5*(iSMy-1)) + (iMod-1)%5 + 1;
   }
   return iyMod;
}

#include <ostream>
std::ostream& operator<<(std::ostream& s,const EKDetId& id) {
  return s << "(EK iz " << ((id.zside()>0)?("+ "):("- ")) << " fiber "
	   << id.fiber() << ", RO " << id.readout() << ", ix " << id.ix() 
	   << ", iy " << id.iy() << ')';
}


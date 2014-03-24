#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
    
#include <iostream>
#include <algorithm>

namespace {
  /** Maximum possibility of Fiber number (0:FIB_MAX-1)
   */
  static const int FIB_MAX=6;
  
  /** Maximum possibility of Read-Out type (0:RO_MAX-1)
   */
  static const int RO_MAX=3;

  const int MAX_SM_SIZE = 21;

  const int MAX_MODULES_ROW = 2*MAX_SM_SIZE*5;

  const int MAX_MODULES = MAX_MODULES_ROW * MAX_MODULES_ROW;

  const int MAX_HASH_INDEX = RO_MAX*FIB_MAX*MAX_MODULES*2;

  const int MODULE_OFFSET = MAX_SM_SIZE * 5;

  const int blackBox[MAX_SM_SIZE] = {
    //109876543210987654321
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111,
    0b111111111111111111111
  };


  const int noTaperEcalEta4[MAX_SM_SIZE] = {
    //109876543210987654321
    0b111111111111111111100, // 1
    0b111111111111111111100, // 2
    0b111111111111111111111, // 3
    0b111111111111111111111, // 4
    0b111111111111111111111, // 5
    0b111111111111111111111, // 6
    0b011111111111111111111, // 7
    0b011111111111111111111, // 8
    0b011111111111111111111, // 9
    0b001111111111111111111, // 10
    0b001111111111111111111, // 11
    0b000111111111111111111, // 12
    0b000011111111111111111, // 13
    0b000011111111111111111, // 14
    0b000001111111111111111, // 15
    0b000000111111111111111, // 16
    0b000000011111111111111, // 17
    0b000000000111111111111, // 18
    0b000000000011111111111, // 19
    0b000000000000111111111, // 20
    0b000000000000000111111  // 21
  };
   
  const int taperEcalEta3[MAX_SM_SIZE] = {
    //109876543210987654321
    0b011111111111111100000, // 1
    0b011111111111111100000, // 2
    0b011111111111111100000, // 3
    0b011111111111111110000, // 4
    0b011111111111111111000, // 5
    0b001111111111111111111, // 6
    0b001111111111111111111, // 7
    0b001111111111111111111, // 8
    0b000111111111111111111, // 9
    0b000011111111111111111, // 10
    0b000011111111111111111, // 11
    0b000011111111111111111, // 12
    0b000001111111111111111, // 13
    0b000000111111111111111, // 14
    0b000000011111111111111, // 15
    0b000000001111111111111, // 16
    0b000000000111111111111, // 17
    0b000000000001111111111, // 18
    0b000000000000011111111, // 19
    0b000000000000000011111, // 20
    0b000000000000000000000  // 21
  };
   
  const int taperEcalEta4[MAX_SM_SIZE] = {
    //109876543210987654321
    0b011111111111111111000, // 1
    0b011111111111111111100, // 2
    0b011111111111111111110, // 3
    0b011111111111111111111, // 4
    0b011111111111111111111, // 5
    0b001111111111111111111, // 6
    0b001111111111111111111, // 7
    0b001111111111111111111, // 8
    0b000111111111111111111, // 9
    0b000011111111111111111, // 10
    0b000011111111111111111, // 11
    0b000011111111111111111, // 12
    0b000001111111111111111, // 13
    0b000000111111111111111, // 14
    0b000000011111111111111, // 15
    0b000000001111111111111, // 16
    0b000000000111111111111, // 17
    0b000000000001111111111, // 18
    0b000000000000011111111, // 19
    0b000000000000000011111, // 20
    0b000000000000000000000  // 21
  };
   
  const int noTaperEcalEta3[MAX_SM_SIZE] = {
    //109876543210987654321
    0b111111111111111100000, // 1
    0b111111111111111100000, // 2
    0b111111111111111110000, // 3
    0b111111111111111110000, // 4
    0b111111111111111111100, // 5
    0b111111111111111111111, // 6
    0b011111111111111111111, // 7
    0b011111111111111111111, // 8
    0b011111111111111111111, // 9
    0b001111111111111111111, // 10
    0b001111111111111111111, // 11
    0b000111111111111111111, // 12
    0b000011111111111111111, // 13
    0b000011111111111111111, // 14
    0b000001111111111111111, // 15
    0b000000111111111111111, // 16
    0b000000011111111111111, // 17
    0b000000000111111111111, // 18
    0b000000000011111111111, // 19
    0b000000000000111111111, // 20
    0b000000000000000111111  // 21
  };

  const int* const EK_CONFIG [] = {blackBox, noTaperEcalEta4, noTaperEcalEta3, taperEcalEta4, taperEcalEta3, 0};
}

bool EKDetId::validHashIndex( int i ) { return ( i < MAX_HASH_INDEX) ; }

int EKDetId::smIndex (int ismCol, int ismRow) {
  if (ismCol == 0 || ismCol > MAX_SM_SIZE || ismCol < -MAX_SM_SIZE || ismRow == 0 || ismRow > MAX_SM_SIZE || ismRow < -MAX_SM_SIZE) {
    throw cms::Exception("InvalidDetId") << "EKDetId::smIndex() called with wrong arguments ismCol:ismRow " << ismCol << ':' << ismRow;
  }
  if (ismCol > 0) --ismCol;
  if (ismRow > 0) --ismRow;
  int result = (ismCol+MAX_SM_SIZE)*2*MAX_SM_SIZE + (ismRow+MAX_SM_SIZE);
  return result;
}

int EKDetId::smXLocation (int iSM) {
  if (iSM >= MAX_MODULES || iSM < 0) {
    throw cms::Exception("InvalidDetId") << "EKDetId::smXLocation() called with wrong arguments SM " << iSM;
  }
  return iSM / (2*MAX_SM_SIZE)-MAX_SM_SIZE;
}

int EKDetId::smYLocation (int iSM) {
  if (iSM >= MAX_MODULES || iSM < 0) {
    throw cms::Exception("InvalidDetId") << "EKDetId::smYLocation() called with wrong arguments SM " << iSM;
  }
  return iSM % (2*MAX_SM_SIZE)-MAX_SM_SIZE;
}

int EKDetId::ix(int iSM, int iMod) const {
  /*
   *  ix() return individual module x-coordinate
   *
   *  Input     : iSM, iMod - SuperModule and module ids
   */
  int smCol = smXLocation (iSM);
  int modCol = (iMod-1) / 5;
  return smCol*5+modCol+MODULE_OFFSET;
}

int EKDetId::iy(int iSM, int iMod) const {
  /*
   *  iy() return individual module y-coordinate
   *
   *  Input     : iSM, iMod - SuperModule and module ids
   */
  int smRow = smYLocation (iSM);
  int modRow = (iMod-1) % 5;
  return smRow*5+modRow+MODULE_OFFSET;
}


   
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
   int ismCol = (ix() - MODULE_OFFSET) / 5;
   int ismRow = (iy() - MODULE_OFFSET) / 5;
   return smIndex (ismCol, ismRow);
 }

int EKDetId::ism(int ix, int iy) {
   int ismCol = (ix - MODULE_OFFSET) / 5;
   int ismRow = (iy - MODULE_OFFSET) / 5;
   return smIndex (ismCol, ismRow);
 }

int EKDetId::imod() const {
  return imod(ix(), iy());
}

int EKDetId::iquadrant() const {
  if (ix() >= MODULE_OFFSET) {
    if (iy() >= MODULE_OFFSET) return 1;
    else            return 4;
  } else {
    if (iy()>=MODULE_OFFSET) return 2;
    else            return 3;
  }
}

int EKDetId::hashedIndex() const {
  return (((((positiveZ() ? 1 : 0)*MAX_MODULES_ROW + ix())*MAX_MODULES_ROW + iy())*FIB_MAX + fiber())*RO_MAX + readout());
}
  
EKDetId EKDetId::unhashIndex(int hi) {

  if (!validHashIndex(hi)) return EKDetId();
  int iRo = hi % RO_MAX;
  hi = (hi-iRo)/RO_MAX;
  int iFib = hi % FIB_MAX;
  hi = (hi-iFib) / FIB_MAX;
  int iy = hi % MAX_MODULES_ROW;
  hi = (hi-iy) / MAX_MODULES_ROW;
  int ix = hi % MAX_MODULES_ROW;
  hi = (hi-ix) / MAX_MODULES_ROW;
  int iz = hi ? 1 : -1;
  return EKDetId(ix, iy, iFib, iRo, iz);
}

bool EKDetId::validSM (int ix, int iy, Configuration conf) {
  if (int (conf) >= int (Configuration::LAST)) return false; 
  int smCol = ix / 5 - MAX_SM_SIZE;
  int smRow = iy / 5 - MAX_SM_SIZE;
  if (smCol < 0) smCol = -smCol-1; //reverse negative
  if (smRow < 0) smRow = -smRow-1;
  if (smCol >= MAX_SM_SIZE || smRow >= MAX_SM_SIZE) return false;
  return (EK_CONFIG [conf] [smRow] & (1<<smCol));
}
  
bool EKDetId::validDetId(int iSM, int iMD, int fib, int ro, int iz, Configuration conf) {
  int smCol = iSM % (2*MAX_SM_SIZE) - MAX_SM_SIZE;
  int smRow = iSM / (2*MAX_SM_SIZE) - MAX_SM_SIZE;
  if (smCol < 0) smCol = -smCol-1; //reverse negative
  if (smRow < 0) smRow = -smRow-1;
  if (smCol >= MAX_SM_SIZE || smRow >= MAX_SM_SIZE) return false;
  if (!EK_CONFIG [conf] [smRow] & (1<<smCol)) return false;
  return (iMD > 0) && (iMD <= 25) && (fib >= 0) && (fib < FIB_MAX) && (ro >= 0) && (ro < RO_MAX) && (abs(iz) == 1);
}
  
bool EKDetId::slowValidDetId(int ix, int iy, int fib, int ro, int iz, Configuration conf) {
  if (!validSM (ix, iy, conf)) return false; 
  return (fib >= 0) && (fib < FIB_MAX) && (ro >= 0) && (ro < RO_MAX) && (abs(iz) == 1);
}

bool EKDetId::isNextToBoundary(EKDetId id, Configuration conf) {
  return isNextToDBoundary(id)  || isNextToRingBoundary(id, conf) ;
}

bool EKDetId::isNextToDBoundary(EKDetId id) {
  // hardcoded values for D boundary
  return id.ix() == MODULE_OFFSET || id.ix() == MODULE_OFFSET-1;
}

bool EKDetId::isNextToRingBoundary(EKDetId id, Configuration conf) {
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      if (i == 0 && j == 0) continue;
      if (!slowValidDetId(id.ix()+i, id.iy()+j, id.fiber(), id.readout(), id.zside(), conf)) {
	return true;
      }
    }
  }
  return false;
}

EKDetId EKDetId::offsetBy(int nrStepsX, int nrStepsY, Configuration conf ) const {
  int newX = ix() + nrStepsX;
  int newY = iy() + nrStepsY;

  if (slowValidDetId(newX, newY, fiber(), readout(), zside(), conf)) {
    return EKDetId(newX, newY, fiber(), readout(), zside());
  } else {
    return EKDetId(0);
  }
}

EKDetId EKDetId::switchZSide() const {
  // assume symmetric detector
  return EKDetId( ix(), iy(), fiber(), readout(), -1 * zside());
}

DetId EKDetId::offsetBy(const DetId startId, int nrStepsX, int nrStepsY, Configuration conf) {
  if (startId.det() == DetId::Ecal && startId.subdetId() == EcalShashlik) {
    EKDetId eeStartId(startId);
    return eeStartId.offsetBy(nrStepsX, nrStepsY, conf).rawId();
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

int EKDetId::imod(int jx, int jy) {
  /*
   *  Return module number from (x,y) coordinates.
   *
   *  Input     : ix, iy - (x,y) position of module
   */
  return (5*(jx%5) + (jy%5) + 1);
}  


#include <ostream>
std::ostream& operator<<(std::ostream& s,const EKDetId& id) {
  return s << "(EK iz " << ((id.zside()>0)?("+ "):("- ")) << " fiber "
	   << id.fiber() << ", RO " << id.readout() << ", ix " << id.ix() 
	   << ", iy " << id.iy() << ')';
}


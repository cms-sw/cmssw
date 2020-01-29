/**
 * General HLS utilities, not specific to KF.
 *
 * Author: Ian Tomalin
 */

#ifdef CMSSW_GIT_HASH
#include "L1Trigger/TrackFindingTMTT/interface/HLS/HLSutilities.h"
#else
#include "HLSutilities.h"
#endif

#ifdef CMSSW_GIT_HASH
namespace TMTT {

namespace KalmanHLS {
#endif

#ifdef PRINT_SUMMARY 

namespace CHECK_AP {

// Map containing info about variable ranges in terms of numbers of bits required to represent them.
std::map<std::string, CHECK_AP::INFO> apCheckMap_ = std::map<std::string, CHECK_AP::INFO>();

// Map containing info about integer ranges expressed as numbers,
std::map<std::string, CHECK_AP::INFO_int> intRangeMap_ = std::map<std::string, CHECK_AP::INFO_int>();

// Print contents of map about variable ranges.

void printCheckMap() {

  // Print summary about all variables & bits needed to represent them.

  std::cout<<std::endl<<"------ checkCalc SUMMARY -----"<<std::endl;
  std::map<std::string, INFO>::const_iterator iter;
  for (iter = apCheckMap_.begin(); iter != apCheckMap_.end(); iter++) {
    const std::string& varName = iter->first;
    const INFO&           info = iter->second;
    std::string status = (info.intBitsCfg_ >= info.intBitsSeenHigh_)  ?  "OK"  :  "BAD";
    int iWantBitRange = info.intBitsSeenHigh_-info.intBitsSeenLow_;
    std::cout<<info.className_<<": "<<std::setw(15)<<varName<<" Int Bits Cfg="<<std::setw(3)<<info.intBitsCfg_<<" should exceed Seen="<<std::setw(3)<<info.intBitsSeenHigh_<<" ; "<<status<<" ; Seen Dynamic Range="<<std::setw(3)<<iWantBitRange<<std::endl;
  }

  // Print summary about integers.

  std::cout<<std::endl<<"------ checkCalc INT RANGE SUMMARY -----"<<std::endl;
  std::map<std::string, INFO_int>::const_iterator iterInt;
  for (iterInt = intRangeMap_.begin(); iterInt != intRangeMap_.end(); iterInt++) {
    const std::string& varName = iterInt->first;
    const INFO_int&       info = iterInt->second;
    std::string status = (info.intCfgHigh_ >= info.intSeenHigh_ && info.intCfgLow_ <= info.intSeenLow_)  ?  "OK"  :  "BAD";
    std::cout<<std::setw(14)<<varName<<" Cfg=("<<std::setw(3)<<info.intCfgLow_<<","<<std::setw(3)<<info.intCfgHigh_<<") should be wider than Seen=("<<std::setw(3)<<info.intSeenLow_<<","<<std::setw(3)<<info.intSeenHigh_<<")  ;  "<<status<<std::endl;
  }
}

// Fill info for summary table & check if fixed bit calculation suffered precision loss.

bool checkCalc(std::string varName, float res_fix, double res_float, double reltol, double tol) {
  double res_float_abs = fabs(res_float);
  bool   res_float_sign = (res_float >= 0);
  int  intBitsSeen = (res_float_abs > 0)  ?  std::ceil(log(res_float_abs)/log(2.))  :  -99;

  std::string cName = "float    ";
  int intBitsCfg = 99;

  if (apCheckMap_.find(varName) == apCheckMap_.end()) {
    apCheckMap_[varName] = INFO(cName, intBitsCfg, intBitsSeen, intBitsSeen); 
  } else {
    int intHighOld = apCheckMap_[varName].intBitsSeenHigh_;
    int intLowOld  = apCheckMap_[varName].intBitsSeenLow_;
    if (intHighOld < intBitsSeen) {
	apCheckMap_[varName] = INFO(cName, intBitsCfg, intBitsSeen, intLowOld); 
    } else if (intLowOld > intBitsSeen) {
      apCheckMap_[varName] = INFO(cName, intBitsCfg, intHighOld, intBitsSeen); 
    }
  }
  return true;
}

// Fill info for integer range summary table.

bool checkIntRange(std::string varName, int intCfgHigh, int intCfgLow, int intValue) {

  bool OK = (intValue <= intCfgHigh && intValue >= intCfgLow);

  if (intRangeMap_.find(varName) == intRangeMap_.end()) {
    intRangeMap_[varName] = INFO_int(intCfgHigh, intCfgLow, intValue, intValue);
  } else {
    int intHighOld = intRangeMap_[varName].intSeenHigh_;
    int intLowOld = intRangeMap_[varName].intSeenLow_;
    if (intHighOld < intValue) {
      intRangeMap_[varName] = INFO_int(intCfgHigh, intCfgLow, intValue, intLowOld);
    } else if (intLowOld > intValue) {
      intRangeMap_[varName] = INFO_int(intCfgHigh, intCfgLow, intHighOld, intValue);
    }
  }

#ifdef PRINT_SUMMARY

#ifdef PRINT
#define NPRINTMAXI 99999
#else
#define NPRINTMAXI 100
#endif

  static unsigned int nErrors = 0;

  // Check -ve numbers aren't stored in unsigned variables.
  if (not OK) {
    nErrors++;
    if (nErrors < NPRINTMAXI) std::cout<<"checkCalc INT RANGE ERROR: "<<varName<<" "<<intValue<<" not in range ("<<intCfgLow<<","<<intCfgHigh<<")"<<std::endl;
  }
#endif 

  return OK;
}

// Check covariance matrix determinants are positive.

bool checkDet(std::string matrixName, double m00, double m11, double m01) {
  const double& m10 = m01; 
  bool OK = (m00 * m11 - m01 * m10 > 0);
#ifdef PRINT_SUMMARY
  static unsigned int detErrCount = 0;
  if ((not OK) && detErrCount < 100) {
    detErrCount++;
    std::cout<<"checkCalc NEGATIVE DETERMINANT "<<matrixName<<" "<<m00<<" "<<m11<<" "<<m01<<std::endl;
  }
#endif
  return OK;
}

bool checkDet(std::string matrixName, double m00, double m11, double m22, double m01, double m02, double m12) {
  const double& m10 = m01; 
  const double& m20 = m02; 
  const double& m21 = m12; 
  // det = epislion_ijk * m0i * m1j * m2k;
  bool OK = ((m00*m11*m22 + m01*m12*m20 + m02*m10*m21) - (m02*m11*m20 + m00*m12*m21 + m01*m10*m22) > 0);
#ifdef PRINT_SUMMARY
  static unsigned int detErrCount = 0;
  if ((not OK) && detErrCount < 100) {
    detErrCount++;
    std::cout<<"checkCalc NEGATIVE DETERMINANT "<<matrixName<<" "<<m00<<" "<<m11<<" "<<m22<<" "<<m01<<" "<<m02<<" "<<m12<<std::endl;
  }
#endif
  return OK;
}

}

#endif

#ifdef CMSSW_GIT_HASH
}

}
#endif


/**
 * General HLS utilities, not specific to KF.
 *
 * Author: Ian Tomalin
 */

#ifndef __HLSutilities__
#define __HLSutilities__

#ifndef __SYNTHESIS__
// Switch on short debug summary printout at end of job.
#define PRINT_SUMMARY
// Switch on verbose debug printout (but never during Vivado synthesis).
//#define PRINT
#endif

#ifdef PRINT
#define PRINT_SUMMARY // Print checkCalc summary of bits used per variable.
#endif

#ifdef PRINT_SUMMARY
//#define PRINT_HLSARGS // Print all input & outputs stubs & states from KF HLS code.
#endif

#ifdef PRINT_SUMMARY
#include <iostream>
#include <string>
#include <typeinfo>
#include <map>
#endif

// Copied from /opt/ppd/tools/xilinx/Vivado_HLS/2016.4/include/
#include "ap_int.h"
#include "ap_fixed.h"
//#include "hls_half.h"
// Gives access to fp_struct used to cast half to ap_fixed.
//#include "hls/utils/x_hls_utils.h"

///=== Hard-wired configuration parameters.
///=== WARNING: Since this code must be used in Vivado HLS, it can't use the Settings class.
///=== Therefore all constants are hard-wired here, so may break if you change the configuration parameters.

#ifdef CMSSW_GIT_HASH
namespace TMTT {

namespace KalmanHLS {
#endif

// Used to calculate number of required bits, (which are constants downloaded into FPGA).
#define MAX2(a,b)     ((a) > (b) ? (a) : (b))
#define MAX3(a,b,c)   (MAX2(MAX2(a,b),c))
#define MAX4(a,b,c,d) (MAX2(MAX2(a,b),MAX2(c,d)))

#ifdef PRINT_SUMMARY 

// Detect poor precision in fixed bit calculations.

namespace CHECK_AP {

class INFO {
public:
  INFO() {}
  INFO(std::string className, int intBitsCfg, int intBitsSeenHigh, int intBitsSeenLow) : className_(className), intBitsCfg_(intBitsCfg), intBitsSeenHigh_(intBitsSeenHigh), intBitsSeenLow_(intBitsSeenLow)  {}
public:
  std::string className_;
  int         intBitsCfg_;
  int         intBitsSeenHigh_;
  int         intBitsSeenLow_;
};

class INFO_int {
public:
  INFO_int() {}
  INFO_int(int intCfgHigh, int intCfgLow, int intSeenHigh, int intSeenLow) : intCfgHigh_(intCfgHigh), intCfgLow_(intCfgLow), intSeenHigh_(intSeenHigh), intSeenLow_(intSeenLow) {}
public:
  int         intCfgHigh_;
  int         intCfgLow_;
  int         intSeenHigh_;
  int         intSeenLow_;
};

// Map containing info about variable ranges in terms of numbers of bits required to represent them.
extern std::map<std::string, INFO> apCheckMap_;

// Map containing info about integer ranges expressed as numbers,
extern std::map<std::string, INFO_int> intRangeMap_;

// Print contents of map about variable ranges at end of job.
void printCheckMap();

// Fill info for summary table & check if fixed bit calculation suffered precision loss.
// (Two versions of this function, one for calculations in float which merely notes max & min
// values of each variable for summary printout, & one for finite bit which also checks precision).

bool checkCalc(std::string varName, float res_fix, double res_float, double reltol = 0.1, double tol = 0.0);

template <class C>
bool checkCalc(std::string varName, C res_fix, double res_float, double reltol = 0.1, double tol = 0.0) {
  bool OK = true;

  double res_float_abs = fabs(res_float);
  bool   res_float_sign = (res_float >= 0);
  int  intBitsSeen = (res_float_abs > 0)  ?  std::ceil(log(res_float_abs)/log(2.))  :  -99;

  std::string cNameUgly = typeid(C).name();
  std::string cName = "unknown";
  int intBitsCfg = 0;
  if (cNameUgly.find("ap_int") != std::string::npos) {
    cName = "ap_int   ";
    intBitsCfg = res_fix.width - 1; // One bit less for sign
  } else if (cNameUgly.find("ap_uint") != std::string::npos) {
    cName = "ap_uint  ";
    intBitsCfg = res_fix.width;
  } else if (cNameUgly.find("ap_fixed") != std::string::npos) {
    cName = "ap_fixed ";
    intBitsCfg = res_fix.iwidth - 1;
  } else if (cNameUgly.find("ap_ufixed") != std::string::npos) {
    cName = "ap_ufixed";
    intBitsCfg = res_fix.iwidth;
  }

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

#ifdef PRINT_SUMMARY

#ifdef PRINT
#define NPRINTMAX 99999
#else
#define NPRINTMAX 100
#endif

  static unsigned int nErrors = 0;

  // Check -ve numbers aren't stored in unsigned variables.
  if (res_float < 0) {
    if (cName == "ap_uint" || cName == "ap_ufixed") {
      OK = false;
      if (nErrors < NPRINTMAX) std::cout<<"checkCalc SIGN ERROR ("<<cName<<"): "<<varName<<" "<<res_float<<std::endl;
    }
  }

  if (varName == "RES0" && intBitsSeen == 13) {
    if (nErrors < NPRINTMAX) std::cout<<"checkCalc WIERD RES0 ERROR "<<res_float<<" "<<res_fix<<std::endl;  
  }

  // Check precision is OK.
  if (res_float != 0.) {
    double err = fabs(double(res_fix) - res_float);
    double relerr = err/fabs(res_float);
    if (relerr > reltol && err > tol) {
      OK = false;
      if (nErrors < NPRINTMAX) {
        if (intBitsSeen > intBitsCfg) {
	  std::cout<<"checkCalc TOO FEW INT BITS ("<<cName<<"): "<<varName<<" "<<intBitsSeen<<" > "<<intBitsCfg<<std::endl;
	} else {
  	  std::cout<<"checkCalc PRECISION LOSS ("<<cName<<"): "<<varName<<" relerr="<<relerr<<" err="<<err<<" fix="<<res_fix<<" flt="<<res_float<<std::endl;
	}
      }
    }
  }
  if (not OK) nErrors++;
#endif
  return OK;
} 

// Fill info for integer range summary table.
bool checkIntRange(std::string varName, int intCfgHigh, int intCfgLow, int intValue);

// Check covariance matrix determinants are positive.
bool checkDet(std::string matrixName, double m00, double m11, double m01);
bool checkDet(std::string matrixName, double m00, double m11, double m22, double m01, double m02, double m12);

}
#endif

#ifdef CMSSW_GIT_HASH
}
}
#endif

#endif

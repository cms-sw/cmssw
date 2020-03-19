#ifndef __KFstate__
#define __KFstate__

/**
 * This defines Helix States for the Kalman Filter HLS code.
 * N.B. It therefore can't use the Settings class or any external libraries! Nor can it be a C++ class.
 *
 * Only the KFstateN, KFstate & KFselect classes are used in the implementation of the KF maths block.
 * The other classes are used in the (failed) attempt at a full KF HLS implementation.
 *
 * All variable names & equations come from Fruhwirth KF paper
 * http://dx.doi.org/10.1016/0168-9002%2887%2990887-4
 * 
 * Author: Ian Tomalin
 */

// Copied from /opt/ppd/tools/xilinx/Vivado_HLS/2016.4/include/
#include "ap_int.h"
#include "ap_fixed.h"

#ifdef CMSSW_GIT_HASH
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFpragmaOpts.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/HLSutilities.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstub.h"
#else
#include "KFpragmaOpts.h"
#include "HLSutilities.h"
#include "KFstub.h"
#endif

#ifndef __SYNTHESIS__
#include <iostream>
#endif

///=== Hard-wired configuration parameters.
///=== WARNING: Since this code must be used in Vivado HLS, it can't use the Settings class.
///=== Therefore all constants are hard-wired here, so may break if you change the configuration parameters.

#ifdef CMSSW_GIT_HASH
namespace TMTT {

namespace KalmanHLS {
#endif

// Ultrascale DSP = (18 bits * 27 bits = 48 bits).
// Though if multiplying unsigned variables, must use 1 bit less than this.

enum B_DSP {
  // Number of bits used by DSP for multiplication of signed numbers in FPGA (Ultrascale).
  B18=18, B27=27, B35=2*B18-1, B48=48,
  // Number of bits used by DSP for multiplication of unsigned numbers in FPGA.
  B17=B18-1, B26=B27-1, B34=B35-1,
  // Number of bits used for interface to VHDL (historic, but increasing may increase VHDL BRAM use).
  B25=25, B24=B25-1
};

namespace KFstateN {

// Data formats from https://gitlab.cern.ch/cms-uk-tracktrigger/firmware/l1tf/blob/master/global_formats.docx .
// Since KF maths rely on same multipliers for helix z0 & stub z, phi0 & phi, 1/2R & (r/phi), 
// extra precision desired for the helix params goes after the decimal point.
// e.g. 15 bits for 1/2R, as is factor pow(2,15) more grenular than stub phi/r. 
//      3 bits for phi0, as is factor pow(2,3) more granular than stub phi.
//      7 bits for z0, as is factor pow(2,7) more granular than stub r. 
// TanL & chi2 have multiplier 1, so no. of integer bits must cover their range.

// Cov. mat. bits can change, but remember to change protostate cov. matrix in VHDL.

// Number of integer+sign bits for helix params & chi2.
// FIX: BCHI should be increased to 11, to allow for loose chi2(rphi) cut.
//      (requires updating HLS IP in KF firmware & removing "bodge" from DigitalTrack.h).
enum {BH0 = 3, BH1 = 15, BH2 = 5, BH3 = 11, BH4=25, BCHI = 10};
// Number of bits needed for integer part of helix covariance matrix
//enum {BC00 = -6, BC11 = 16, BC22 = -1, BC33=16, BC44=41+2, BC01=6, BC23=8, BC04=18, BC14=20+8+2};
enum {BC00 = -6, BC11 = 16, BC22 = -1, BC33=16, BC44=41, BC01=5, BC23=7, BC04=17, BC14=29};

// Total number of bits (integer + fractional) needed for helix covariance matrix.
// Increasing to 26/27 doesn't increase DSP use, but does increase BRAM use in KF VHDL.
enum {BLENCOV = 20};

enum {BEV   = KFstubN::BEV, 
      BTRK  = KFstubN::BTRK,  
      NTRK  = KFstubN::NTRK,  
      BLAY  = KFstubN::BLAY,    
      NLAY  = KFstubN::NLAY,
      BID   = KFstubN::BLAY,    
      NID   = KFstubN::NLAY,
      BSEC  = KFstubN::BSEC, 
      BSEC1 = KFstubN::BSEC1, 
      BM = KFstubN::BM, BC = KFstubN::BC};  

typedef ap_fixed<B18,BH0> TR;   
typedef ap_fixed<B18,BH1> TP;   
typedef ap_fixed<B18,BH2> TT;   
typedef ap_fixed<B18,BH3> TZ;   
typedef ap_fixed<B18,BH4> TD;   

typedef ap_ufixed<BLENCOV,BC00>  TC00; 
typedef ap_ufixed<BLENCOV,BC11>  TC11;
typedef ap_ufixed<BLENCOV,BC22>  TC22;
typedef ap_ufixed<BLENCOV,BC33>  TC33;
typedef ap_fixed <BLENCOV,BC01>  TC01;
typedef ap_fixed <BLENCOV,BC23>  TC23;
typedef ap_ufixed<BLENCOV,BC44>  TC44;
typedef ap_fixed <BLENCOV,BC04>  TC04;
typedef ap_fixed <BLENCOV,BC14>  TC14;

// Additional type with extra bit, for internal use.
typedef ap_ufixed<1+BLENCOV,BC00>  TC00EX; 
typedef ap_ufixed<1+BLENCOV,BC11>  TC11EX;
typedef ap_ufixed<1+BLENCOV,BC22>  TC22EX;
typedef ap_ufixed<1+BLENCOV,BC33>  TC33EX;
typedef ap_fixed <1+BLENCOV,BC01>  TC01EX;
typedef ap_fixed <1+BLENCOV,BC23>  TC23EX;
typedef ap_ufixed<1+BLENCOV,BC44>  TC44EX;
typedef ap_fixed <1+BLENCOV,BC04>  TC04EX;
typedef ap_fixed <1+BLENCOV,BC14>  TC14EX;

typedef ap_ufixed<B17,BCHI> TCHI;

typedef ap_uint<BEV>     TEV;
typedef ap_uint<BTRK>    TTRK;
typedef ap_uint<BLAY>    TLAY;
typedef ap_uint<BID>     TID;
typedef ap_uint<NLAY>    TNLAY;
typedef ap_uint<BSEC>    TSEC;
typedef ap_uint<BSEC1>   TSEC1;

typedef ap_int<BM>       TM;
typedef ap_int<BC>       TC;
};

#ifdef ALL_HLS // Used only for full KF implementation in HLS

//--- Extra info about proto-state.

class ProtoInfo {

public:

  ProtoInfo() { 
    for (unsigned int i = 0; i < KFstubN::NLAY; i++) numStubsPerLay[i] = 0;
  }

public:

  KFstateN::TID numStubsPerLay[KFstubN::NLAY];
};

#endif

//--- Format of KF helix state to match VHDL, for both 4 & 5 param helix states.

template <unsigned int NPAR> class KFstate;

template <> class KFstate<4> {

public:

  KFstate<4>() : inv2R(0), phi0(0), tanL(0), z0(0),
    cov_00(0), cov_11(0), cov_22(0), cov_33(0), cov_01(0), cov_23(0),
    chiSquaredRphi(0), chiSquaredRz(0), cBin_ht(0), mBin_ht(0), layerID(0), nSkippedLayers(0), hitPattern(0),
    trackID(0), eventID(0), phiSectID(0), etaSectID(0), etaSectZsign(0),
    valid(0) {}

public:

  // The digitized helix & covariance parameters specified here are scaled relative to the floating 
  // point ones by factors appearing in KF4ParamsCombHLS::getDigiState().

  KFstateN::TR    inv2R; // This is misnamed as rInv in Maxeller. Integer bits = 1+ceil(log2(51));
  KFstateN::TP    phi0;  // Measured with respect to centre of phi sector. Integer bits = 1+ceil(log2(8191));
  KFstateN::TT    tanL;  // This is misnamed as tanTheta in Maxeller. Integer bits = 1+ceil(log2(12));
  KFstateN::TZ    z0;    // Integer bits = 1+ceil(log2(150));

  KFstateN::TC00  cov_00; 
  KFstateN::TC11  cov_11;
  KFstateN::TC22  cov_22;
  KFstateN::TC33  cov_33;
  KFstateN::TC01  cov_01; // (inv2R, phi0) -- other off-diagonal elements assumed negligible.
  KFstateN::TC23  cov_23; // (tanL,  z0)   -- other off-diagonal elements assumed negligible.

  KFstateN::TCHI  chiSquaredRphi; // Chi2 in r-phi plane + small contributions from r-phi & r-z correlations.
  KFstateN::TCHI  chiSquaredRz;   // Chi2 in r-z plane   

  KFstateN::TC    cBin_ht;  // The HT cell (cbin, mbin) are centred on zero here.
  KFstateN::TM    mBin_ht;     

  // This is the KF layer that the KF updator next wants to take a stub from, encoded by L1KalmanComb::doKF(), which in any eta region increases from 0-7 as a particle goes through each layer in turn. It is updated by the StateStubAssociator.
  KFstateN::TLAY  layerID;  
  // This is the number of skipped layers assuming we find a stub in the layer the KF updator is currently searched. The KF updator in HLS/Maxeller does not incremement it.
  ap_uint<2>      nSkippedLayers;
  // Hit pattern 
  KFstateN::TNLAY hitPattern;
  KFstateN::TTRK  trackID;    // Not used by KF updator. Just helps VHDL keep track of which state this is. 
  KFstateN::TEV   eventID;        // Not used by KF updator. Just helps VHDL keep track of which event this is.
  ap_uint<1>      phiSectID;
  KFstateN::TSEC  etaSectID; // Eta sector ID, but counting away from 0 near theta=PI/2 & increasing to 8 near endcap. (Named SectorID in Maxeller).
  ap_uint<1>      etaSectZsign;  // True if eta sector is in +ve z side of tracker; False otherwise. (Named zSign in Maxeller).
  ap_uint<1>      valid; // Used by external code when calculation finished on valid input state & stub.

#ifdef ALL_HLS
  ProtoInfo protoInfo; // Extra info about proto state.
#endif

#ifndef __SYNTHESIS__
public:
  void print(const char* text) const {
    if (valid) {
      std::cout<<text<<std::dec
           <<" trackID="<<trackID
	   <<" phiSectID="<<phiSectID<<" etaSectID="<<etaSectID<<" etaSectZsign="<<etaSectZsign
	   <<" HT (m,c)=("<<mBin_ht<<","<<cBin_ht<<")"
           <<" layers (ID, skip)=("<<layerID<<","<<nSkippedLayers<<")"
	   <<std::hex<<" hitPattern="<<hitPattern<<std::dec
	   <<" 1/2R="<<ap_int<B18>(inv2R.range())
	   <<" phi0="<<ap_int<B18>(phi0.range())
	   <<" tanL="<<ap_int<B18>(tanL.range())
	   <<" z0="  <<ap_int<B18>(z0.range())
	   <<" chi2rphi="<<ap_uint<B17>(chiSquaredRphi.range())
	   <<" chi2rz="<<ap_uint<B17>(chiSquaredRz.range())
	   <<std::endl;
      std::cout<<"      "<<std::dec
           <<" cov00="<<ap_uint<KFstateN::BLENCOV>(cov_00.range())
	   <<" cov11="<<ap_uint<KFstateN::BLENCOV>(cov_11.range())
           <<" cov22="<<ap_uint<KFstateN::BLENCOV>(cov_22.range())
           <<" cov33="<<ap_uint<KFstateN::BLENCOV>(cov_33.range())
           <<" cov01="<<ap_int<KFstateN::BLENCOV>(cov_01.range())
	   <<" cov23="<<ap_int<KFstateN::BLENCOV>(cov_23.range())
 	   <<std::endl;
#ifdef ALL_HLS
      std::cout<<"       Proto #stubs/layer:";
      for (ap_uint<1+KFstateN::BLAY> i = 0; i < KFstateN::NLAY; i++) {
	if (protoInfo.numStubsPerLay[i] > 0) std::cout<<std::dec<<" (L"<<i<<",#S="<<protoInfo.numStubsPerLay[i]<<")";
      }
      std::cout<<std::endl;
#endif
    }
  }
#endif
};


template <> class KFstate<5> : public KFstate<4> {

public:
  KFstateN::TD  d0;

  KFstateN::TC44 cov_44; // (d0,    d0)   
  KFstateN::TC04 cov_04; // (inv2R, d0)   -- other off-diagonal elements assumed negligible.
  KFstateN::TC14 cov_14; // (phi0,  d0)   -- other off-diagonal elements assumed negligible.

#ifndef __SYNTHESIS__
public:
  void print(const char* text) const {
    this->KFstate<4>::print(text);
    if (valid) std::cout<<text
             <<" d0="<<ap_int<B18>(d0.range())
             <<" cov44="<<ap_uint<KFstateN::BLENCOV>(cov_44.range())
             <<" cov04="<<ap_int<KFstateN::BLENCOV>(cov_04.range())
	     <<" cov14="<<ap_int<KFstateN::BLENCOV>(cov_14.range())
             <<std::endl;
  }
#endif
};

//--- Additional output parameters returned by KF updated, for both 4 & 5 param helix fits.
//--- https://svnweb.cern.ch/cern/wsvn/UK-TrackTrig/firmware/trunk/cactusupgrades/projects/tracktrigger/kalmanfit/firmware/hdl/KalmanFilter/KalmanWorker.vhd?peg=4914

template <unsigned int NPAR> class KFselect;

template <> class KFselect<4> {
public:
  // Must use ap_uint<1> instead of bool, due to bug in HLS IP export.
  ap_uint<1>      z0Cut; // Did updated state pass cut on z0 etc.
  ap_uint<1>      ptCut;
  ap_uint<1>      chiSquaredCut;
  ap_uint<1>      sufficientPScut; // Enough PS layers

  //-- The following are now calculated at end of KF VHDL, so no longer needed here.
  //KFstateN::TM    mBin_fit;    // HT bin that fitted helix params lie within.
  //KFstateN::TC    cBin_fit;
  //ap_uint<1>      sectorCut;   // Helix parameters lie within Sector.
  //ap_uint<1>      consistent;  // Duplicate removal -- helix parameters lie within original HT cell.

#ifndef __SYNTHESIS__
public:
  void print(const char* text) const {
    //    std::cout<<"HLS OUTPUT EXTRA:"
    //             <<" Helix (m,c)=("<<mBin_fit<<","<<cBin_fit<<")"
    //             <<std::endl;
  }
#endif
};

template <> class KFselect<5> : public KFselect<4> {
public:
  ap_uint<1>    d0Cut;
};

#ifdef CMSSW_GIT_HASH
}

}
#endif

#endif

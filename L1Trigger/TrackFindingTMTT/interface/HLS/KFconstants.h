#ifndef __KFconstants__
#define __KFconstants__

#ifdef CMSSW_GIT_HASH
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFpragmaOpts.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/HLSutilities.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstub.h"
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFstate.h"
#else
#include "KFpragmaOpts.h"
#include "HLSutilities.h"
#include "KFstub.h"
#include "KFstate.h"
#include "hls_math.h" // Provides hls::exp(), hls::pow(), hls::abs()
#endif

#ifdef CMSSW_GIT_HASH
namespace TMTT {

namespace KalmanHLS {
#endif

//--- Number of helix parameters for track fit (ignored if running inside CMSSW, aside from z0 cut below).

static const unsigned int N_HELIX_PAR = 4;

//=== Configuration of KF data-handling ===

// These params are not used in the KF maths block, but only in the failed attempt at a 
// full KF HLS implementation.

static const unsigned int nKalmanWorkers = 18;
static const unsigned int TMperiod = 143; // clk cycles. Why not 144?

// Set accumulation period. Latency = TMperiod*numAccEvents + extraClkCounts
static const unsigned int numAccEvents= 5; // No. of events processed in parallel within KF
static const unsigned int extraClkCounts = 50; // Time to get states through worker pipeline.

// Max #stubs considered by KF per layer per HT track. (Keep less than KFstubN::NID)
static const unsigned int maxStubsPerLayPerTrack = 4;
// Max #stubs considered by KF per HT track.
static const unsigned int maxStubsPerTrack = 16;

// Max #stubs HT track must have to be defined "hard"
static const unsigned int maxStubsEasy         = 10; // If reduced, some tracks have wrong z0?

// Max allowed skipped layers on easy/hard tracks.
static const unsigned int maxSkipLayersEasy    = 2;
static const unsigned int maxSkipLayersHard    = 1;

// Min/max stubs required to output final fitted track.
static const unsigned int minStubsPerFitTrack  = 4;
static const unsigned int maxStubsPerFitTrack  = 4;  

//=== Configuration of KF maths ===

// Return number of bits needed to contain variable.
//int width(float x){return 1 + hls::ilogb(x);} 

// Digitisation multipliers (from data format doc).
// KF uses same multiplier for r as for stubs in DTC, but one extra bit to accomodate larger range,
// since KF measures r w.r.t. beamline. And it uses r multiplier for z too.

#ifdef HYBRID_FORMAT
// Taken from smallest stub granularity in DTC in any region of tracker.
// https://twiki.cern.ch/twiki/bin/viewauth/CMS/HybridDataFormat
static const float rMult = 1. / 0.02929688;
static const float phiMult = 1. / (7.828293e-6 * 8); // Degrade tracklet granularity by factor 8 to save bits.
#else
static const float rMult = pow(2.,KFstubN::BR-1)/91.652837;
static const float phiMult = pow(2.,KFstubN::BPHI)/0.6981317;
#endif

static const float rphiMult = rMult*phiMult;
static const float inv2R_Mult = (phiMult/rMult);
static const float chi2_Mult = 1.;

// Beam spot length & reference radii w.r.t. beamline.
static const float beamSpotLength= 15.0;
static const float chosenRofPhi_flt = 67.240;
static const KFstubN::TR chosenRofPhi = chosenRofPhi_flt*rMult + 0.5;
static const float chosenRofZ_flt = 50.0;
static const KFstubN::TR chosenRofZ = chosenRofZ_flt*rMult;

// These boundaries are for tilted tracker geometry.
static const KFstubN::TZ zBarrel     = rMult*125.; // Largest z in barrel.
static const KFstubN::TZ zWheel12    = rMult*170.; // Largest z of endcap wheels 1 or 2.
static const KFstubN::TR rPSbarrel   = rMult*60.0;  // r of PS-2S transition in barrel.
static const KFstubN::TR rPSwheel12  = rMult*66.4; // r below which stub certain to be PS if in endcap wheels 1 or 2.
static const KFstubN::TR rPSwheel345 = rMult*64.6; // r below which stub certain to be PS if in endcap wheels 3, 4 or 5.

static const float bField = 3.81120228767395;
static const float cSpeed = 2.99792458e10; // Speed of light (cm/s)
static const float invPtToInv2R = bField*(cSpeed/2.0e13);
#ifdef PT_2GEV
static const float minPt_HT = 2.; // Range of Hough transform
#else
static const float minPt_HT = 3.; // Range of Hough transform
#endif
static const float inv2Rmin_HT = invPtToInv2R*(1./minPt_HT);

static const float kalmanMultScatTerm = 0.00075; // Same as cfg param of same name in CMSSW TMTT code.

// Phi sectors
static const float TWO_PI = 2*3.14159265;
static const int   numPhiSectors = 18;
static const float phiSectorWidth = TWO_PI / numPhiSectors;

// Bit shift *_bitShift to calculate HT cell from ap_fixed (phi, inv2R) of helix params.
// Chosen such that pow(2,+-shift) = (dcBin_digi, dmBin_digi) calculated below.
// (where sign diff is because in KalmanUpdate.cc, one is used to shift right & one to shift left).
// Adjust if you change bits assigned to stubs.
enum {phiToCbin_bitShift = 7, inv2RToMbin_bitShift = 4}; // Shift right & left respectively to get (c,m)
enum {BCH=KFstateN::BH1-phiToCbin_bitShift, BMH=KFstateN::BH0+inv2RToMbin_bitShift};

// Size of HT array
static const int numPhiBins = 64; 
#ifdef PT_2GEV
static const int numPtBins = 48;  
#else
static const int numPtBins = 32;  
#endif
static const ap_int<BCH> minPhiBin = -numPhiBins/2; // BCH & BMH should be larger than BC & BM to monitor overflow.
static const ap_int<BCH> maxPhiBin =  numPhiBins/2 - 1;
static const ap_int<BMH> minPtBin  = -numPtBins/2;
static const ap_int<BMH> maxPtBin  =  numPtBins/2 - 1;

/*
static const float dcBin = numPhiBins / phiSectorWidth; 
static const float dmBin = numPtBins / (inv2Rmin_HT); 
static const float dcBin_digi = dcBin/phiMult; // = pow(2,-7)
static const float dmBin_digi = dmBin/inv2R_Mult; // = pow(2,4)
*/

// Eta sector boundaries in z at reference radius (assumed symmetric).
// (As this is complex, ROM initialization fails unless stored in a class ...)

class EtaBoundaries {
public:
  enum {nSec=8};

  EtaBoundaries() {
    static const float eta[nSec+1] = {0.0, 0.20, 0.41, 0.62, 0.90, 1.26, 1.68, 2.08, 2.4};
    for (unsigned int i = 0; i <= nSec; i++) {
      float zAtRefR = chosenRofZ_flt/tan(2 * atan(exp(-eta[i])));
      z_[i] = rMult*zAtRefR;
    }
    for (unsigned int j = 0; j < nSec; j++) {
      tanL_[j] = 0.5*(1/tan(2*atan(exp(-eta[j]))) + 1/tan(2*atan(exp(-eta[j+1])))); 
    }
  }

public:
  KFstubN::TZ  z_[nSec+1]; // z at ref. radius
  KFstateN::TT tanL_[nSec]; // central tanL in eta sector.
};

// Also failed in VHDL
//static const EtaBoundaries etaBoundaries;

//--- Cuts to select acceptable fitted track states.
//--- (Array vs #stubs on track, where element 0 is never used).
//--- N.B. If cut value is zero, this indicates cut is not applied. (Trick to avoid Vivado timing failure).

// Pt or 1/2R cut.
static const float ptCut_flt_tight = minPt_HT - 0.05; // Smaller than HT cut to allow for resolution during KF fit.
static const float ptCut_flt_loose = minPt_HT - 0.10;
static const float inv2Rcut_flt_tight = invPtToInv2R*(1./ptCut_flt_tight);
static const float inv2Rcut_flt_loose = invPtToInv2R*(1./ptCut_flt_loose);
static const KFstateN::TR inv2Rcut_tight = inv2R_Mult*inv2Rcut_flt_tight;
static const KFstateN::TR inv2Rcut_loose = inv2R_Mult*inv2Rcut_flt_loose;
static const KFstateN::TR inv2Rcut[]      = {0, 0,  inv2Rcut_loose,  inv2Rcut_loose,  inv2Rcut_tight,  inv2Rcut_tight,  inv2Rcut_tight};
static const KFstateN::TR inv2RcutMinus[] = {0, 0, -inv2Rcut_loose, -inv2Rcut_loose, -inv2Rcut_tight, -inv2Rcut_tight, -inv2Rcut_tight};

// z0 cut
static const KFstateN::TZ z0Cut_tight = (N_HELIX_PAR == 4) ? rMult*beamSpotLength : 1.7*rMult*beamSpotLength; // r multiplier used for z in KF. 
static const KFstateN::TZ z0Cut[]      = {0, 0,  z0Cut_tight,  z0Cut_tight,  z0Cut_tight,  z0Cut_tight,  z0Cut_tight}; 
static const KFstateN::TZ z0CutMinus[] = {0, 0, -z0Cut_tight, -z0Cut_tight, -z0Cut_tight, -z0Cut_tight, -z0Cut_tight}; 

// d0 cut
static const float d0Cut_flt_tight = 5.;
static const float d0Cut_flt_loose = 10.;
static const KFstateN::TD d0Cut_tight = rphiMult*d0Cut_flt_tight;
static const KFstateN::TD d0Cut_loose = rphiMult*d0Cut_flt_loose;
static const KFstateN::TD d0Cut[]      = {0, 0, 0,  d0Cut_loose,  d0Cut_tight,  d0Cut_tight,  d0Cut_tight};
static const KFstateN::TD d0CutMinus[] = {0, 0, 0, -d0Cut_loose, -d0Cut_tight, -d0Cut_tight, -d0Cut_tight};

// Chi2 cut
static const KFstateN::TCHI chi2Cut[] = {0, 0, chi2_Mult*10, chi2_Mult*30, chi2_Mult*80, chi2_Mult*120, chi2_Mult*160}; 
// Scale down chi2 in r-phi plane by this factor when applying chi2 cut (to improve electron efficiency).
static const unsigned int chi2rphiScale = 8; // Must be power of 2!

#ifdef CMSSW_GIT_HASH
}
}
#endif

#endif

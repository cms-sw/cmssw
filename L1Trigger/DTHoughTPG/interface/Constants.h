#ifndef DTCHT_Constants_H
#define DTCHT_Constants_H

//#define ALSO_THETA
#define MAX_MACROCELLS 50
#define MAX_CHAMBERHITS 200
#define NUM_MACROCELLWIRES 14 /// was 18 /// FIXME maybe type size can be reduced
#define BINNUM_TIME 16
#define BINNUM_TANPHI 384
#define BINNUM_HALF_TANPHI ( BINNUM_TANPHI / 2 )
#define MAX_TANPHI_CLUSTERS 3
#define BINNUM_X0 128
#define HISTLOW_X0 2048
#define BINSIZE_X0_1 32
#define BINSIZE_X0_1_DIV 5
#define BINSIZE_HALF_X0_1 ( BINSIZE_X0_1 / 2 )

#include <inttypes.h>
#include <bitset>

typedef std::bitset<BINNUM_TANPHI> uINTBitSet_t;
typedef bool BOOL_t;

typedef uint8_t b2_Idx_t;
typedef uint8_t b3_Idx_t;
typedef uint8_t b4_Idx_t;
typedef uint8_t b5_Idx_t;
typedef uint8_t b6_Idx_t;
typedef int8_t sb5_Idx_t;
typedef uint8_t b7_Idx_t;
const b7_Idx_t MAX_b7 = 0xFF;
typedef uint8_t b8_Idx_t;
typedef uint16_t b9_Idx_t;

typedef uint32_t NHitPerChmb_t;
typedef int8_t MCellIdx_t;
typedef int8_t WireIdx_t;
typedef int32_t MCellPos_t;
const MCellPos_t DEF_MCellPos = 0x7FFFFFFF;
typedef int32_t CompPos_t;
typedef int8_t WireShift_t;

typedef uint32_t InputBMask_t;
const InputBMask_t DEF_InputBMask = 0xFFFFFFFF;
typedef int32_t TimeTDC_t;
const TimeTDC_t DEF_TimeTDC = 0x7FFFFFFF;
typedef int32_t CompTDC_t;
typedef int16_t TimeMMT_t;
const TimeMMT_t DEF_TimeMMT = 0x7FFF;
typedef int32_t CompMMT_t;
typedef int16_t BX_t;
typedef uint8_t MMTHist_t;
const MMTHist_t MAX_MMTHist = 0xFF;
typedef int8_t MMTBin_t;
const MMTBin_t DEF_MMTBin = 0x7F;
typedef uint8_t CHTHist_t;
const CHTHist_t MAX_CHTHist = 0xFF;
typedef int16_t CHTBin_t;
const CHTBin_t DEF_CHTBin = 0x7FFF;
typedef int32_t CompBin_t;

typedef uINTBitSet_t CHTBitset_t;

typedef uint8_t Qual_t;
typedef uint16_t TanPhiClu_t;
typedef int32_t TanPhi_t;
const TanPhi_t DEF_TanPhi = 0x7FFFFFFF;
typedef uint8_t X0Hist_t;
const X0Hist_t MAX_X0Hist = 0xFF;
typedef int16_t X0Bin_t;
const X0Bin_t DEF_X0Bin = 0x7FFF;
typedef uint16_t X0Clu_t;
typedef int16_t CompX0_t;
typedef int8_t ZLocSL_t;
typedef uint16_t WiBits_t;

const WireShift_t ZERO_WireShift = 0;
const MCellIdx_t ZERO_MCellIdx = 0;
const TimeMMT_t ZERO_TimeMMT = 0;
const MMTHist_t ZERO_MMTHist = 0;

const double defDTCellWidth = 4.2; /// cm
const double defUnitX = defDTCellWidth / 1024.;
const double defDTSuperLayerDistance = 23.50; /// cm

enum HoughQuality
{
  qDummy = 0,
  qLongPoor = 1,     /// 3/4 Theta
  qLong = 2,         /// 4/4 Theta
  qSinglePoor = 3,   /// 3/4
  qSingle = 4,       /// 4/4
  qDoublePoor = 5,   /// 3/4 + 2/4
  qDoubleLoose = 6,  /// 3/4 + 3/4
  qDoubleTight6 = 7, /// 4/4 + 2/4
  qDoubleTight7 = 8, /// 4/4 + 3/4
  qDoubleTight8 = 9  /// 4/4 + 8/4
};

const InputBMask_t bmaskSuperLayer = 0xC0000000; // 2 bits
const InputBMask_t bmaskLayer =      0x38000000; // 3 bits
const InputBMask_t bmaskWire =       0x07F00000; // 7 bits
const InputBMask_t bmaskTime =       0x0001FFFF; // 17 bits
const InputBMask_t bit1SuperLayer = 30;
const InputBMask_t bit1Layer = 27;
const InputBMask_t bit1Wire = 20;
const InputBMask_t bit1Time = 0;

const InputBMask_t bmaskSL =  0x0C000000; // 2 bits
const InputBMask_t bmaskCh =  0x03FE0000; // 9 bits
const InputBMask_t bmaskBXi =  0x0001FFE0; // 12 bits
const InputBMask_t bmaskTDC = 0x0000001F; // 5 bits
const InputBMask_t bmaskBXTDC = 0x0001FFFF; // 17 bits
const InputBMask_t bit1SL = 26;
const InputBMask_t bit1Ch = 17;
const InputBMask_t bit1BXi = 5;
const InputBMask_t bit1TDC = 0;

const uint16_t bmaskSize = 0xFC00; // 6 bits
const uint16_t bmaskFirst = 0x03FF; // 10 bits
const uint16_t bit1Size = 10;
const uint16_t bit1First = 0;

TimeMMT_t vDefMaxDriftTime[4] = { 121, 124, 124, 126 }; /// 55.4, 54.4, 54.2, 53.2 um/ns
CompTDC_t vDefVDrift[4] = { 1081, 1061, 1057, 1038 };
TanPhi_t vDefSlopeToTime[4] = { 75, 76, 77, 78 };
TimeMMT_t vMmtNonLinCorr[464] =
{
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15
};

const CompBin_t chtWinCorr[22] =
{
  179, 179, 179, 179, 179, 952, 100, 93, 93, 669, 669, 669, 316, 316, 316, 316, 75, 75, 75, 569, 569, 17
};

const b3_Idx_t idxToLayerMap[ NUM_MACROCELLWIRES ] = 
{
  4, 2, 3, 1, 4, 2, 3, 1, 4, 2, 3, 1, 4, 2
};

const MCellPos_t idxToCoordMap[ NUM_MACROCELLWIRES ] = 
{
  -1536, -1536, -1024, -1024, -512, -512, 0, 0, 512, 512, 1024, 1024, 1536, 1536
};

#endif

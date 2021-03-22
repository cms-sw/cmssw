/**
 * Project:
 * File name:  constants.h
 * Language:   C++
 *
 * *********************************************************************
 * Description:
 *
 *
 * To Do:
 *
 * Author: Jose Manuel Cela <josemanuel.cela@ciemat.es>
 *
 * *********************************************************************
 * Copyright (c) 2015-08-07 Jose Manuel Cela <josemanuel.cela@ciemat.es>
 *
 * For internal use, all rights reserved.
 * *********************************************************************
 */
#ifndef L1Trigger_DTTriggerPhase2_constants_h
#define L1Trigger_DTTriggerPhase2_constants_h
#include <cstdint>

// Compiler option to select program mode: PRUEBA_MEZCLADOR, PRUEBA_ANALIZADOR,
// or NONE

/* Quality of the trayectories:
   NOPATH => Not valid trayectory
   LOWQGHOST => 3h (multiple lateralities)
   LOWQ   => 3h
   HIGHQGHOST => 4h (multiple lateralities)
   HIGHQ  => 4h
   CLOWQ  => 3h + 2h/1h
   LOWLOWQ => 3h + 3h
   CHIGHQ => 4h + 2h/1h
   HIGHLOWQ => 4h + 3h
   HIGHHIGHQ => 4h + 4h
*/
namespace cmsdt {

  // enum MP_QUALITY { NOPATH = 0, LOWQGHOST, LOWQ, HIGHQGHOST, HIGHQ, CLOWQ, LOWLOWQ, CHIGHQ, HIGHLOWQ, HIGHHIGHQ };
  enum MP_QUALITY { NOPATH = 0, LOWQ = 1, CLOWQ = 2, HIGHQ = 3, CHIGHQ = 4, LOWLOWQ = 6, HIGHLOWQ = 7, HIGHHIGHQ = 8 };

  // Tipos de lateralidad de traza de partícula al pasar por una celda
  enum LATERAL_CASES { LEFT = 0, RIGHT, NONE };

  enum RPC_QUALITY { NORPC = 0, RPC_TIME, RPC_ONLY, RPC_HIT, RPC_CONFIRM, RPC_ASSOCIATE };

  struct metaPrimitive {
    metaPrimitive(uint32_t id,
                  double t,
                  double pos,
                  double tan,
                  double ph,
                  double phb,
                  double chi,
                  int q,
                  int w1,
                  int t1,
                  int l1,
                  int w2,
                  int t2,
                  int l2,
                  int w3,
                  int t3,
                  int l3,
                  int w4,
                  int t4,
                  int l4,
                  int w5 = 0,
                  int t5 = -1,
                  int l5 = 0,
                  int w6 = 0,
                  int t6 = -1,
                  int l6 = 0,
                  int w7 = 0,
                  int t7 = -1,
                  int l7 = 0,
                  int w8 = 0,
                  int t8 = -1,
                  int l8 = 0,
                  int idx = 0,
                  int rpc = 0)
        : rawId(id),
          t0(t),
          x(pos),
          tanPhi(tan),
          phi(ph),
          phiB(phb),
          chi2(chi),
          quality(q),
          wi1(w1),
          tdc1(t1),
          lat1(l1),
          wi2(w2),
          tdc2(t2),
          lat2(l2),
          wi3(w3),
          tdc3(t3),
          lat3(l3),
          wi4(w4),
          tdc4(t4),
          lat4(l4),
          wi5(w5),
          tdc5(t5),
          lat5(l5),
          wi6(w6),
          tdc6(t6),
          lat6(l6),
          wi7(w7),
          tdc7(t7),
          lat7(l7),
          wi8(w8),
          tdc8(t8),
          lat8(l8),
          index(idx),
          rpcFlag(rpc) {}

    uint32_t rawId;
    double t0;
    double x;
    double tanPhi;
    double phi;
    double phiB;
    double chi2;
    int quality;
    int wi1;
    int tdc1;
    int lat1;
    int wi2;
    int tdc2;
    int lat2;
    int wi3;
    int tdc3;
    int lat3;
    int wi4;
    int tdc4;
    int lat4;
    int wi5;
    int tdc5;
    int lat5;
    int wi6;
    int tdc6;
    int lat6;
    int wi7;
    int tdc7;
    int lat7;
    int wi8;
    int tdc8;
    int lat8;
    int index;
    int rpcFlag = 0;
  };
  struct PARTIAL_LATQ_TYPE {
    bool latQValid;
    int bxValue;
  };
  struct LATQ_TYPE {
    bool valid;
    int bxValue;
    int invalidateHitIdx;
    MP_QUALITY quality;
  };

  enum algo { Standard = 0, PseudoBayes = 1, HoughTrans = 2 };

  enum scenario { MC = 0, DATA = 1, SLICE_TEST = 2 };

  /* En nanosegundos */
  constexpr int LHC_CLK_FREQ = 25;

  /* Adimensional */
  constexpr int MAX_BX_IDX = 3564;

  // In ns (maximum drift time inside the cell)
  constexpr float MAXDRIFT = 387;
  // In mm (cell dimmensions)
  constexpr int CELL_HEIGHT = 13;
  constexpr float CELL_SEMIHEIGHT = 6.5;
  constexpr int CELL_LENGTH = 42;
  constexpr int CELL_SEMILENGTH = 21;
  // In mm / ns (velocidad de deriva)
  constexpr float DRIFT_SPEED = 0.0542;
  // With 4 bits for the decimal part
  constexpr int DRIFT_SPEED_X4 = 889; // 55.5 * 2 ** 4 

  // distance between SLs, cm
  constexpr float VERT_PHI1_PHI3 = 23.5;

  // inverse of the distance between SLs, FW units
  constexpr int VERT_PHI1_PHI3_INV = 558; 

  // distance between center of the chamber and each SL in mm, 2 bit precision for the decimal part
  constexpr int CH_CENTER_TO_MID_SL_X2 = 470; // 117.5 * 2 ** 2
  
  /*
  This is the maximum value than internal time can take. This is because
  internal time is cyclical due to the limited size of the time counters and
  the limited value of the bunch crossing index.
  It should be, approximately, the LHC's clock frequency multiplied by the
  maximum BX index, plus an arbitrary amount for taking into account the
  muon traveling time and muon's signal drift time.
 */
  constexpr int MAX_VALUE_OF_TIME = (LHC_CLK_FREQ * MAX_BX_IDX + 5000);

  /*
 * Total BTI number and total channel number must be coordinated. One BTI
 * works over 10 channels, but 2 consecutive BTI's overlap many of their
 * channels.
 */
  constexpr int TOTAL_BTI = 100;         // Should be the same value as NUM_CH_PER_LAYER
  constexpr int NUM_CH_PER_LAYER = 100;  // Should be the same value as TOTAL_BTI
  constexpr int NUM_LAYERS = 4;
  constexpr int NUM_LATERALITIES = 16;
  constexpr int NUM_CELL_COMB = 3;
  constexpr int TOTAL_CHANNELS = (NUM_LAYERS * NUM_CH_PER_LAYER);
  constexpr int NUM_SUPERLAYERS = 3;
  constexpr float PHIRES_CONV = 65536. / 0.8;
  constexpr float PHIBRES_CONV = 2048. / 1.4;
  constexpr int CHI2RES_CONV = 1000000;
  constexpr int TDCTIME_REDUCED_SIZE = 10;
  constexpr float ZRES_CONV = 65536. / 1500;
  constexpr float KRES_CONV = 65536. / 2;
 
  /*
 * Size of pre-mixer buffers for DTPrimitives
 *
 * As first approach, this value should be evaluated in order to allow storing
 * enough elements to avoid saturating its size. It will be dependent on the
 * noise level, the number of good data injected in the system, as well as on
 * the processing speed of the final analyzer.
 */
  constexpr int SIZE_SEEKT_BUFFER = 32;

  // Number of cells for a analysis block (BTI)
  constexpr int NUM_CELLS_PER_BLOCK = 10;

  /*
 * Number of entries for the payload inside DTPrimitive.
 * This value is also used in other code places to manage reading and writing
 * from/to files
 */
  constexpr int PAYLOAD_ENTRIES = 9;

  /*
   * Size of muon primitive 
   */
  constexpr int NUM_LAYERS_2SL = 8;
  constexpr double PHI_CONV = 0.5235988;

  constexpr int BX_SHIFT = 20;
  constexpr float Z_SHIFT_MB4 = -1.8;
  constexpr float Z_POS_SL = 11.75;
  constexpr double X_POS_L3 = 0.65;
  constexpr double X_POS_L4 = 1.95;

  /*
   * Analyzer precision constants
   */
  constexpr int DIV_SHR_BITS_T0 = 16;
  constexpr int DIV_SHR_BITS_POS = 21;
  constexpr int DIV_SHR_BITS_SLOPE = 21;
  constexpr int DIV_SHR_BITS_SLOPE_XHH = 18;
  
  constexpr int INCREASED_RES_T0 = 0;
  constexpr int INCREASED_RES_POS = 4;
  constexpr int INCREASED_RES_SLOPE = 12;
  constexpr int INCREASED_RES_SLOPE_XHH = 4;
  
  constexpr int INCREASED_RES_POS_POW = 16;
  constexpr int INCREASED_RES_SLOPE_POW = 4096;

  // Values to compute drift distances from drift times
  constexpr int DTDD_PREADD = 9;
  constexpr int DTDD_MULT = 445;
  constexpr int DTDD_SHIFTR_BITS = 13;
  
  /*
   * Local to global coordinates transformation
   */
  
  constexpr int X_SIZE = 18;
  constexpr int TANPSI_SIZE  = 15;
  constexpr int PHI_SIZE = 17; // (1 / 2 ** 17)
  constexpr int PHIB_SIZE = 11; // (2 ** 2) / (2 ** 13)

  constexpr int PHI_LUT_ADDR_WIDTH = 12;
  constexpr int PHI_B_SHL_BITS = 7;
  constexpr int PHI_MULT_SHR_BITS = 10;
  constexpr int PHI_LUT_A_BITS = 12;
  constexpr int PHI_LUT_B_BITS = 20;

  constexpr int PHIB_LUT_ADDR_WIDTH = 9;
  constexpr int PHIB_B_SHL_BITS = 7;
  constexpr int PHIB_MULT_SHR_BITS = 10;
  constexpr int PHIB_LUT_A_BITS = 10;
  constexpr int PHIB_LUT_B_BITS = 16;

  constexpr int PHI_PHIB_RES_DIFF_BITS = 6;


}  // namespace cmsdt

#endif

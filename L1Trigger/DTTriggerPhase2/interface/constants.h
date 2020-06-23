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
enum MP_QUALITY { NOPATH = 0, LOWQGHOST, LOWQ, HIGHQGHOST, HIGHQ, CLOWQ, LOWLOWQ, CHIGHQ, HIGHLOWQ, HIGHHIGHQ };

// Tipos de lateralidad de traza de partícula al pasar por una celda
enum LATERAL_CASES { LEFT = 0, RIGHT, NONE };

namespace cmsdt {
  struct metaPrimitive {
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

  /* En nanosegundos */
  constexpr int LHC_CLK_FREQ = 25;

  /* Adimensional */
  constexpr int MAX_BX_IDX = 3564;

  // En nanosegundos (tiempo de deriva en la celda)
  constexpr float MAXDRIFT = 386.75;
  // En milímetros (dimensiones de la celda)
  constexpr int CELL_HEIGHT = 13;
  constexpr float CELL_SEMIHEIGHT = 6.5;
  constexpr int CELL_LENGTH = 42;
  constexpr int CELL_SEMILENGTH = 21;
  // En milímetros / nanosegundo (velocidad de deriva)
  constexpr float DRIFT_SPEED = 0.0542;
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
}  // namespace cmsdt

#endif

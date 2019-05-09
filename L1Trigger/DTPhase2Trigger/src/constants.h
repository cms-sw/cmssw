/**
 * Project:
 * Subproject: tracrecons
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
#ifndef CONSTANTS_H
#define CONSTANTS_H

// Compiler option to select program mode: PRUEBA_MEZCLADOR, PRUEBA_ANALIZADOR,
// or NONE
// #define PRUEBA_MEZCLADOR 1
// #define PRUEBA_ANALIZADOR 1

namespace CMS {

/* En nanosegundos */
#define LHC_CLK_FREQ 25

/* Adimensional */
#define MAX_BX_IDX 3564

// En nanosegundos (tiempo de deriva en la celda)
#define MAXDRIFT ((float)(386.74))
// En milímetros (dimensiones de la celda)
#define CELL_HEIGHT     13
#define CELL_SEMIHEIGHT 6.5
#define CELL_LENGTH     42
#define CELL_SEMILENGTH 21
// En milímetros / nanosegundo (velocidad de deriva)
#define DRIFT_SPEED (CELL_SEMILENGTH/MAXDRIFT)
/*
  This is the maximum value than internal time can take. This is because
  internal time is cyclical due to the limited size of the time counters and
  the limited value of the bunch crossing index.
  It should be, approximately, the LHC's clock frequency multiplied by the
  maximum BX index, plus an arbitrary amount for taking into account the
  muon traveling time and muon's signal drift time.
 */
#define MAX_VALUE_OF_TIME (LHC_CLK_FREQ*MAX_BX_IDX+5000)

/*
 * Total BTI number and total channel number must be coordinated. One BTI
 * works over 10 channels, but 2 consecutive BTI's overlap many of their
 * channels.
 */
#define TOTAL_BTI           100  // Should be the same value as NUM_CH_PER_LAYER
#define NUM_CH_PER_LAYER    100  // Should be the same value as TOTAL_BTI
#define NUM_LAYERS          4
#define TOTAL_CHANNELS      (NUM_LAYERS*NUM_CH_PER_LAYER)
#define NUM_SUPERLAYERS     3

/*
 * Size of pre-mixer buffers for DTPrimitives
 *
 * As first approach, this value should be evaluated in order to allow storing
 * enough elements to avoid saturating its size. It will be dependent on the
 * noise level, the number of good data injected in the system, as well as on
 * the processing speed of the final analyzer.
 */
#define SIZE_SEEKT_BUFFER 32

// Number of cells for a analysis block (BTI)
#define NUM_CELLS_PER_BLOCK 10

/*
 * Number of entries for the payload inside DTPrimitive.
 * This value is also used in other code places to manage reading and writing
 * from/to files
 */
#define PAYLOAD_ENTRIES 9
// #define PAYLOAD_ENTRIES 1

}

#endif

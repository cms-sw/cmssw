#ifndef ANALTYPDEFS_H
#define ANALTYPDEFS_H
#include "constants.h"
#include <stdint.h>

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
typedef enum {NOPATH = 0, LOWQGHOST, LOWQ, HIGHQGHOST, HIGHQ, CLOWQ, LOWLOWQ, CHIGHQ, HIGHLOWQ, HIGHHIGHQ} MP_QUALITY;

// Tipos de lateralidad de traza de partícula al pasar por una celda
typedef enum {LEFT=0, RIGHT, NONE} LATERAL_CASES;
struct metaPrimitive
{
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
typedef struct {
    bool latQValid;
    int  bxValue;
} PARTIAL_LATQ_TYPE;
typedef struct {
    bool valid;
    int bxValue;
    int invalidateHitIdx;
    MP_QUALITY quality;
} LATQ_TYPE;
#endif

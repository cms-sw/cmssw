#ifndef TrackProbabilityCategoryData_H
#define TrackProbabilityCategoryData_H

#include "CondFormats/Serialization/interface/Serializable.h"

struct TrackProbabilityCategoryData {
  float pMin, pMax, etaMin, etaMax;
  int nHitsMin, nHitsMax, nPixelHitsMin, nPixelHitsMax;
  float chiMin, chiMax;
  float withFirstPixel;
  signed short trackQuality;

  COND_SERIALIZABLE;
};

#endif

#ifndef TrackProbabilityCategoryData_H
#define TrackProbabilityCategoryData_H


struct TrackProbabilityCategoryData {
  float pMin, pMax, etaMin, etaMax;
  int   nHitsMin, nHitsMax, nPixelHitsMin, nPixelHitsMax;
  float chiMin,chiMax;
  float withFirstPixel;
  signed short trackQuality;
};


#endif









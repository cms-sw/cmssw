#ifndef __GCTERRORANALYZERDEFINITIONS_HH_
#define __GCTERRORANALYZERDEFINITIONS_HH_

const unsigned int GCT_OBJECT_QUANTA = 4;
const unsigned int GCT_SUMS_QUANTA = 1;
const unsigned int RCT_REGION_QUANTA = 396;
const unsigned int RCT_EM_OBJECT_QUANTA = 144;
const unsigned int NUM_GCT_RINGS = 4;
const unsigned int NUM_INT_JETS = 108;

struct GctErrorAnalyzerMBxInfo {
  int RCTTrigBx;
  int EmuTrigBx;
  int GCTTrigBx;
};

struct jetData {
 unsigned int et;
 unsigned int eta;
 unsigned int phi;
};

#endif

#ifndef EcalHFNoise_h
#define EcalHFNoise_h


#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

struct EcalHFChannelNoise {

  struct Zero { float z1; float z2;};

  static Zero zero;


        float rms_x12;
        float rms_x6;
        float rms_x1;

        public:

        float rms(int i) const {
                if (i==0) return 0.;
                return *(&rms_x12+(i-1));
        }
};

typedef EcalCondObjectContainer<EcalHFChannelNoise> EcalHFNoiseMap;
typedef EcalHFNoiseMap::const_iterator EcalHFNoiseMapIterator;
typedef EcalHFNoiseMap EcalHFNoise;

#endif

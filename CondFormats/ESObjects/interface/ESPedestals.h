#ifndef ESPedestals_h
#define ESPedestals_h


#include "CondFormats/ESObjects/interface/ESCondObjectContainer.h"

struct ESPedestal {
        struct Zero { float z1; float z2;};

        static Zero zero;

        float mean;
        float rms;

        public:

        float getMean() const {
                return mean;
        }

        float getRms() const {
                return rms;
        }
};

typedef ESCondObjectContainer<ESPedestal> ESPedestalsMap;
typedef ESPedestalsMap::const_iterator ESPedestalsMapIterator;
typedef ESPedestalsMap ESPedestals;

#endif

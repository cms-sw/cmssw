// federico.ferri@cern.ch
// largely copied from a class written by T. Tabarelli de Fatis

#ifndef DRing_h
#define DRing_h

#include "DataFormats/DetId/interface/DetId.h"

class DRings {
        public:
                DRings() : init_(false) {};
                ~DRings() {};

                void setEERings(const char * filename);
                int ieta(DetId id);
                int ring(DetId id);

                static const int nHalfIEta = 124;

        private:
                int eeRings_[101][101][2];
                bool init_;
};

#endif

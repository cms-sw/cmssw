#ifndef EVF_UTILITIES_AUXILIARYMAKERS_H
#define EVF_UTILITIES_AUXILIARYMAKERS_H

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/TCDS/interface/TCDSRaw.h"

namespace evf{
  namespace evtn{
    edm::EventAuxiliary makeEventAuxiliary(const tcds::Raw_v1*,
					   unsigned int runNumber,
					   unsigned int lumiSection,
					   std::string const &processGUID,
                                           bool verifyLumiSection);
  }
}
#endif

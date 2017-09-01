#ifndef EVF_UTILITIES_AUXILIARYMAKERS_H
#define EVF_UTILITIES_AUXILIARYMAKERS_H

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/TCDS/interface/TCDSRecord.h"

namespace evf{
  namespace evtn{
    edm::EventAuxiliary makeEventAuxiliary(TCDSRecord *record,
					   unsigned int runNumber,
					   unsigned int lumiSection,
					   std::string const &processGUID,
                                           bool verifyLumiSection);
  }
}
#endif

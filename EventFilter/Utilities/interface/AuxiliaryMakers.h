#ifndef EVF_UTILITIES_AUXILIARYMAKERS_H
#define EVF_UTILITIES_AUXILIARYMAKERS_H

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "EventFilter/FEDInterface/interface/FED1024.h"

namespace evf{
  namespace evtn{
    edm::EventAuxiliary makeEventAuxiliary(TCDSRecord *record, 
					   unsigned int runNumber,
					   unsigned int lumiSection,
					   std::string const &processGUID);
  }
}
#endif

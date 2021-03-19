#ifndef EvFMicroStateService_H
#define EvFMicroStateService_H 1

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
//#include "DataFormats/Provenance/interface/EventID.h"
//#include "DataFormats/Provenance/interface/Timestamp.h"
//#include "DataFormats/Provenance/interface/ModuleDescription.h"
//#include "DataFormats/Provenance/interface/ParameterSetID.h"

namespace evf {

  class MicroStateService {
  public:
    // the names of the states - some of them are never reached in an online app
    MicroStateService(const edm::ParameterSet &, edm::ActivityRegistry &);
    virtual ~MicroStateService();

  protected:
  };

}  // namespace evf

#endif

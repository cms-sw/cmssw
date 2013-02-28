#ifndef RecoAlgos_MassiveCandidateConverter_h
#define RecoAlgos_MassiveCandidateConverter_h
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include <string>

namespace edm { class EventSetup; }

namespace converter {
  struct MassiveCandidateConverter {
    MassiveCandidateConverter( const edm::ParameterSet & );
    void beginFirstRun( const edm::EventSetup & );

  protected:
    double massSqr_;
    PdtEntry particle_;
  };
}

#endif

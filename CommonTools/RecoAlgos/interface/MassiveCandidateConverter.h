#ifndef RecoAlgos_MassiveCandidateConverter_h
#define RecoAlgos_MassiveCandidateConverter_h
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include <string>

namespace edm {
  class EventSetup;
  class ConsumesCollector;
}  // namespace edm

namespace converter {
  struct MassiveCandidateConverter {
    MassiveCandidateConverter(const edm::ParameterSet&, edm::ConsumesCollector);
    void beginFirstRun(const edm::EventSetup&);

  protected:
    double massSqr_;
    PdtEntry particle_;

  private:
    const edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> tableToken_;
  };
}  // namespace converter

#endif

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace CalibFormats_CaloObjects {
  struct dictionary {
    CaloSamples cs;
    CaloSamplesCollection vcs;
    edm::Wrapper<CaloSamplesCollection> wcvs;
  };
}

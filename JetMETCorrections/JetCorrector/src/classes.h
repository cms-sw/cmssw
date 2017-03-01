#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace JetMETCorrections_JetCorrector {
  struct dictionary {
    edm::Wrapper<reco::JetCorrector> wslsn;
  };
}

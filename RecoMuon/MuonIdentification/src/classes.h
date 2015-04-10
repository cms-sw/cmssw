#include "DataFormats/Common/interface/Wrapper.h"

#include "RecoMuon/MuonIdentification/interface/VersionedMuonSelectors.h"

namespace RecoMuon_MuonIdentification {
  struct dictionary {
    //for using the selectors in python
    VersionedRecoMuonSelector vRecoMuonSelector;
    VersionedPatMuonSelector  vPatMuonSelector ;
  };
}

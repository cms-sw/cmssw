#ifndef RecoCandidate_IsoDepositFwd_h
#define RecoCandidate_IsoDepositFwd_h
#include <vector>
#include "DataFormats/Common/interface/ValueMap.h"

#define HAVE_COMMON_ISODEPOSITMAP 1

namespace reco {
  class IsoDeposit;

  //! keep it only as a part of ValueMap
  typedef edm::ValueMap<reco::IsoDeposit> IsoDepositMap; 


}

#endif

#ifndef RecoCandidate_IsoDepositFwd_h
#define RecoCandidate_IsoDepositFwd_h
#include <vector>
#include "DataFormats/Common/interface/ValueMap.h"

#define HAVE_COMMON_ISODEPOSITMAP 1

namespace reco {
  namespace io_v1 {
    class IsoDeposit;
  }
  using IsoDeposit = io_v1::IsoDeposit;

  //! keep it only as a part of ValueMap
  typedef edm::ValueMap<reco::IsoDeposit> IsoDepositMap;

}  // namespace reco

#endif

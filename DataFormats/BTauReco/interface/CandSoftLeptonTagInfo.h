#ifndef DataFormats_BTauReco_CandSoftLeptonTagInfo_h
#define DataFormats_BTauReco_CandSoftLeptonTagInfo_h

#include <vector>
#include <limits>

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/BTauReco/interface/TemplatedSoftLeptonTagInfo.h"

namespace reco {

  typedef TemplatedSoftLeptonTagInfo<CandidatePtr> CandSoftLeptonTagInfo;

  DECLARE_EDM_REFS(CandSoftLeptonTagInfo)

}  // namespace reco

#endif  // DataFormats_BTauReco_CandSoftLeptonTagInfo_h

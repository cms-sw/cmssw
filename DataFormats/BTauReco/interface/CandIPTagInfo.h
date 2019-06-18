#ifndef BTauReco_CandIpTagInfo_h
#define BTauReco_CandIpTagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/IPTagInfo.h"

namespace reco {
  typedef IPTagInfo<std::vector<CandidatePtr>, JetTagInfo> CandIPTagInfo;

  DECLARE_EDM_REFS(CandIPTagInfo)

}  // namespace reco

#endif

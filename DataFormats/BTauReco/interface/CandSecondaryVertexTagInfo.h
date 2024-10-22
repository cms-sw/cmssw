#ifndef DataFormats_BTauReco_CandSecondaryVertexTagInfo_h
#define DataFormats_BTauReco_CandSecondaryVertexTagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/BTauReco/interface/TemplatedSecondaryVertexTagInfo.h"

namespace reco {

  typedef reco::TemplatedSecondaryVertexTagInfo<reco::CandIPTagInfo, reco::VertexCompositePtrCandidate>
      CandSecondaryVertexTagInfo;

  DECLARE_EDM_REFS(CandSecondaryVertexTagInfo)

}  // namespace reco
#endif  // DataFormats_BTauReco_CandSecondaryVertexTagInfo_h

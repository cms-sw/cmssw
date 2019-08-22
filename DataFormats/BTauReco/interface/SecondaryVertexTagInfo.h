#ifndef DataFormats_BTauReco_SecondaryVertexTagInfo_h
#define DataFormats_BTauReco_SecondaryVertexTagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/BTauReco/interface/TemplatedSecondaryVertexTagInfo.h"

namespace reco {

  typedef reco::TemplatedSecondaryVertexTagInfo<reco::TrackIPTagInfo, reco::Vertex> SecondaryVertexTagInfo;

  DECLARE_EDM_REFS(SecondaryVertexTagInfo)

}  // namespace reco
#endif  // DataFormats_BTauReco_SecondaryVertexTagInfo_h

#include "FWPFTrack3DProxyBuilder.h"

//______________________________________________________________________________
void FWPFTrack3DProxyBuilder::build(const reco::Track &iData,
                                    unsigned int iIndex,
                                    TEveElement &oItemHolder,
                                    const FWViewContext *vc) {
  FWPFTrackUtils *utils = new FWPFTrackUtils();
  TEveTrack *trk = utils->setupTrack(iData);
  TEvePointSet *ps = utils->getCollisionMarkers(trk);
  setupAddElement(trk, &oItemHolder);
  if (ps->GetN() != 0)
    setupAddElement(ps, &oItemHolder);
  else
    delete ps;

  delete utils;
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER(FWPFTrack3DProxyBuilder, reco::Track, "PF Tracks", FWViewType::k3DBit);

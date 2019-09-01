#include "FWPFTrackRPZProxyBuilder.h"

//______________________________________________________________________________
void FWPFTrackRPZProxyBuilder::build(const reco::Track &iData,
                                     unsigned int iIndex,
                                     TEveElement &oItemHolder,
                                     const FWViewContext *vc) {
  FWPFTrackUtils *utils = new FWPFTrackUtils();
  TEveTrack *trk = utils->setupTrack(iData);
  TEvePointSet *ps = utils->getCollisionMarkers(trk);
  TEvePointSet *rzps = new TEvePointSet();
  setupAddElement(trk, &oItemHolder);

  Float_t *trackPoints = trk->GetP();
  unsigned int last = (trk->GetN() - 1) * 3;
  float y = trackPoints[last + 1];
  float z = trackPoints[last + 2];

  // Reposition any points that have been translated in RhoZ
  for (signed int i = 0; i < ps->GetN(); ++i) {
    // WORKS BUT DOESN'T HANDLE ALL SCENARIOS.....
    Float_t a, b, c;
    ps->GetPoint(i, a, b, c);

    if (y < 0 && b > 0)
      b *= -1;
    else if (y > 0 && b < 0)
      b *= -1;

    if (z < 0 && c > 0)
      c *= -1;
    else if (z > 0 && c < 0)
      c *= -1;

    rzps->SetNextPoint(a, b, c);
  }

  if (rzps->GetN() != 0)
    setupAddElement(rzps, &oItemHolder);
  else
    delete rzps;

  delete ps;
  delete utils;
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER(FWPFTrackRPZProxyBuilder,
                        reco::Track,
                        "PF Tracks",
                        FWViewType::kRhoPhiPFBit | FWViewType::kRhoZBit);

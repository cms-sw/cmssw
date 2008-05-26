#include "Fireworks/Muons/interface/MuonsProxy3DBuilder.h"
#include "Fireworks/Muons/interface/MuonsProxyRhoPhiZ2DBuilder.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveManager.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TEveStraightLineSet.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "RVersion.h"
#include "TEveGeoNode.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "TColor.h"
#include "TEvePolygonSetProjected.h"

MuonsProxy3DBuilder::MuonsProxy3DBuilder()
{
}

MuonsProxy3DBuilder::~MuonsProxy3DBuilder()
{
}

void MuonsProxy3DBuilder::build(const FWEventItem* iItem, 
				TEveElementList** product)
{
   MuonsProxyRhoPhiZ2DBuilder::build(iItem, product, false);
}


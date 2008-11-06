#include "Fireworks/Muons/interface/MuonsProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Muons/interface/PatMuonsProxyRhoPhiZ2DBuilder.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveManager.h"
#include "TEveCompound.h"
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TEveStraightLineSet.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "RVersion.h"
#include "TEveGeoNode.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "TColor.h"
#include "TEvePolygonSetProjected.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/TracksProxy3DBuilder.h"

PatMuonsProxyRhoPhiZ2DBuilder::PatMuonsProxyRhoPhiZ2DBuilder()
{
}

PatMuonsProxyRhoPhiZ2DBuilder::~PatMuonsProxyRhoPhiZ2DBuilder()
{
}

void PatMuonsProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem, TEveElementList** product)
{
   build(iItem, product, false);
}

void PatMuonsProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem, TEveElementList** product)
{
   build(iItem, product, true);
}

void PatMuonsProxyRhoPhiZ2DBuilder::build(const FWEventItem* iItem,
					  TEveElementList** product,
					  bool showEndcap,
					  bool tracksOnly)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"trackerMuons",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const std::vector<pat::Muon>* muons=0;
   iItem->get(muons);

   if(0 == muons ) return;

   // if auto field estimation mode, do extra loop over muons.
   if ( CmsShowMain::isAutoField() )
     for ( std::vector<pat::Muon>::const_iterator muon = muons->begin();
	   muon != muons->end(); ++muon) {
	if ( fabs( muon->eta() ) > 2.0 || muon->pt() < 3 ||
	     !muon->standAloneMuon().isAvailable()) continue;
	double estimate = fw::estimate_field(*(muon->standAloneMuon()));
	if ( estimate < 0 ) continue;
	CmsShowMain::guessFieldIsOn( estimate > 0.5 );
     }

   fw::NamedCounter counter("muon");
   for ( std::vector<pat::Muon>::const_iterator muon = muons->begin(); muon != muons->end(); ++muon, ++counter )
     {
	MuonsProxyRhoPhiZ2DBuilder::buildMuon(iItem, &*muon, tList, counter, showEndcap, tracksOnly );
     }
}

REGISTER_FWRPZ2DDATAPROXYBUILDER(PatMuonsProxyRhoPhiZ2DBuilder,std::vector<pat::Muon>,"PatMuons");

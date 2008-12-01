#include "Fireworks/Muons/interface/Muons3DProxyBuilder.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveManager.h"
#include "TEveCompound.h"
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

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
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/TracksProxy3DBuilder.h"

Muons3DProxyBuilder::Muons3DProxyBuilder()
{
}

Muons3DProxyBuilder::~Muons3DProxyBuilder()
{
}

void Muons3DProxyBuilder::build(const FWEventItem* iItem,
				       TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Muons",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }

   const reco::MuonCollection* muons=0;
   iItem->get(muons);

   if(0 == muons ) return;

   // if auto field estimation mode, do extra loop over muons.
   if ( CmsShowMain::isAutoField() )
     for ( reco::MuonCollection::const_iterator muon = muons->begin();
	   muon != muons->end(); ++muon) {
	if ( fabs( muon->eta() ) > 2.0 || muon->pt() < 3 ||
	     !muon->standAloneMuon().isAvailable()) continue;
	double estimate = fw::estimate_field(*(muon->standAloneMuon()));
	if ( estimate < 0 ) continue;
	CmsShowMain::guessFieldIsOn( estimate > 0.5 );
     }

   fw::NamedCounter counter("muon");
   for ( reco::MuonCollection::const_iterator muon = muons->begin(); muon != muons->end(); ++muon, ++counter )
     {
	m_builder.buildMuon(iItem, &*muon, tList, counter, true, false );
     }
}
REGISTER_FW3DDATAPROXYBUILDER(Muons3DProxyBuilder,reco::MuonCollection,"Muons");


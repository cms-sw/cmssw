#include "RVersion.h"
#include "TColor.h"
#include "TEvePolygonSetProjected.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveManager.h"
#include "TEveStraightLineSet.h"
#include "TEveGeoNode.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Electrons/interface/ElectronsProxy3DBuilder.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

const reco::GsfElectronCollection *ElectronsProxy3DBuilder::electrons = 0;

ElectronsProxy3DBuilder::ElectronsProxy3DBuilder()
{
}

ElectronsProxy3DBuilder::~ElectronsProxy3DBuilder()
{
}

void ElectronsProxy3DBuilder::build (const FWEventItem* iItem, 
				     TEveElementList** product)
{
     TEveElementList* tList = *product;
     // printf("Calling electron proxy\n");
     
     if(0 == tList) {
	  tList =  new TEveElementList(iItem->name().c_str(), "GSF Electrons", true);
	  *product = tList;
	  tList->SetMainColor(iItem->defaultDisplayProperties().color());
	  gEve->AddElement(tList);
     } else {
	  tList->DestroyElements();
     }
     
     using reco::GsfElectronCollection;
     const GsfElectronCollection *electrons = 0;
     // printf("getting electrons\n");
     iItem->get(electrons);
     // printf("got electrons\n");
     ElectronsProxy3DBuilder::electrons = electrons;
   
     if (electrons == 0) {
	  std::cout <<"failed to get GSF electrons" << std::endl;
	  return;
     }
     // printf("%d GSF electrons\n", electrons->size());
   
     TEveTrackPropagator *propagator = new TEveTrackPropagator();
     propagator->SetMagField( -4.0);
     propagator->SetMaxR( 122 );
     propagator->SetMaxZ( 300 );

     int index=0;
     TEveRecTrack t;
     t.fBeta = 1.;
     for(GsfElectronCollection::const_iterator i = electrons->begin();
	 i != electrons->end(); ++i, ++index) {
	  std::stringstream s;
	  s << "electron" << index;
	  TEveElementList *elList = new TEveElementList(s.str().c_str());
	  gEve->AddElement( elList, tList );
	  assert(i->gsfTrack().isNonnull());
	  t.fP = TEveVector(i->gsfTrack()->px(),
			    i->gsfTrack()->py(),
			    i->gsfTrack()->pz());
	  t.fV = TEveVector(i->gsfTrack()->vx(),
			    i->gsfTrack()->vy(),
			    i->gsfTrack()->vz());
	  t.fSign = i->gsfTrack()->charge();
	  TEveTrack* trk = new TEveTrack(&t, propagator);
	  trk->SetMainColor(iItem->defaultDisplayProperties().color());
 	  trk->MakeTrack();
	  elList->AddElement(trk);
// 	  gEve->AddElement(trk,tList);
	  //cout << it->px()<<" "
	  //   <<it->py()<<" "
	  //   <<it->pz()<<endl;
	  //cout <<" *";
	  assert(i->superCluster().isNonnull());
#if 0
	  std::vector<DetId> detids = i->superCluster()->getHitsByDetId();
	  for (std::vector<DetId>::const_iterator k = detids.begin();
	       k != detids.end(); ++k) {
// 	       const TGeoHMatrix* matrix = m_item->getGeom()->getMatrix( k->rawId() );
	       TEveGeoShapeExtract* extract = m_item->getGeom()->getExtract( k->rawId() );
	       if(0!=extract) {
		    TEveElement* shape = TEveGeoShape::ImportShapeExtract(extract,0);
		    shape->SetMainTransparency(50);
		    shape->SetMainColor(tList->GetMainColor());
		    tList->AddElement(shape);
	       }
	  }
#endif
     }
}

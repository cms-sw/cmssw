// -*- C++ -*-
//
// Package:     Muons
// Class  :     MuonsProxyEveLegoBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: MuonsProxyEveLegoBuilder.cc,v 1.8 2008/11/10 18:07:58 amraktad Exp $
//

// system include files
#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"
#include "TEveStraightLineSet.h"
#include "TEveCompound.h"
#include "TEvePointSet.h"
#include "TEveManager.h"

// user include files
#include "Fireworks/Muons/interface/MuonsProxyEveLegoBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonsProxyEveLegoBuilder::MuonsProxyEveLegoBuilder()
{
}

// MuonsProxyEveLegoBuilder::MuonsProxyEveLegoBuilder(const MuonsProxyEveLegoBuilder& rhs)
// {
//    // do actual copying here;
// }

MuonsProxyEveLegoBuilder::~MuonsProxyEveLegoBuilder()
{
}

void
MuonsProxyEveLegoBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"MuonsLego",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
   } else {
      tList->DestroyElements();
   }

   const reco::MuonCollection* muons=0;
   iItem->get(muons);
   if(0 == muons ) return;

   fw::NamedCounter counter("moun");
   for ( reco::MuonCollection::const_iterator muon = muons->begin();
	 muon != muons->end(); ++muon, ++counter )
     {
	char title[1024];
	sprintf(title,"Muon %d, Pt: %0.1f GeV",counter.index(), muon->pt());
	TEveCompound* muonList = new TEveCompound(counter.str().c_str(), title);
        muonList->OpenCompound();
	gEve->AddElement(muonList, tList);
        //guarantees that CloseCompound will be called no matter what happens
        boost::shared_ptr<TEveCompound> sentry(muonList,boost::mem_fn(&TEveCompound::CloseCompound));
	muonList->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
	muonList->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());
	TEvePointSet* points = new TEvePointSet("points");
	gEve->AddElement(points, muonList);
	points->SetMarkerStyle(2);
	points->SetMarkerSize(0.2);
	if ( muon->track().isAvailable() && muon->track()->extra().isAvailable() ) {
	   // get position of the muon at surface of the tracker
	   points->SetNextPoint(muon->track()->outerPosition().eta(),
				muon->track()->outerPosition().phi(),
				0.1);
	} else {
	   if ( muon->standAloneMuon().isAvailable() && muon->standAloneMuon()->extra().isAvailable() ) {
	      // get position of the inner state of the stand alone muon
	      if (  muon->standAloneMuon()->innerPosition().R() <  muon->standAloneMuon()->outerPosition().R() )
		points->SetNextPoint(muon->standAloneMuon()->innerPosition().eta(),
				     muon->standAloneMuon()->innerPosition().phi(),
				     0.1);
	      else
		points->SetNextPoint(muon->standAloneMuon()->outerPosition().eta(),
				     muon->standAloneMuon()->outerPosition().phi(),
				     0.1);
	   } else {
	      // WARNING: use direction at POCA as the last option
	      points->SetNextPoint(muon->eta(),muon->phi(),0.1);
	   }
	}
	points->SetMarkerColor(  iItem->defaultDisplayProperties().color() );
	// muonList->AddElement( points );
	// tList->AddElement(muonList);
     }
}

//REGISTER_FW3DLEGODATAPROXYBUILDER(MuonsProxyEveLegoBuilder,reco::MuonCollection,"Muons");


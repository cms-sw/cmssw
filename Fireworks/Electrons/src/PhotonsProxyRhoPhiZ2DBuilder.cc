// -*- C++ -*-
//
// Package:     Calo
// Class  :     PhotonsProxyRhoPhiZ2DBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: PhotonsProxyRhoPhiZ2DBuilder.cc,v 1.5 2008/11/09 19:10:37 jmuelmen Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEveTrack.h"
#include "TEveCompound.h"
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

// user include files
#include "Fireworks/Electrons/interface/PhotonsProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PhotonsProxyRhoPhiZ2DBuilder::PhotonsProxyRhoPhiZ2DBuilder()
{
}

// PhotonsProxyRhoPhiZ2DBuilder::PhotonsProxyRhoPhiZ2DBuilder(const PhotonsProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

PhotonsProxyRhoPhiZ2DBuilder::~PhotonsProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void
PhotonsProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
     TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
     TEveElementList* tList = *product;

     // printf("calling PhotonsProxyRhoPhiZ2DBuilder::buildRhiPhi\n");
     if(0 == tList) {
	  tList =  new TEveElementList(iItem->name().c_str(),"Photon RhoPhi",true);
	  *product = tList;
	  tList->SetMainColor(iItem->defaultDisplayProperties().color());
	  gEve->AddElement(tList);
     } else {
	  tList->DestroyElements();
     }
     // get photons
     const reco::PhotonCollection *photons = 0;
     iItem->get(photons);
     if (photons == 0) return;
     fw::NamedCounter counter("photon");
     const double r = 122;
     // loop over photons
     for (reco::PhotonCollection::const_iterator photon = photons->begin();
	  photon != photons->end(); ++photon,++counter) {
	const unsigned int nBuffer = 1024;
	char title[nBuffer];
	snprintf(title, nBuffer, "Photon %d, Pt: %0.1f GeV",counter.index(), photon->pt());
	TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
        container->OpenCompound();
        //guarantees that CloseCompound will be called no matter what happens
        boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

	assert(photon->superCluster().isNonnull());
	std::vector<DetId> detids = photon->superCluster()->getHitsByDetId();
	std::vector<double> phis;
	for (std::vector<DetId>::const_iterator id = detids.begin(); id != detids.end(); ++id) {
	   const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( id->rawId() );
	   if ( matrix ) phis.push_back( atan2(matrix->GetTranslation()[1], matrix->GetTranslation()[0]) );
	}

	std::pair<double,double> phiRange = fw::getPhiRange( phis, photon->phi() );
	TGeoBBox *sc_box = new TGeoTubeSeg(r - 1, r + 1, 1,
					   phiRange.first * 180 / M_PI - 0.5,
					   phiRange.second * 180 / M_PI + 0.5 ); // 0.5 is roughly half size of a crystal
	TEveGeoShape *sc = fw::getShape( "supercluster", sc_box, tList->GetMainColor() );
	sc->SetPickable(kTRUE);
	container->AddElement(sc);

// 	TEveTrack* track = fw::getEveTrack( *(photon->gsfTrack()) );
// 	track->SetMainColor( iItem->defaultDisplayProperties().color() );
// 	container->AddElement(track);
	container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
	container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
	tList->AddElement(container);
     }

}

void
PhotonsProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					  TEveElementList** product)
{
     // printf("calling PhotonsProxyRhoPhiZ2DBuilder::buildRhoZ\n");
     TEveElementList* tList = *product;
     if(0 == tList) {
	  tList =  new TEveElementList(iItem->name().c_str(),"Photon RhoZ",true);
	  *product = tList;
	  tList->SetMainColor(iItem->defaultDisplayProperties().color());
	  gEve->AddElement(tList);
     } else {
	  tList->DestroyElements();
     }
     const reco::PhotonCollection *photons = 0;
     iItem->get(photons);
     if (photons == 0) return;
     fw::NamedCounter counter("photon");
     double z_ecal = 302; // ECAL endcap inner surface
     double r_ecal = 122;
     for (reco::PhotonCollection::const_iterator photon = photons->begin();
	  photon != photons->end(); ++photon, ++counter) {
	const unsigned int nBuffer = 1024;
	char title[nBuffer];
	snprintf(title, nBuffer, "Photon %d, Pt: %0.1f GeV",counter.index(), photon->pt());
	TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
        container->OpenCompound();
        //guarantees that CloseCompound will be called no matter what happens
        boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

	assert(photon->superCluster().isNonnull());
	std::vector<DetId> detids = photon->superCluster()->getHitsByDetId();
	double theta_max = 0;
	double theta_min = 10;
	for (std::vector<DetId>::const_iterator id = detids.begin(); id != detids.end(); ++id) {
	   const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( id->rawId() );
	   if ( matrix ) {
	      double r = sqrt( matrix->GetTranslation()[0]*matrix->GetTranslation()[0] +
			       matrix->GetTranslation()[1]*matrix->GetTranslation()[1] );
	      double theta = atan2(r,matrix->GetTranslation()[2]);
	      if ( theta > theta_max ) theta_max = theta;
	      if ( theta < theta_min ) theta_min = theta;
	   }
	}

// 	TEveTrack* track = fw::getEveTrack( *(photon->gsfTrack()) );
// 	track->SetMainColor( iItem->defaultDisplayProperties().color() );
// 	container->AddElement(track);

	// expand theta range by the size of a crystal to avoid segments of zero length
	if ( theta_min <= theta_max )
	  fw::addRhoZEnergyProjection( container, r_ecal, z_ecal, theta_min-0.003, theta_max+0.003,
				       photon->phi(), iItem->defaultDisplayProperties().color() );
	container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
	container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
	tList->AddElement(container);
     }
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(PhotonsProxyRhoPhiZ2DBuilder,reco::PhotonCollection,"Photons");

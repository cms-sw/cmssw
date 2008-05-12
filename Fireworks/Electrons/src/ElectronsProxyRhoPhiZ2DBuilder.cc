// -*- C++ -*-
//
// Package:     Calo
// Class  :     ElectronsProxyRhoPhiZ2DBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: ElectronsProxyRhoPhiZ2DBuilder.cc,v 1.5 2008/02/28 18:29:03 dmytro Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEveTrack.h"

// user include files
#include "Fireworks/Electrons/interface/ElectronsProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
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
ElectronsProxyRhoPhiZ2DBuilder::ElectronsProxyRhoPhiZ2DBuilder()
{
}

// ElectronsProxyRhoPhiZ2DBuilder::ElectronsProxyRhoPhiZ2DBuilder(const ElectronsProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

ElectronsProxyRhoPhiZ2DBuilder::~ElectronsProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void 
ElectronsProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
     TEveElementList* tList = *product;

     // printf("calling ElectronsProxyRhoPhiZ2DBuilder::buildRhiPhi\n");
     if(0 == tList) {
	  tList =  new TEveElementList(iItem->name().c_str(),"Electron RhoPhi",true);
	  *product = tList;
	  tList->SetMainColor(iItem->defaultDisplayProperties().color());
	  gEve->AddElement(tList);
     } else {
	  tList->DestroyElements();
     }
     // get electrons
     const reco::GsfElectronCollection *electrons = 0;
     iItem->get(electrons);
     if (electrons == 0) {
	  std::cout <<"failed to get GSF electrons" << std::endl;
	  return;
     }
     fw::NamedCounter counter("electron");
     const double r = 122;
     // loop over electrons
     for (reco::GsfElectronCollection::const_iterator electron = electrons->begin();
	  electron != electrons->end(); ++electron,++counter) {
	TEveElementList* container = new TEveElementList( counter.str().c_str() );
	assert(electron->superCluster().isNonnull());
	std::vector<DetId> detids = electron->superCluster()->getHitsByDetId();
	std::vector<double> phis;
	for (std::vector<DetId>::const_iterator id = detids.begin(); id != detids.end(); ++id) {
	   const TGeoHMatrix* matrix = m_item->getGeom()->getMatrix( id->rawId() );
	   if ( matrix ) phis.push_back( atan2(matrix->GetTranslation()[1], matrix->GetTranslation()[0]) );
	}
	
	std::pair<double,double> phiRange = fw::getPhiRange( phis, electron->phi() );
	TGeoBBox *sc_box = new TGeoTubeSeg(r - 1, r + 1, 1, 
					   phiRange.first * 180 / M_PI - 0.5,  
					   phiRange.second * 180 / M_PI + 0.5 ); // 0.5 is roughly half size of a crystal 
	TEveGeoShapeExtract *sc = fw::getShapeExtract( "supercluster", sc_box, tList->GetMainColor() );
	TEveElement* element = TEveGeoShape::ImportShapeExtract(sc, 0);
	element->SetPickable(kTRUE);
	container->AddElement(element);
	
	TEveTrack* track = fw::getEveTrack( *(electron->gsfTrack()) );
	track->SetMainColor( iItem->defaultDisplayProperties().color() );
	container->AddElement(track);
	tList->AddElement(container);
     }

}

void 
ElectronsProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					  TEveElementList** product)
{
     // printf("calling ElectronsProxyRhoPhiZ2DBuilder::buildRhoZ\n");
     TEveElementList* tList = *product;
     if(0 == tList) {
	  tList =  new TEveElementList(iItem->name().c_str(),"Electron RhoZ",true);
	  *product = tList;
	  tList->SetMainColor(iItem->defaultDisplayProperties().color());
	  gEve->AddElement(tList);
     } else {
	  tList->DestroyElements();
     }
     const reco::GsfElectronCollection *electrons = 0;
     iItem->get(electrons);
     if (electrons == 0) {
	  std::cout <<"failed to get GSF electrons" << std::endl;
	  return;
     }
     fw::NamedCounter counter("electron");
     double z_ecal = 302; // ECAL endcap inner surface
     double r_ecal = 122;
     for (reco::GsfElectronCollection::const_iterator electron = electrons->begin();
	  electron != electrons->end(); ++electron, ++counter) {
	TEveElementList* container = new TEveElementList( counter.str().c_str() );
	assert(electron->superCluster().isNonnull());
	std::vector<DetId> detids = electron->superCluster()->getHitsByDetId();
	double theta_max = 0;
	double theta_min = 10;
	for (std::vector<DetId>::const_iterator id = detids.begin(); id != detids.end(); ++id) {
	   const TGeoHMatrix* matrix = m_item->getGeom()->getMatrix( id->rawId() );
	   if ( matrix ) {
	      double r = sqrt( matrix->GetTranslation()[0]*matrix->GetTranslation()[0] +
			       matrix->GetTranslation()[1]*matrix->GetTranslation()[1] );
	      double theta = atan2(r,matrix->GetTranslation()[2]);
	      if ( theta > theta_max ) theta_max = theta;
	      if ( theta < theta_min ) theta_min = theta;
	   }
	}
	
	TEveTrack* track = fw::getEveTrack( *(electron->gsfTrack()) );
	track->SetMainColor( iItem->defaultDisplayProperties().color() );
	container->AddElement(track);
	
	// expand theta range by the size of a crystal to avoid segments of zero length
	if ( theta_min <= theta_max ) 
	  fw::addRhoZEnergyProjection( container, r_ecal, z_ecal, theta_min-0.003, theta_max+0.003, 
				       electron->phi(), iItem->defaultDisplayProperties().color() );

	tList->AddElement(container);
     }
}

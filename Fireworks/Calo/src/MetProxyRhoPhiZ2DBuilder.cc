// -*- C++ -*-
//
// Package:     Calo
// Class  :     MetProxyRhoPhiZ2DBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: MetProxyRhoPhiZ2DBuilder.cc,v 1.2 2008/07/03 02:06:41 dmytro Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"
#include "TGeoArb8.h"
#include "TEveManager.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEvePointSet.h"
#include "TEveScalableStraightLineSet.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Calo/interface/MetProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Calo/interface/ECalCaloTowerProxyRhoPhiZ2DBuilder.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MetProxyRhoPhiZ2DBuilder::MetProxyRhoPhiZ2DBuilder()
{
}

// MetProxyRhoPhiZ2DBuilder::MetProxyRhoPhiZ2DBuilder(const MetProxyRhoPhiZ2DBuilder& rhs)
// {
//    // do actual copying here;
// }

MetProxyRhoPhiZ2DBuilder::~MetProxyRhoPhiZ2DBuilder()
{
}

//
// member functions
//
void 
MetProxyRhoPhiZ2DBuilder::buildRhoPhi(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Mets RhoPhi",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }
   
   const reco::CaloMETCollection* mets=0;
   iItem->get(mets);
   if(0==mets) {
      std::cout <<"Failed to get METs"<<std::endl;
      return;
   }
   
   double r_ecal = 126;
   
   fw::NamedCounter counter("met");

   for(unsigned int i = 0; i < mets->size(); ++i, ++counter) {
      char title[1024]; 
      sprintf(title,"MET: %0.1f GeV",mets->at(i).et());
      TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
      container->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));
      
      double phi = mets->at(i).phi();
      double min_phi = phi-M_PI/36/2;
      double max_phi = phi+M_PI/36/2;

      double size = mets->at(i).et();
      TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
      TEveGeoShapeExtract *sc = fw::getShapeExtract( "spread", sc_box, iItem->defaultDisplayProperties().color() );
      
      TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
      marker->SetLineWidth(2);
      // marker->SetLineStyle(kDotted);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      TEveElement* element = TEveGeoShape::ImportShapeExtract(sc, 0);
      element->SetPickable(kTRUE);
      container->AddElement(element);
      marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
      marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
      marker->AddLine( (r_ecal+size*0.9)*cos(phi+0.01), (r_ecal+size*0.9)*sin(phi+0.01), 0, 
		       (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
      marker->AddLine( (r_ecal+size*0.9)*cos(phi-0.01), (r_ecal+size*0.9)*sin(phi-0.01), 0, 
		       (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
      container->AddElement(marker);

      tList->AddElement(container);
   }
}

void 
MetProxyRhoPhiZ2DBuilder::buildRhoZ(const FWEventItem* iItem,
					    TEveElementList** product)
{
   TEveElementList* tList = *product;
   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Mets RhoZ",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }
   
   const reco::CaloMETCollection* mets=0;
   iItem->get(mets);
   if(0==mets) {
      std::cout <<"Failed to get METs"<<std::endl;
      return;
   }
   
   double r = 126;
   
   fw::NamedCounter counter("met");

   for(unsigned int i = 0; i < mets->size(); ++i, ++counter) {
      char title[1024]; 
      sprintf(title,"MET: %0.1f GeV",mets->at(i).et());
      TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
      container->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

      double phi = mets->at(i).phi();
      double size = mets->at(i).et();
      
      TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
      marker->SetLineWidth(2);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      marker->SetScaleCenter(0., (phi>0 ? r : -r), 0);
      marker->AddLine(0., (phi>0 ? r : -r), 0,
		      0., (phi>0 ? (r+size) : -(r+size)), 0 );
      marker->AddLine(0., (phi>0 ? r+size*0.9 : -(r+size*0.9) ), r*0.01,
		      0., (phi>0 ? (r+size) : -(r+size)), 0 );
      marker->AddLine(0., (phi>0 ? r+size*0.9 : -(r+size*0.9) ), -r*0.01,
		      0., (phi>0 ? (r+size) : -(r+size)), 0 );
      container->AddElement( marker );
      tList->AddElement(container);
   }
}
   
REGISTER_FWRPZ2DDATAPROXYBUILDER(MetProxyRhoPhiZ2DBuilder,reco::CaloMETCollection,"MET");

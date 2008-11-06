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
// $Id: MetProxyRhoPhiZ2DBuilder.cc,v 1.7 2008/11/04 11:46:33 amraktad Exp $
//

// system include files
#include "TEveGeoNode.h"
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
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Mets RhoPhi",true);
      *product = tList;
      tList->SetMainColor(   iItem->defaultDisplayProperties().color() );
      gEve->AddElement(tList);
   } else {
      tList->DestroyElements();
   }
   
   const reco::CaloMETCollection* mets=0;
   iItem->get(mets);
   if(0==mets) return;

   double r_ecal = 126;
   
   fw::NamedCounter counter("met");

   for(unsigned int i = 0; i < mets->size(); ++i, ++counter) {
      const unsigned int nBuffer = 1024;
      char title[nBuffer]; 
      snprintf(title, nBuffer, "MET: %0.1f GeV", mets->at(i).et());
      TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
      container->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));
      
      double phi = mets->at(i).phi();
      double min_phi = phi-M_PI/36/2;
      double max_phi = phi+M_PI/36/2;

      double size = mets->at(i).et();
      TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
      TEveGeoShape *element = fw::getShape( "spread", sc_box, iItem->defaultDisplayProperties().color() );
      element->SetPickable(kTRUE);
      container->AddElement(element);

      TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
      marker->SetLineWidth(2);
      // marker->SetLineStyle(kDotted);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );

      marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
      const double dx = 0.9*size*0.05;
      const double dy = 0.9*size*cos(0.05);
      marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0, 
		       (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
      marker->AddLine( dx*sin(phi) + (dy+r_ecal)*cos(phi), -dx*cos(phi) + (dy+r_ecal)*sin(phi), 0,
		       (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
      marker->AddLine( -dx*sin(phi) + (dy+r_ecal)*cos(phi), dx*cos(phi) + (dy+r_ecal)*sin(phi), 0,
		       (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
		       
      container->AddElement(marker);
      container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );

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
   if(0==mets) return;
   
   double r = 126;
   
   fw::NamedCounter counter("met");

   for(unsigned int i = 0; i < mets->size(); ++i, ++counter) {
      const unsigned int nBuffer = 1024;
      char title[nBuffer]; 
      snprintf(title, nBuffer, "MET: %0.1f GeV", mets->at(i).et());
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
      const double dx = 0.9*size*0.05;
      const double dy = 0.9*size*cos(0.05);
      marker->AddLine(0., (phi>0 ? r : -r), 0,
		      0., (phi>0 ? (r+size) : -(r+size)), 0 );
      marker->AddLine(0., (phi>0 ? r+dy : -(r+dy) ), dx,
		      0., (phi>0 ? (r+size) : -(r+size)), 0 );
      marker->AddLine(0., (phi>0 ? r+dy : -(r+dy) ), -dx,
		      0., (phi>0 ? (r+size) : -(r+size)), 0 );
      container->AddElement( marker );
      container->SetRnrSelf(     iItem->defaultDisplayProperties().isVisible() );
      container->SetRnrChildren( iItem->defaultDisplayProperties().isVisible() );
      tList->AddElement(container);
   }
}
   
REGISTER_FWRPZ2DDATAPROXYBUILDER(MetProxyRhoPhiZ2DBuilder,reco::CaloMETCollection,"MET");

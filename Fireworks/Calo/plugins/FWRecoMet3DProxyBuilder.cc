// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWRecoMet3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWRecoMet3DProxyBuilder.cc,v 1.1 2009/05/14 Yanjun Tu Exp $
//

// system include files
#include "TEveManager.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveElement.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FW3DView.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"

#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/MET.h"

class FWRecoMet3DProxyBuilder : public FW3DDataProxyBuilder
{

public:
   FWRecoMet3DProxyBuilder();
   virtual ~FWRecoMet3DProxyBuilder();

   // ---------- const member functions ---------------------
   REGISTER_PROXYBUILDER_METHODS();

   // ---------- static member functions --------------------
private:
   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product);

   FWRecoMet3DProxyBuilder(const FWRecoMet3DProxyBuilder&);    // stop default

   const FWRecoMet3DProxyBuilder& operator=(const FWRecoMet3DProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWRecoMet3DProxyBuilder::FWRecoMet3DProxyBuilder()
{
}

FWRecoMet3DProxyBuilder::~FWRecoMet3DProxyBuilder()
{
}

void
FWRecoMet3DProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Met 3D",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
   } else {
      tList->DestroyElements();
   }

   const reco::METCollection* mets=0;
   iItem->get(mets);
   if(0==mets) return;
   
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

     
      // double size = mets->at(i).et();
      double size = mets->at(i).et()*2;
      TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
      TEveGeoShape *element = fw::getShape( "spread", sc_box, iItem->defaultDisplayProperties().color() );
      element->SetPickable(kTRUE);
      container->AddElement(element);

      TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
      marker->SetLineWidth(2);
      // marker->SetLineStyle(kDotted);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );

      marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
    //   const double dx = 0.9*size*0.05;
//       const double dy = 0.9*size*cos(0.05);
      const double dx = 0.9*size*0.1;
      const double dy = 0.9*size*cos(0.1);
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

REGISTER_FW3DDATAPROXYBUILDER(FWRecoMet3DProxyBuilder,reco::METCollection,"recoMET");


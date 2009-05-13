// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWRecoMetGlimpseProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWRecoMetGlimpseProxyBuilder.cc,v 1.2 2009/01/23 21:35:40 amraktad Exp $
//

// system include files
#include "TEveElement.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWGlimpseView.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"

#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/MET.h"

class FWRecoMetGlimpseProxyBuilder : public FWGlimpseDataProxyBuilder
{

public:
   FWRecoMetGlimpseProxyBuilder();
   virtual ~FWRecoMetGlimpseProxyBuilder();

   // ---------- const member functions ---------------------
   REGISTER_PROXYBUILDER_METHODS();

   // ---------- static member functions --------------------
private:
   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product);

   FWRecoMetGlimpseProxyBuilder(const FWRecoMetGlimpseProxyBuilder&);    // stop default

   const FWRecoMetGlimpseProxyBuilder& operator=(const FWRecoMetGlimpseProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWRecoMetGlimpseProxyBuilder::FWRecoMetGlimpseProxyBuilder()
{
}

FWRecoMetGlimpseProxyBuilder::~FWRecoMetGlimpseProxyBuilder()
{
}

void
FWRecoMetGlimpseProxyBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"Met",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
   } else {
      tList->DestroyElements();
   }

   const reco::METCollection* mets=0;
   iItem->get(mets);
   if(0==mets) return;

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
      // marker->SetLineStyle(kDotted);
      marker->SetLineColor(  iItem->defaultDisplayProperties().color() );
      marker->AddLine( 0, 0, 0, size*cos(phi), size*sin(phi), 0);
      marker->AddLine( size*0.9*cos(phi+0.03), size*0.9*sin(phi+0.03), 0, size*cos(phi), size*sin(phi), 0);
      marker->AddLine( size*0.9*cos(phi-0.03), size*0.9*sin(phi-0.03), 0, size*cos(phi), size*sin(phi), 0);
      container->AddElement(marker);

      tList->AddElement(container);
   }

}

REGISTER_FWGLIMPSEDATAPROXYBUILDER(FWRecoMetGlimpseProxyBuilder,reco::METCollection,"recoMET");


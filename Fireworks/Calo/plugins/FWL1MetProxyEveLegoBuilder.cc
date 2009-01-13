// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWL1MetProxyEveLegoBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Jan  6 23:57:00 EST 2008
//
// $Id: FWL1MetProxyEveLegoBuilder.cc,v 1.1 2009/01/13 20:10:02 amraktad Exp $

// system include files
#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"
#include "TEveStraightLineSet.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

class FWL1MetProxyEveLegoBuilder : public FW3DLegoEveElementProxyBuilder
{
   public:
      FWL1MetProxyEveLegoBuilder();
      virtual ~FWL1MetProxyEveLegoBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      FWL1MetProxyEveLegoBuilder(const FWL1MetProxyEveLegoBuilder&); // stop default

      const FWL1MetProxyEveLegoBuilder& operator=(const FWL1MetProxyEveLegoBuilder&); // stop default

      // ---------- member data --------------------------------
};

FWL1MetProxyEveLegoBuilder::FWL1MetProxyEveLegoBuilder()
{
}

FWL1MetProxyEveLegoBuilder::~FWL1MetProxyEveLegoBuilder()
{
}

void
FWL1MetProxyEveLegoBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"JetsLego",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
   } else {
      tList->DestroyElements();
   }

   // Get the particle map collection for L1EtMissParticles
   l1extra::L1EtMissParticleCollection const * mets=0;
   iItem->get(mets);
   if(0==mets) return;

   fw::NamedCounter counter("met");

   for(unsigned int i = 0; i < mets->size(); ++i, ++counter) {
      char title[1024];
      sprintf(title,"L1 MET: %0.1f GeV",mets->at(i).et());
      TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
      container->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(container,boost::mem_fn(&TEveCompound::CloseCompound));

      TEveStraightLineSet* mainLine = new TEveStraightLineSet( "MET phi" );
      // mainLine->SetLineWidth(2);
      mainLine->SetLineColor(  iItem->defaultDisplayProperties().color() );
      mainLine->AddLine(-5.191, mets->at(i).phi(), 0.1, 5.191, mets->at(i).phi(), 0.1 );
      container->AddElement( mainLine );

      double phi = mets->at(i).phi();
      phi = phi > 0 ? phi - M_PI : phi + M_PI;
      TEveStraightLineSet* secondLine = new TEveStraightLineSet( "MET opposite phi" );
      // secondLine->SetLineWidth(2);
      secondLine->SetLineStyle(7);
      secondLine->SetLineColor(  iItem->defaultDisplayProperties().color() );
      secondLine->AddLine(-5.191, phi, 0.1, 5.191, phi, 0.1 );
      container->AddElement( secondLine );

      tList->AddElement(container);
   }
}

REGISTER_FW3DLEGODATAPROXYBUILDER(FWL1MetProxyEveLegoBuilder,l1extra::L1EtMissParticleCollection,"L1-MET");

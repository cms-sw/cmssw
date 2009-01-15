// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWMetProxyEveLegoBuilder

//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWMetProxyEveLegoBuilder.cc,v 1.1 2009/11/15 22:05:21 amraktad Exp $
//

// system include files
#include "TEveElement.h"
#include "TEveStraightLineSet.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"

class FWMetProxyEveLegoBuilder : public FW3DLegoEveElementProxyBuilder
{
   public:
      FWMetProxyEveLegoBuilder();
      virtual ~FWMetProxyEveLegoBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
                         TEveElementList** product);

      FWMetProxyEveLegoBuilder(const FWMetProxyEveLegoBuilder&); // stop default

      const FWMetProxyEveLegoBuilder& operator=(const FWMetProxyEveLegoBuilder&); // stop default

      // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWMetProxyEveLegoBuilder::FWMetProxyEveLegoBuilder()
{
}

// FWMetProxyEveLegoBuilder::FWMetProxyEveLegoBuilder(const FWMetProxyEveLegoBuilder& rhs)
// {
//    // do actual copying here;
// }

FWMetProxyEveLegoBuilder::~FWMetProxyEveLegoBuilder()
{
}

//
// assignment operators
//
// const FWMetProxyEveLegoBuilder& FWMetProxyEveLegoBuilder::operator=(const FWMetProxyEveLegoBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWMetProxyEveLegoBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }


void
FWMetProxyEveLegoBuilder::build(const FWEventItem* iItem, TEveElementList** product)
{
   TEveElementList* tList = *product;

   if(0 == tList) {
      tList =  new TEveElementList(iItem->name().c_str(),"JetsLego",true);
      *product = tList;
      tList->SetMainColor(iItem->defaultDisplayProperties().color());
   } else {
      tList->DestroyElements();
   }

   const reco::CaloMETCollection* mets=0;
   iItem->get(mets);
   if(0==mets) return;

   fw::NamedCounter counter("met");

   for(unsigned int i = 0; i < mets->size(); ++i, ++counter) {
      char title[1024];
      sprintf(title,"MET: %0.1f GeV",mets->at(i).et());
      TEveCompound* container = new TEveCompound( counter.str().c_str(), title );
      container->SetRnrSelf(iItem->defaultDisplayProperties().isVisible());
      container->SetRnrChildren(iItem->defaultDisplayProperties().isVisible());

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

REGISTER_FW3DLEGODATAPROXYBUILDER(FWMetProxyEveLegoBuilder,reco::CaloMETCollection,"MET");

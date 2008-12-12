// -*- C++ -*-
// $Id: FWLegoProxyBuilder.template,v 1.1 2008/12/10 13:58:53 dmytro Exp $
//

// include files
#include "Fireworks/Core/interface/FW3DLegoSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"
#include "TEveStraightLineSet.h"

#include "DataFormats/JetReco/interface/Jet.h"

class FWJetsLegoProxyBuilder : public FW3DLegoSimpleProxyBuilderTemplate<reco::Jet> {
   
public:
   FWJetsLegoProxyBuilder(){}
   virtual ~FWJetsLegoProxyBuilder(){}
   REGISTER_PROXYBUILDER_METHODS();
  
private:
   FWJetsLegoProxyBuilder(const FWJetsLegoProxyBuilder&); // stop default
   const FWJetsLegoProxyBuilder& operator=(const FWJetsLegoProxyBuilder&); // stop default
   
   void build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

};

void 
FWJetsLegoProxyBuilder::build(const reco::Jet& iData, unsigned int iIndex,TEveElement& tList) const
{
   const unsigned int nLineSegments = 20;
   const double jetRadius = 0.5;
   TEveStraightLineSet* container = new TEveStraightLineSet();
   container->SetLineColor( item()->defaultDisplayProperties().color() );
   container->SetRnrSelf( item()->defaultDisplayProperties().isVisible() );
   container->SetRnrChildren( item()->defaultDisplayProperties().isVisible() );
   for ( unsigned int iphi = 0; iphi < nLineSegments; ++iphi ) {
      container->AddLine(iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*iphi),
			 iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*iphi),
			 0.1,
			 iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*(iphi+1)),
			 iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*(iphi+1)),
			 0.1);
   }
   tList.AddElement(container);
}

REGISTER_FW3DLEGODATAPROXYBUILDER(FWJetsLegoProxyBuilder,reco::Jet,"Jets");

// -*- C++ -*-
// $Id: FWJetsGlimpseProxyBuilder.cc,v 1.1 2008/12/12 06:06:02 dmytro Exp $
//

// include files
#include "Fireworks/Core/interface/FWGlimpseSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"
#include "TEveStraightLineSet.h"
#include "TEveBoxSet.h"

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "Fireworks/Calo/interface/FWGlimpseEveJet.h"

class FWJetsGlimpseProxyBuilder : public FWGlimpseSimpleProxyBuilderTemplate<reco::Jet> {

public:
   FWJetsGlimpseProxyBuilder(){
   }
   virtual ~FWJetsGlimpseProxyBuilder(){
   }
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWJetsGlimpseProxyBuilder(const FWJetsGlimpseProxyBuilder&); // stop default
   const FWJetsGlimpseProxyBuilder& operator=(const FWJetsGlimpseProxyBuilder&); // stop default

   virtual void build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void
FWJetsGlimpseProxyBuilder::build(const reco::Jet& iData, unsigned int iIndex,TEveElement& tList) const
{
   FWGlimpseEveJet* cone = new FWGlimpseEveJet(&iData,"jet","jet");
   cone->SetPickable(kTRUE);
   cone->SetMainColor(item()->defaultDisplayProperties().color());
   cone->SetMainTransparency(50);
   cone->SetRnrSelf(item()->defaultDisplayProperties().isVisible());
   cone->SetRnrChildren(item()->defaultDisplayProperties().isVisible());
   cone->SetDrawConeCap(kFALSE);
   cone->SetMainTransparency(50);
   tList.AddElement(cone);
}

REGISTER_FWGLIMPSEDATAPROXYBUILDER(FWJetsGlimpseProxyBuilder,reco::Jet,"Jets");

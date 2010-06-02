// -*- C++ -*-
// $Id: FWPFTauGlimpseProxyBuilder.cc,v 1.1 2009/05/04 23:57:42 yanjuntu Exp $
//

// include files
#include "Fireworks/Core/interface/FWGlimpseSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

#include "TEveElement.h"
#include "TColor.h"
#include "TGeoTube.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TROOT.h"
#include "TEveStraightLineSet.h"
#include "TEveBoxSet.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Calo/interface/FWGlimpseEveJet.h"
// #include "TEvePointSet.h"

class FWPFTauGlimpseProxyBuilder : public FWGlimpseSimpleProxyBuilderTemplate<reco::PFTau> {
   
public:
   FWPFTauGlimpseProxyBuilder(){}
   virtual ~FWPFTauGlimpseProxyBuilder(){}
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFTauGlimpseProxyBuilder(const FWPFTauGlimpseProxyBuilder&); // stop default
   const FWPFTauGlimpseProxyBuilder& operator=(const FWPFTauGlimpseProxyBuilder&); // stop default
   
   virtual void build(const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void 
FWPFTauGlimpseProxyBuilder::build(const reco::PFTau& tau, unsigned int iIndex,TEveElement& tList) const
{
  const reco::PFTauTagInfo *tau_tag_info = (const reco::PFTauTagInfo*)(tau.pfTauTagInfoRef().get());
  const reco::PFJet* jet = (const reco::PFJet*)(tau_tag_info->pfjetRef().get());
  FWGlimpseEveJet* cone = new FWGlimpseEveJet(jet,"jet","jet");
  cone->SetPickable(kTRUE);
  cone->SetMainColor(item()->defaultDisplayProperties().color());
  cone->SetMainTransparency(50);
  cone->SetRnrSelf(item()->defaultDisplayProperties().isVisible());
  cone->SetRnrChildren(item()->defaultDisplayProperties().isVisible());
  cone->SetDrawConeCap(kFALSE);
  cone->SetMainTransparency(50);
  tList.AddElement(cone);
}

REGISTER_FWGLIMPSEDATAPROXYBUILDER(FWPFTauGlimpseProxyBuilder,reco::PFTau,"PFTau");

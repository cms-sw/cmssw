// -*- C++ -*-
// $Id: FWGlimpseProxyBuilder 2009/04/27 Yanjun Tu $
//

// include files
#include "Fireworks/Core/interface/FWGlimpseSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"

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

class FWCaloTauGlimpseProxyBuilder : public FWGlimpseSimpleProxyBuilderTemplate<reco::CaloTau> {
   
public:
   FWCaloTauGlimpseProxyBuilder(){}
   virtual ~FWCaloTauGlimpseProxyBuilder(){}
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloTauGlimpseProxyBuilder(const FWCaloTauGlimpseProxyBuilder&); // stop default
   const FWCaloTauGlimpseProxyBuilder& operator=(const FWCaloTauGlimpseProxyBuilder&); // stop default
   
   virtual void build(const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void 
FWCaloTauGlimpseProxyBuilder::build(const reco::CaloTau& tau, unsigned int iIndex,TEveElement& tList) const
{
  const reco::CaloTauTagInfo *tau_tag_info = (const reco::CaloTauTagInfo*)(tau.caloTauTagInfoRef().get());
  const reco::CaloJet* jet = (const reco::CaloJet*)(tau_tag_info->calojetRef().get());
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

REGISTER_FWGLIMPSEDATAPROXYBUILDER(FWCaloTauGlimpseProxyBuilder,reco::CaloTau,"CaloTau");

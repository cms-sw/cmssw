// -*- C++ -*-
// $Id: FW3DProxyBuilder. 2009/04/27 13:58:53 Yanjun Tu Exp $
//

// include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Calo/interface/FW3DEveJet.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"


class FWPFTau3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::PFTau>  {
   
public:
   FWPFTau3DProxyBuilder(){}
   virtual ~FWPFTau3DProxyBuilder(){}
   REGISTER_PROXYBUILDER_METHODS();
  static void buildTau3D(const FWEventItem* iItem,
			 const reco::PFTau* iData,
			 TEveElement& tList);
 
private:
   FWPFTau3DProxyBuilder(const FWPFTau3DProxyBuilder&); // stop default
   const FWPFTau3DProxyBuilder& operator=(const FWPFTau3DProxyBuilder&); // stop default
   
   // ---------- member data --------------------------------
   void build(const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void 
FWPFTau3DProxyBuilder::build(const reco::PFTau& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
  buildTau3D( item(), &iData, oItemHolder);
}

void 
FWPFTau3DProxyBuilder::buildTau3D(const FWEventItem* iItem,
				      const reco::PFTau* tau,
				      TEveElement& container)
{
  const reco::PFTauTagInfo *tau_tag_info = (const reco::PFTauTagInfo*)(tau->pfTauTagInfoRef().get());
  const reco::PFJet *jet = (const reco::PFJet*)(tau_tag_info->pfjetRef().get());
  
   FW3DEveJet* cone = new FW3DEveJet(*jet,"jet","jet");
   cone->SetPickable(kTRUE);
   cone->SetMainColor( container.GetMainColor() );
   cone->SetMainTransparency(75); 
  
  container.AddElement( cone );
}

REGISTER_FW3DDATAPROXYBUILDER(FWPFTau3DProxyBuilder,reco::PFTau,"PFTau");

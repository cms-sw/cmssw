// -*- C++ -*-
// $Id: FWCaloTau3DProxyBuilder.cc,v 1.1 2009/05/04 23:57:42 yanjuntu Exp $
//

// include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Calo/interface/FW3DEveJet.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"


class FWCaloTau3DProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::CaloTau>  {
   
public:
   FWCaloTau3DProxyBuilder(){}
   virtual ~FWCaloTau3DProxyBuilder(){}
   REGISTER_PROXYBUILDER_METHODS();
  static void buildTau3D(const FWEventItem* iItem,
			 const reco::CaloTau* iData,
			 TEveElement& tList);
 
private:
   FWCaloTau3DProxyBuilder(const FWCaloTau3DProxyBuilder&); // stop default
   const FWCaloTau3DProxyBuilder& operator=(const FWCaloTau3DProxyBuilder&); // stop default
   
   // ---------- member data --------------------------------
   void build(const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder) const;
};

void 
FWCaloTau3DProxyBuilder::build(const reco::CaloTau& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
  buildTau3D( item(), &iData, oItemHolder);
}

void 
FWCaloTau3DProxyBuilder::buildTau3D(const FWEventItem* iItem,
				      const reco::CaloTau* tau,
				      TEveElement& container)
{
  const reco::CaloTauTagInfo *tau_tag_info = (const reco::CaloTauTagInfo*)(tau->caloTauTagInfoRef().get());
  const reco::CaloJet *jet = (const reco::CaloJet*)(tau_tag_info->calojetRef().get());
  
   FW3DEveJet* cone = new FW3DEveJet(*jet,"jet","jet");
   cone->SetPickable(kTRUE);
   cone->SetMainColor( container.GetMainColor() );
   cone->SetMainTransparency(75); 
  
  container.AddElement( cone );
}

REGISTER_FWPROXYBUILDER(FWCaloTau3DProxyBuilder,reco::CaloTau,"CaloTau", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);

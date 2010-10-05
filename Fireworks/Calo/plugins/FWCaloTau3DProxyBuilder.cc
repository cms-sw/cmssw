// -*- C++ -*-
// $Id: FW3DProxyBuilder. 2009/04/27 13:58:53 Yanjun Tu Exp $
//

// include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Calo/interface/FW3DEveJet.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"


class FWCaloTau3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::CaloTau>  {
   
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

REGISTER_FW3DDATAPROXYBUILDER(FWCaloTau3DProxyBuilder,reco::CaloTau,"CaloTau");

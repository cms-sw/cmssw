// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWPhotonLegoProxyBuilder
//
// $Id: FWPhotonLegoProxyBuilder.cc,v 1.1 2010/05/06 14:13:26 mccauley Exp $
//

#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

class FWPhotonLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Photon>  
{
public:
  FWPhotonLegoProxyBuilder() {}
  virtual ~FWPhotonLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPhotonLegoProxyBuilder(const FWPhotonLegoProxyBuilder&);
   const FWPhotonLegoProxyBuilder& operator=(const FWPhotonLegoProxyBuilder&);

   virtual void build(const reco::Photon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);
};

void FWPhotonLegoProxyBuilder::build(const reco::Photon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   TEvePointSet* points = new TEvePointSet("points");
   setupAddElement(points, &oItemHolder);
 
   points->SetMarkerStyle(2);
   points->SetMarkerSize(0.2);
    
   points->SetNextPoint(iData.eta(), iData.phi(), 0.1);
}

REGISTER_FWPROXYBUILDER(FWPhotonLegoProxyBuilder, reco::Photon, "Photons", FWViewType::kLegoBit);




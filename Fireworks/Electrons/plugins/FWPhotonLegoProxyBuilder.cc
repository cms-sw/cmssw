// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWPhotonLegoProxyBuilder
//
// $Id: FWPhotonLegoProxyBuilder.cc,v 1.1 2010/05/07 09:04:32 mccauley Exp $
//

#include "TEveStraightLineSet.h"

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
   TEveStraightLineSet *marker = new TEveStraightLineSet("marker");
   setupAddElement(marker, &oItemHolder);
 
   const double delta = 0.1;
   marker->AddLine(iData.eta()-delta, iData.phi()-delta, 0.1,
		   iData.eta()+delta, iData.phi()+delta, 0.1);
   marker->AddLine(iData.eta()-delta, iData.phi()+delta, 0.1,
		   iData.eta()+delta, iData.phi()-delta, 0.1);
}

REGISTER_FWPROXYBUILDER(FWPhotonLegoProxyBuilder, reco::Photon, "Photons", FWViewType::kLegoBit);




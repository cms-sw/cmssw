// -*- C++ -*-
// $Id: FWLegoProxyBuilder 2009/4/27 Yanjun Tu 
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

#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"


class FWCaloTauLegoProxyBuilder : public FW3DLegoSimpleProxyBuilderTemplate<reco::CaloTau> {
   
public:
   FWCaloTauLegoProxyBuilder(){}
   virtual ~FWCaloTauLegoProxyBuilder(){}
   REGISTER_PROXYBUILDER_METHODS();
  
private:
   FWCaloTauLegoProxyBuilder(const FWCaloTauLegoProxyBuilder&); // stop default
   const FWCaloTauLegoProxyBuilder& operator=(const FWCaloTauLegoProxyBuilder&); // stop default
   
   void build(const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

};

void 
FWCaloTauLegoProxyBuilder::build(const reco::CaloTau& iData, unsigned int iIndex,TEveElement& tList) const
{
  const unsigned int nLineSegments = 20;
  const double jetRadius = 0.17;   //10 degree
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

REGISTER_FW3DLEGODATAPROXYBUILDER(FWCaloTauLegoProxyBuilder,reco::CaloTau,"CaloTau");

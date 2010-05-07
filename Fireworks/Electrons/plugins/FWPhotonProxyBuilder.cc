// -*- C++ -*-
//
// Package:     Photons
// Class  :     FWPhotonProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 26 14:52:01 EST 2008
// $Id: FWPhotonProxyBuilder.cc,v 1.11 2010/05/06 14:13:26 mccauley Exp $
//

#include "TEveCompound.h"
#include "TEveStraightLineSet.h"
#include "TEveBox.h"

#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Calo/interface/CaloUtils.h"
#include "Fireworks/Electrons/interface/makeSuperCluster.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "Fireworks/Core/interface/DetIdToMatrix.h"

class FWPhotonProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Photon> 
{
public:
  FWPhotonProxyBuilder() {}

  virtual ~FWPhotonProxyBuilder() {}
  
  virtual bool haveSingleProduct() const { return false; }
  
  REGISTER_PROXYBUILDER_METHODS();

private:
  FWPhotonProxyBuilder(const FWPhotonProxyBuilder&);
  const FWPhotonProxyBuilder& operator=(const FWPhotonProxyBuilder&);
  
  virtual void buildViewType(const reco::Photon& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*);
};

void
FWPhotonProxyBuilder::buildViewType(const reco::Photon& photon, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*)
{  
  if( type == FWViewType::kRhoPhi )
  {
    fireworks::makeRhoPhiSuperCluster(this,
                                      photon.superCluster(),
                                      photon.phi(),
                                      oItemHolder);
  }
  
  else if( type == FWViewType::kRhoZ )
    fireworks::makeRhoZSuperCluster(this,
                                    photon.superCluster(),
                                    photon.phi(),
                                    oItemHolder);

  else if ( type == FWViewType::kISpy )
  {
    std::vector<std::pair<DetId, float> > detIds = photon.superCluster()->hitsAndFractions();
    
    for ( std::vector<std::pair<DetId, float> >::iterator id = detIds.begin(), ide = detIds.end();
          id != ide; ++id )      
    {
      std::vector<TEveVector> corners = item()->getGeom()->getPoints((*id).first);
      
      if ( corners.empty() )
      {
        fwLog(fwlog::kWarning)<<"No corners available for supercluster constituent"<<std::endl;
        continue;
      }
      
      fireworks::drawEnergyTower3D(corners, (*id).second, &oItemHolder, this);
    }
  }
}

REGISTER_FWPROXYBUILDER( FWPhotonProxyBuilder, reco::Photon, "Photons", FWViewType::kAllRPZBits |  FWViewType::kAll3DBits );

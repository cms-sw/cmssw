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
// $Id: FWPhotonProxyBuilder.cc,v 1.8 2010/04/29 12:16:52 mccauley Exp $
//
#include "TEveCompound.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"

#include "Fireworks/Electrons/interface/makeSuperCluster.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "Fireworks/Calo/interface/CaloUtils.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

class FWPhotonProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Photon> {

public:
   FWPhotonProxyBuilder() {}

   virtual ~FWPhotonProxyBuilder() {}

   virtual bool haveSingleProduct() const { return false; }

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPhotonProxyBuilder(const FWPhotonProxyBuilder&); // stop default
   const FWPhotonProxyBuilder& operator=(const FWPhotonProxyBuilder&); // stop default

   virtual void buildViewType(const reco::Photon& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type );
};

void
FWPhotonProxyBuilder::buildViewType(const reco::Photon& photon, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type )
{  
  // FIXME: these numbers also appear in makeSuperCluster
  double lEB = 300.0;  // half-length of the EB (cm)
  double rEB = 124.0;  // inner radius of the EB (cm)

  double eta = photon.eta();
  double phi = photon.phi();

  double t = 0.0;
    
  double x0 = photon.vx();
  double y0 = photon.vy();
  double z0 = photon.vz();
  
  double px = cos(phi);
  double py = sin(phi);
  double pz = sinh(eta);

  // FIXME: magic number
  if ( fabs(eta) > 1.48 ) // i.e. not in the EB, so propagate to ES
    t = fabs((lEB - z0)/pz); 

  else // propagate to EB
  {
    double a = px*px + py*py;
    double b = 2*x0*px + 2*y0*py;
    double c = x0*x0 + y0*y0 - rEB*rEB;
    t = (-b+sqrt(b*b-4*a*c))/2*a;
  }

  std::stringstream s;
  s << "photon" << iIndex;

  TEveStraightLineSet* lineSet = new TEveStraightLineSet(s.str().c_str());
  lineSet->SetLineWidth(3);    
  lineSet->AddLine(x0, y0, z0, x0+px*t, y0+py*t, z0+pz*t);
  setupAddElement(lineSet, &oItemHolder);

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
    std::vector<std::pair<DetId, float> > detIds = photon.superCluster()->hitsAndFractions ();
    
    for ( std::vector<std::pair<DetId, float> >::iterator id = detIds.begin(), ide = detIds.end();
          id != ide; ++id )      
    {
      std::vector<TEveVector> corners = item()->getGeom()->getPoints((*id).first);
      
      if ( corners.empty() )
        continue;

      fireworks::drawEnergyTower3D(corners, (*id).second, &oItemHolder, this);
    }
  }
}

REGISTER_FWPROXYBUILDER( FWPhotonProxyBuilder, reco::Photon, "Photons", FWViewType::kAllRPZBits |  FWViewType::kAll3DBits );

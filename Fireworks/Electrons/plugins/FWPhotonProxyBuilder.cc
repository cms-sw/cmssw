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
// $Id: FWPhotonProxyBuilder.cc,v 1.20 2010/11/11 20:25:28 amraktad Exp $
//

#include "TEveBoxSet.h"

#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Electrons/interface/makeSuperCluster.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "Fireworks/Core/interface/FWGeometry.h"

class FWPhotonProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Photon> 
{
public:
   FWPhotonProxyBuilder( void ) {}

   virtual ~FWPhotonProxyBuilder( void ) {}
  
   virtual bool haveSingleProduct( void ) const { return false; }
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPhotonProxyBuilder( const FWPhotonProxyBuilder& );
   const FWPhotonProxyBuilder& operator=( const FWPhotonProxyBuilder& );
  
   virtual void buildViewType( const reco::Photon& photon, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*);
};

void
FWPhotonProxyBuilder::buildViewType( const reco::Photon& photon, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*)
{  
   const FWGeometry *geom = item()->getGeom();
 
   if( type == FWViewType::kRhoPhi || type == FWViewType::kRhoPhiPF )
   {
      fireworks::makeRhoPhiSuperCluster( this,
                                         photon.superCluster(),
                                         photon.phi(),
                                         oItemHolder );
   }
  
   else if( type == FWViewType::kRhoZ )
      fireworks::makeRhoZSuperCluster( this,
                                       photon.superCluster(),
                                       photon.phi(),
                                       oItemHolder );

   else if( type == FWViewType::kISpy )
   {
      std::vector<std::pair<DetId, float> > detIds = photon.superCluster()->hitsAndFractions();

      TEveBoxSet* boxset = new TEveBoxSet();
      boxset->Reset(TEveBoxSet::kBT_FreeBox, true, 64);
      boxset->UseSingleColor();
      boxset->SetPickable(1);

      for( std::vector<std::pair<DetId, float> >::iterator id = detIds.begin(), ide = detIds.end();
           id != ide; ++id )      
      {
         const float* corners = geom->getCorners( id->first.rawId() );
      
         if( corners == 0 )
         {
            fwLog( fwlog::kWarning )
               << "No corners available for supercluster constituent" << std::endl;
            continue;
         }
         boxset->AddBox( &corners[0]);
      }

      boxset->RefitPlex();
      setupAddElement(boxset, &oItemHolder);
   }
}

REGISTER_FWPROXYBUILDER( FWPhotonProxyBuilder, reco::Photon, "Photons", FWViewType::kAllRPZBits |  FWViewType::kAll3DBits );

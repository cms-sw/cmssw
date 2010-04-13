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
// $Id: FWPhotonsProxyBuilder.cc,v 1.2 2009/01/23 21:35:46 amraktad Exp $
//

// system include files

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Electrons/interface/makeSuperCluster.h"

class FWPhotonRhoPhiProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Photon> {

public:
   FWPhotonRhoPhiProxyBuilder() :
     m_reportedNoSuperClusters(false) {}
  
   virtual ~FWPhotonRhoPhiProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPhotonRhoPhiProxyBuilder(const FWPhotonRhoPhiProxyBuilder&); // stop default

   const FWPhotonRhoPhiProxyBuilder& operator=(const FWPhotonRhoPhiProxyBuilder&); // stop default

   virtual void build(const reco::Photon& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   mutable bool m_reportedNoSuperClusters;
};

void
FWPhotonRhoPhiProxyBuilder::build(const reco::Photon& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   if( fireworks::makeRhoPhiSuperCluster(*item(),
                                         iData.superCluster(),
                                         iData.phi(),
                                         oItemHolder) ) {
   } else {
      if(!m_reportedNoSuperClusters) {
         m_reportedNoSuperClusters=true;
         std::cout <<"Can not draw photons because could not get super cluster"<<std::endl;
      }
   }
}

class FWPhotonRhoZProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Photon> {

public:
   FWPhotonRhoZProxyBuilder() :
     m_reportedNoSuperClusters(false) {}
  
   virtual ~FWPhotonRhoZProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPhotonRhoZProxyBuilder(const FWPhotonRhoZProxyBuilder&); // stop default

   const FWPhotonRhoZProxyBuilder& operator=(const FWPhotonRhoZProxyBuilder&); // stop default

   virtual void build(const reco::Photon& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   mutable bool m_reportedNoSuperClusters;
};

void
FWPhotonRhoZProxyBuilder::build(const reco::Photon& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   if( fireworks::makeRhoZSuperCluster(*item(),
                                       iData.superCluster(),
                                       iData.phi(),
                                       oItemHolder) ) {
   } else {
      if(!m_reportedNoSuperClusters) {
         m_reportedNoSuperClusters=true;
         std::cout <<"Can not draw photons because could not get super cluster"<<std::endl;
      }
   }
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWPhotonRhoPhiProxyBuilder, reco::Photon, "Photons", FWViewType::kRhoPhiBit);
REGISTER_FWPROXYBUILDER(FWPhotonRhoZProxyBuilder, reco::Photon, "Photons", FWViewType::kRhoZBit);

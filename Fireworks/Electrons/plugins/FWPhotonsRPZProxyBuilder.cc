// -*- C++ -*-
//
// Package:     Photons
// Class  :     FWPhotonRPZProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 26 14:52:01 EST 2008
// $Id: FWPhotonRPZProxyBuilder.cc,v 1.4 2008/12/01 14:46:08 chrjones Exp $
//

// system include files

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


// user include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Electrons/interface/makeSuperCluster.h"

class FWPhotonRPZProxyBuilder : public FWRPZ2DSimpleProxyBuilderTemplate<reco::Photon> {
      
public:
   FWPhotonRPZProxyBuilder();
   virtual ~FWPhotonRPZProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWPhotonRPZProxyBuilder(const FWPhotonRPZProxyBuilder&); // stop default
   
   const FWPhotonRPZProxyBuilder& operator=(const FWPhotonRPZProxyBuilder&); // stop default
   
   void buildRhoPhi(const reco::Photon& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   void buildRhoZ(const reco::Photon& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   // ---------- member data --------------------------------
   mutable bool m_reportedNoSuperClusters;
   
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWPhotonRPZProxyBuilder::FWPhotonRPZProxyBuilder():
m_reportedNoSuperClusters(false)
{
}

// FWPhotonRPZProxyBuilder::FWPhotonRPZProxyBuilder(const FWPhotonRPZProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWPhotonRPZProxyBuilder::~FWPhotonRPZProxyBuilder()
{
}

//
// assignment operators
//
// const FWPhotonRPZProxyBuilder& FWPhotonRPZProxyBuilder::operator=(const FWPhotonRPZProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWPhotonRPZProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//
void 
FWPhotonRPZProxyBuilder::buildRhoPhi(const reco::Photon& iData, unsigned int iIndex,TEveElement& oItemHolder) const
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

void 
FWPhotonRPZProxyBuilder::buildRhoZ(const reco::Photon& iData, unsigned int iIndex,TEveElement& oItemHolder) const
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
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWPhotonRPZProxyBuilder,reco::Photon,"Photons");

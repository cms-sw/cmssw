// -*- C++ -*-
//
// Package:     Electrons
// Class  :     FWElectronRPZProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 26 14:52:01 EST 2008
// $Id: FWElectronRPZProxyBuilder.cc,v 1.4 2008/12/01 14:46:08 chrjones Exp $
//

// system include files
#include "TColor.h"
#include "TROOT.h"
#include "TEveTrack.h"
#include "TEveCompound.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"


// user include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Electrons/interface/makeSuperCluster.h"


class FWElectronRPZProxyBuilder : public FWRPZ2DSimpleProxyBuilderTemplate<reco::GsfElectron> {
      
public:
   FWElectronRPZProxyBuilder();
   virtual ~FWElectronRPZProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWElectronRPZProxyBuilder(const FWElectronRPZProxyBuilder&); // stop default
   
   const FWElectronRPZProxyBuilder& operator=(const FWElectronRPZProxyBuilder&); // stop default
   
   void buildRhoPhi(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   void buildRhoZ(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
      // ---------- member data --------------------------------
   
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
FWElectronRPZProxyBuilder::FWElectronRPZProxyBuilder()
{
}

// FWElectronRPZProxyBuilder::FWElectronRPZProxyBuilder(const FWElectronRPZProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWElectronRPZProxyBuilder::~FWElectronRPZProxyBuilder()
{
}

//
// assignment operators
//
// const FWElectronRPZProxyBuilder& FWElectronRPZProxyBuilder::operator=(const FWElectronRPZProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWElectronRPZProxyBuilder temp(rhs);
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
FWElectronRPZProxyBuilder::buildRhoPhi(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   fireworks::makeRhoPhiSuperCluster(*item(),
                                     iData.superCluster(),
                                     iData.phi(),
                                     oItemHolder);
   TEveTrack* track(0);
   if ( iData.gsfTrack().isAvailable() )
      track = fw::getEveTrack( *(iData.gsfTrack()) );
   else
      track = fw::getEveTrack( iData );
   oItemHolder.AddElement(track);   
}

void 
FWElectronRPZProxyBuilder::buildRhoZ(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   fireworks::makeRhoZSuperCluster(*item(),
                                   iData.superCluster(),
                                   iData.phi(),
                                   oItemHolder);
   TEveTrack* track(0);
   if ( iData.gsfTrack().isAvailable() )
      track = fw::getEveTrack( *(iData.gsfTrack()) );
   else
      track = fw::getEveTrack( iData );
   oItemHolder.AddElement(track);
   
}

//
// static member functions
//
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWElectronRPZProxyBuilder,reco::GsfElectron,"Electrons");

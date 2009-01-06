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
// $Id: FWElectronRPZProxyBuilder.cc,v 1.5 2008/12/05 21:00:00 chrjones Exp $
//

// system include files
#include "TColor.h"
#include "TROOT.h"
#include "TEveTrack.h"
#include "TEveCompound.h"
#include "TEveTrackPropagator.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"


// user include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Candidates/interface/prepareSimpleTrack.h"
#include "Fireworks/Tracks/interface/prepareTrack.h"
//#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Electrons/interface/makeSuperCluster.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/src/CmsShowMain.h"


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
   FWEvePtr<TEveTrackPropagator> m_propagator;   

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
FWElectronRPZProxyBuilder::FWElectronRPZProxyBuilder() :
m_propagator( new TEveTrackPropagator)

{
   m_propagator->SetMagField( - CmsShowMain::getMagneticField() );
   m_propagator->SetMaxR(123.0);
   m_propagator->SetMaxZ(300.0);
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
   m_propagator->SetMagField( - CmsShowMain::getMagneticField() );
   TEveTrack* track(0);
   if ( iData.gsfTrack().isAvailable() )
      track = fireworks::prepareTrack( *(iData.gsfTrack()),
				      m_propagator.get(), 
				      &oItemHolder, 
				      item()->defaultDisplayProperties().color() );
   else
      track = fireworks::prepareSimpleTrack( iData,
					    m_propagator.get(), 
					    &oItemHolder, 
					    item()->defaultDisplayProperties().color() );
   track->MakeTrack();
   oItemHolder.AddElement(track);   
}

void 
FWElectronRPZProxyBuilder::buildRhoZ(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   fireworks::makeRhoZSuperCluster(*item(),
                                   iData.superCluster(),
                                   iData.phi(),
                                   oItemHolder);
   m_propagator->SetMagField( - CmsShowMain::getMagneticField() );
   TEveTrack* track(0);
   if ( iData.gsfTrack().isAvailable() )
      track = fireworks::prepareTrack( *(iData.gsfTrack()),
				      m_propagator.get(), 
				      &oItemHolder, 
				      item()->defaultDisplayProperties().color() );
   else
      track = fireworks::prepareSimpleTrack( iData,
					    m_propagator.get(), 
					    &oItemHolder, 
					    item()->defaultDisplayProperties().color() );
   track->MakeTrack();
   oItemHolder.AddElement(track);
   
}

//
// static member functions
//
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWElectronRPZProxyBuilder,reco::GsfElectron,"Electrons");

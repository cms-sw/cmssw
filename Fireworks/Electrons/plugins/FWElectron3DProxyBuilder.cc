// -*- C++ -*-
//
// Package:     Electrons
// Class  :     FWElectron3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWElectron3DProxyBuilder.cc,v 1.2 2009/01/06 20:07:48 chrjones Exp $
//

// system include files
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

// user include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "TEveTrackPropagator.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Candidates/interface/prepareSimpleTrack.h"
#include "Fireworks/Tracks/interface/prepareTrack.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "TEveTrack.h"

class FWElectron3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::GsfElectron> {

public:
   FWElectron3DProxyBuilder();
   //virtual ~FWElectron3DProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWElectron3DProxyBuilder(const FWElectron3DProxyBuilder&); // stop default

   const FWElectron3DProxyBuilder& operator=(const FWElectron3DProxyBuilder&); // stop default

   virtual void build(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

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
FWElectron3DProxyBuilder::FWElectron3DProxyBuilder() :
   m_propagator( new TEveTrackPropagator)
{
   m_propagator->SetMagField( -CmsShowMain::getMagneticField() );
   m_propagator->SetMaxR(123.0);
   m_propagator->SetMaxZ(300.0);

}

// FWElectron3DProxyBuilder::FWElectron3DProxyBuilder(const FWElectron3DProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

//FWElectron3DProxyBuilder::~FWElectron3DProxyBuilder()
//{
//}

//
// assignment operators
//
// const FWElectron3DProxyBuilder& FWElectron3DProxyBuilder::operator=(const FWElectron3DProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWElectron3DProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWElectron3DProxyBuilder::build(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   // make sure we use current magnetic field
   m_propagator->SetMagField( -CmsShowMain::getMagneticField() );
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
   oItemHolder.AddElement( track );
}


//
// const member functions
//

//
// static member functions
//

REGISTER_FW3DDATAPROXYBUILDER(FWElectron3DProxyBuilder,std::vector<reco::GsfElectron>,"Electrons");

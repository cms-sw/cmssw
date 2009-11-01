// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWTrack3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 25 14:42:13 EST 2008
// $Id: FWTrack3DProxyBuilder.cc,v 1.3 2009/01/06 21:38:40 chrjones Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "DataFormats/TrackReco/interface/Track.h"

// user include files
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "Fireworks/Core/interface/FW3DDataProxyBuilderFactory.h"

#include "Fireworks/Core/src/CmsShowMain.h"

#include "Fireworks/Tracks/interface/prepareTrack.h"

class FWTrack3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::Track> {

public:
   FWTrack3DProxyBuilder();
   virtual ~FWTrack3DProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWTrack3DProxyBuilder(const FWTrack3DProxyBuilder&); // stop default

   const FWTrack3DProxyBuilder& operator=(const FWTrack3DProxyBuilder&); // stop default

   void build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
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
FWTrack3DProxyBuilder::FWTrack3DProxyBuilder() :
   m_propagator( new TEveTrackPropagator)
{
   m_propagator->SetMagField( -CmsShowMain::getMagneticField() );
   m_propagator->SetMaxR(123.0);
   m_propagator->SetMaxZ(300.0);

}

// FWTrack3DProxyBuilder::FWTrack3DProxyBuilder(const FWTrack3DProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWTrack3DProxyBuilder::~FWTrack3DProxyBuilder()
{
}

//
// assignment operators
//
// const FWTrack3DProxyBuilder& FWTrack3DProxyBuilder::operator=(const FWTrack3DProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWTrack3DProxyBuilder temp(rhs);
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
FWTrack3DProxyBuilder::build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   if(CmsShowMain::isAutoField()) {
      if ( fabs( iData.eta() ) < 2.0 && iData.pt() > 1 ) {
         double estimate = fw::estimate_field(iData);
         if ( estimate >= 0 ) {
            bool fieldIsOn = CmsShowMain::getMagneticField() > 0;
            bool measuredFieldIsOn = estimate > 2.0;
            if(fieldIsOn != measuredFieldIsOn) {
               CmsShowMain::guessFieldIsOn(measuredFieldIsOn);
               m_propagator->SetMagField( -CmsShowMain::getMagneticField() );
            }
         }
      }
   }

   TEveTrack* trk = fireworks::prepareTrack( iData, m_propagator.get(), &oItemHolder, item()->defaultDisplayProperties().color() );
   trk->MakeTrack();
   oItemHolder.AddElement( trk );
}

//
// static member functions
//
REGISTER_FW3DDATAPROXYBUILDER(FWTrack3DProxyBuilder,reco::Track,"Tracks");

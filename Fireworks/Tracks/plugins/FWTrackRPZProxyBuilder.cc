// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWTrackRPZProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 25 14:42:13 EST 2008
// $Id: FWTrackRPZProxyBuilder.cc,v 1.5 2009/08/21 16:57:50 dmytro Exp $
//

// system include files
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "DataFormats/TrackReco/interface/Track.h"

// user include files
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/Core/interface/FWRPZSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilderBaseFactory.h"

#include "Fireworks/Core/src/CmsShowMain.h"

#include "Fireworks/Tracks/interface/prepareTrack.h"
#include "Fireworks/Tracks/interface/CmsMagField.h"
class FWTrackRPZProxyBuilder : public FWRPZSimpleProxyBuilderTemplate<reco::Track> {

public:
   FWTrackRPZProxyBuilder();
   virtual ~FWTrackRPZProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWTrackRPZProxyBuilder(const FWTrackRPZProxyBuilder&); // stop default

   const FWTrackRPZProxyBuilder& operator=(const FWTrackRPZProxyBuilder&); // stop default

   void build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   // ---------- member data --------------------------------

   FWEvePtr<TEveTrackPropagator> m_propagator;
   CmsMagField* m_cmsMagField;
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
FWTrackRPZProxyBuilder::FWTrackRPZProxyBuilder() :
   m_propagator( new TEveTrackPropagator),
   m_cmsMagField( new CmsMagField)
{
   m_cmsMagField->setReverseState( true );
   m_propagator->SetMagFieldObj( m_cmsMagField );
   m_propagator->SetMaxR(850);
   m_propagator->SetMaxZ(1100);
}

// FWTrackRPZProxyBuilder::FWTrackRPZProxyBuilder(const FWTrackRPZProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWTrackRPZProxyBuilder::~FWTrackRPZProxyBuilder()
{
}

//
// assignment operators
//
// const FWTrackRPZProxyBuilder& FWTrackRPZProxyBuilder::operator=(const FWTrackRPZProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWTrackRPZProxyBuilder temp(rhs);
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
FWTrackRPZProxyBuilder::build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   if(CmsShowMain::isAutoField()) {
      if ( fabs( iData.eta() ) < 2.0 && iData.pt() > 1 ) {
         double estimate = fw::estimate_field(iData);
         if ( estimate >= 0 ) {
	    CmsShowMain::guessFieldIsOn(estimate > 2.0);
	    m_cmsMagField->setMagnetState( CmsShowMain::getMagneticField() > 0 );
         }
      }
   }

   if ( iData.extra().isAvailable() ){
     m_propagator->SetMaxR(850);
     m_propagator->SetMaxZ(1100);
   } else {
     m_propagator->SetMaxR(123);
     m_propagator->SetMaxZ(300);
   }     
   TEveTrack* trk = fireworks::prepareTrack( iData, m_propagator.get(), item()->defaultDisplayProperties().color() );
   trk->MakeTrack();
   oItemHolder.AddElement( trk );
}

//
// static member functions
//
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWTrackRPZProxyBuilder,reco::Track,"Tracks");

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
// $Id: FWTrackRPZProxyBuilder.cc,v 1.10 2009/08/26 22:23:08 dmytro Exp $
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

#include "Fireworks/Tracks/interface/TrackUtils.h"
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

   FWEvePtr<TEveTrackPropagator> m_defaultPropagator;
   FWEvePtr<TEveTrackPropagator> m_trackerPropagator;
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
   m_defaultPropagator( new TEveTrackPropagator),
   m_trackerPropagator( new TEveTrackPropagator),
   m_cmsMagField( new CmsMagField)
{
   m_cmsMagField->setReverseState( true );
   
   m_defaultPropagator->SetMagFieldObj( m_cmsMagField );
   m_defaultPropagator->SetStepper(TEveTrackPropagator::kRungeKutta);
   m_defaultPropagator->IncRefCount();
   m_defaultPropagator->IncDenyDestroy();
   m_defaultPropagator->SetMaxR(850);
   m_defaultPropagator->SetMaxZ(1100);

   m_trackerPropagator->SetMagFieldObj( m_cmsMagField );
   m_trackerPropagator->SetStepper(TEveTrackPropagator::kRungeKutta);
   m_trackerPropagator->IncRefCount();
   m_trackerPropagator->IncDenyDestroy();
   m_trackerPropagator->SetMaxR(123);
   m_trackerPropagator->SetMaxZ(300);
}

FWTrackRPZProxyBuilder::~FWTrackRPZProxyBuilder()
{
   m_defaultPropagator->DecRefCount();
   m_trackerPropagator->DecRefCount();
}

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
   TEveTrackPropagator* propagator = m_defaultPropagator.get();
   if ( ! iData.extra().isAvailable() ) propagator = m_trackerPropagator.get();
   TEveTrack* trk = fireworks::prepareTrack( iData, propagator, item()->defaultDisplayProperties().color() );
   trk->MakeTrack();
   oItemHolder.AddElement( trk );
}

//
// static member functions
//
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWTrackRPZProxyBuilder,reco::Track,"Tracks");

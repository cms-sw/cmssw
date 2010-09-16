/*
 *  FWSimTrackProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/9/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "FWCore/Common/interface/EventBase.h"

#include "TEveTrack.h"
#include "TParticle.h"

class FWSimTrackProxyBuilder : public FWProxyBuilderBase
{
public:
   FWSimTrackProxyBuilder( void ) {} 
   virtual ~FWSimTrackProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWSimTrackProxyBuilder( const FWSimTrackProxyBuilder& );
   // Disable default assignment operator
   const FWSimTrackProxyBuilder& operator=( const FWSimTrackProxyBuilder& );

   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );

   void getVertices( void );
   std::vector<SimVertex> m_vertices;
};

void
FWSimTrackProxyBuilder::getVertices( void )
{
   edm::Handle<edm::SimVertexContainer> collection;
   const edm::EventBase *event = item()->getEvent();
   event->getByLabel( edm::InputTag( "g4SimHits" ), collection );
   
   if( collection.isValid())
   {
      for( std::vector<SimVertex>::const_iterator isimv = collection->begin(), isimvEnd = collection->end();
	   isimv != isimvEnd; ++isimv )
      {	
	 m_vertices.push_back( *isimv );
      }
   }
}

void
FWSimTrackProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
   const edm::SimTrackContainer* collection = 0;
   iItem->get( collection );

   if( 0 == collection )
   {
      return;
   }

   TEveTrackPropagator* propagator = context().getTrackPropagator();
   getVertices();
   int i = 0;
   for( std::vector<SimTrack>::const_iterator it = collection->begin(), end = collection->end(); it != end; ++it )
   {
     const SimTrack& iData = (*it);
     double vx = 0.0;
     double vy = 0.0;
     double vz = 0.0;
     double vt = 0.0;
     if(! iData.noVertex() && ! m_vertices.empty())
     {
       int vInd = iData.vertIndex();
       vx = ( m_vertices.at( vInd )).position().x() * 0.01;
       vy = ( m_vertices.at( vInd )).position().y() * 0.01;
       vz = ( m_vertices.at( vInd )).position().z() * 0.01;
       vt = ( m_vertices.at( vInd )).position().t();
     }
   
     TParticle* particle = new TParticle;
     particle->SetPdgCode( iData.type());
     particle->SetMomentum( iData.momentum().px(), iData.momentum().py(), iData.momentum().pz(), iData.momentum().e());
     particle->SetProductionVertex( vx, vy, vz, vt );
  
     TEveTrack* track = new TEveTrack( particle, ++i, propagator );
     switch( iData.type())
     {
     case  2112: //"neutron"
     case -2112: //"antineutron"
     case    22: //"gamma"
       track->SetLineStyle( 7 );
       break;
     default:
       break;
     }
   
     track->MakeTrack();
     setupAddElement( track, product );
   }
}

REGISTER_FWPROXYBUILDER( FWSimTrackProxyBuilder, edm::SimTrackContainer, "SimTracks", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );

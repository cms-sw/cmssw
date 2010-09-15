/*
 *  FWSimTrackProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/9/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TEveTrack.h"
#include "TParticle.h"

class FWSimTrackProxyBuilder : public FWSimpleProxyBuilderTemplate<SimTrack>
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

   void build( const SimTrack& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );

   void getVertices( void );
   std::vector<SimVertex> m_vertices;
};

void
FWSimTrackProxyBuilder::getVertices( void )
{
   std::vector<edm::Handle<edm::SimVertexContainer> > vertexCollections;
//    const edm::EventBase *event = item()->getEvent();
//    event->getManyByType( vertexCollections );
   
   if(! vertexCollections.empty())
   {
      for( std::vector<edm::Handle<edm::SimVertexContainer> >::iterator i = vertexCollections.begin(), iEnd = vertexCollections.end();
	   i != iEnd; ++i ) 
      {
	 const edm::SimVertexContainer& c = *(*i);

	 for( std::vector<SimVertex>::const_iterator isimv = c.begin(), isimvEnd = c.end();
	      isimv != isimvEnd; ++isimv )
	 {
	    m_vertices.push_back( *isimv );
	 }
      }
   }
}

void
FWSimTrackProxyBuilder::build( const SimTrack& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
   double vx = 0.0;
   double vy = 0.0;
   double vz = 0.0;
   double vt = 0.0;
   if(! iData.noVertex() && ! m_vertices.empty())
   {
      int vInd = iData.vertIndex();
      // FIXME: get SimTrack vertex from cached vertices
      vx = ( m_vertices.at( vInd )).position().x() * 0.01;
      vy = ( m_vertices.at( vInd )).position().y() * 0.01;
      vz = ( m_vertices.at( vInd )).position().z() * 0.01;
      vt = ( m_vertices.at( vInd )).position().t();
   }
   
   TParticle* particle = new TParticle;
   particle->SetPdgCode( iData.type());
   particle->SetMomentum( iData.momentum().px(), iData.momentum().py(), iData.momentum().pz(), iData.momentum().e());
   particle->SetProductionVertex( vx, vy, vz, vt );

   TEveTrackPropagator* propagator = context().getTrackPropagator();
  
   TEveTrack* track = new TEveTrack( particle, iIndex, propagator );
   setupAddElement( track, &oItemHolder );
}

REGISTER_FWPROXYBUILDER( FWSimTrackProxyBuilder, SimTrack, "SimTracks", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );

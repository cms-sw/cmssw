// -*- C++ -*-
// $Id: FWSecVertexRPZProxyBuilder.cc,v 1.1 2009/08/29 21:00:18 dmytro Exp $
//

// include files

#include "TEvePointSet.h"
#include "TEveTrack.h"

#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

class FWSecVertexRPZProxyBuilder : public FWRPZDataProxyBuilder
{   
public:
   FWSecVertexRPZProxyBuilder() {}
   virtual ~FWSecVertexRPZProxyBuilder(){}
   
   REGISTER_PROXYBUILDER_METHODS();
 
private:
   virtual void build (const FWEventItem* item, TEveElementList** product);

   FWSecVertexRPZProxyBuilder(const FWSecVertexRPZProxyBuilder&); // stop default
   const FWSecVertexRPZProxyBuilder& operator=(const FWSecVertexRPZProxyBuilder&); // stop default 
};

void 
FWSecVertexRPZProxyBuilder::build(const FWEventItem* item, TEveElementList** product)
{
   const reco::SecondaryVertexTagInfo *vertices;
   item->get(vertices);

   TEveElementList *  oItemHolder = *product;
   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetMainColor(item->defaultDisplayProperties().color());
   for(unsigned int i=0; i < vertices->nVertices(); i++)
   {
      const reco::Vertex & v = vertices->secondaryVertex(i);
      pointSet->SetNextPoint( v.x(), v.y(), v.z() );
      for(reco::Vertex::trackRef_iterator it = v.tracks_begin() ;
          it != v.tracks_end()  ; ++it)
      {
         const reco::Track & track = *it->get();
         TEveRecTrack t;
         t.fBeta = 1.;
         t.fV = TEveVector(track.vx(), track.vy(), track.vz());
         t.fP = TEveVector(track.px(), track.py(), track.pz());
         t.fSign = track.charge();
         TEveTrack* trk = new TEveTrack(&t, item->context().getTrackPropagator());
         trk->SetMainColor(item->defaultDisplayProperties().color());
         trk->MakeTrack();
         oItemHolder->AddElement( trk );
      }

   }
   oItemHolder->AddElement( pointSet );
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWSecVertexRPZProxyBuilder,reco::SecondaryVertexTagInfo,"SecVertex");

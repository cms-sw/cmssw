// -*- C++ -*-
// $Id: FWRPZProxyBuilder.template,v 1.1 2008/12/10 13:58:53 dmytro Exp $
//

// include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

#include "TEvePointSet.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "DataFormats/TrackReco/interface/Track.h"

class FWSecVertexRPZProxyBuilder : public FWRPZ2DSimpleProxyBuilderTemplate<reco::SecondaryVertexTagInfo>  {
   
public:
   FWSecVertexRPZProxyBuilder():
    m_propagator( new TEveTrackPropagator)
{
   m_propagator->SetMagField( - CmsShowMain::getMagneticField() );
   m_propagator->SetMaxR(123.0);
   m_propagator->SetMaxZ(300.0);
}
   virtual ~FWSecVertexRPZProxyBuilder(){}
   
   REGISTER_PROXYBUILDER_METHODS();
 
private:
   FWEvePtr<TEveTrackPropagator> m_propagator;
   FWSecVertexRPZProxyBuilder(const FWSecVertexRPZProxyBuilder&); // stop default
   
   const FWSecVertexRPZProxyBuilder& operator=(const FWSecVertexRPZProxyBuilder&); // stop default

 
   void buildRhoPhi(const reco::SecondaryVertexTagInfo& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   void buildRhoZ(const reco::SecondaryVertexTagInfo& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
};

void 
FWSecVertexRPZProxyBuilder::buildRhoPhi(const reco::SecondaryVertexTagInfo& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
    TEvePointSet* pointSet = new TEvePointSet();
    pointSet->SetMainColor(item()->defaultDisplayProperties().color());
    for(unsigned int i=0;i<iData.nVertices();i++)
    {
      const reco::Vertex & v = iData.secondaryVertex(i);
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
      TEveTrack* trk = new TEveTrack(&t,m_propagator.get());
      trk->SetMainColor(item()->defaultDisplayProperties().color());
      trk->MakeTrack();
      oItemHolder.AddElement( trk );
      }

    }
    oItemHolder.AddElement( pointSet );

}

void 
FWSecVertexRPZProxyBuilder::buildRhoZ(const reco::SecondaryVertexTagInfo& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
    TEvePointSet* pointSet = new TEvePointSet();
    pointSet->SetMainColor(item()->defaultDisplayProperties().color());
    for(unsigned int i=0;i<iData.nVertices();i++)
    {
      const reco::Vertex & v = iData.secondaryVertex(i);
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
      TEveTrack* trk = new TEveTrack(&t,m_propagator.get());
      trk->SetMainColor(item()->defaultDisplayProperties().color());
      trk->MakeTrack();
      oItemHolder.AddElement( trk );
      }

    }
    oItemHolder.AddElement( pointSet );

}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWSecVertexRPZProxyBuilder,reco::SecondaryVertexTagInfo,"SecVertex");

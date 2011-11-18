// -*- C++ -*-
//
// Package:     Vertexs
// Class  :     FWVertexProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWVertexProxyBuilder.cc,v 1.6 2011/08/11 03:39:51 amraktad Exp $
//
// user include files// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWParameters.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TEvePointSet.h"

#include "TMatrixDEigen.h"
#include "TMatrixDSym.h"
#include "TDecompSVD.h"
#include "TVectorD.h"
#include "TEveTrans.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveBoxSet.h"
#include "TGeoSphere.h"
#include "TEveGeoNode.h"
#include "TEveVSDStructs.h"

class FWVertexProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Vertex> 
{
public:
   FWVertexProxyBuilder() {}
   virtual ~FWVertexProxyBuilder() {}

   virtual void setItem(const FWEventItem* iItem)
   {
      FWProxyBuilderBase::setItem(iItem);
      iItem->getConfig()->assertParam("Draw Tracks", false);
   }

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWVertexProxyBuilder(const FWVertexProxyBuilder&); // stop default
   const FWVertexProxyBuilder& operator=(const FWVertexProxyBuilder&); // stop default

   virtual void build(const reco::Vertex& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*);
};

//
// member functions
//
void
FWVertexProxyBuilder::build(const reco::Vertex& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   const reco::Vertex & v = iData;

   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetMainColor(item()->defaultDisplayProperties().color());
   pointSet->SetNextPoint( v.x(), v.y(), v.z() );
   oItemHolder.AddElement( pointSet );

   if ( item()->getConfig()->value<bool>("Draw Tracks")) 
   {
      // do we need this stuff?
      TGeoSphere * sphere = new TGeoSphere(0, 0.002); //would that leak?
      TEveGeoShape * shape = new TEveGeoShape();
      sphere->SetBoxDimensions(2.5,2.5,2.5);
      shape->SetShape(sphere);
      shape->SetMainColor(item()->defaultDisplayProperties().color());
      shape->SetMainTransparency(10);

      TEveTrans & t =   shape->RefMainTrans();
      reco::Vertex::Error e= v.error();
      TMatrixDSym m(3);
      for(int i=0;i<3;i++)
         for(int j=0;j<3;j++)
         {
            m(i,j) = e(i,j);
         }
      TMatrixDEigen eig(m);
      TDecompSVD svd(m);
      TMatrixD mm = svd.GetU();
      //   TMatrixD mm =  eig.GetEigenVectors().Print();
      for(int i=0;i<3;i++)
         for(int j=0;j<3;j++)
         {
            t(i+1,j+1) = mm(i,j);
         }

      TVectorD vv ( eig.GetEigenValuesRe())   ;
      t.Scale(sqrt(vv(0))*1000.,sqrt(vv(1))*1000.,sqrt(vv(2))*1000.);
      t.SetPos(v.x(),v.y(),v.z());
      oItemHolder.AddElement(shape);
      for(reco::Vertex::trackRef_iterator it = v.tracks_begin() ;
          it != v.tracks_end()  ; ++it)
      {
         float w = v.trackWeight(*it);
         if (w < 0.5) continue;

         const reco::Track & track = *it->get();
         TEveRecTrack t;
         t.fBeta = 1.;
         t.fV = TEveVector(track.vx(), track.vy(), track.vz());
         t.fP = TEveVector(track.px(), track.py(), track.pz());
         t.fSign = track.charge();
         TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator());
         trk->SetMainColor(item()->defaultDisplayProperties().color());
         trk->MakeTrack();
         oItemHolder.AddElement( trk );
      }
   }
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWVertexProxyBuilder, reco::Vertex, "Vertices", FWViewType::k3DBit | FWViewType::kAllRPZBits);

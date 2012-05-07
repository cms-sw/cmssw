// -*- C++ -*-
//
// Package:     Vertexs
// Class  :     FWVertexWithTracksProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWVertexWithTracksProxyBuilder.cc,v 1.1 2011/03/21 14:58:08 arizzi Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
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

// include files
#include "Fireworks/Core/interface/FWEvePtr.h"
#include "Fireworks/Core/src/CmsShowMain.h"

#include "DataFormats/TrackReco/interface/Track.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"

//class FWVertexProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Vertex> {

class FWVertexWithTracksProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Vertex> {

public:
   FWVertexWithTracksProxyBuilder();
   //virtual ~FWVertexWithTracksProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWVertexWithTracksProxyBuilder(const FWVertexWithTracksProxyBuilder&); // stop default

   const FWVertexWithTracksProxyBuilder& operator=(const FWVertexWithTracksProxyBuilder&); // stop default

   virtual void build(const reco::Vertex& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*) ;

   // ---------- member data --------------------------------
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
FWVertexWithTracksProxyBuilder::FWVertexWithTracksProxyBuilder()
{
}


//
// member functions
//
void
FWVertexWithTracksProxyBuilder::build(const reco::Vertex& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*) 
{

  TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetMainColor(item()->defaultDisplayProperties().color());
//   for(unsigned int i=0;i<iData.nVertices();i++)
//   {
      const reco::Vertex & v = iData;
      // do we need this stuff?
      TGeoSphere * sphere = new TGeoSphere(0, 0.002); //would that leak?
      TGeoTranslation position(v.x(), v.y(), v.z() );
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
      pointSet->SetNextPoint( v.x(), v.y(), v.z() );
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

//   }
   oItemHolder.AddElement( pointSet );




}


//
// const member functions
//

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWVertexWithTracksProxyBuilder, reco::Vertex, "VerticesWithTracks", FWViewType::k3DBit | FWViewType::kAllRPZBits);

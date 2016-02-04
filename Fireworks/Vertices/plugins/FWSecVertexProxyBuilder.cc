// -*- C++ -*-
// $Id: FWSecVertexProxyBuilder.cc,v 1.5 2011/03/21 14:57:42 arizzi Exp $
//
#include <vector>

#include "TMatrixDEigen.h"
#include "TMatrixDSym.h"  
#include "TDecompSVD.h"
#include "TEveTrans.h"
#include "TEveTrack.h"
#include "TGeoSphere.h"
#include "TGeoMatrix.h"
#include "TEveGeoNode.h"
#include "TEveVSDStructs.h"

// include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

class FWSecVertexProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::SecondaryVertexTagInfo>  {
   
public:
   FWSecVertexProxyBuilder(){}
   virtual ~FWSecVertexProxyBuilder(){}
  
   REGISTER_PROXYBUILDER_METHODS();
 
private:
   FWSecVertexProxyBuilder(const FWSecVertexProxyBuilder&); // stop default
   const FWSecVertexProxyBuilder& operator=(const FWSecVertexProxyBuilder&); // stop default
   
   // ---------- member data --------------------------------
   void build(const reco::SecondaryVertexTagInfo& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);
};

void 
FWSecVertexProxyBuilder::build(const reco::SecondaryVertexTagInfo& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*) 
{
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetMainColor(item()->defaultDisplayProperties().color());
   for(unsigned int i=0;i<iData.nVertices();i++)
   {
      const reco::Vertex & v = iData.secondaryVertex(i);
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

      setupAddElement(shape, &oItemHolder);

      pointSet->SetNextPoint( v.x(), v.y(), v.z() );

      for(reco::Vertex::trackRef_iterator it = v.tracks_begin(), itEnd = v.tracks_end(); 
          it != itEnd; ++it)
      {
         const reco::Track & track = *it->get();
         TEveRecTrack t;
         t.fBeta = 1.;
         t.fV = TEveVector(track.vx(), track.vy(), track.vz());
         t.fP = TEveVector(track.px(), track.py(), track.pz());
         t.fSign = track.charge();
         TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator());
         trk->SetMainColor(item()->defaultDisplayProperties().color());
         trk->MakeTrack();
         setupAddElement(trk, &oItemHolder);
      }
   }
   setupAddElement(pointSet, &oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWSecVertexProxyBuilder, reco::SecondaryVertexTagInfo, "SecVertex", FWViewType::k3DBit | FWViewType::kAllRPZBits);

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
// $Id: FWVertexProxyBuilder.cc,v 1.10 2012/03/01 05:11:52 amraktad Exp $
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
#include "TEveStraightLineSet.h"
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
      if (iItem)
      {
         iItem->getConfig()->assertParam("Draw Tracks", false);
         iItem->getConfig()->assertParam("Draw Ellipse", false);
         iItem->getConfig()->assertParam("Scale Ellipse", 3l, 1l, 100l);
      }
   }
   
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWVertexProxyBuilder(const FWVertexProxyBuilder&); // stop default
   const FWVertexProxyBuilder& operator=(const FWVertexProxyBuilder&); // stop default

   // virtual void build(const reco::Vertex& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*);

   virtual bool haveSingleProduct() const { return false; }
   virtual void buildViewType(const reco::Vertex& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*);

};

//
// member functions
//

namespace 
{
  TEveStraightLineSet* make_ellipse(double* e)
  {
    TEveStraightLineSet* ls = new TEveStraightLineSet("Ellipse");
    const Int_t   N = 32;
    const Float_t S = 2*TMath::Pi()/N;

    Float_t a = e[0], b = e[1];
    for (Int_t i = 0; i<N; i++) {
      ls->AddLine(a*TMath::Cos(i*S)  , b*TMath::Sin(i*S)  , 0,
                  a*TMath::Cos(i*S+S), b*TMath::Sin(i*S+S), 0);
    }

    a = e[0]; b = e[2];
    for (Int_t i = 0; i<N; i++) {
      ls->AddLine(a*TMath::Cos(i*S)  , 0, b*TMath::Sin(i*S),
                  a*TMath::Cos(i*S+S), 0, b*TMath::Sin(i*S+S));
    }
    a = e[1]; b = e[2];
    for (Int_t i = 0; i<N; i++) {
      ls->AddLine(0, a*TMath::Cos(i*S)  ,  b*TMath::Sin(i*S),
                  0, a*TMath::Cos(i*S+S),  b*TMath::Sin(i*S+S));
    }


    ls->SetLineWidth(2);
    return ls;
  }
}

void
FWVertexProxyBuilder::buildViewType(const reco::Vertex& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*)
{
   const reco::Vertex & v = iData;
  
   // marker
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetNextPoint( v.x(), v.y(), v.z() );  
   setupAddElement(pointSet, &oItemHolder);
  

   // ellipse
   if ( item()->getConfig()->value<bool>("Draw Ellipse"))
   {
      double ellipseScale = 1.;
      if ( item()->getConfig()->value<long>("Scale Ellipse")){
         ellipseScale = item()->getConfig()->value<long>("Scale Ellipse");
      }

      if(type == FWViewType::kRhoZ )
      {

         reco::Vertex::Error e= v.error();
         TMatrixDSym m(3);
         for(int i=1;i<3;i++)
            for(int j=1;j<3;j++)
            {
               m(i,j) = e(i,j);
            }
         //m.Print();

         TMatrixDEigen eig(m);
         TDecompSVD svd(m);
         // svd.GetU().Print();

         TVectorD vv ( eig.GetEigenValuesRe())   ;
         // vv.Print();        

         // build line-set
         double erng[] = {sqrt(vv(0))*ellipseScale,sqrt(vv(1))*ellipseScale,sqrt(vv(2))*ellipseScale};
         TEveStraightLineSet* sl = make_ellipse(&erng[0]);
         TEveTrans & t =   sl->RefMainTrans();
         {
            TMatrixD mm = svd.GetU();
            for(int i=0;i<3;i++)
               for(int j=0;j<3;j++)
               {
                  t(i+1,j+1) = mm(i,j);
               }
         }
         t.SetPos(v.x(),v.y(),v.z());
         setupAddElement(sl, &oItemHolder);
      }
      else 
      {
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

         TEveTrans  t;
         for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
            {
               t(i+1,j+1) = mm(i,j);
            }

         t.SetPos(v.x(),v.y(),v.z());
         TVectorD vv ( eig.GetEigenValuesRe());
        
         // line-set
         double erng[] = {sqrt(vv(0))*ellipseScale,sqrt(vv(1))*ellipseScale,sqrt(vv(2))*ellipseScale};         
         TEveStraightLineSet* sl = make_ellipse(&erng[0]);
         setupAddElement(sl, &oItemHolder);
         sl->SetTransMatrix(t.Array());

         // sphere commented out, problems rendering in 3D
         /*
         double s0 = sqrt(vv(0));
         TGeoSphere * sphere = new TGeoSphere(0, ellipseScale*s0); //would that leak?
         TEveGeoShape * shape = new TEveGeoShape();
         shape->SetShape(sphere);
         shape->SetDrawFrame(false);
         shape->SetHighlightFrame(false);
         shape->SetMainColor(item()->defaultDisplayProperties().color());
         shape->SetFillColor(item()->defaultDisplayProperties().color());
         // shape->SetMainTransparency(90);

         sphere->SetBoxDimensions(s0, s0, s0); 
         // t.Scale(1,sqrt(vv(1))/s0,sqrt(vv(2))/s0);
         printf("%f \n", vv(0));
         shape->SetTransMatrix(t.Array());
         setupAddElement(shape, &oItemHolder);
         */

      }
   }
   // tracks
   if ( item()->getConfig()->value<bool>("Draw Tracks")) 
   {    
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
         setupAddElement(trk, &oItemHolder);
      }
   }
  
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWVertexProxyBuilder, reco::Vertex, "Vertices", FWViewType::k3DBit | FWViewType::kAllRPZBits);

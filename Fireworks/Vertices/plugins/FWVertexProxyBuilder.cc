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
// $Id: FWVertexProxyBuilder.cc,v 1.8 2011/11/21 08:37:14 amraktad Exp $
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
      }
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

namespace 
{
  TEveStraightLineSet* make_ellipse(double* e)
  {
    TEveStraightLineSet* ls = new TEveStraightLineSet("Ellipse");;
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
    return ls;
  }
}

void
FWVertexProxyBuilder::build(const reco::Vertex& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
  const reco::Vertex & v = iData;
  
  // marker
  TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
  TEvePointSet* pointSet = new TEvePointSet();
  pointSet->SetNextPoint( v.x(), v.y(), v.z() );  
  setupAddElement(pointSet, &oItemHolder);
  
  // ellipse
  if ( item()->getConfig()->value<bool>("Draw Ellipse")){
    reco::Vertex::Error vError= v.error();
    TMatrixDSym m(3);
    for(int i=0;i<3;i++) {
      for(int j=0;j<3;j++)
        m(i,j) = vError(i,j);
    }
    TMatrixDEigen eig(m);
    TDecompSVD svd(m);
    TVectorD vv ( eig.GetEigenValuesRe());
    double erng[] = {sqrt(vv(0))*2,sqrt(vv(1))*2,sqrt(vv(2))*2};
    
    TEveStraightLineSet* sl = make_ellipse(&erng[0]);    //    t.Scale(sqrt(vv(0))*1000.,sqrt(vv(1))*1000.,sqrt(vv(2))*1000.); //!!!!
    TEveTrans & t =   sl->RefMainTrans();
    
    {
      TMatrixD mm = svd.GetU();
      //   TMatrixD mm =  eig.GetEigenVectors().Print();
      for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
        {
          t(i+1,j+1) = mm(i,j);
        }
    }
    t.SetPos(v.x(),v.y(),v.z());
    setupAddElement(sl, &oItemHolder);
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

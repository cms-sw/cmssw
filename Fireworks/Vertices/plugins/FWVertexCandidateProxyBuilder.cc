// -*- C++ -*-
//
// Package:     Vertexs
// Class  :     FWVertexCandidateProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
//
// user include files// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWParameters.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Fireworks/Vertices/interface/TEveEllipsoid.h"
#include "Fireworks/Candidates/interface/CandidateUtils.h"

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

class FWVertexCandidateProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::VertexCompositePtrCandidate> 
{
public:
   FWVertexCandidateProxyBuilder() {}
   virtual ~FWVertexCandidateProxyBuilder() {}
   
   virtual void setItem(const FWEventItem* iItem) override
   {
      FWProxyBuilderBase::setItem(iItem);
      if (iItem)
      {
         iItem->getConfig()->assertParam("Draw Tracks", false);
         iItem->getConfig()->assertParam("Draw Pseudo Track", false);
         iItem->getConfig()->assertParam("Draw Ellipse", false);
         iItem->getConfig()->assertParam("Scale Ellipse",2l, 1l, 10l);
         iItem->getConfig()->assertParam("Ellipse Color Index",  6l, 0l, (long)context().colorManager()->numberOfLimitedColors());
      }
   }
   
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWVertexCandidateProxyBuilder(const FWVertexCandidateProxyBuilder&); // stop default
   const FWVertexCandidateProxyBuilder& operator=(const FWVertexCandidateProxyBuilder&); // stop default

   using FWSimpleProxyBuilderTemplate<reco::VertexCompositePtrCandidate> ::build;
   virtual void build(const reco::VertexCompositePtrCandidate& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*) override;

   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc) override;

};


void
FWVertexCandidateProxyBuilder::build(const reco::VertexCompositePtrCandidate& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*)
{
   const reco::VertexCompositePtrCandidate & v = iData;
  
   // marker
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetNextPoint( v.vx(), v.vy(), v.vz() );  
   setupAddElement(pointSet, &oItemHolder);
  

   // ellipse
   if ( item()->getConfig()->value<bool>("Draw Ellipse"))
   {
    
      TEveEllipsoid* eveEllipsoid = new TEveEllipsoid("Ellipsoid", Form("Ellipsoid %d", iIndex)); 

      eveEllipsoid->RefPos().Set(v.vx(),v.vy(),v.vz());

      reco::Vertex::Error e= v.error();      
      TMatrixDSym m(3);
      for(int i=0;i<3;i++)
         for(int j=0;j<3;j++)
         {
            m(i,j) = e(i,j);
            eveEllipsoid->RefEMtx()(i+1, j+1) =  e(i,j);
         }

      // external scaling
      double ellipseScale = 1.;
      if ( item()->getConfig()->value<long>("Scale Ellipse"))
         ellipseScale = item()->getConfig()->value<long>("Scale Ellipse");
     
      eveEllipsoid->SetScale(ellipseScale);

      // cache 3D extend used in eval bbox and render 3D
      TMatrixDEigen eig(m);
      TVectorD vv ( eig.GetEigenValuesRe());
      eveEllipsoid->RefExtent3D().Set(sqrt(vv(0))*ellipseScale,sqrt(vv(1))*ellipseScale,sqrt(vv(2))*ellipseScale); 

      eveEllipsoid->SetLineWidth(2);
      setupAddElement(eveEllipsoid, &oItemHolder);
      eveEllipsoid->SetMainTransparency(TMath::Min(100, 80 + item()->defaultDisplayProperties().transparency() / 5)); 
      
      
      
      Color_t color = item()->getConfig()->value<long>("Ellipse Color Index");
     // eveEllipsoid->SetFillColor(item()->defaultDisplayProperties().color());
     // eveEllipsoid->SetLineColor(item()->defaultDisplayProperties().color());    
      eveEllipsoid->SetMainColor(color + context().colorManager()->offsetOfLimitedColors());
   }

   // tracks
   if ( item()->getConfig()->value<bool>("Draw Tracks")) 
   {    
      for(unsigned int j=0;j<v.numberOfDaughters();j++)
      {
        const reco::Candidate * c =  v.daughter(j);
        std::cout << c << std::endl;
        TEveTrack* trk = fireworks::prepareCandidate( *c, context().getTrackPropagator() );

         trk->SetMainColor(item()->defaultDisplayProperties().color());
         trk->MakeTrack();
         setupAddElement(trk, &oItemHolder);
      }
   }
   if ( item()->getConfig()->value<bool>("Draw Pseudo Track"))
   {
      TEveRecTrack t;
      t.fBeta = 1.;
      t.fV = TEveVector(v.vx(),v.vy(),v.vz());
      t.fP = TEveVector(-v.p4().px(), -v.p4().py(), -v.p4().pz());
      t.fSign = 1;
      TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator());
      trk->SetLineStyle(7);
      trk->MakeTrack();
      setupAddElement(trk, &oItemHolder);
      
   }
}

void
FWVertexCandidateProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                     FWViewType::EType viewType, const FWViewContext* vc)
{
   increaseComponentTransparency(iId.index(), iCompound, "Ellipsoid", 80);
   TEveElement* el = iCompound->FindChild("Ellipsoid");
   if (el)
      el->SetMainColor(item()->getConfig()->value<long>("Ellipse Color Index") + context().colorManager()->offsetOfLimitedColors());
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWVertexCandidateProxyBuilder, reco::VertexCompositePtrCandidate, "CandVertices", FWViewType::k3DBit | FWViewType::kAllRPZBits);

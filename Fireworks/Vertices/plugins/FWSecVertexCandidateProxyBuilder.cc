// -*- C++ -*-
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

#include "Fireworks/Candidates/interface/CandidateUtils.h"


// include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"

class FWSecVertexCandidateProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::CandSecondaryVertexTagInfo>  {
   
public:
   FWSecVertexCandidateProxyBuilder(){}
   virtual ~FWSecVertexCandidateProxyBuilder(){}
  
   REGISTER_PROXYBUILDER_METHODS();
 
private:
   FWSecVertexCandidateProxyBuilder(const FWSecVertexCandidateProxyBuilder&); // stop default
   const FWSecVertexCandidateProxyBuilder& operator=(const FWSecVertexCandidateProxyBuilder&); // stop default
   
   // ---------- member data --------------------------------
   using FWSimpleProxyBuilderTemplate<reco::CandSecondaryVertexTagInfo>::build;
   void build(const reco::CandSecondaryVertexTagInfo& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) override;
};

void 
FWSecVertexCandidateProxyBuilder::build(const reco::CandSecondaryVertexTagInfo& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*) 
{
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetMainColor(item()->defaultDisplayProperties().color());
   for(unsigned int i=0;i<iData.nVertices();i++)
   {
      const reco::VertexCompositePtrCandidate & v = iData.secondaryVertex(i);
      // do we need this stuff?
      TGeoSphere * sphere = new TGeoSphere(0, 0.002); //would that leak?
      TGeoTranslation position(v.vx(), v.vy(), v.vz() );
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
      t.SetPos(v.vx(),v.vy(),v.vz());

      setupAddElement(shape, &oItemHolder);

      pointSet->SetNextPoint( v.vx(), v.vy(), v.vz() );

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
   setupAddElement(pointSet, &oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWSecVertexCandidateProxyBuilder, reco::CandSecondaryVertexTagInfo, "SecVertexCand", FWViewType::k3DBit | FWViewType::kAllRPZBits);

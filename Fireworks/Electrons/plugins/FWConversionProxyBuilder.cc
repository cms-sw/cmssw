// -*- C++ -*-
//
// Package:     Conversions
// Class  :     FWConversionProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWConversionProxyBuilder.cc,v 1.1 2011/02/25 19:31:15 fgolf Exp $
//
#include "TEveCompound.h"
#include "TEveLine.h"
#include "TEveScalableStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/Context.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Electrons/interface/makeSuperCluster.h" 

#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

////////////////////////////////////////////////////////////////////////////////
//
//   3D and RPZ proxy builder with shared track list
// 
////////////////////////////////////////////////////////////////////////////////

class FWConversionProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Conversion> {

public:
   FWConversionProxyBuilder() ;
   virtual ~FWConversionProxyBuilder();

   virtual bool haveSingleProduct() const { return false; }
   virtual void cleanLocal();

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWConversionProxyBuilder( const FWConversionProxyBuilder& ); // stop default
   const FWConversionProxyBuilder& operator=( const FWConversionProxyBuilder& ); // stop default
  
   virtual void buildViewType(const reco::Conversion& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*);

   TEveElementList* requestCommon();

   TEveElementList* m_common;
};


FWConversionProxyBuilder::FWConversionProxyBuilder():
   m_common(0)
{
   m_common = new TEveElementList( "common conversion scene" );
   m_common->IncDenyDestroy();
}

FWConversionProxyBuilder::~FWConversionProxyBuilder()
{
   m_common->DecDenyDestroy();
}

TEveElementList*
FWConversionProxyBuilder::requestCommon()
{
   if( m_common->HasChildren() == false )
   {
      for (int i = 0; i < static_cast<int>(item()->size()); ++i)
      {
         const reco::Conversion& conversion = modelData(i);
         TEveLine* line = new TEveLine(0);
         if (conversion.nTracks() == 2) {
			  if (fabs(conversion.zOfPrimaryVertexFromTracks()) < fireworks::Context::caloZ1())
				   line->SetNextPoint(0., 0., conversion.zOfPrimaryVertexFromTracks());
			  else
				   line->SetNextPoint(0., 0., 0.);
			  
			  float phi = conversion.pairMomentum().phi();
			  if (fabs(conversion.pairMomentum().eta()) < fireworks::Context::caloTransEta()) {
				   float radius = fireworks::Context::caloR1();
				   float z = radius/tan(conversion.pairMomentum().theta());
				   line->SetNextPoint(radius*cos(phi), radius*sin(phi), z);
			  }
			  else {
				   float z = fireworks::Context::caloZ1();
				   float radius = z*tan(conversion.pairMomentum().theta());
				   z *= (conversion.pairMomentum().eta()/fabs(conversion.pairMomentum().eta()));
				   line->SetNextPoint(radius*cos(phi), radius*sin(phi), z);
			  }
		 }
         else {
			  line->SetNextPoint(0., 0., 0.);
			  line->SetNextPoint(0., 0., 0.);
		 }

         setupElement( line );
         m_common->AddElement( line );
      }
   }

   return m_common;
}

void
FWConversionProxyBuilder::cleanLocal()
{
   m_common->DestroyElements();
}

void
FWConversionProxyBuilder::buildViewType(const reco::Conversion& conversion, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*)
{
   TEveElementList*   lines = requestCommon();
   TEveElement::List_i linIt = lines->BeginChildren();
   std::advance(linIt, iIndex);
   TEveLine* line = (TEveLine*)((*linIt)->CloneElement());
   TEveVector bvec = line->GetLineStart();
   TEveVector evec = line->GetLineEnd();
   if (bvec.Mag() != evec.Mag())
		setupAddElement(*linIt, &oItemHolder );
}

REGISTER_FWPROXYBUILDER( FWConversionProxyBuilder, reco::Conversion, "Conversions", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );

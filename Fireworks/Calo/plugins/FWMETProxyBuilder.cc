// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWMETProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWMETProxyBuilder.cc,v 1.18 2010/09/16 15:42:20 yana Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveScalableStraightLineSet.h"
#include "TGeoTube.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

#include "DataFormats/METReco/interface/MET.h"

////////////////////////////////////////////////////////////////////////////////
//
//   3D and RPZ proxy builder with shared MET shape
// 
////////////////////////////////////////////////////////////////////////////////

class FWMETProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWMETProxyBuilder() {}
   virtual ~FWMETProxyBuilder() {}

   virtual bool haveSingleProduct() const { return false; } // use buildViewType instead of buildView

   virtual bool havePerViewProduct(FWViewType::EType) const { return true; } // used energy scaling
   
   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMETProxyBuilder( const FWMETProxyBuilder& );    // stop default
   const FWMETProxyBuilder& operator=( const FWMETProxyBuilder& );    // stop default

   virtual void buildViewType(const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*);
};

void
FWMETProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   for (TEveElement::List_i i = parent->BeginChildren(); i!= parent->EndChildren(); ++i)
   {
      TEveElement* comp = (*i);
      for (TEveElement::List_i j = comp->BeginChildren(); j!= comp->EndChildren(); ++j)
      {
         TEveScalableStraightLineSet* ls = dynamic_cast<TEveScalableStraightLineSet*> (*j);
         if (ls ) 
         {
            ls->SetScale(vc->getEnergyScale("Calo")->getValToHeight());
            if (FWViewType::isProjected(type))
            {
               TEveProjected* proj = *ls->BeginProjecteds();
               proj->UpdateProjection();
            }
         }
      }
   }
}

void
FWMETProxyBuilder::buildViewType(const reco::MET& met, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext* vc)
{
   float r_ecal = context().caloR1();
   double phi  = met.phi();
   double size = met.et();

   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet( "energy" );
   marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
   marker->SetLineWidth( 2 );
   const double dx = 0.9*size*0.1;
   const double dy = 0.9*size*cos(0.1);
   marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0,
                    (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
   marker->AddLine( dx*sin(phi) + (dy+r_ecal)*cos(phi), -dx*cos(phi) + (dy+r_ecal)*sin(phi), 0,
                    (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
   marker->AddLine( -dx*sin(phi) + (dy+r_ecal)*cos(phi), dx*cos(phi) + (dy+r_ecal)*sin(phi), 0,
                    (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);

   marker->SetScale(vc->getEnergyScale("Calo")->getValToHeight());
   setupAddElement( marker, &oItemHolder );
      
   if( type == FWViewType::kRhoPhi )
   {
      double min_phi = phi-M_PI/36/2;
      double max_phi = phi+M_PI/36/2;

      TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
      TEveGeoShape *element = fireworks::getShape( "spread", new TGeoTubeSeg( r_ecal - 2, r_ecal, 1, min_phi*180/M_PI, max_phi*180/M_PI ), 0 );
      element->SetPickable( kTRUE );
      setupAddElement( element, &oItemHolder );
   }
   
   else if ( type == FWViewType::kRhoZ ) 
   {
      TEveScalableStraightLineSet* tip = new TEveScalableStraightLineSet( "tip" );
      tip->SetLineWidth(2);
      tip->SetScaleCenter(0., (phi>0 ? r_ecal : -r_ecal), 0);
      tip->AddLine(0., (phi>0 ? r_ecal+dy : -(r_ecal+dy) ), dx,
                   0., (phi>0 ? (r_ecal+size) : -(r_ecal+size)), 0 );
      tip->AddLine(0., (phi>0 ? r_ecal+dy : -(r_ecal+dy) ), -dx,
                   0., (phi>0 ? (r_ecal+size) : -(r_ecal+size)), 0 );
      tip->SetScale(vc->getEnergyScale("Calo")->getValToHeight());
      setupAddElement( tip, &oItemHolder );
   }   

}

REGISTER_FWPROXYBUILDER( FWMETProxyBuilder, reco::MET, "recoMET", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );

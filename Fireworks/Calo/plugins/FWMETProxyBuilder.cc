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
// $Id: FWMETProxyBuilder.cc,v 1.12 2010/05/07 16:54:02 amraktad Exp $
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
            ls->SetScale(vc->getEnergyScale());
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
   float r_ecal = fireworks::Context::s_ecalR;
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

   marker->SetScale(vc->getEnergyScale());
   setupAddElement( marker, &oItemHolder );
      
   if( type == FWViewType::kRhoPhi )
   {
      double min_phi = phi-M_PI/36/2;
      double max_phi = phi+M_PI/36/2;

      TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
      TEveGeoShape *element = fw::getShape( "spread", new TGeoTubeSeg( r_ecal - 1, r_ecal + 1, 1, min_phi*180/M_PI, max_phi*180/M_PI ), 0 );
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
      tip->SetScale(vc->getEnergyScale());
      setupAddElement( tip, &oItemHolder );
   }   

}

////////////////////////////////////////////////////////////////////////////////
//
//   GLIMPSE specific proxy builder
// 
////////////////////////////////////////////////////////////////////////////////


class FWMETGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWMETGlimpseProxyBuilder() {}
   virtual ~FWMETGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMETGlimpseProxyBuilder( const FWMETGlimpseProxyBuilder& );    // stop default
   const FWMETGlimpseProxyBuilder& operator=( const FWMETGlimpseProxyBuilder& );    // stop default

   virtual void build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*);
};

void 
FWMETGlimpseProxyBuilder::build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*) 
{
   double phi = iData.phi();
   double size = iData.et();
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet( "energy" );
   marker->SetLineWidth( 2 );
   marker->AddLine( 0, 0, 0, size*cos(phi), size*sin(phi), 0);
   marker->AddLine( size*0.9*cos(phi+0.03), size*0.9*sin(phi+0.03), 0, size*cos(phi), size*sin(phi), 0);
   marker->AddLine( size*0.9*cos(phi-0.03), size*0.9*sin(phi-0.03), 0, size*cos(phi), size*sin(phi), 0);
   setupAddElement( marker, &oItemHolder );
}

////////////////////////////////////////////////////////////////////////////////
//
//   LEGO specific proxy builder
// 
////////////////////////////////////////////////////////////////////////////////

class FWMETLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWMETLegoProxyBuilder() {}
   virtual ~FWMETLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMETLegoProxyBuilder( const FWMETLegoProxyBuilder& );    // stop default
   const FWMETLegoProxyBuilder& operator=( const FWMETLegoProxyBuilder& );    // stop default

   virtual void build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*);
};

void
FWMETLegoProxyBuilder::build( const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*) 
{
   TEveStraightLineSet* mainLine = new TEveStraightLineSet( "MET phi" );
   mainLine->AddLine(-5.191, iData.phi(), 0.1, 5.191, iData.phi(), 0.1 );
   setupAddElement( mainLine, &oItemHolder );

   double phi = iData.phi();
   phi = phi > 0 ? phi - M_PI : phi + M_PI;
   TEveStraightLineSet* secondLine = new TEveStraightLineSet( "MET opposite phi" );
   secondLine->SetLineStyle( 7 );
   secondLine->AddLine(-5.191, phi, 0.1, 5.191, phi, 0.1 );
   setupAddElement( secondLine, &oItemHolder );
}

REGISTER_FWPROXYBUILDER( FWMETProxyBuilder, reco::MET, "recoMET", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
REGISTER_FWPROXYBUILDER( FWMETGlimpseProxyBuilder, reco::MET, "recoMET", FWViewType::kGlimpseBit );
REGISTER_FWPROXYBUILDER( FWMETLegoProxyBuilder, reco::MET, "recoMET", FWViewType::kLegoBit );

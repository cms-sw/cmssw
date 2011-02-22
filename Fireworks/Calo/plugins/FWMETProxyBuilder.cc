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
// $Id: FWMETProxyBuilder.cc,v 1.25 2010/10/01 09:45:19 amraktad Exp $
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
#include "Fireworks/Calo/interface/scaleMarker.h"

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
 
   virtual void cleanLocal() { m_lines.clear(); }

   REGISTER_PROXYBUILDER_METHODS();

private:
 
   FWMETProxyBuilder( const FWMETProxyBuilder& );    // stop default
   const FWMETProxyBuilder& operator=( const FWMETProxyBuilder& );    // stop default

   virtual void buildViewType(const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*);
   
   std::vector<fireworks::scaleMarker> m_lines;
};

void
FWMETProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   typedef std::vector<fireworks::scaleMarker> Lines_t;
   FWViewEnergyScale* caloScale = vc->getEnergyScale("Calo");  

   // printf("MET %p -> %f\n", vc, caloScale->getValToHeight() );
   for (Lines_t::iterator i = m_lines.begin(); i!= m_lines.end(); ++ i)
   {
      if ( vc == (*i).m_vc )
      { 
         //    printf("lineset %p \n",(*i).m_ls );
         (*i).m_ls->SetScale(caloScale->getValToHeight()*(*i).m_et);

         TEveProjectable *pable = static_cast<TEveProjectable*>((*i).m_ls);
         for (TEveProjectable::ProjList_i j = pable->BeginProjecteds(); j != pable->EndProjecteds(); ++j)
         {
            (*j)->UpdateProjection();
         }
      }
   }
}
 
void
FWMETProxyBuilder::buildViewType(const reco::MET& met, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext* vc)
{
   if (type == FWViewType::kISpy)
      return;

   float r_ecal = context().caloR1();
   double phi  = met.phi();
   double size = 1.f;

   FWViewEnergyScale* caloScale = vc->getEnergyScale("Calo");   
        
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
   
   marker->SetScale(caloScale->getValToHeight()*met.et());
   m_lines.push_back(fireworks::scaleMarker(marker, met.energy(), met.et(), vc));  // register for scales
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
      
      m_lines.push_back(fireworks::scaleMarker(tip, met.energy(), met.et(), vc)); //register for scaes 

      tip->SetScale(caloScale->getValToHeight()*met.et());
      setupAddElement( tip, &oItemHolder );
   }   

}

REGISTER_FWPROXYBUILDER( FWMETProxyBuilder, reco::MET, "recoMET", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );

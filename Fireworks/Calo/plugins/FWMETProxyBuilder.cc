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
// $Id: FWMETProxyBuilder.cc,v 1.31 2011/02/03 15:15:12 amraktad Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TEveScalableStraightLineSet.h"
#include "TGeoTube.h"
#include "TMath.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Calo/interface/scaleMarker.h"

#include "DataFormats/METReco/interface/MET.h"

////////////////////////////////////////////////////////////////////////////////
//
//   RPZ proxy builder with shared MET shape
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
   FWViewEnergyScale* caloScale = vc->getEnergyScale();  

   //   printf("MET %s %p -> %f\n",  item()->name().c_str(),  vc, caloScale->getScaleFactor3D() );
   for (Lines_t::iterator i = m_lines.begin(); i!= m_lines.end(); ++ i)
   {
      if ( vc == (*i).m_vc )
      { 
         // printf("lineset %s  %p val %f ...%f\n", item()->name().c_str(), (*i).m_ls , (*i).m_et, caloScale->getScaleFactor3D()*(*i).m_et);
         float value = caloScale->getPlotEt() ? (*i).m_et : (*i).m_energy;      

         (*i).m_ls->SetScale(caloScale->getScaleFactor3D()*value);

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
   using namespace  TMath;
   double phi  = met.phi();
   double theta = met.theta();
   double size = 1.f;

   FWViewEnergyScale* caloScale = vc->getEnergyScale();   
        
   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet( "MET marker" );
   marker->SetLineWidth( 2 );

   
  
   if ( type == FWViewType::kRhoZ ) 
   {
      // body 
      double r0;
      if (TMath::Abs(met.eta()) < context().caloTransEta())
      {
         r0  = context().caloR1()/sin(theta);
      }
      else
      {
         r0  = context().caloZ1()/fabs(cos(theta));
      }
      marker->SetScaleCenter( 0., Sign(r0*sin(theta), phi), r0*cos(theta) );
      double r1 = r0 + 1;
      marker->AddLine( 0., Sign(r0*sin(theta), phi), r0*cos(theta),
                       0., Sign(r1*sin(theta), phi), r1*cos(theta) );

      // arrow pointer
      double r2 = r1 - 0.1;
      double dy = 0.05*size;
      marker->AddLine( 0., Sign(r2*sin(theta) + dy*cos(theta), phi), r2*cos(theta) -dy*sin(theta),
                       0., Sign(r1*sin(theta), phi), r1*cos(theta) );
      dy = -dy;
      marker->AddLine( 0., Sign(r2*sin(theta) + dy*cos(theta), phi), r2*cos(theta) -dy*sin(theta),
                       0., Sign(r1*sin(theta), phi), r1*cos(theta) );

      // segment  
      fireworks::addRhoZEnergyProjection( this, &oItemHolder, context().caloR1() -1, context().caloZ1() -1,
                                          theta - 0.04, theta + 0.04,
                                          phi );
   }
   else
   { 
      // body
      double r0 = context().caloR1();
      double r1 = r0 + 1;
      marker->SetScaleCenter( r0*cos(phi), r0*sin(phi), 0 );
      marker->AddLine( r0*cos(phi), r0*sin(phi), 0,
                       r1*cos(phi), r1*sin(phi), 0);
       
      // arrow pointer, xy  rotate offset point ..
      double r2 = r1 - 0.1;
      double dy = 0.05*size;

      marker->AddLine( r2*cos(phi) -dy*sin(phi), r2*sin(phi) + dy*cos(phi), 0,
                       r1*cos(phi), r1*sin(phi), 0);
      dy = -dy;
      marker->AddLine( r2*cos(phi) -dy*sin(phi), r2*sin(phi) + dy*cos(phi), 0,
                       r1*cos(phi), r1*sin(phi), 0);

      // segment
      double min_phi = phi-M_PI/36/2;
      double max_phi = phi+M_PI/36/2;
      TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
      TEveGeoShape *element = fireworks::getShape( "spread", new TGeoTubeSeg( r0 - 2, r0, 1, min_phi*180/M_PI, max_phi*180/M_PI ), 0 );
      element->SetPickable( kTRUE );
      setupAddElement( element, &oItemHolder );
   }

   marker->SetScale(caloScale->getScaleFactor3D()*(caloScale->getPlotEt() ? met.et() : met.energy()));
   setupAddElement( marker, &oItemHolder );

   // printf("add line %s  %f %f .... eta %f theta %f\n", item()->name().c_str(), met.et(), met.energy(), met.eta(), met.theta());
   m_lines.push_back(fireworks::scaleMarker(marker, met.et(), met.energy(), vc));  // register for scales

   context().voteMaxEtAndEnergy(met.et(), met.energy());
}

REGISTER_FWPROXYBUILDER( FWMETProxyBuilder, reco::MET, "recoMET", FWViewType::kAllRPZBits );

// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWJetProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWJetProxyBuilder.cc,v 1.27 2010/10/22 14:34:44 amraktad Exp $
//

#include "TEveJetCone.h"
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Calo/interface/makeEveJetCone.h"
#include "Fireworks/Calo/interface/scaleMarker.h"

#include "DataFormats/JetReco/interface/Jet.h"

class FWJetProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetProxyBuilder();
   virtual ~FWJetProxyBuilder();

   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
   virtual bool haveSingleProduct() const { return false; } // different view types
   virtual void cleanLocal();

   REGISTER_PROXYBUILDER_METHODS();
   
protected:
   virtual void buildViewType(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*);


   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);

   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);

private:
   FWJetProxyBuilder( const FWJetProxyBuilder& ); // stop default
   const FWJetProxyBuilder& operator=( const FWJetProxyBuilder& ); // stop default

   TEveElementList* requestCommon();
   TEveElementList* m_common;

   std::vector<fireworks::scaleMarker> m_lines;
};

//______________________________________________________________________________
FWJetProxyBuilder::FWJetProxyBuilder():
   m_common(0)
{
   m_common = new TEveElementList( "common electron scene" );
   m_common->IncDenyDestroy();
}

FWJetProxyBuilder::~FWJetProxyBuilder()
{
   m_common->DecDenyDestroy();
}

TEveElementList*
FWJetProxyBuilder::requestCommon()
{
   if( m_common->HasChildren() == false )
   {
      for (int i = 0; i < static_cast<int>(item()->size()); ++i)
      {
         TEveJetCone* cone = fireworks::makeEveJetCone(modelData(i), context());

         m_common->AddElement(cone);
         cone->SetFillColor(item()->defaultDisplayProperties().color());
         cone->SetLineColor(item()->defaultDisplayProperties().color());

      }
   }
   return m_common;
}

void
FWJetProxyBuilder::buildViewType(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext* vc)
{
   // add cone from shared pool
   TEveElementList*    cones = requestCommon();
   TEveElement::List_i coneIt = cones->BeginChildren();
   std::advance(coneIt, iIndex);

   const FWDisplayProperties &dp = item()->defaultDisplayProperties();
   setupAddElement( *coneIt, &oItemHolder );
   (*coneIt)->SetMainTransparency(TMath::Min(100, 80 + dp.transparency() / 5)); 

   // scale markers in projected views
   if (FWViewType::isProjected(type))
   {
      TEveScalableStraightLineSet* marker =new TEveScalableStraightLineSet("jet lineset");
      float size = 1.f; // values are saved in scale
      double theta = iData.theta();
      double phi = iData.phi();

      if ( type == FWViewType::kRhoZ )
      {  
         static const float_t offr = 4;
         float r_ecal = context().caloR1() + offr;
         float z_ecal = context().caloZ1() + offr/tan(context().caloTransAngle());

         double r(0);
         if ( theta < context().caloTransAngle() || M_PI-theta < context().caloTransAngle())
         {
            z_ecal = context().caloZ2() + offr/tan(context().caloTransAngle());
            r = z_ecal/fabs(cos(theta));
         }
         else
         {
            r = r_ecal/sin(theta);
         }

         marker->SetScaleCenter( 0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );
         marker->AddLine( 0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
                          0., (phi>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
      }
      else
      {
         float ecalR = context().caloR1() + 4;
         marker->SetScaleCenter(ecalR*cos(phi), ecalR*sin(phi), 0);
         marker->AddLine(ecalR*cos(phi), ecalR*sin(phi), 0, (ecalR+size)*cos(phi), (ecalR+size)*sin(phi), 0);
      }

      marker->SetLineWidth(4);  

      marker->SetLineColor(dp.color());
      FWViewEnergyScale* caloScale = vc->getEnergyScale("Calo");    
      marker->SetScale(caloScale->getValToHeight()*(caloScale->getPlotEt() ?  iData.et() : iData.energy()));
      setupAddElement( marker, &oItemHolder );
      m_lines.push_back(fireworks::scaleMarker(marker, iData.et(), iData.energy(), vc));
   }
}

void
FWJetProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                           FWViewType::EType viewType, const FWViewContext* vc)
{
   increaseComponentTransparency(iId.index(), iCompound, "TEveJetCone", 80);
}

void
FWJetProxyBuilder::cleanLocal()
{
   m_lines.clear();
   m_common->DestroyElements();
}

void
FWJetProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{ 
   typedef std::vector<fireworks::scaleMarker> Lines_t;  
   FWViewEnergyScale* caloScale = vc->getEnergyScale("Calo");
   // printf("%p -> %f\n", this,caloScale->getValToHeight() );
   for (Lines_t::iterator i = m_lines.begin(); i!= m_lines.end(); ++ i)
   {
      if (vc == (*i).m_vc)
      { 
         float value = caloScale->getPlotEt() ? (*i).m_et : (*i).m_energy;      
         (*i).m_ls->SetScale(caloScale->getValToHeight()*value);
         TEveProjected* proj = *(*i).m_ls->BeginProjecteds();
         proj->UpdateProjection();
      }
   }
}

REGISTER_FWPROXYBUILDER( FWJetProxyBuilder, reco::Jet, "Jets", FWViewType::kAll3DBits  | FWViewType::kAllRPZBits | FWViewType::kGlimpseBit);

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
// $Id: FWJetProxyBuilder.cc,v 1.25 2010/09/29 16:19:48 amraktad Exp $
//

#include "TEveJetCone.h"
#include "TEveScalableStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
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
   struct  SLines
   {
      SLines(TEveScalableStraightLineSet* ls, float et, float e, const FWViewContext* vc) : m_ls(ls), m_et(et), m_energy(e), m_vc(vc) {}
      
      TEveScalableStraightLineSet* m_ls;
      float m_et, m_energy;
      const FWViewContext* m_vc;
   };
   
   FWJetProxyBuilder( const FWJetProxyBuilder& ); // stop default
   const FWJetProxyBuilder& operator=( const FWJetProxyBuilder& ); // stop default

   TEveElementList* requestCommon();
   TEveElementList* m_common;

   std::vector<SLines> m_lines;
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
         const reco::Jet& iData = modelData(i);

         TEveJetCone* jet = new TEveJetCone();
         jet->SetApex(TEveVector(iData.vertex().x(),iData.vertex().y(),iData.vertex().z()));

         reco::Jet::Constituents c = iData.getJetConstituents();
         bool haveData = true;
         for ( reco::Jet::Constituents::const_iterator itr = c.begin(); itr != c.end(); ++itr )
         {
            if ( !itr->isAvailable() ) {
               haveData = false;
               break;
            }
         }

         double eta_size = 0.2;
         double phi_size = 0.2;
         if ( haveData ){
            eta_size = sqrt(iData.etaetaMoment());
            phi_size = sqrt(iData.phiphiMoment());
         }

         static const float offr = 5;
         static const float offz = offr/tan(context().caloTransAngle());
         if (iData.eta() < context().caloMaxEta())
            jet->SetCylinder(context().caloR1(false) -offr, context().caloZ1(false)-offz);
         else
            jet->SetCylinder(context().caloR2(false)-offr, context().caloZ2(false)-offz);


         jet-> AddEllipticCone(iData.eta(), iData.phi(), eta_size, phi_size);
         jet->SetPickable(kTRUE);
         m_common->AddElement(jet);
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

   TEveJetCone* cone = static_cast<TEveJetCone*>(*coneIt);
   const FWDisplayProperties &dp = item()->defaultDisplayProperties();
   cone->SetFillColor(dp.color());
   cone->SetLineColor(dp.color());
   setupAddElement( cone, &oItemHolder );
   cone->SetMainTransparency(TMath::Min(100, 80 + dp.transparency() / 5)); 

   // scale markers in projected views
   if (FWViewType::isProjected(type))
   {
      TEveScalableStraightLineSet* marker =new TEveScalableStraightLineSet("jet lineset");
      float size = 1.f; // values are saved in scale
      double theta = iData.theta();
      double phi = iData.phi();

      if( type == FWViewType::kRhoPhi )
      {  
         float ecalR = context().caloR1() + 4;
         marker->SetScaleCenter(ecalR*cos(phi), ecalR*sin(phi), 0);
         marker->AddLine(ecalR*cos(phi), ecalR*sin(phi), 0, (ecalR+size)*cos(phi), (ecalR+size)*sin(phi), 0);

      }
      else
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

      marker->SetLineWidth(4);  

      marker->SetLineColor(dp.color());
      FWViewEnergyScale* caloScale = vc->getEnergyScale("Calo");    
      marker->SetScale(caloScale->getValToHeight()*(caloScale->getPlotEt() ?  iData.et() : iData.energy()));
      setupAddElement( marker, &oItemHolder );
      m_lines.push_back(SLines(marker, iData.et(), iData.energy(), vc));
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
   typedef std::vector<SLines> Lines_t;  
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

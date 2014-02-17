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
// $Id: FWJetProxyBuilder.cc,v 1.35 2011/11/21 08:36:37 amraktad Exp $
//

#include "TEveJetCone.h"
#include "TEveScalableStraightLineSet.h"

#include "Fireworks/Core/interface/FWTextProjected.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWParameters.h"
// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Calo/interface/makeEveJetCone.h"
#include "Fireworks/Calo/interface/scaleMarker.h"

#include "DataFormats/JetReco/interface/Jet.h"

namespace fireworks {

struct jetScaleMarker : public  scaleMarker {
   jetScaleMarker(TEveScalableStraightLineSet* ls, float et, float e, const FWViewContext* vc):
      scaleMarker(ls, et, e, vc) , m_text(0) {}
   
   FWEveText* m_text;
};
}

static const std::string kJetLabelsRhoPhiOn("Draw Labels in RhoPhi View");
static const std::string kJetLabelsRhoZOn("Draw Labels in RhoZ View");
static const std::string kJetOffset("Label Offset");


class FWJetProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetProxyBuilder();
   virtual ~FWJetProxyBuilder();

   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
   virtual bool haveSingleProduct() const { return false; } // different view types
   virtual void cleanLocal();

   virtual void setItem(const FWEventItem* iItem)
   {
      FWProxyBuilderBase::setItem(iItem);
      if (iItem) {
      iItem->getConfig()->assertParam(kJetLabelsRhoPhiOn, false);
      iItem->getConfig()->assertParam(kJetLabelsRhoZOn, false);
      iItem->getConfig()->assertParam(kJetOffset, 2.1, 1.0, 5.0);
      }
   }

   REGISTER_PROXYBUILDER_METHODS();
   
protected:
   virtual void buildViewType(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*);


   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);

   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);

private:
   typedef std::vector<fireworks::jetScaleMarker> Lines_t;  

   FWJetProxyBuilder( const FWJetProxyBuilder& ); // stop default
   const FWJetProxyBuilder& operator=( const FWJetProxyBuilder& ); // stop default

   TEveElementList* requestCommon();
   void setTextPos(fireworks::jetScaleMarker& s, const FWViewContext* vc, FWViewType::EType);

   TEveElementList* m_common;

   std::vector<fireworks::jetScaleMarker> m_lines;
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
      m_lines.push_back(fireworks::jetScaleMarker(new TEveScalableStraightLineSet("jetline"), iData.et(), iData.energy(), vc));
      fireworks::jetScaleMarker& markers =  m_lines.back();

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

         markers.m_ls->SetScaleCenter( 0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );
         markers.m_ls->AddLine( 0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
                                0., (phi>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
      }
      else
      {
         float ecalR = context().caloR1() + 4;
         markers.m_ls->SetScaleCenter(ecalR*cos(phi), ecalR*sin(phi), 0);
         markers.m_ls->AddLine(ecalR*cos(phi), ecalR*sin(phi), 0, (ecalR+size)*cos(phi), (ecalR+size)*sin(phi), 0);
      }

      markers.m_ls->SetLineWidth(4);  
      markers.m_ls->SetLineColor(dp.color());
      FWViewEnergyScale* caloScale = vc->getEnergyScale();    
      markers.m_ls->SetScale(caloScale->getScaleFactor3D()*(caloScale->getPlotEt() ?  iData.et() : iData.energy()));

      if ((type == FWViewType::kRhoZ && item()->getConfig()->value<bool>(kJetLabelsRhoZOn))||
          (type == FWViewType::kRhoPhi && item()->getConfig()->value<bool>(kJetLabelsRhoPhiOn) ) )
      {
         markers.m_text = new FWEveText(Form("%.1f", vc->getEnergyScale()->getPlotEt() ? iData.et() : iData.energy()));
         markers.m_text->SetMainColor( item()->defaultDisplayProperties().color());         
         setTextPos(markers, vc, type);
      }


      markers.m_ls->SetMarkerColor(markers.m_ls->GetMainColor());
      setupAddElement( markers.m_ls, &oItemHolder );
      if (markers.m_text)  setupAddElement( markers.m_text, &oItemHolder , false);

   }
   context().voteMaxEtAndEnergy(iData.et(), iData.energy());
 
}

void
FWJetProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                     FWViewType::EType viewType, const FWViewContext* vc)
{
   increaseComponentTransparency(iId.index(), iCompound, "TEveJetCone", 80);

   for (Lines_t::iterator i = m_lines.begin(); i!= m_lines.end(); ++ i)
   {
      TEveStraightLineSetProjected* projLineSet = (TEveStraightLineSetProjected*)(*(*i).m_ls->BeginProjecteds());
      if (projLineSet) projLineSet->UpdateProjection();
   }
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
   for (Lines_t::iterator i = m_lines.begin(); i!= m_lines.end(); ++ i)
   {
      if (vc == (*i).m_vc)
      {  
         float value = vc->getEnergyScale()->getPlotEt() ? (*i).m_et : (*i).m_energy;

         (*i).m_ls->SetScale(vc->getEnergyScale()->getScaleFactor3D() * value );
         if ((*i).m_text) 
         {
            (*i).m_text->SetText(Form("%.1f", value));
            setTextPos(*i, vc, type);
         }
         TEveStraightLineSetProjected* projLineSet = (TEveStraightLineSetProjected*)(*(*i).m_ls->BeginProjecteds());
         projLineSet->UpdateProjection();
      }
   }
}


void FWJetProxyBuilder::setTextPos(fireworks::jetScaleMarker& s, const FWViewContext* vc, FWViewType::EType type)
{   
   TEveChunkManager::iterator li( s.m_ls->GetLinePlex() );
   li.next();
   TEveStraightLineSet::Line_t &l = * ( TEveStraightLineSet::Line_t* ) li();
   TEveVector v(l.fV2[0] - l.fV1[0], l.fV2[1] - l.fV1[1],  l.fV2[2] - l.fV1[2] );
   v.Normalize();


   double off = item()->getConfig()->value<double>(kJetOffset) -1;
   float value = vc->getEnergyScale()->getPlotEt() ? s.m_et : s.m_energy;
   double trs = off * 130 * value/context().getMaxEnergyInEvent(vc->getEnergyScale()->getPlotEt());
   v *= trs;

   float x = l.fV1[0] + v[0];
   float y = l.fV1[1] + v[1];
   float z = l.fV1[2] + v[2];

   s.m_text->m_offsetZ =  value/context().getMaxEnergyInEvent(vc->getEnergyScale()->getPlotEt());
   s.m_text->RefMainTrans().SetPos(x, y, z);
   if ((s.m_text)->BeginProjecteds() != (s.m_text)->EndProjecteds()) {
      FWEveTextProjected* textProjected = (FWEveTextProjected*)(*(s.m_text)->BeginProjecteds());
      textProjected->UpdateProjection();
   }
   
}

REGISTER_FWPROXYBUILDER( FWJetProxyBuilder, reco::Jet, "Jets", FWViewType::kAll3DBits  | FWViewType::kAllRPZBits | FWViewType::kGlimpseBit);

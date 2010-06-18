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
// $Id: FWJetProxyBuilder.cc,v 1.14 2010/06/18 12:31:56 matevz Exp $
//
#include "TGeoArb8.h"
#include "TEveGeoNode.h"
#include "TEveScalableStraightLineSet.h"
#include "TEveElement.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

#include "Fireworks/Calo/interface/FW3DEveJet.h"
#include "Fireworks/Calo/interface/FWGlimpseEveJet.h"
#include "Fireworks/Calo/interface/thetaBins.h"

#include "DataFormats/JetReco/interface/Jet.h"

class FWJetProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetProxyBuilder() {}
   virtual ~FWJetProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();
   
protected:
   virtual void build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder,
                      const FWViewContext*);
   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);

private:
   FWJetProxyBuilder( const FWJetProxyBuilder& ); // stop default
   const FWJetProxyBuilder& operator=( const FWJetProxyBuilder& ); // stop default
};

void
FWJetProxyBuilder::build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*) 
{
   FW3DEveJet* cone = new FW3DEveJet(iData, "Cone");
   cone->SetPickable(kTRUE);
   setupAddElement(cone, &oItemHolder);

   Char_t transp = item()->defaultDisplayProperties().transparency();
   cone->SetMainTransparency(TMath::Min(100, 80 + transp / 5));
}

void
FWJetProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                           FWViewType::EType viewType, const FWViewContext* vc)
{
   increaseComponentTransparency(iId.index(), iCompound, "Cone", 80);
}


//------------------------------------------------------------------------------

class FWJetRPZProxyBuilderBase : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetRPZProxyBuilderBase() {}
   virtual ~FWJetRPZProxyBuilderBase() {}

   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }

   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);

private:
   FWJetRPZProxyBuilderBase( const FWJetRPZProxyBuilderBase& ); // stop default
   const FWJetRPZProxyBuilderBase& operator=( const FWJetRPZProxyBuilderBase& ); // stop default
};

void
FWJetRPZProxyBuilderBase::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
  for (TEveElement::List_i i = parent->BeginChildren(); i!= parent->EndChildren(); ++i)
   {
      TEveElement* comp = (*i);
      for (TEveElement::List_i j = comp->BeginChildren(); j!= comp->EndChildren(); ++j)
      {
         TEveScalableStraightLineSet* ls = dynamic_cast<TEveScalableStraightLineSet*> (*j);
         if (ls ) 
         { 
            ls->SetScale(vc->getEnergyScale("Calo")->getVal());
            TEveProjected* proj = *ls->BeginProjecteds();
            proj->UpdateProjection();
         }
      }
   }
}

//______________________________________________________________________________

class FWJetRhoPhiProxyBuilder : public FWJetRPZProxyBuilderBase
{
public:
   FWJetRhoPhiProxyBuilder() {}
   virtual ~FWJetRhoPhiProxyBuilder() {}
   REGISTER_PROXYBUILDER_METHODS();

protected:
   virtual void build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder,
                      const FWViewContext* vc);
   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);

private:
   FWJetRhoPhiProxyBuilder( const FWJetRhoPhiProxyBuilder& ); // stop default
   const FWJetRhoPhiProxyBuilder& operator=( const FWJetRhoPhiProxyBuilder& ); // stop default
};

void
FWJetRhoPhiProxyBuilder::build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder,
                               const FWViewContext* vc) 
{
   float ecalR = fireworks::Context::s_ecalR;
   double phi = iData.phi();
   
   std::vector<double> phis;
   double phiSize = sqrt(iData.phiphiMoment() );
   phis.push_back(phi + phiSize);
   phis.push_back(phi - phiSize);
   std::pair<double, double> phiRange = fw::getPhiRange(phis, iData.phi());

   double min_phi = phiRange.first  - M_PI / 72;
   double max_phi = phiRange.second + M_PI / 72;
   if (fabs(phiRange.first-phiRange.second) < 1e-3)
   {
      min_phi = phi - M_PI / 72;
      max_phi = phi + M_PI / 72;
   }
 
   double size = iData.et();

   Double_t points[16];
   points[0] = iData.vertex().x();
   points[1] = iData.vertex().y();
   points[2] = ecalR*cos(min_phi);
   points[3] = ecalR*sin(min_phi);
   points[4] = ecalR*cos(max_phi);
   points[5] = ecalR*sin(max_phi);
   points[6] = points[0];
   points[7] = points[1];
   for (int i = 0; i<8; ++i) {
      points[i+8] = points[i];
   }

   TEveGeoManagerHolder gmgr( TEveGeoShape::GetGeoMangeur() );
   const FWDisplayProperties &dp = item()->defaultDisplayProperties();
   TEveGeoShape *element = fw::getShape("cone", new TGeoArb8(0, points),
                                        dp.color());
   setupAddElement(element, &oItemHolder);

   element->SetMainTransparency(TMath::Min(100, 90 + dp.transparency() / 10));

   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
   marker->SetLineWidth(4);
   marker->SetLineColor(dp.color());

   marker->SetScaleCenter(ecalR*cos(phi), ecalR*sin(phi), 0);
   marker->AddLine(ecalR*cos(phi), ecalR*sin(phi), 0, (ecalR+size)*cos(phi), (ecalR+size)*sin(phi), 0);
   marker->SetScale(vc->getEnergyScale("Calo")->getVal());
   setupAddElement(marker, &oItemHolder);
}

void
FWJetRhoPhiProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                           FWViewType::EType viewType, const FWViewContext* vc)
{
  const FWDisplayProperties& dp = item()->modelInfo(iId.index()).displayProperties();

  TEveElement* cone = iCompound->FindChild("cone");
  if (cone)
     cone->SetMainTransparency(TMath::Min(100, 90 + dp.transparency() / 10));
}


//______________________________________________________________________________


class FWJetRhoZProxyBuilder : public FWJetRPZProxyBuilderBase
{
public:
   FWJetRhoZProxyBuilder() {}
   virtual ~FWJetRhoZProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

protected:
   virtual void build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder,
                      const FWViewContext*);
   virtual void localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                  FWViewType::EType viewType, const FWViewContext* vc);

private:
   FWJetRhoZProxyBuilder( const FWJetRhoZProxyBuilder& ); // stop default
   const FWJetRhoZProxyBuilder& operator=( const FWJetRhoZProxyBuilder& ); // stop default
};


void
FWJetRhoZProxyBuilder::build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext* vc) 
{
   static const std::vector<std::pair<double,double> > thetaBins = fireworks::thetaBins();

   float r_ecal = fireworks::Context::s_ecalR;
   float z_ecal = fireworks::Context::s_ecalZ;
   float transition_angle = fireworks::Context::s_transitionAngle;

   double theta = iData.theta();
   double eta = iData.eta();
   double phi = iData.phi();
   
   // distance from the origin of the jet centroid
   // energy is measured from this point
   // if jet is made of a single tower, the length of the jet will
   // be identical to legth of the displayed tower
   double r(0);
   ( theta < transition_angle || M_PI-theta < transition_angle ) ?
     r = z_ecal/fabs(cos(theta)) :
     r = r_ecal/sin(theta);
   
   double size = iData.et();
   double etaSize = sqrt( iData.etaetaMoment() );
   
   

   TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
   marker->SetLineWidth(4);
   marker->SetLineColor(  item()->defaultDisplayProperties().color() );
   marker->SetScaleCenter( 0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );

   marker->AddLine( 0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
		    0., (phi>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );

   
   marker->SetScale(vc->getEnergyScale("Calo")->getVal());
   setupAddElement( marker, &oItemHolder );

   double min_theta = 2*atan(exp(-( eta+etaSize )));
   double max_theta = 2*atan(exp(-( eta-etaSize )));
   Double_t points[16];
   points[0] = iData.vertex().z();
   points[1] = phi>0 ? iData.vertex().rho() : -iData.vertex().rho();
   
   if( max_theta > M_PI - transition_angle )
   {
      points[6] = -z_ecal;
      points[7] = phi>0 ? z_ecal*fabs(tan(max_theta)) : -z_ecal*fabs(tan(max_theta));
   }
   else if( max_theta < transition_angle ) {
      points[6] = z_ecal;
      points[7] = phi>0 ? z_ecal*fabs(tan(max_theta)) : -z_ecal*fabs(tan(max_theta));
   }
   else {
      points[6] = r_ecal/tan(max_theta);
      points[7] = phi>0 ? r_ecal : -r_ecal;
   }
   
   if( min_theta > M_PI - transition_angle )
   {
      points[2] = -z_ecal;
      points[3] = phi>0 ? z_ecal*fabs(tan(min_theta)) : -z_ecal*fabs(tan(min_theta));
   }
   else if ( min_theta < transition_angle ) {
      points[2] = z_ecal;
      points[3] = phi>0 ? z_ecal*fabs(tan(min_theta)) : -z_ecal*fabs(tan(min_theta));
   }
   else {
      points[2] = r_ecal/tan(min_theta);
      points[3] = phi>0 ? r_ecal : -r_ecal;
   }
   
   if( min_theta < M_PI - transition_angle && max_theta > M_PI - transition_angle )
   {
      points[4] = -z_ecal;
      points[5] = phi>0 ? r_ecal : -r_ecal;
   }
   else if( min_theta < transition_angle && max_theta > transition_angle ) {
      points[4] = z_ecal;
      points[5] = phi>0 ? r_ecal : -r_ecal;
   }
   else {
      points[4] = points[2];
      points[5] = points[3];
   }
   
   for( int i = 0; i<8; ++i ) {
     points[i+8] = points[i];
   }
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   const FWDisplayProperties &p = item()->defaultDisplayProperties();
   TEveGeoShape *element = fw::getShape("cone", new TGeoArb8( 0, points ),
                                        p.color());
   element->RefMainTrans().RotateLF( 1, 3, M_PI/2 );
   setupAddElement(element, &oItemHolder);

   element->SetMainTransparency(TMath::Min(100, 90 + p.transparency() / 10));

}

void
FWJetRhoZProxyBuilder::localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                         FWViewType::EType viewType, const FWViewContext* vc)
{
  const FWDisplayProperties& dp = item()->modelInfo(iId.index()).displayProperties();

  TEveElement* cone = iCompound->FindChild("cone");
  if (cone)
     cone->SetMainTransparency(TMath::Min(100, 90 + dp.transparency() / 10));
}


//______________________________________________________________________________


class FWJetGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetGlimpseProxyBuilder() {}
   virtual ~FWJetGlimpseProxyBuilder() {}
  
   virtual bool havePerViewProduct() const { return true; }
   
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWJetGlimpseProxyBuilder( const FWJetGlimpseProxyBuilder& ); // stop default
   const FWJetGlimpseProxyBuilder& operator=( const FWJetGlimpseProxyBuilder& ); // stop default

   virtual void build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*);
};

void
FWJetGlimpseProxyBuilder::build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*) 
{
   FWGlimpseEveJet* cone = new FWGlimpseEveJet( &iData, "jet" );
   cone->SetMainTransparency( 50 );
   cone->SetDrawConeCap( kFALSE );
   cone->SetMainTransparency( 50 );
   setupAddElement( cone, &oItemHolder );
}

//______________________________________________________________________________


class FWJetLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetLegoProxyBuilder() {}
   virtual ~FWJetLegoProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

protected:
   virtual void build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder,
                      const FWViewContext*);

private:
   FWJetLegoProxyBuilder( const FWJetLegoProxyBuilder& ); // stop default
   const FWJetLegoProxyBuilder& operator=( const FWJetLegoProxyBuilder& ); // stop default
};

void
FWJetLegoProxyBuilder::build(const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder,
                             const FWViewContext*) 
{
   TEveStraightLineSet* container = new TEveStraightLineSet( "circle" );

   const unsigned int nLineSegments = 20;
   const double jetRadius = 0.5;
   for (unsigned int iphi = 0; iphi < nLineSegments; ++iphi)
   {
      container->AddLine(iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*iphi),
                         iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*iphi),
                         0.01,
                         iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*(iphi+1)),
                         iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*(iphi+1)),
                         0.01);
   }
   setupAddElement(container, &oItemHolder);
}

REGISTER_FWPROXYBUILDER( FWJetProxyBuilder, reco::Jet, "Jets", FWViewType::kAll3DBits );
REGISTER_FWPROXYBUILDER( FWJetRhoPhiProxyBuilder, reco::Jet, "Jets", FWViewType::kRhoPhiBit );
REGISTER_FWPROXYBUILDER( FWJetRhoZProxyBuilder, reco::Jet, "Jets", FWViewType::kRhoZBit );
REGISTER_FWPROXYBUILDER( FWJetGlimpseProxyBuilder, reco::Jet, "Jets", FWViewType::kGlimpseBit );
REGISTER_FWPROXYBUILDER( FWJetLegoProxyBuilder, reco::Jet, "Jets", FWViewType::kLegoBit | FWViewType::kLegoHFBit );

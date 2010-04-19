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
// $Id: FWJetProxyBuilder.cc,v 1.4 2010/04/16 11:28:03 amraktad Exp $
//
#include "TGeoArb8.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/Context.h"

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

private:
   FWJetProxyBuilder( const FWJetProxyBuilder& ); // stop default
   const FWJetProxyBuilder& operator=( const FWJetProxyBuilder& ); // stop default

   virtual void build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWJetProxyBuilder::build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   FW3DEveJet* cone = new FW3DEveJet( iData, "jet", "jet");
   cone->SetPickable(kTRUE);
   cone->SetMainTransparency(75);

   oItemHolder.AddElement( cone );
}

class FWJetRhoPhiProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetRhoPhiProxyBuilder() {}
   virtual ~FWJetRhoPhiProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
   FWJetRhoPhiProxyBuilder( const FWJetRhoPhiProxyBuilder& ); // stop default
   const FWJetRhoPhiProxyBuilder& operator=( const FWJetRhoPhiProxyBuilder& ); // stop default

   virtual void build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWJetRhoPhiProxyBuilder::build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   float ecalR = fireworks::Context::s_ecalR;
   double phi = iData.phi();
   
   std::vector<double> phis;
   double phiSize = sqrt( iData.phiphiMoment() );
   phis.push_back( phi+phiSize );
   phis.push_back( phi-phiSize );
   std::pair<double, double> phiRange = fw::getPhiRange( phis, iData.phi() );

   double min_phi = phiRange.first-M_PI/36/2;
   double max_phi = phiRange.second+M_PI/36/2;
   if( fabs(phiRange.first-phiRange.second)<1e-3 ) {
      min_phi = phi-M_PI/36/2;
      max_phi = phi+M_PI/36/2;
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
   for( int i = 0; i<8; ++i ) {
     points[i+8] = points[i];
   }
   
   TEveElementList* comp = new TEveElementList;
   TEveGeoShape *element = fw::getShape( "cone", 
					 new TGeoArb8( 0, points ),
					 item()->defaultDisplayProperties().color() );
   element->SetMainTransparency( 90 );
   element->SetPickable( kTRUE );
   comp->AddElement( element );
   
   TEveStraightLineSet* marker = new TEveStraightLineSet( "energy" );
   marker->SetLineWidth( 4 );
   marker->AddLine( ecalR*cos(phi), ecalR*sin(phi), 0,
		    (ecalR+size)*cos(phi), (ecalR+size)*sin(phi), 0);
   comp->AddElement( marker );
   oItemHolder.AddElement( comp );
}

class FWJetRhoZProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetRhoZProxyBuilder() {}
   virtual ~FWJetRhoZProxyBuilder() {}

  REGISTER_PROXYBUILDER_METHODS();

private:
   FWJetRhoZProxyBuilder( const FWJetRhoZProxyBuilder& ); // stop default
   const FWJetRhoZProxyBuilder& operator=( const FWJetRhoZProxyBuilder& ); // stop default

   virtual void build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWJetRhoZProxyBuilder::build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   static const std::vector<std::pair<double,double> > thetaBins = fireworks::thetaBins();

   float r_ecal = fireworks::Context::s_ecalR;
   float z_ecal = fireworks::Context::s_ecalZ;
   float transition_angle = atan( r_ecal/z_ecal ); //FIXME This Context number is wrong: fireworks::Context::s_transitionAngle;

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
   double etaSize = sqrt(iData.etaetaMoment());
   
   TEveStraightLineSet* marker = new TEveStraightLineSet( "energy" );
   marker->SetLineWidth(4);
   marker->AddLine(0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
		   0., (phi>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
   oItemHolder.AddElement( marker );
   
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
   TEveGeoShape *element = fw::getShape( "cone2", 
					 new TGeoArb8( 0, points ),
					 item()->defaultDisplayProperties().color() );
   element->RefMainTrans().RotateLF( 1, 3, M_PI/2 );
   element->SetMainTransparency( 90 );
   element->SetPickable( kTRUE );

   oItemHolder.AddElement( element );
}

class FWJetGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetGlimpseProxyBuilder() {}
   virtual ~FWJetGlimpseProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWJetGlimpseProxyBuilder(const FWJetGlimpseProxyBuilder&); // stop default
   const FWJetGlimpseProxyBuilder& operator=(const FWJetGlimpseProxyBuilder&); // stop default

   virtual void build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWJetGlimpseProxyBuilder::build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   FWGlimpseEveJet* cone = new FWGlimpseEveJet(&iData, "jet", "jet");
   oItemHolder.AddElement( cone );
   
   cone->SetPickable( kTRUE );
   cone->SetMainTransparency( 50 );
   cone->SetRnrSelf( item()->defaultDisplayProperties().isVisible() );
   cone->SetRnrChildren( item()->defaultDisplayProperties().isVisible() );
   cone->SetDrawConeCap( kFALSE );
   cone->SetMainTransparency( 50 );
}

class FWJetLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Jet>
{
public:
   FWJetLegoProxyBuilder() {}
   virtual ~FWJetLegoProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWJetLegoProxyBuilder(const FWJetLegoProxyBuilder&); // stop default
   const FWJetLegoProxyBuilder& operator=(const FWJetLegoProxyBuilder&); // stop default

   virtual void build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWJetLegoProxyBuilder::build( const reco::Jet& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   TEveStraightLineSet* container = new TEveStraightLineSet("circle");
   oItemHolder.AddElement(container);

   const unsigned int nLineSegments = 20;
   const double jetRadius = 0.5;
   for( unsigned int iphi = 0; iphi < nLineSegments; ++iphi ) {
      container->AddLine(iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*iphi),
                         iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*iphi),
                         0.1,
                         iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*(iphi+1)),
                         iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*(iphi+1)),
                         0.1);
   }
}

REGISTER_FWPROXYBUILDER( FWJetProxyBuilder, reco::Jet, "Jets", FWViewType::kAll3DBits );
REGISTER_FWPROXYBUILDER( FWJetRhoPhiProxyBuilder, reco::Jet, "Jets", FWViewType::kRhoPhiBit );
REGISTER_FWPROXYBUILDER( FWJetRhoZProxyBuilder, reco::Jet, "Jets", FWViewType::kRhoZBit );
REGISTER_FWPROXYBUILDER( FWJetGlimpseProxyBuilder, reco::Jet, "Jets", FWViewType::kGlimpseBit );
REGISTER_FWPROXYBUILDER( FWJetLegoProxyBuilder, reco::Jet, "Jets", FWViewType::kLegoBit );

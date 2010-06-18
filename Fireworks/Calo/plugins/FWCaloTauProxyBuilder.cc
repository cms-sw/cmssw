// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWCaloTauProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWCaloTauProxyBuilder.cc,v 1.10 2010/05/03 15:47:33 amraktad Exp $
//

// system include files
#include "TEveCompound.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TEveTrack.h"

// user include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "Fireworks/Calo/interface/FW3DEveJet.h"
#include "Fireworks/Calo/interface/FWGlimpseEveJet.h"
#include "Fireworks/Calo/interface/thetaBins.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"

class FWCaloTauProxyBuilder : public FWProxyBuilderBase
{
public:
   FWCaloTauProxyBuilder() {}
   virtual ~FWCaloTauProxyBuilder() {}

   virtual bool haveSingleProduct() const { return false; }
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloTauProxyBuilder(const FWCaloTauProxyBuilder&);    // stop default
   const FWCaloTauProxyBuilder& operator=(const FWCaloTauProxyBuilder&);    // stop default

   virtual void buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type , const FWViewContext*);

   // Add Tracks which passed quality cuts and
   // are inside a tracker signal cone around leading Track
   void addConstituentTracks( const reco::CaloTau &caloTau, class TEveElement *product );
   // Add leading Track
   void addLeadTrack( const reco::CaloTau &caloTau, class TEveElement *product );
};

void
FWCaloTauProxyBuilder::buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type , const FWViewContext*)
{
   reco::CaloTauCollection const * caloTaus = 0;
   iItem->get( caloTaus );
   if( caloTaus == 0 ) return;

   float r_ecal = fireworks::Context::s_ecalR;
   float z_ecal = fireworks::Context::s_ecalZ;
   float transition_angle = fireworks::Context::s_transitionAngle;
      
   Int_t idx = 0;
   for( reco::CaloTauCollection::const_iterator it = caloTaus->begin(), itEnd = caloTaus->end(); it != itEnd; ++it, ++idx)
   { 
      const reco::CaloTauTagInfo *tauTagInfo = dynamic_cast<const reco::CaloTauTagInfo*>(((*it).caloTauTagInfoRef().get()));
      const reco::CaloJet *jet = dynamic_cast<const reco::CaloJet*>((tauTagInfo->calojetRef().get()));

      int min =  100;
      int max = -100;
      std::vector<double> phis;
      std::vector<CaloTowerPtr> towers = jet->getCaloConstituents();
      for( std::vector<CaloTowerPtr>::const_iterator tower = towers.begin(), towerEnd = towers.end();
	   tower != towerEnd; ++tower )
      {
	 unsigned int ieta = 41 + (*tower)->id().ieta();
	 if( ieta > 40 ) --ieta;
	 assert( ieta <= 82 );
	 
	 if( int(ieta) > max ) max = ieta;
	 if( int(ieta) < min ) min = ieta;
	 phis.push_back( (*tower)->phi() );
      }
      if( min > max ) {	
	 min = 0; max = 0;
      }
      
      double phi = (*it).phi();
      double theta = (*it).theta();
      double size = (*it).et();

      TEveCompound* comp = createCompound();

      // FIXME: Should it be only in RhoPhi and RhoZ?
      addLeadTrack( *it, comp );
      addConstituentTracks( *it, comp );

      if( ( type == FWViewType::k3D ) | ( type == FWViewType::kISpy ) ) {
	 FW3DEveJet* cone = new FW3DEveJet( *jet, "cone" );
	 cone->SetPickable( kTRUE );
	 cone->SetMainTransparency( 75 ); 
	 setupAddElement( cone, comp );
      }
      else if( type == FWViewType::kRhoPhi ) 
      {	 
	 std::pair<double,double> phiRange = fw::getPhiRange( phis, phi );
	 double min_phi = phiRange.first-M_PI/36/2;
	 double max_phi = phiRange.second+M_PI/36/2;
	 if( fabs(phiRange.first-phiRange.first)<1e-3 ) {
	    min_phi = phi-M_PI/36/2;
	    max_phi = phi+M_PI/36/2;
	 }
	 TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
	 TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
	 TEveGeoShape *element = fw::getShape( "spread", sc_box, iItem->defaultDisplayProperties().color() );
	 element->SetPickable(kTRUE);
	 setupAddElement( element, comp );

	 TEveStraightLineSet* marker = new TEveStraightLineSet( "energy" );
	 marker->SetLineWidth( 4 );
	 marker->AddLine( r_ecal*cos( phi ), r_ecal*sin( phi ), 0,
			  ( r_ecal+size )*cos( phi ), ( r_ecal+size )*sin( phi ), 0);
	 setupAddElement( marker, comp );
      }
      else if( type == FWViewType::kRhoZ )
      {
	 double r(0);
	 ( theta < transition_angle || M_PI-theta < transition_angle ) ?
	    r = z_ecal/fabs(cos(theta)) :
	    r = r_ecal/sin(theta);
   
	 TEveStraightLineSet* marker = new TEveStraightLineSet( "energy" );
	 marker->SetLineWidth(4);
	 marker->AddLine(0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
			 0., (phi>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
	 setupAddElement( marker, comp );

	 const std::vector<std::pair<double, double> > thetaBins = fireworks::thetaBins();
	 double max_theta = thetaBins[min].first;
	 double min_theta = thetaBins[max].second;
	 fw::addRhoZEnergyProjection( this, comp, r_ecal, z_ecal, min_theta-0.003, max_theta+0.003,
				      phi);
      }            
      setupAddElement( comp, product );
   }
}

// Tracks which passed quality cuts and
// are inside a tracker signal cone around leading Track
void
FWCaloTauProxyBuilder::addConstituentTracks( const reco::CaloTau &caloTau, class TEveElement* product )
{
   for( reco::TrackRefVector::iterator i = caloTau.signalTracks().begin(), iEnd = caloTau.signalTracks().end();
	i != iEnd; ++i ) {
     TEveTrack* track( 0 );
     if( i->isAvailable() ) {
        track = fireworks::prepareTrack( **i, context().getTrackPropagator() );
     }
     track->MakeTrack();
     if( track )
        product->AddElement( track );
   }	
}

// Leading Track
void
FWCaloTauProxyBuilder::addLeadTrack( const reco::CaloTau &caloTau, class TEveElement *product ) 
{
   const reco::TrackRef leadTrack = caloTau.leadTrack();
   if( !leadTrack ) return;

   TEveTrack* track = fireworks::prepareTrack( *leadTrack, context().getTrackPropagator() );
   track->MakeTrack();
   if( track )
      product->AddElement( track );
}

////////////////////////////////////////////////////////////////////////////////
//
//   GLIMPSE specific proxy builder
// 
////////////////////////////////////////////////////////////////////////////////

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"

class FWCaloTauGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::CaloTau>
{
public:
   FWCaloTauGlimpseProxyBuilder() {}
   virtual ~FWCaloTauGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloTauGlimpseProxyBuilder(const FWCaloTauGlimpseProxyBuilder&);    // stop default
   const FWCaloTauGlimpseProxyBuilder& operator=(const FWCaloTauGlimpseProxyBuilder&);    // stop default

   virtual void build( const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*);
};

void
FWCaloTauGlimpseProxyBuilder::build( const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*) 
{
   const reco::CaloTauTagInfo *tauTagInfo = dynamic_cast<const reco::CaloTauTagInfo*>((iData.caloTauTagInfoRef().get()));
   const reco::CaloJet *jet = dynamic_cast<const reco::CaloJet*>((tauTagInfo->calojetRef().get()));

   FWGlimpseEveJet* cone = new FWGlimpseEveJet( jet, "jet", "jet");
   cone->SetPickable( kTRUE );
   cone->SetMainTransparency( 50 ); 
   cone->SetDrawConeCap( kFALSE );

   setupAddElement(cone, &oItemHolder);
}

////////////////////////////////////////////////////////////////////////////////
//
//   LEGO specific proxy builder
// 
////////////////////////////////////////////////////////////////////////////////

class FWCaloTauLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::CaloTau>
{
public:
   FWCaloTauLegoProxyBuilder() {}
   virtual ~FWCaloTauLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloTauLegoProxyBuilder(const FWCaloTauLegoProxyBuilder&);    // stop default
   const FWCaloTauLegoProxyBuilder& operator=(const FWCaloTauLegoProxyBuilder&);    // stop default

   virtual void build( const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*);
};

void
FWCaloTauLegoProxyBuilder::build( const reco::CaloTau& iData, unsigned int iIndex, TEveElement& oItemHolder , const FWViewContext*) 
{
   const unsigned int nLineSegments = 20;
   const double jetRadius = 0.17;   //10 degree
   TEveStraightLineSet* container = new TEveStraightLineSet();
   for( unsigned int iphi = 0; iphi < nLineSegments; ++iphi ) {
      container->AddLine(iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*iphi),
			 iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*iphi),
			 0.1,
			 iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*(iphi+1)),
			 iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*(iphi+1)),
			 0.1);
   }
   setupAddElement(container, &oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWCaloTauProxyBuilder, reco::CaloTauCollection, "CaloTau", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
REGISTER_FWPROXYBUILDER(FWCaloTauGlimpseProxyBuilder, reco::CaloTau, "CaloTau", FWViewType::kGlimpseBit);
REGISTER_FWPROXYBUILDER(FWCaloTauLegoProxyBuilder, reco::CaloTau, "CaloTau", FWViewType::kLegoBit);

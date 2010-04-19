// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWPFTauProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWPFTauProxyBuilder.cc,v 1.4 2010/04/16 11:28:03 amraktad Exp $
//

// system include files
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

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"

class FWPFTauProxyBuilder : public FWProxyBuilderBase
{
public:
   FWPFTauProxyBuilder() {}
   virtual ~FWPFTauProxyBuilder() {}

   virtual bool hasSingleProduct() { return false; }

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFTauProxyBuilder(const FWPFTauProxyBuilder&);    // stop default
   const FWPFTauProxyBuilder& operator=(const FWPFTauProxyBuilder&);    // stop default

   virtual void buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type );
};

void
FWPFTauProxyBuilder::buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type )
{
   reco::PFTauCollection const * pfTaus = 0;
   iItem->get( pfTaus );
   if( pfTaus == 0 ) return;

   float r_ecal = fireworks::Context::s_ecalR;
   float z_ecal = fireworks::Context::s_ecalZ;
   float transition_angle = atan( r_ecal/z_ecal ); //FIXME This Context number is wrong: fireworks::Context::s_transitionAngle;
      
   Int_t idx = 0;
   for( reco::PFTauCollection::const_iterator it = pfTaus->begin(), itEnd = pfTaus->end(); it != itEnd; ++it, ++idx)
   { 
      const reco::PFTauTagInfo *tauTagInfo = dynamic_cast<const reco::PFTauTagInfo*>(((*it).pfTauTagInfoRef().get()));
      const reco::PFJet *jet = dynamic_cast<const reco::PFJet*>((tauTagInfo->pfjetRef().get()));
      int min =  100;
      int max = -100;
      std::vector<double> phis;
      std::vector <const reco::Candidate*> candidates = jet->getJetConstituentsQuick();
      for( std::vector<const reco::Candidate*>::const_iterator candidate = candidates.begin(), candidateEnd = candidates.end();
	   candidate != candidateEnd; ++candidate )
      {
	 double itheta = (*candidate)->theta();
	 if( itheta > max ) max = itheta;
	 if( itheta < min ) min = itheta;

	 phis.push_back( (*candidate)->phi() );
      }
      if( min > max ) {	
	 min = 0; max = 0;
      }
      
      double phi = (*it).phi();
      double theta = (*it).theta();
      double size = (*it).et();

      const char* name = Form( "PFTau %d", idx );
      TEveElementList* comp = new TEveElementList( name, name );

      // FIXME: Should it be only in RhoPhi and RhoZ?
      const reco::TrackRef lead_track = (*it).leadTrack();
      reco::TrackRefVector::iterator tracks_end = (*it).signalTracks().end(); 
      for( reco::TrackRefVector::iterator i = (*it).signalTracks().begin(); i != tracks_end; ++i ) {
	TEveTrack* track(0);
	if( i->isAvailable() ) {
	  track = fireworks::prepareTrack(**i,
					  context().getTrackPropagator());
	}
	track->MakeTrack();
	if( track )
	  comp->AddElement( track );
      }	

      if( ( type == FWViewType::k3D ) | ( type == FWViewType::kISpy ) ) {
	 FW3DEveJet* cone = new FW3DEveJet( *jet, name, name);
	 cone->SetPickable( kTRUE );
	 cone->SetMainTransparency( 75 ); 
	 comp->AddElement( cone );
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
	 TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
	 TEveGeoShape *element = fw::getShape( "spread", sc_box, iItem->defaultDisplayProperties().color() );
	 element->SetPickable(kTRUE);
	 comp->AddElement(element);

	 TEveStraightLineSet* marker = new TEveStraightLineSet( "energy", name );
	 marker->SetLineWidth( 4 );
	 marker->AddLine( r_ecal*cos( phi ), r_ecal*sin( phi ), 0,
			  ( r_ecal+size )*cos( phi ), ( r_ecal+size )*sin( phi ), 0);
	 comp->AddElement( marker );
      }
      else if( type == FWViewType::kRhoZ )
      {
	 double r(0);
	 ( theta < transition_angle || M_PI-theta < transition_angle ) ?
	    r = z_ecal/fabs(cos(theta)) :
	    r = r_ecal/sin(theta);
   
	 TEveStraightLineSet* marker = new TEveStraightLineSet( "energy", name );
	 marker->SetLineWidth(4);
	 marker->AddLine(0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
			 0., (phi>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );
	 comp->AddElement( marker );

	 const std::vector<std::pair<double, double> > thetaBins = fireworks::thetaBins();
	 double max_theta = thetaBins[min].first;
	 double min_theta = thetaBins[max].second;
	 fw::addRhoZEnergyProjection( comp, r_ecal, z_ecal, min_theta-0.003, max_theta+0.003,
				      phi, item()->defaultDisplayProperties().color() );
      }            
      product->AddElement( comp );
   }
}

////////////////////////////////////////////////////////////////////////////////
//
//   GLIMPSE specific proxy builder
// 
////////////////////////////////////////////////////////////////////////////////

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"

class FWPFTauGlimpseProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFTau>
{
public:
   FWPFTauGlimpseProxyBuilder() {}
   virtual ~FWPFTauGlimpseProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFTauGlimpseProxyBuilder(const FWPFTauGlimpseProxyBuilder&);    // stop default
   const FWPFTauGlimpseProxyBuilder& operator=(const FWPFTauGlimpseProxyBuilder&);    // stop default

   virtual void build( const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWPFTauGlimpseProxyBuilder::build( const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
{
   const reco::PFTauTagInfo *tauTagInfo = dynamic_cast<const reco::PFTauTagInfo*>((iData.pfTauTagInfoRef().get()));
   const reco::PFJet *jet = dynamic_cast<const reco::PFJet*>((tauTagInfo->pfjetRef().get()));

   FWGlimpseEveJet* cone = new FWGlimpseEveJet( jet, "jet", "jet");
   cone->SetPickable( kTRUE );
   cone->SetMainTransparency( 50 ); 
   cone->SetDrawConeCap( kFALSE );

   oItemHolder.AddElement( cone );
}

////////////////////////////////////////////////////////////////////////////////
//
//   LEGO specific proxy builder
// 
////////////////////////////////////////////////////////////////////////////////

class FWPFTauLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::PFTau>
{
public:
   FWPFTauLegoProxyBuilder() {}
   virtual ~FWPFTauLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPFTauLegoProxyBuilder(const FWPFTauLegoProxyBuilder&);    // stop default
   const FWPFTauLegoProxyBuilder& operator=(const FWPFTauLegoProxyBuilder&);    // stop default

   virtual void build( const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const;
};

void
FWPFTauLegoProxyBuilder::build( const reco::PFTau& iData, unsigned int iIndex, TEveElement& oItemHolder ) const
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
   oItemHolder.AddElement( container );
}

REGISTER_FWPROXYBUILDER(FWPFTauProxyBuilder, reco::PFTauCollection, "PFTau", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
REGISTER_FWPROXYBUILDER(FWPFTauGlimpseProxyBuilder, reco::PFTau, "PFTau", FWViewType::kGlimpseBit);
REGISTER_FWPROXYBUILDER(FWPFTauLegoProxyBuilder, reco::PFTau, "PFTau", FWViewType::kLegoBit);

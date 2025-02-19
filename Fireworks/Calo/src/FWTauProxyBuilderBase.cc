// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWTauProxyBuilderBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Thu Oct 21 20:40:28 CEST 2010
// $Id: FWTauProxyBuilderBase.cc,v 1.5 2012/03/23 00:08:29 amraktad Exp $
//

// system include files

// user include files
#include "TGeoTube.h"
#include "TEveJetCone.h"
#include "TEveGeoNode.h"
#include "TEveScalableStraightLineSet.h"
#include "TEveTrack.h"

#include "Fireworks/Calo/interface/FWTauProxyBuilderBase.h"
#include "Fireworks/Calo/interface/makeEveJetCone.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Calo/interface/thetaBins.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TrackReco/interface/Track.h"


FWTauProxyBuilderBase::FWTauProxyBuilderBase():
   m_minTheta(0),
   m_maxTheta(0)
{
}

FWTauProxyBuilderBase::~FWTauProxyBuilderBase()
{
}

void
FWTauProxyBuilderBase::buildBaseTau( const reco::BaseTau& iTau, const reco::Jet* iJet, TEveElement* comp, FWViewType::EType type, const FWViewContext* vc)
{
   // track
   addLeadTrack( iTau, comp );
   addConstituentTracks( iTau, comp );
 
   // projected markers
   if (FWViewType::isProjected(type))
   {
      double phi   = iTau.phi();
      double theta = iTau.theta();
      double size  = 1;

      bool  barrel = (theta< context().caloTransAngle() || theta > (TMath::Pi() - context().caloTransAngle()));
      float ecalR  = barrel ? context().caloR1() : context().caloR2();
      float ecalZ  = barrel ? context().caloZ1() : context().caloZ2();
  
      TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet( "energy" );
     
      if( type == FWViewType::kRhoZ )
      {
         double r(0);
         ( theta <  context().caloTransAngle() || M_PI-theta < context().caloTransAngle()) ?
            r = ecalZ/fabs(cos(theta)) :
            r = ecalR/sin(theta);
   
         fireworks::addRhoZEnergyProjection( this, comp, ecalR, ecalZ, m_minTheta-0.003, m_maxTheta+0.003, phi);

         marker->SetScaleCenter( 0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta) );
         marker->AddLine(0., (phi>0 ? r*fabs(sin(theta)) : -r*fabs(sin(theta))), r*cos(theta),
                         0., (phi>0 ? (r+size)*fabs(sin(theta)) : -(r+size)*fabs(sin(theta))), (r+size)*cos(theta) );

      }
      else
      {
         std::pair<double,double> phiRange = fireworks::getPhiRange( m_phis, phi );
         double min_phi = phiRange.first-M_PI/36/2;
         double max_phi = phiRange.second+M_PI/36/2;
         if( fabs(phiRange.first-phiRange.first)<1e-3 ) {
            min_phi = phi-M_PI/36/2;
            max_phi = phi+M_PI/36/2;
         }
         TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
         TGeoBBox *sc_box = new TGeoTubeSeg(ecalR - 1, ecalR + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
         TEveGeoShape *shape = fireworks::getShape( "spread", sc_box, item()->defaultDisplayProperties().color() );
         shape->SetPickable(kTRUE);
         setupAddElement( shape, comp );

         marker->SetScaleCenter(ecalR*cos(phi), ecalR*sin(phi), 0);      
         marker->AddLine( ecalR*cos( phi ), ecalR*sin( phi ), 0,
                          ( ecalR+size )*cos( phi ), ( ecalR+size )*sin( phi ), 0);
      }
      marker->SetLineWidth(4);  
      FWViewEnergyScale* caloScale = vc->getEnergyScale();    
      marker->SetScale(caloScale->getScaleFactor3D()*(caloScale->getPlotEt() ?  iTau.et() : iTau.energy()));
      setupAddElement( marker, comp );
      m_lines.push_back(fireworks::scaleMarker(marker, iTau.et(), iTau.energy(), vc));

      context().voteMaxEtAndEnergy( iTau.et(), iTau.energy());
   }
   else if (iJet)
   {
      // jet
      TEveJetCone* cone = fireworks::makeEveJetCone(*iJet, context());
      const FWDisplayProperties &dp = item()->defaultDisplayProperties();
      cone->SetFillColor(dp.color());
      cone->SetLineColor(dp.color());
      setupAddElement( cone, comp );
      cone->SetMainTransparency(TMath::Min(100, 80 + dp.transparency() / 5)); 
   }
}

// Tracks which passed quality cuts and are inside a tracker signal cone around leading Track
void
FWTauProxyBuilderBase::addConstituentTracks( const reco::BaseTau &tau, class TEveElement* product )
{
   for( reco::TrackRefVector::iterator i = tau.signalTracks().begin(), iEnd = tau.signalTracks().end();
	i != iEnd; ++i ) {
      TEveTrack* track( 0 );
      if( i->isAvailable() ) {
         track = fireworks::prepareTrack( **i, context().getTrackPropagator() );
         track->MakeTrack();
         setupAddElement( track, product );
      }
   }	
}

// Leading Track
void
FWTauProxyBuilderBase::addLeadTrack( const reco::BaseTau &tau, class TEveElement *product ) 
{
   const reco::TrackRef leadTrack = tau.leadTrack();
   if( !leadTrack ) return;

   TEveTrack* track = fireworks::prepareTrack( *leadTrack, context().getTrackPropagator() );
   if( track ) 
   {
      track->MakeTrack();
      setupAddElement( track, product );
   }
}

void
FWTauProxyBuilderBase::localModelChanges(const FWModelId& iId, TEveElement* iCompound,
                                         FWViewType::EType viewType, const FWViewContext* vc)
{
   if (FWViewType::isProjected(viewType))
      increaseComponentTransparency(iId.index(), iCompound, "TEveJetCone", 80);
}

void
FWTauProxyBuilderBase::scaleProduct(TEveElementList* parent, FWViewType::EType viewType, const FWViewContext* vc)
{ 
   if (FWViewType::isProjected(viewType))
   {
      typedef std::vector<fireworks::scaleMarker> Lines_t;  
      FWViewEnergyScale* caloScale = vc->getEnergyScale();
      // printf("%p -> %f\n", this,caloScale->getValToHeight() );
      for (Lines_t::iterator i = m_lines.begin(); i!= m_lines.end(); ++ i)
      {
         if (vc == (*i).m_vc)
         { 
            float value = caloScale->getPlotEt() ? (*i).m_et : (*i).m_energy;      
            (*i).m_ls->SetScale(caloScale->getScaleFactor3D()*value);
            TEveProjected* proj = *(*i).m_ls->BeginProjecteds();
            proj->UpdateProjection();
         }
      }
   }
}

void
FWTauProxyBuilderBase::cleanLocal()
{
   m_lines.clear();
}

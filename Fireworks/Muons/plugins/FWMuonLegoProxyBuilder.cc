// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonLegoProxyBuilder
//
// $Id: FWMuonLegoProxyBuilder.cc,v 1.2 2010/12/01 11:41:36 amraktad Exp $
//

#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

class FWMuonLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Muon>
{
public:
   FWMuonLegoProxyBuilder( void ) {}
   virtual ~FWMuonLegoProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWMuonLegoProxyBuilder( const FWMuonLegoProxyBuilder& );
   // Disable default assignment operator
   const FWMuonLegoProxyBuilder& operator=( const FWMuonLegoProxyBuilder& );

   virtual void build( const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWMuonLegoProxyBuilder::build( const reco::Muon& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
   TEvePointSet* points = new TEvePointSet;
   setupAddElement( points, &oItemHolder );
 
   points->SetMarkerStyle( 2 );
   points->SetMarkerSize( 0.2 );
    
   // get ECAL position of the propagated trajectory if available
   if( iData.isEnergyValid() && iData.calEnergy().ecal_position.r() > 100 )
   {
      points->SetNextPoint( iData.calEnergy().ecal_position.eta(),
			    iData.calEnergy().ecal_position.phi(),
			    0.1 );
     return;
   }
    
   // get position of the muon at surface of the tracker
   if( iData.track().isAvailable() && iData.track()->extra().isAvailable())
   {
      points->SetNextPoint( iData.track()->outerPosition().eta(),
			    iData.track()->outerPosition().phi(),
			    0.1 );
      return;
   } 

   // get position of the inner state of the stand alone muon
   if( iData.standAloneMuon().isAvailable() && iData.standAloneMuon()->extra().isAvailable())
   {
      if( iData.standAloneMuon()->innerPosition().R() <  iData.standAloneMuon()->outerPosition().R())
         points->SetNextPoint( iData.standAloneMuon()->innerPosition().eta(),
			       iData.standAloneMuon()->innerPosition().phi(),
			       0.1 );
      else
         points->SetNextPoint( iData.standAloneMuon()->outerPosition().eta(),
			       iData.standAloneMuon()->outerPosition().phi(),
			       0.1 );
      return;
   } 
   
   // WARNING: use direction at POCA as the last option
   points->SetNextPoint( iData.eta(), iData.phi(), 0.1 );
}

REGISTER_FWPROXYBUILDER( FWMuonLegoProxyBuilder, reco::Muon, "Muons", FWViewType::kAllLegoBits );


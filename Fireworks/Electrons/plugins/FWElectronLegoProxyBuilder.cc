// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWElectronLegoProxyBuilder
//
// $Id: FWElectronLegoProxyBuilder.cc,v 1.5 2010/12/01 11:41:36 amraktad Exp $
//

#include "TEveStraightLineSet.h"
#include "TEveTrack.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

class FWElectronLegoProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::GsfElectron>  
{
public:
  FWElectronLegoProxyBuilder() {}
  virtual ~FWElectronLegoProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWElectronLegoProxyBuilder(const FWElectronLegoProxyBuilder&);
   const FWElectronLegoProxyBuilder& operator=(const FWElectronLegoProxyBuilder&);

   virtual void build(const reco::GsfElectron& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);
};

void FWElectronLegoProxyBuilder::build(const reco::GsfElectron& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   TEveStraightLineSet *marker = new TEveStraightLineSet("marker");
   setupAddElement(marker, &oItemHolder);
 
   TEveTrack* track(0);
   
   if( iData.gsfTrack().isAvailable() )
     track = fireworks::prepareTrack(*iData.gsfTrack(), context().getTrackPropagator());     
   else
     track = fireworks::prepareCandidate(iData, context().getTrackPropagator());
         
   track->MakeTrack();
   const double delta = 0.1;
   marker->AddLine(track->GetEndMomentum().Eta()-delta, track->GetEndMomentum().Phi()-delta, 0.1,
		   track->GetEndMomentum().Eta()+delta, track->GetEndMomentum().Phi()+delta, 0.1);
   marker->AddLine(track->GetEndMomentum().Eta()-delta, track->GetEndMomentum().Phi()+delta, 0.1,
		   track->GetEndMomentum().Eta()+delta, track->GetEndMomentum().Phi()-delta, 0.1);
   marker->AddLine(track->GetEndMomentum().Eta(), track->GetEndMomentum().Phi()-delta, 0.1,
		   track->GetEndMomentum().Eta(), track->GetEndMomentum().Phi()+delta, 0.1);
   marker->AddLine(track->GetEndMomentum().Eta()-delta, track->GetEndMomentum().Phi(), 0.1,
		   track->GetEndMomentum().Eta()+delta, track->GetEndMomentum().Phi(), 0.1);
}

REGISTER_FWPROXYBUILDER(FWElectronLegoProxyBuilder, reco::GsfElectron, "Electrons", FWViewType::kAllLegoBits);

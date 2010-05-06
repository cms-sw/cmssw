// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWElectronLegoProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu Dec  4 19:28:07 EST 2008
// $Id: FWElectronLegoBuilder.cc,v 1.6 2010/05/03 15:47:42 amraktad Exp $
//

#include "TEvePointSet.h"
#include "TEveTrack.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "Fireworks/Candidates/interface/CandidateUtils.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
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
   TEvePointSet* points = new TEvePointSet("points");
   setupAddElement(points, &oItemHolder);
 
   points->SetMarkerStyle(2);
   points->SetMarkerSize(0.2);
    
   TEveTrack* track;
   
   if( iData.gsfTrack().isAvailable() )
     track = fireworks::prepareTrack(*iData.gsfTrack(), context().getTrackPropagator());     
   else
     track = fireworks::prepareCandidate(iData, context().getTrackPropagator());
         
   track->MakeTrack();
   points->SetNextPoint(track->GetEndMomentum().Eta(), track->GetEndMomentum().Phi(), 0.1);
}

REGISTER_FWPROXYBUILDER(FWElectronLegoProxyBuilder, reco::GsfElectron, "Electrons", FWViewType::kLegoBit);

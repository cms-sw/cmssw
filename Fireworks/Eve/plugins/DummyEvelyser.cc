// -*- C++ -*-
//
// Package:     Fireworks/Eve
// Class  :     DummyEvelyser
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Matevz Tadel
//         Created:  Mon Jun 28 18:17:47 CEST 2010
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Fireworks/Eve/interface/EveService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"


class DummyEvelyser : public edm::EDAnalyzer
{
public:
  explicit DummyEvelyser(const edm::ParameterSet&);
  ~DummyEvelyser();

private:
  virtual void beginJob() {}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}


   edm::InputTag trackTags_;
};

DEFINE_FWK_MODULE(DummyEvelyser);


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//==============================================================================
// constructors and destructor
//==============================================================================

DummyEvelyser::DummyEvelyser(const edm::ParameterSet& iConfig) :
   trackTags_(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))
{
}

DummyEvelyser::~DummyEvelyser()
{
}


//==============================================================================
// member functions
//==============================================================================

void DummyEvelyser::analyze(const edm::Event& iEvent, const edm::EventSetup&)
{
   edm::Service<EveService> eve;
   eve->getManager(); // Returns TEveManager, it is also set in global gEve.

   // Stripped down demo from twiki

   using namespace edm;
   // using reco::TrackCollection;

   edm::Handle<View<reco::Track> >  tracks;
   iEvent.getByLabel(trackTags_,tracks);


   TEveTrackList *tl = new TEveTrackList("Tracks"); 
   tl->SetMainColor(6);
   tl->SetMarkerColor(kYellow);
   tl->SetMarkerStyle(4);
   tl->SetMarkerSize(0.5);

   gEve->AddElement(tl);

   TEveTrackPropagator *prop = tl->GetPropagator();
   // Guess mag field ...
   prop->SetMagField(-3.8);
   prop->SetFitReferences(kFALSE);
   prop->SetFitDaughters(kFALSE);
   prop->SetFitDecay(kFALSE);
   prop->SetStepper(TEveTrackPropagator::kRungeKutta);


   int cnt = 0;
   for (View<reco::Track>::const_iterator itTrack = tracks->begin();
        itTrack != tracks->end(); ++itTrack, ++cnt)
   {
      TEveTrack* trk = fireworks::prepareTrack(*itTrack, prop);
      trk->SetElementName (TString::Format("Track %d", cnt));
      trk->SetElementTitle(TString::Format("Track %d, pt=%.3f", cnt, itTrack->pt()));
      trk->MakeTrack();
      trk->SetAttLineAttMarker(tl);
      tl->AddElement(trk);
   }

   tl->MakeTracks();
}


//
// const member functions
//

//
// static member functions
//

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
// $Id: DummyEvelyser.cc,v 1.4 2010/07/07 18:11:13 matevz Exp $
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Fireworks/Eve/interface/EveService.h"

#include "Fireworks/Geometry/interface/DisplayGeomRecord.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TEveGeoNode.h"
#include "TEveTrans.h"
#include "TEveScene.h"
#include "TGLScenePad.h"
#include "TGLRnrCtx.h"

class DummyEvelyser : public edm::EDAnalyzer
{
public:
  explicit DummyEvelyser(const edm::ParameterSet&);
  ~DummyEvelyser();

protected:
   TEveGeoTopNode* make_node(const TString& path, Int_t vis_level, Bool_t global_cs);

private:
   virtual void beginJob();
   virtual void endJob();

   virtual void beginRun(const edm::Run&, const edm::EventSetup&);
   virtual void endRun  (const edm::Run&, const edm::EventSetup&);

   virtual void analyze(const edm::Event&, const edm::EventSetup&);

   edm::InputTag  trackTags_;
   TEveElement   *m_geomList;
   TEveTrackList *m_trackList;

   edm::ESWatcher<DisplayGeomRecord> m_geomWatcher;
   void remakeGeometry(const DisplayGeomRecord& dgRec);
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
   trackTags_(iConfig.getUntrackedParameter<edm::InputTag>("tracks")),
   m_geomList(0),
   m_trackList(0),
   m_geomWatcher(this, &DummyEvelyser::remakeGeometry)
{}

DummyEvelyser::~DummyEvelyser()
{}


//==============================================================================
// Protected helpers
//==============================================================================

TEveGeoTopNode* DummyEvelyser::make_node(const TString& path, Int_t vis_level, Bool_t global_cs)
{
   if (! gGeoManager->cd(path))
   {
      Warning("make_node", "Path '%s' not found.", path.Data());
      return 0;
   }

   TEveGeoTopNode* tn = new TEveGeoTopNode(gGeoManager, gGeoManager->GetCurrentNode());
   tn->SetVisLevel(vis_level);
   if (global_cs)
   {
      tn->RefMainTrans().SetFrom(*gGeoManager->GetCurrentMatrix());
   }
   m_geomList->AddElement(tn);

   return tn;
}


//==============================================================================
// member functions
//==============================================================================

void DummyEvelyser::beginJob()
{
   printf("DummyEvelyser::beginJob\n");

   edm::Service<EveService> eve;
   eve->getManager(); // Returns TEveManager, it is also set in global gEve.

   // Make a track-list container we'll hold on until the end of the job.
   // This allows us to preserve settings done by user via GUI.
   m_trackList = new TEveTrackList("Tracks"); 
   m_trackList->SetMainColor(6);
   m_trackList->SetMarkerColor(kYellow);
   m_trackList->SetMarkerStyle(4);
   m_trackList->SetMarkerSize(0.5);

   m_trackList->IncDenyDestroy();

   TEveTrackPropagator *prop = m_trackList->GetPropagator();
   prop->SetStepper(TEveTrackPropagator::kRungeKutta);
   // Use simplified magnetic field.
   eve->setupFieldForPropagator(prop);
}

void DummyEvelyser::endJob()
{
   printf("DummyEvelyser::endJob\n");

   m_trackList->DecDenyDestroy();
   m_trackList = 0;
}

//------------------------------------------------------------------------------

void DummyEvelyser::beginRun(const edm::Run&, const edm::EventSetup& iSetup)
{
   printf("DummyEvelyser::beginRun\n");

   m_geomList = new TEveElementList("DummyEvelyzer Geom");
   gEve->AddGlobalElement(m_geomList);
   gEve->GetGlobalScene()->GetGLScene()->SetStyle(TGLRnrCtx::kWireFrame);
}

void DummyEvelyser::endRun(const edm::Run&, const edm::EventSetup&)
{
   printf("DummyEvelyser::endRun\n");
}

//------------------------------------------------------------------------------

void DummyEvelyser::remakeGeometry(const DisplayGeomRecord& dgRec)
{
   m_geomList->DestroyElements();

   edm::ESHandle<TGeoManager> geom;
   dgRec.get(geom);
   TEveGeoManagerHolder _tgeo(const_cast<TGeoManager*>(geom.product()));

   // To have a full one, all detectors in one top-node:
   // make_node("/cms:World_1/cms:CMSE_1", 4, kTRUE);

   make_node("/cms:World_1/cms:CMSE_1/tracker:Tracker_1", 1, kTRUE);
   make_node("/cms:World_1/cms:CMSE_1/caloBase:CALO_1",   1, kTRUE);
   make_node("/cms:World_1/cms:CMSE_1/muonBase:MUON_1",   1, kTRUE);
}

void DummyEvelyser::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   printf("DummyEvelyser::analyze\n");


   // Remake geometry if it has changed.

   m_geomWatcher.check(iSetup);


   // Stripped down demo from Tracking twiki.

   using namespace edm;

   Handle<View<reco::Track> >  tracks;
   iEvent.getByLabel(trackTags_,tracks);

   m_trackList->DestroyElements();

   int cnt = 0;
   for (View<reco::Track>::const_iterator itTrack = tracks->begin();
        itTrack != tracks->end(); ++itTrack, ++cnt)
   {
      TEveTrack* trk = fireworks::prepareTrack(*itTrack, m_trackList->GetPropagator());
      trk->SetElementName (TString::Format("Track %d", cnt));
      trk->SetElementTitle(TString::Format("Track %d, pt=%.3f", cnt, itTrack->pt()));
      trk->MakeTrack();
      trk->SetAttLineAttMarker(m_trackList);
      m_trackList->AddElement(trk);
   }

   m_trackList->MakeTracks();

   gEve->AddElement(m_trackList);
}

//
// const member functions
//

//
// static member functions
//

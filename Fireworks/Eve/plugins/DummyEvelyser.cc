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
#include "DataFormats/TrackReco/interface/TrackFwd.h"

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
  ~DummyEvelyser() override;
   edm::EDGetTokenT<reco::TrackCollection > trackCollectionToken_;
protected:
   TEveGeoTopNode* make_node(const TString& path, Int_t vis_level, Bool_t global_cs);

   

private:
   void beginJob() override;
   void endJob() override;

   void beginRun(const edm::Run&, const edm::EventSetup&) override;
   void endRun  (const edm::Run&, const edm::EventSetup&) override;

   void analyze(const edm::Event&, const edm::EventSetup&) override;

   edm::Service<EveService>  m_eve;

   edm::InputTag  m_trackTags;
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
   m_eve(),
   m_trackTags(iConfig.getUntrackedParameter<edm::InputTag>("tracks")),
   m_geomList(nullptr),
   m_trackList(nullptr),
   m_geomWatcher(this, &DummyEvelyser::remakeGeometry)
{
   trackCollectionToken_ =  consumes<reco::TrackCollection >(m_trackTags);
}

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
      return nullptr;
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

   if (m_eve)
   {
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
      // Use simplified magnetic field provided by EveService.
      m_eve->setupFieldForPropagator(prop);
   }
}

void DummyEvelyser::endJob()
{
   printf("DummyEvelyser::endJob\n");

   if (m_trackList)
   {
      m_trackList->DecDenyDestroy();
      m_trackList = nullptr;
   }
}

//------------------------------------------------------------------------------

void DummyEvelyser::beginRun(const edm::Run&, const edm::EventSetup& iSetup)
{
   printf("DummyEvelyser::beginRun\n");

   if (m_eve)
   {
      m_geomList = new TEveElementList("DummyEvelyzer Geom");
      m_eve->AddGlobalElement(m_geomList);
      m_eve->getManager()->GetGlobalScene()->GetGLScene()->SetStyle(TGLRnrCtx::kWireFrame);
   }
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

  edm::Handle<reco::TrackCollection> trackHandle;
   iEvent.getByToken(trackCollectionToken_, trackHandle);
   const reco::TrackCollection trackCollection = *(trackHandle.product());
   if (!trackHandle.isValid()) {
      edm::LogError("DummyEvelyser")
         << "Error! Can't get Track collection "
         << std::endl;
      return;
   }
   
   if (m_eve)
   {
      // Remake geometry if it has changed.
      m_geomWatcher.check(iSetup);

      // Stripped down demo from Tracking twiki.
 
      using namespace edm;

      m_trackList->DestroyElements();

      // All top-level elements are removed from default event-store at
      // the end of each event.
      m_eve->AddElement(m_trackList);

      int cnt = 0;
      
      for (auto itTrack = trackCollection.begin();
           itTrack != trackCollection.end(); ++itTrack, ++cnt)
      {
         TEveTrack* trk = fireworks::prepareTrack(*itTrack, m_trackList->GetPropagator());
         trk->SetElementName (TString::Format("Track %d", cnt));
         trk->SetElementTitle(TString::Format("Track %d, pt=%.3f", cnt, itTrack->pt()));
         trk->MakeTrack();
         
         trk->SetAttLineAttMarker(m_trackList);
         m_trackList->AddElement(trk);   
      }

      
      // The display() function runs the GUI event-loop and shows
      // whatever has been registered so far to eve.
      // It returns when user presses the "Step" button (or "Continue" or
      // "Next Event").
      m_eve->display(Form("DummyEvelyser::analyze done for %d tracks\n", m_trackList->NumChildren()) );
   }
}

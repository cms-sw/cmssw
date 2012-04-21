
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
#include "TEvePointSet.h"
#include "TRandom.h"
#include "TEveUtil.h"

// class decleration
//
       
class DisplayGeom : public edm::EDAnalyzer {

public:
   explicit DisplayGeom(const edm::ParameterSet&);
   ~DisplayGeom();

protected:
   TEveGeoTopNode* make_node(const TString& path, Int_t vis_level, Bool_t global_cs);

  
private:
   virtual void beginJob() ;
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void endJob() ;

   edm::Service<EveService>  m_eve;
   TEveElement   *m_geomList;
   edm::ESWatcher<DisplayGeomRecord> m_geomWatcher;

   void remakeGeometry(const DisplayGeomRecord& dgRec);

};

DisplayGeom::DisplayGeom(const edm::ParameterSet& iConfig):
   m_eve(),
   m_geomList(0),
   m_geomWatcher(this, &DisplayGeom::remakeGeometry)
{}


DisplayGeom::~DisplayGeom()
{
}


//==============================================================================
// Protected helpers
//==============================================================================

TEveGeoTopNode* DisplayGeom::make_node(const TString& path, Int_t vis_level, Bool_t global_cs)
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


void
DisplayGeom::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   printf("DisplayGeom::analyze\n");

   if (m_eve)
   {
      // Add a test obj
      if (!gRandom)
         gRandom = new TRandom(0);
      TRandom& r= *gRandom;

      Float_t s = 100;

      TEvePointSet* ps = new TEvePointSet();
      ps->SetOwnIds(kTRUE);

      for(Int_t i = 0; i< 100; i++)
      {
         ps->SetNextPoint(r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
         ps->SetPointId(new TNamed(Form("Point %d", i), ""));
      }

      ps->SetMarkerColor(TMath::Nint(r.Uniform(2, 9)));
      ps->SetMarkerSize(r.Uniform(1, 2));
      ps->SetMarkerStyle(4);
      m_eve->AddElement(ps);
   }
}

// ------------ method called once each job just before starting event loop  ------------
void 
DisplayGeom::beginJob()
{ 
   if (m_eve)
   {
      m_geomList = new TEveElementList("Display Geom");
      m_eve->AddGlobalElement(m_geomList);
      m_eve->getManager()->GetGlobalScene()->GetGLScene()->SetStyle(TGLRnrCtx::kWireFrame);
   }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DisplayGeom::endJob() {
}

//------------------------------------------------------------------------------
void DisplayGeom::remakeGeometry(const DisplayGeomRecord& dgRec)
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

// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDisplayEvent
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Mon Dec  3 08:38:38 PST 2007
// $Id: FWDisplayEvent.cc,v 1.3 2007/12/06 20:18:44 chrjones Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TEveBrowser.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveGeoNode.h"
#include "TSystem.h"
#include "TEveProjectionManager.h"
#include "TEveScene.h"
#include "TGLViewer.h"

//geometry
#include "TFile.h"
#include "TEveGeoShapeExtract.h"
#include "TROOT.h"

#include "TGButton.h"

//needed to work around a bug
#include "TApplication.h"

// user include files
#include "Fireworks/Core/interface/FWDisplayEvent.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
static
void make_tracks_rhophi(const fwlite::Event* iEvent,
			TEveElementList** oList)
{
  fwlite::Handle<reco::TrackCollection> tracks;
  tracks.getByLabel(*iEvent,"ctfWithMaterialTracks");
  
  if(0 == tracks.ptr() ) {
    std::cout <<"failed to get Tracks"<<std::endl;
  }

  if(0 == *oList) {
    TEveTrackList* tlist =  new TEveTrackList("Tracks");
    *oList =tlist;
    (*oList)->SetMainColor(Color_t(3));
    TEveTrackPropagator* rnrStyle = tlist->GetPropagator();
    //units are kG
    rnrStyle->SetMagField( -4.0*10.);
    //get this from geometry, units are CM
    rnrStyle->SetMaxR(120.0);
    rnrStyle->SetMaxZ(300.0);
    
    gEve->AddElement(*oList);
  } else {
    (*oList)->DestroyElements();
  }
  //since we created it, we know the type (would like to do this better)
  TEveTrackList* tlist = dynamic_cast<TEveTrackList*>(*oList);

  TEveTrackPropagator* rnrStyle = tlist->GetPropagator();
  
  int index=0;
  //cout <<"----"<<endl;
  TEveRecTrack t;
  t.beta = 1.;
  for(reco::TrackCollection::const_iterator it = tracks->begin();
      it != tracks->end();++it,++index) {
    t.P = TEveVector(it->px(),
		     it->py(),
		     it->pz());
    t.V = TEveVector(it->vx(),
		     it->vy(),
		     it->vz());
    t.sign = it->charge();

    TEveTrack* trk = new TEveTrack(&t,rnrStyle);
    trk->SetMainColor((*oList)->GetMainColor());
    gEve->AddElement(trk,(*oList));
    //cout << it->px()<<" "
    //   <<it->py()<<" "
    //   <<it->pz()<<endl;
    //cout <<" *";
  }
}


//
// constructors and destructor
//
FWDisplayEvent::FWDisplayEvent() :
  m_continueProcessingEvents(false),
  m_waitForUserAction(true),
  m_rhoPhiProjMgr(0)

{
  m_physicsTypes.push_back("Tracks");
  m_physicsElements.push_back(0);

  //These are only needed temporarilty to work around a problem which 
  // Matevz has patched in a later version of the code
  TApplication::NeedGraphicsLibs();
  gApplication->InitializeGraphics();

  TEveManager::Create();
  TEveBrowser* browser = gEve->GetBrowser();
  //should check to see if already has our tab
  {
    browser->StartEmbedding(TRootBrowser::kLeft);
    {
      TGMainFrame* frmMain=new TGMainFrame(gClient->GetRoot(),
					   1000,
					   600);
      frmMain->SetWindowName("GUI");
      frmMain->SetCleanup(kDeepCleanup);

      TGHorizontalFrame* hf = new TGHorizontalFrame(frmMain);
      //We need an error handling system which can properly report
      // errors and decide what to do
      // given that we are an interactive system we need to leave
      // the code in a good state so that users can decided to 
      // continue or not
      {
	if(0==gSystem->Getenv("ROOTSYS")) {
	  std::cerr<<"environment variable ROOTSYS is not set" <<
	    std::endl;
	  throw std::runtime_error("ROOTSYS environment variable not set");
	}
	TString icondir(Form("%s/icons/",gSystem->Getenv("ROOTSYS")));
	
	m_advanceButton= new TGPictureButton(hf,
			 gClient->GetPicture(icondir+"GoForward.gif"));
	hf->AddFrame(m_advanceButton);
	m_advanceButton->Connect("Clicked()",
				 "FWDisplayEvent",
				 this,
				 "continueProcessingEvents()");
      }
      frmMain->AddFrame(hf);

      frmMain->MapSubwindows();
      frmMain->Resize();
      frmMain->MapWindow();
    }
    browser->StopEmbedding();
    browser->SetTabTitle("Event Control",0);
  }

  //setup projection
  TEveViewer* nv = gEve->SpawnNewViewer("Rho Phi");
  nv->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
  TEveScene* ns = gEve->SpawnNewScene("Rho Phi");
  nv->AddScene(ns);

  m_rhoPhiProjMgr = new TEveProjectionManager;
  gEve->AddToListTree(m_rhoPhiProjMgr,kTRUE);
  gEve->AddElement(m_rhoPhiProjMgr,ns);

  //handle geometry
  /*
  gGeoManager = gEve->GetGeometry("cmsGeom20.root");

  TEveElementList* gL = 
    new TEveElementList("CMS");
  gEve->AddGlobalElement(gL);

  TGeoNode* node = gGeoManager->GetTopVolume()->GetNode(0);
  TEveGeoTopNode* re = new TEveGeoTopNode(gGeoManager,
					  node);
  re->UseNodeTrans();
  gEve->AddGlobalElement(re,gL);
  */
  TFile f("tracker.root");
  if(not f.IsOpen()) {
    std::cerr <<"failed to open 'tracker.root'"<<std::endl;
    throw std::runtime_error("Failed to open 'tracker.root' geometry file");
  }
  TEveGeoShapeExtract* gse = dynamic_cast<TEveGeoShapeExtract*>(f.Get("Tracker"));
  TEveGeoShape* gsre = TEveGeoShape::ImportShapeExtract(gse,0);
  f.Close();
  m_geom = gsre;

  //kTRUE tells it to reset the camera so we see everything 
  gEve->Redraw3D(kTRUE);  

  gSystem->ProcessEvents();
}

// FWDisplayEvent::FWDisplayEvent(const FWDisplayEvent& rhs)
// {
//    // do actual copying here;
// }

FWDisplayEvent::~FWDisplayEvent()
{
}

//
// assignment operators
//
// const FWDisplayEvent& FWDisplayEvent::operator=(const FWDisplayEvent& rhs)
// {
//   //An exception safe implementation is
//   FWDisplayEvent temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWDisplayEvent::continueProcessingEvents()
{
  m_continueProcessingEvents = true;
}

void
FWDisplayEvent::waitForUserAction()
{
  m_waitForUserAction = true;
}

void
FWDisplayEvent::doNotWaitForUserAction()
{
  m_waitForUserAction = false;
}

//
// const member functions
//
bool
FWDisplayEvent::waitingForUserAction() const
{
  return m_waitForUserAction;
}

namespace {
  //guarantee that no matter how we go back to Cint that
  // we have disabled these buttons
  struct EnableButton {
    EnableButton( TGButton* iButton):
      m_button(iButton)
    {
      if(0!=m_button) {
	m_button->SetEnabled();
      }
    }
    ~EnableButton()
    {
      m_button->SetEnabled(kFALSE);
      gSystem->DispatchOneEvent(kFALSE);
    }

  private:
    TGButton* m_button;
  };

  struct DisableRedraw {
    DisableRedraw() {
      gEve->DisableRedraw();
    }
    ~DisableRedraw() {
      gEve->EnableRedraw();
    }
  };
}
void
FWDisplayEvent::draw(const fwlite::Event& iEvent) const
{
  //need to reset 
  m_continueProcessingEvents = false;
  EnableButton advancedB(m_advanceButton);

  using namespace std;
  if(0==gEve) {
    cout <<"Eve not initialized"<<endl;
  }
  {
    //while inside this scope, do not let
    // Eve do any redrawing
    DisableRedraw disableRedraw;

    make_tracks_rhophi(&iEvent,
		       &(m_physicsElements.front()));
    
    //setup the projection
    m_rhoPhiProjMgr->DestroyElements();
    m_rhoPhiProjMgr->ImportElements(m_geom);
    for(std::vector<TEveElementList*>::iterator it = m_physicsElements.begin();
	it != m_physicsElements.end();
	++it) {
      m_rhoPhiProjMgr->ImportElements(*it);
    }

    //At the end of this scope, redrawing will be enabled
  }
  
  //check for input at least once
  gSystem->ProcessEvents();
  while(not gROOT->IsInterrupted() and
	m_waitForUserAction and 
	not m_continueProcessingEvents) {
    //gSystem->ProcessEvents();
    gSystem->DispatchOneEvent(kFALSE);
  }
}

//
// static member functions
//

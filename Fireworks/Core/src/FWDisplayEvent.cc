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
// $Id: FWDisplayEvent.cc,v 1.6 2007/12/09 22:49:23 chrjones Exp $
//

// system include files
#include <sstream>
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TEveBrowser.h"
#include "TEveGeoNode.h"
#include "TSystem.h"
#include "TEveProjectionManager.h"
#include "TEveScene.h"
#include "TGLViewer.h"
#include "TClass.h"

//geometry
#include "TFile.h"
#include "TEveGeoShapeExtract.h"
#include "TROOT.h"

#include "TGButton.h"

//needed to work around a bug
#include "TApplication.h"

// user include files
#include "Fireworks/Core/interface/FWDisplayEvent.h"
#include "Fireworks/Core/interface/FWDataProxyBuilder.h"
#include "DataFormats/FWLite/interface/Event.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWDisplayEvent::FWDisplayEvent() :
  m_continueProcessingEvents(false),
  m_waitForUserAction(true),
  m_rhoPhiProjMgr(0)

{
  const char* cmspath = gSystem->Getenv("CMSSW_BASE");
  if(0 == cmspath) {
    throw std::runtime_error("CMSSW_BASE environment variable not set");
  }
  //tell ROOT where to find our macros
  std::string macPath(cmspath);
  macPath += "/src/Fireworks/Core/macros";
  gROOT->SetMacroPath(macPath.c_str());  

  //will eventually get this info from the configuration
  m_physicsTypes.push_back("Tracks");
  m_physicsElements.push_back(0);
  m_physicsProxyBuilderNames.push_back("TracksProxy3DBuilder");

  // register muons
  m_physicsTypes.push_back("Muons");
  m_physicsElements.push_back(0);
  m_physicsProxyBuilderNames.push_back("MuonsProxy3DBuilder");

   
  //create proxy builders
  Int_t error;
  TClass *baseClass = TClass::GetClass(typeid(FWDataProxyBuilder));
  assert(baseClass !=0);

  for(std::vector<std::string>::iterator itBuilderName=
	m_physicsProxyBuilderNames.begin();
      itBuilderName !=m_physicsProxyBuilderNames.end();
      ++itBuilderName) {
    //does the class already exist?
    TClass *c = TClass::GetClass(itBuilderName->c_str());
    if(0==c) {
      //try to load a macro of that name
      
      //How can I tell if this succeeds or failes? error and value are always 0!
      // I could not make the non-compiled mechanism to work without seg-faults
      Int_t value = gROOT->LoadMacro(((*itBuilderName)+".C+").c_str(),&error);
      c = TClass::GetClass(itBuilderName->c_str());
      if(0==c ) {
	std::cerr <<"failed to find "<<*itBuilderName<<std::endl;
	continue;
      }
    }
    FWDataProxyBuilder* builder=0;
    void* inst = c->New();
    builder = reinterpret_cast<FWDataProxyBuilder*>(c->DynamicCast(baseClass,inst));
    if(0==builder) {
      std::cerr<<"conversion to FWDataProxyBuilder failed"<<std::endl;
      continue;
    }
    m_builders.push_back(boost::shared_ptr<FWDataProxyBuilder>(builder));
  }


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

}
void
FWDisplayEvent::draw(const fwlite::Event& iEvent)
{
  const FWDisplayEvent* c = this;
  c->draw(iEvent);
}

void
FWDisplayEvent::draw(const fwlite::Event& iEvent) const
{
  // FIXIT: vectors should be aligned a bit better than that
  if ( m_physicsTypes.size() != m_physicsElements.size() ||
       m_physicsElements.size() != m_physicsProxyBuilderNames.size() ||
       m_physicsProxyBuilderNames.size() != m_builders.size() ) 
     {
	std::cout << "Vectors of proxy data have different size - fatal error" << std::endl;
	return;
     }
   
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
    TEveManager::TRedrawDisabler disableRedraw(gEve);
    
    // build models
    for ( unsigned int iProxy = 0; iProxy < m_builders.size(); ++iProxy )
       m_builders[iProxy]->build( &iEvent, &(m_physicsElements[iProxy]) );
     
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

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
// $Id: FWDisplayEvent.cc,v 1.18 2008/01/22 16:34:08 chrjones Exp $
//

// system include files
#include <sstream>
#include <boost/bind.hpp>
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TEveBrowser.h"
#include "TSystem.h"
#include "TClass.h"

//geometry
#include "TFile.h"
#include "TROOT.h"

#include "TGButton.h"
#include "TGComboBox.h"
#include "TGTextEntry.h"

//needed to work around a bug
#include "TApplication.h"

// user include files
#include "Fireworks/Core/interface/FWDisplayEvent.h"
#include "Fireworks/Core/interface/FWRhoPhiZViewManager.h"
#include "Fireworks/Core/interface/FW3DLegoViewManager.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"
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
  m_changeManager(new FWModelChangeManager),
  m_selectionManager(new FWSelectionManager(m_changeManager.get())),
  m_eiManager(new FWEventItemsManager(m_changeManager.get(),
                                      m_selectionManager.get())),
  m_viewManager( new FWViewManagerManager(m_changeManager.get())),
  m_continueProcessingEvents(false),
  m_waitForUserAction(true),
  m_code(0)
{
  //connect up the managers
   m_eiManager->newItem_.connect(boost::bind(&FWModelChangeManager::newItemSlot,
                                             m_changeManager.get(), _1) );
   
  //figure out where to find macros
  const char* cmspath = gSystem->Getenv("CMSSW_BASE");
  if(0 == cmspath) {
    throw std::runtime_error("CMSSW_BASE environment variable not set");
  }
  //tell ROOT where to find our macros
  std::string macPath(cmspath);
  macPath += "/src/Fireworks/Core/macros";
  gROOT->SetMacroPath(macPath.c_str());  


  // These are only needed temporarilty to work around a problem which 
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
	
	m_homeButton= new TGPictureButton(hf, gClient->GetPicture(icondir+"GoHome.gif"));
	hf->AddFrame(m_homeButton);
	m_homeButton->Connect("Clicked()", "FWDisplayEvent", this, "goHome()");
	 
	m_advanceButton= new TGPictureButton(hf, gClient->GetPicture(icondir+"GoForward.gif"));
	hf->AddFrame(m_advanceButton);
	m_advanceButton->Connect("Clicked()", "FWDisplayEvent", this, "goForward()");
	
	m_backwardButton= new TGPictureButton(hf, gClient->GetPicture(icondir+"GoBack.gif"));
	hf->AddFrame(m_backwardButton);
	m_backwardButton->Connect("Clicked()", "FWDisplayEvent", this, "goBack()");
	 
	m_stopButton= new TGPictureButton(hf, gClient->GetPicture(icondir+"StopLoading.gif"));
	hf->AddFrame(m_stopButton);
	m_stopButton->Connect("Clicked()", "FWDisplayEvent", this, "stop()");
	
      }
      frmMain->AddFrame(hf);

      TGGroupFrame* vf = new TGGroupFrame(frmMain,"Selection",kVerticalFrame);
      {
        TGGroupFrame* vf2 = new TGGroupFrame(vf,"Expression");
        m_selectionItemsComboBox = new TGComboBox(vf2,200);
        m_selectionItemsComboBox->Resize(200,20);
        vf2->AddFrame(m_selectionItemsComboBox, new TGLayoutHints(kLHintsTop | kLHintsLeft,5,5,5,5));
        m_selectionExpressionEntry = new TGTextEntry(vf2,"$.pt() > 10");
        vf2->AddFrame(m_selectionExpressionEntry);
        m_selectionRunExpressionButton = new TGTextButton(vf2,"Select by Expression");
        vf2->AddFrame(m_selectionRunExpressionButton);
        m_selectionRunExpressionButton->Connect("Clicked()","FWDisplayEvent",this,"selectByExpression()");
        vf->AddFrame(vf2);
        
        m_unselectAllButton = new TGTextButton(vf,"Unselect All");
        m_unselectAllButton->Connect("Clicked()", "FWDisplayEvent",this,"unselectAll()");
        vf->AddFrame(m_unselectAllButton);
      }
      frmMain->AddFrame(vf);
      frmMain->MapSubwindows();
      frmMain->Resize();
      frmMain->MapWindow();
    }
    browser->StopEmbedding();
    browser->SetTabTitle("Fireworks",0);
  }

   
  // prepare geometry service
  // ATTN: this should be made configurable
  const char* geomtryFile = "cmsGeom10.root";
  m_detIdToGeo.loadGeometry( geomtryFile );
  m_detIdToGeo.loadMap( geomtryFile );
   
  boost::shared_ptr<FWViewManagerBase> rpzViewManager( new FWRhoPhiZViewManager() );
  rpzViewManager->setGeom(&m_detIdToGeo);
  m_viewManager->add(rpzViewManager);
  m_viewManager->add( boost::shared_ptr<FWViewManagerBase>( new FW3DLegoViewManager()));
   
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
void FWDisplayEvent::registerPhysicsObject(const FWPhysicsObjectDesc&iItem)
{
  const FWEventItem* newItem = m_eiManager->add(iItem);
  m_viewManager->registerEventItem(newItem);
  m_selectionItemsComboBox->AddEntry(newItem->name().c_str(),newItem->id());
  if(newItem->id()==0) {
    m_selectionItemsComboBox->Select(0);
  }
}

void FWDisplayEvent::registerProxyBuilder(const std::string& type, 
					  const std::string& proxyBuilderName)
{
  m_viewManager->registerProxyBuilder(type,proxyBuilderName);
}

void
FWDisplayEvent::goForward()
{
  m_continueProcessingEvents = true;
  m_code = 1;
}

void
FWDisplayEvent::goBack()
{
  m_continueProcessingEvents = true;
  m_code = -1;
}

void
FWDisplayEvent::goHome()
{
  m_continueProcessingEvents = true;
  m_code = -2;
}

void
FWDisplayEvent::stop()
{
  m_continueProcessingEvents = true;
  m_code = -3;
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

void 
FWDisplayEvent::selectByExpression()
{
  FWModelExpressionSelector selector;
  selector.select(*(m_eiManager->begin()+m_selectionItemsComboBox->GetSelected()),
                  m_selectionExpressionEntry->GetText());
}

void 
FWDisplayEvent::unselectAll()
{
  m_selectionManager->clearSelection();
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
int
FWDisplayEvent::draw(const fwlite::Event& iEvent)
{
  const FWDisplayEvent* c = this;
  return c->draw(iEvent);
}

int
FWDisplayEvent::draw(const fwlite::Event& iEvent) const
{
  //need to reset
  m_continueProcessingEvents = false;
  EnableButton homeB(m_homeButton);
  EnableButton advancedB(m_advanceButton);
  EnableButton backwardB(m_backwardButton);
  EnableButton stopB(m_stopButton);
  EnableButton stopUnselect(m_unselectAllButton);
  EnableButton stopSelect(m_selectionRunExpressionButton);
  
  using namespace std;
  if(0==gEve) {
    cout <<"Eve not initialized"<<endl;
  }
   
  m_eiManager->newEvent(&iEvent);
  m_eiManager->setGeom(&m_detIdToGeo);
  m_viewManager->newEventAvailable();

  //check for input at least once
  gSystem->ProcessEvents();
  while(not gROOT->IsInterrupted() and
	m_waitForUserAction and 
	not m_continueProcessingEvents) {
     // gSystem->ProcessEvents();
     gSystem->DispatchOneEvent(kFALSE);
  }
  m_selectionManager->eventDone();
  return m_code;
}

//
// static member functions
//


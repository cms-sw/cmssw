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
// $Id: FWDisplayEvent.cc,v 1.24 2008/02/11 20:19:33 chrjones Exp $
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
#include "Fireworks/Core/interface/FWGUIManager.h"
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
  m_guiManager(new FWGUIManager(m_selectionManager.get(),
                                m_eiManager.get())),
  m_viewManager( new FWViewManagerManager(m_changeManager.get())) //,
//  m_continueProcessingEvents(false),
//  m_waitForUserAction(true),
//  m_code(0)
{
  //connect up the managers
   m_eiManager->newItem_.connect(boost::bind(&FWModelChangeManager::newItemSlot,
                                             m_changeManager.get(), _1) );
   
  //m_selectionManager->selectionChanged_.connect(boost::bind(&FWDisplayEvent::selectionChanged,this,_1));
  //figure out where to find macros
  const char* cmspath = gSystem->Getenv("CMSSW_BASE");
  if(0 == cmspath) {
    throw std::runtime_error("CMSSW_BASE environment variable not set");
  }
  //tell ROOT where to find our macros
  std::string macPath(cmspath);
  macPath += "/src/Fireworks/Core/macros";
  gROOT->SetMacroPath(macPath.c_str());  

  // prepare geometry service
  // ATTN: this should be made configurable
  const char* geomtryFile = "cmsGeom10.root";
  m_detIdToGeo.loadGeometry( geomtryFile );
  m_detIdToGeo.loadMap( geomtryFile );
   
  boost::shared_ptr<FWViewManagerBase> rpzViewManager( new FWRhoPhiZViewManager(m_guiManager.get()) );
  rpzViewManager->setGeom(&m_detIdToGeo);
  m_viewManager->add(rpzViewManager);
  m_viewManager->add( boost::shared_ptr<FWViewManagerBase>( new FW3DLegoViewManager(m_guiManager.get())));
   
  m_guiManager->processGUIEvents();
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
}

void FWDisplayEvent::registerProxyBuilder(const std::string& type, 
					  const std::string& proxyBuilderName)
{
  m_viewManager->registerProxyBuilder(type,proxyBuilderName);
}

//
// const member functions
//
int
FWDisplayEvent::draw(const fwlite::Event& iEvent)
{
  const FWDisplayEvent* c = this;
  return c->draw(iEvent);
}

int
FWDisplayEvent::draw(const fwlite::Event& iEvent) const
{
  m_eiManager->setGeom(&m_detIdToGeo);
  m_eiManager->newEvent(&iEvent);
  return m_guiManager->allowInteraction();
}

//
// static member functions
//


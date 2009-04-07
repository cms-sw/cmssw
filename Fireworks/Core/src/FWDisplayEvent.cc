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
// $Id: FWDisplayEvent.cc,v 1.64 2009/01/23 21:35:42 amraktad Exp $
//

// system include files
#include <sstream>
#include <boost/bind.hpp>
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TEveBrowser.h"
#include "TSystem.h"
#include "TClass.h"
#include "TEveTrackProjected.h"
#include "TEveSelection.h"

//geometry
#include "TFile.h"
#include "TROOT.h"

#include "TGButton.h"
#include "TGComboBox.h"
#include "TGTextEntry.h"
#include "TStopwatch.h"

//needed to work around a bug
#include "TApplication.h"

// user include files
#include "Fireworks/Core/interface/FWDisplayEvent.h"
#include "Fireworks/Core/interface/FWRhoPhiZViewManager.h"
#include "Fireworks/Core/interface/FWEveLegoViewManager.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"
#include "Fireworks/Core/interface/FWTextView.h"
#include "DataFormats/FWLite/interface/Event.h"

#include "Fireworks/Core/interface/FWConfigurationManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
double FWDisplayEvent::m_magneticField = 4;
double FWDisplayEvent::m_caloScale = 2;

//
// constructors and destructor
//
FWDisplayEvent::FWDisplayEvent(const std::string& iConfigFileName,
                               bool iEnableDebug,
                               bool iNewLego) :
   m_configurationManager(new FWConfigurationManager),
   m_changeManager(new FWModelChangeManager),
   m_colorManager( new FWColorManager(m_changeManager.get())),
   m_selectionManager(new FWSelectionManager(m_changeManager.get())),
   m_eiManager(new FWEventItemsManager(m_changeManager.get(),
                                       m_selectionManager.get())),
   m_viewManager( new FWViewManagerManager(m_changeManager.get(),m_colorManager.get())),
   m_guiManager(new FWGUIManager(m_selectionManager.get(),
                                 m_eiManager.get(),
                                 m_changeManager.get(),
                                 m_colorManager.get(),
                                 m_viewManager.get(),
                                 iEnableDebug)),
//   m_textView(new FWTextView(this, &*m_selectionManager)),
   m_configFileName(iConfigFileName)
{
   //connect up the managers
   m_eiManager->newItem_.connect(boost::bind(&FWModelChangeManager::newItemSlot,
                                             m_changeManager.get(), _1) );

   m_eiManager->newItem_.connect(boost::bind(&FWViewManagerManager::registerEventItem,
                                             m_viewManager.get(), _1));
   m_configurationManager->add("EventItems",m_eiManager.get());
   m_configurationManager->add("GUI",m_guiManager.get());
   m_guiManager->writeToConfigurationFile_.connect(boost::bind(&FWConfigurationManager::writeToFile,
                                                               m_configurationManager.get(),_1));
   //m_selectionManager->selectionChanged_.connect(boost::bind(&FWDisplayEvent::selectionChanged,this,_1));
   //figure out where to find macros
   const char* cmspath = gSystem->Getenv("CMSSW_BASE");
   if(0 == cmspath) {
      throw std::runtime_error("CMSSW_BASE environment variable not set");
   }
   //tell ROOT where to find our macros
   std::string macPath(cmspath);
   macPath += "/src/Fireworks/Core/macros";
   gROOT->SetMacroPath((std::string("./:")+macPath).c_str());

   // prepare geometry service
   // ATTN: this should be made configurable
   const char* geomtryFile = "cmsGeom10.root";
   m_detIdToGeo.loadGeometry( geomtryFile );
   m_detIdToGeo.loadMap( geomtryFile );

   boost::shared_ptr<FWViewManagerBase> rpzViewManager( new FWRhoPhiZViewManager(m_guiManager.get()) );
   rpzViewManager->setGeom(&m_detIdToGeo);
   m_viewManager->add(rpzViewManager);

   m_viewManager->add( boost::shared_ptr<FWViewManagerBase>( new FWEveLegoViewManager(m_guiManager.get()) ) );

   if(iConfigFileName.empty() ) {
      std::cout << "WARNING: no configuration is loaded." << std::endl;
      m_configFileName = "newconfig.fwc";
      m_guiManager->createView("Rho Z");
      m_guiManager->createView("Glimpse");
      m_guiManager->createView("3D Lego");
      m_guiManager->createView("Rho Phi");
   } else {
      std::string configFileName(iConfigFileName);
      char* whereConfig = gSystem->Which(TROOT::GetMacroPath(), configFileName.c_str(), kReadPermission);
      if(0==whereConfig) {
         configFileName = "default.fwc";
      }

      delete [] whereConfig;
      m_configurationManager->readFromFile(configFileName);
   }

   if(not m_configFileName.empty() ) {
      /* //when the program quits we will want to save the configuration automatically
         m_guiManager->goingToQuit_.connect(
                                          boost::bind(&FWConfigurationManager::writeToFile,
                                                      m_configurationManager.get(),
                                                      m_configFileName));
       */
      m_guiManager->writeToPresentConfigurationFile_.connect(
         boost::bind(&FWConfigurationManager::writeToFile,
                     m_configurationManager.get(),
                     m_configFileName));
   }
   gEve->GetHighlight()->SetPickToSelect(TEveSelection::kPS_PableCompound);
   TEveTrack::SetDefaultBreakProjectedTracks(kFALSE);
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
   m_eiManager->add(iItem);
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
   TStopwatch stopwatch;
   m_eiManager->setGeom(&m_detIdToGeo);
   m_eiManager->newEvent(&iEvent);
//   if (m_textView.get() != 0)
//        m_textView->newEvent(iEvent, this);
   stopwatch.Stop();
   printf("Processing time: \n");
   stopwatch.Print("m");
   return 0;
}

void
FWDisplayEvent::writeConfigurationFile(const std::string& iFileName) const
{
   m_configurationManager->writeToFile(iFileName);
}

//
// static member functions
//


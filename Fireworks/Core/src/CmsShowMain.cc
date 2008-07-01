// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowMain
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Mon Dec  3 08:38:38 PST 2007
// $Id: CmsShowMain.cc,v 1.13 2008/07/01 04:23:57 chrjones Exp $
//

// system include files
#include <sstream>
#include <sigc++/sigc++.h>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>
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
#include "TGFileDialog.h"

//needed to work around a bug
#include "TApplication.h"

// user include files
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/FWRhoPhiZViewManager.h"
#include "Fireworks/Core/interface/FWEveLegoViewManager.h"
#include "Fireworks/Core/interface/FWGlimpseViewManager.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWTextView.h"
#include "Fireworks/Core/interface/ElectronSCViewManager.h"
#include "Fireworks/Core/interface/MuonPUViewManager.h"
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"
#include "Fireworks/Core/interface/ElectronDetailView.h"
#include "Fireworks/Core/interface/TrackDetailView.h"
#include "Fireworks/Core/interface/MuonDetailView.h"
#include "Fireworks/Core/interface/GenParticleDetailView.h"

#include "DataFormats/FWLite/interface/Event.h"

#include "Fireworks/Core/interface/FWConfigurationManager.h"

#include "Fireworks/Core/interface/CmsShowNavigator.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/CSGNumAction.h"

#include "Fireworks/Core/interface/ActionsList.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
double CmsShowMain::m_magneticField = 4;
double CmsShowMain::m_caloScale = 2;

//
// constructors and destructor
//
CmsShowMain::CmsShowMain(int argc, char *argv[]) :
  m_configurationManager(new FWConfigurationManager),
  m_changeManager(new FWModelChangeManager),
  m_selectionManager(new FWSelectionManager(m_changeManager.get())),
  m_eiManager(new FWEventItemsManager(m_changeManager.get(),
                                      m_selectionManager.get())),
  m_guiManager(new FWGUIManager(m_selectionManager.get(),
                                m_eiManager.get(),
                                m_changeManager.get(),
                                false)),
  m_viewManager( new FWViewManagerManager(m_changeManager.get())),
  m_textView(0)
  //  m_configFileName(iConfigFileName)
{
  try {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("input-file",    po::value<std::string>(), "Input root file")
    ("config-file,c", po::value<std::string>(), "Include configuration file")
    ("geom-file,g",   po::value<std::string>(), "Include geometry file")
    ("noconfig,n",    "Don't load any configuration file")
    ("fast,f",        "Load fast")
    ("debug,d",       "Show Eve browser to help debug problems");
  po::positional_options_description p;
  p.add("input-file", -1);

  int newArgc = argc;
  char **newArgv = argv;
  po::variables_map vm;
  po::store(po::parse_command_line(newArgc, newArgv, desc), vm);
  po::notify(vm);
  po::store(po::command_line_parser(newArgc, newArgv).
	    options(desc).positional(p).run(), vm);
  po::notify(vm);

  if (vm.count("input-file"))
    m_inputFileName = vm["input-file"].as<std::string>();
  else {
    printf("No file name.  Choosing default.\n");
    m_inputFileName = "data.root";
  }
  if (vm.count("config-file")) 
    m_configFileName = vm["config-file"].as<std::string>();
  else {
     if (vm.count("noconfig")){
	printf("No configiguration is loaded, show everything.\n");
	m_configFileName = "";
     } else {
	m_configFileName = "src/Fireworks/Core/macros/default.fwc";
     }
  }
  if (vm.count("geom-file"))
    m_geomFileName = vm["geom-file"].as<std::string>();
  else {
    printf("No geom file name.  Choosing default.\n");
    m_geomFileName = "cmsGeom10.root";
  }
  bool debugMode = vm.count("debug");
   

  if ( !vm.count("fast") )
     m_textView = std::auto_ptr<FWTextView>( new FWTextView(this, &*m_selectionManager, &*m_guiManager) );

  printf("Input: %s\n", m_inputFileName.c_str());
  printf("Config: %s\n", m_configFileName.c_str());
  printf("Geom: %s\n", m_geomFileName.c_str());
  //connect up the managers
   m_eiManager->newItem_.connect(boost::bind(&FWModelChangeManager::newItemSlot,
                                             m_changeManager.get(), _1) );

  m_eiManager->newItem_.connect(boost::bind(&FWViewManagerManager::registerEventItem,
                                              m_viewManager.get(), _1));
  m_configurationManager->add("EventItems",m_eiManager.get());
  m_configurationManager->add("GUI",m_guiManager.get());
  m_guiManager->writeToConfigurationFile_.connect(boost::bind(&FWConfigurationManager::writeToFile,
                                                              m_configurationManager.get(),_1));
  //m_selectionManager->selectionChanged_.connect(boost::bind(&CmsShowMain::selectionChanged,this,_1));
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
  m_detIdToGeo.loadGeometry( m_geomFileName.c_str() );
  m_detIdToGeo.loadMap( m_geomFileName.c_str() );
   
  boost::shared_ptr<FWViewManagerBase> rpzViewManager( new FWRhoPhiZViewManager(m_guiManager.get()) );
  rpzViewManager->setGeom(&m_detIdToGeo);
  m_viewManager->add(rpzViewManager);
//   m_viewManager->add( boost::shared_ptr<FWViewManagerBase>( new MuonPUViewManager));

  m_viewManager->add( boost::shared_ptr<FWViewManagerBase>( new FWEveLegoViewManager(m_guiManager.get()) ) );
   
  m_viewManager->add( boost::shared_ptr<FWViewManagerBase>( new FWGlimpseViewManager(m_guiManager.get()) ) );
   
  if(m_configFileName.empty() ) {
    std::cout << "WARNING: no configuration is loaded." << std::endl;
    m_configFileName = "newconfig.fwc";
    m_guiManager->createView("Rho Phi");
    m_guiManager->createView("Rho Z");
    m_guiManager->createView("3D Lego");
    m_guiManager->createView("Glimpse");

   FWPhysicsObjectDesc ecal("ECal",
                            TClass::GetClass("CaloTowerCollection"),
			    "ECal",
			    FWDisplayProperties(kRed),
                            "towerMaker",
                            "",
                            "",
                            "",
                            2);

   FWPhysicsObjectDesc hcal("HCal",
		    TClass::GetClass("CaloTowerCollection"),
			    "HCal",
                            FWDisplayProperties(kBlue),
                            "towerMaker",
                            "",
                            "",
                            "",
                            2);

   FWPhysicsObjectDesc jets("Jets",
                            TClass::GetClass("reco::CaloJetCollection"),
			    "Jets",
                            FWDisplayProperties(kYellow),
                            "iterativeCone5CaloJets",
                            "",
                            "",
                            "$.pt()>15",
                            3);


   FWPhysicsObjectDesc l1EmTrigs("L1EmTrig",
                            TClass::GetClass("l1extra::L1EmParticleCollection"),
                            "L1EmTrig",
                            FWDisplayProperties(kOrange),
                            "hltL1extraParticles",
                            "Isolated",
                            "",
                            "",
                            3);

   FWPhysicsObjectDesc l1MuonTrigs("L1MuonTrig",
                            TClass::GetClass("l1extra::L1MuonParticleCollection"),
                            "L1MuonTrig",
                            FWDisplayProperties(kViolet),
                            "hltL1extraParticles",
                            "",
                            "",
                            "",
                            3);

   FWPhysicsObjectDesc l1EtMissTrigs("L1EtMissTrig",
                            TClass::GetClass("l1extra::L1EtMissParticleCollection"),
                            "L1EtMissTrig",
                            FWDisplayProperties(kTeal),
                            "hltL1extraParticles",
                            "",
                            "",
                            "",
                            3);

   FWPhysicsObjectDesc l1JetTrigs("L1JetTrig",
                            TClass::GetClass("l1extra::L1JetParticleCollection"),
                            "L1JetTrig",
                            FWDisplayProperties(kMagenta),
                            "hltL1extraParticles",
                            "Central",
                            "",
                            "",
                            3);


   FWPhysicsObjectDesc tracks("Tracks",
                              TClass::GetClass("reco::TrackCollection"),
			      "Tracks",
                              FWDisplayProperties(kGreen),
                              "generalTracks",
                              "",
                              "",
                              "$.pt()>2",
                              1);

   FWPhysicsObjectDesc muons("Muons",
                             TClass::GetClass("reco::MuonCollection"),
			     "Muons",
                             FWDisplayProperties(kRed),
                             "muons",
                             "",
                             "",
                             "$.isGlobalMuon()",
                             5);

   FWPhysicsObjectDesc electrons("Electrons",
				 TClass::GetClass("reco::GsfElectronCollection"),
				 "Electrons",
				 FWDisplayProperties(kCyan),
				 "pixelMatchGsfElectrons",
                                 "",
                                 "",
                                 "$.hadronicOverEm()<0.05",
                                 3);

   FWPhysicsObjectDesc genParticles("GenParticles",
				    TClass::GetClass("reco::GenParticleCollection"),
				    "GenParticles",
				    FWDisplayProperties(kMagenta),
				    "genParticles",
				    "",
				    "",
				    "$.pt()>1 && $.status() == 3",
				    6);

   // Vertices
   FWPhysicsObjectDesc vertices("Vertices",
				TClass::GetClass("std::vector<reco::Vertex>"),
				"Vertices",
				FWDisplayProperties(kYellow),
				"offlinePrimaryVertices",
				"",
				"",
	 			"",
				10);

   FWPhysicsObjectDesc mets("METs",
			    TClass::GetClass("reco::CaloMETCollection"),
			    "METs",
			    FWDisplayProperties(kRed),
			    "metNoHF",
			    "",
			    "",
			    "",
			    3);

   registerPhysicsObject(ecal);
   registerPhysicsObject(hcal);
   registerPhysicsObject(jets);
   registerPhysicsObject(l1EmTrigs);
   registerPhysicsObject(l1MuonTrigs);
   registerPhysicsObject(l1EtMissTrigs);
   registerPhysicsObject(l1JetTrigs);
   registerPhysicsObject(tracks);
   registerPhysicsObject(muons);
   registerPhysicsObject(electrons);
   registerPhysicsObject(genParticles);
   registerPhysicsObject(vertices);
   registerPhysicsObject(mets);
   
  } else {
    char* whereConfig = gSystem->Which(TROOT::GetMacroPath(), m_configFileName.c_str(), kReadPermission);
    if(0==whereConfig) {
       m_configFileName = "default.fwc";
    } 
    
    delete [] whereConfig;
    m_configurationManager->readFromFile(m_configFileName);
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
  TEveTrackProjected::SetBreakTracks(kFALSE);
  
  
  // register detail viewers
  registerDetailView("Electrons", new ElectronDetailView);
  registerDetailView("Muons", new MuonDetailView);
  registerDetailView("Tracks", new TrackDetailView);
  registerDetailView("GenParticles", new GenParticleDetailView);
  
  m_navigator = new CmsShowNavigator();
  m_navigator->oldEvent.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::loadEvent));
  m_navigator->newEvent.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::loadEvent));
  m_navigator->newEvent.connect(sigc::mem_fun(*this, &CmsShowMain::draw));
  m_navigator->newFileLoaded.connect(sigc::mem_fun(*this, &CmsShowMain::resetInitialization));
  m_navigator->atBeginning.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::disablePrevious));
  m_navigator->atEnd.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::disableNext));
  if (m_guiManager->getAction(cmsshow::sOpenData) != 0) m_guiManager->getAction(cmsshow::sOpenData)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::openData));
  if (m_guiManager->getAction(cmsshow::sNextEvent) != 0) m_guiManager->getAction(cmsshow::sNextEvent)->activated.connect(sigc::mem_fun(*m_navigator, &CmsShowNavigator::nextEvent));
  if (m_guiManager->getAction(cmsshow::sPreviousEvent) != 0) m_guiManager->getAction(cmsshow::sPreviousEvent)->activated.connect(sigc::mem_fun(*m_navigator, &CmsShowNavigator::previousEvent));
  if (m_guiManager->getAction(cmsshow::sHome) != 0) m_guiManager->getAction(cmsshow::sHome)->activated.connect(sigc::mem_fun(*m_navigator, &CmsShowNavigator::firstEvent));
  if (m_guiManager->getAction(cmsshow::sQuit) != 0) m_guiManager->getAction(cmsshow::sQuit)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::quit));
  if (m_guiManager->getAction(cmsshow::sShowEventDisplayInsp) != 0) m_guiManager->getAction(cmsshow::sShowEventDisplayInsp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createEDIFrame));
  if (m_guiManager->getAction(cmsshow::sShowMainViewCtl) != 0) m_guiManager->getAction(cmsshow::sShowMainViewCtl)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createViewPopup));
  if (m_guiManager->getRunEntry() != 0) m_guiManager->getRunEntry()->activated.connect(sigc::mem_fun(*m_navigator, &CmsShowNavigator::goToRun));
  if (m_guiManager->getEventEntry() != 0) m_guiManager->getEventEntry()->activated.connect(sigc::mem_fun(*m_navigator, &CmsShowNavigator::goToEvent));
  m_navigator->loadFile(m_inputFileName);
   
   if(debugMode) {
      m_guiManager->openEveBrowserForDebugging();
   }
   } catch(std::exception& iException) {
      std::cerr <<"CmsShowMain caught exception "<<iException.what()<<std::endl;
      throw;
   }
}

// CmsShowMain::CmsShowMain(const CmsShowMain& rhs)
// {
//    // do actual copying here;
// }

CmsShowMain::~CmsShowMain()
{
  delete m_navigator;
}

//
// assignment operators
//
// const CmsShowMain& CmsShowMain::operator=(const CmsShowMain& rhs)
// {
//   //An exception safe implementation is
//   CmsShowMain temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void CmsShowMain::resetInitialization() {
  printf("Need to reset\n");
} 

void CmsShowMain::draw(const fwlite::Event& event) 
{
  TStopwatch stopwatch;
   
  m_guiManager->enableActions(false);
  m_eiManager->setGeom(&m_detIdToGeo);
  m_eiManager->newEvent(&event);
  if (m_textView.get() != 0)
       m_textView->newEvent(event, this);
  m_guiManager->enableActions();
  stopwatch.Stop(); printf("Total event draw time: "); stopwatch.Print("m");
}

void CmsShowMain::openData()
{
  const char* kRootType[] = {"ROOT files","*.root",
			     0,0};
  TGFileInfo fi;
  fi.fFileTypes = kRootType;
  fi.fIniDir = ".";
  new TGFileDialog(gClient->GetDefaultRoot(), gClient->GetDefaultRoot(), kFDOpen, &fi);
  if (fi.fFilename) m_navigator->loadFile(fi.fFilename);
}

void CmsShowMain::quit() 
{
  // m_configurationManager->writeToFile(m_configFileName);
  gApplication->Terminate(0);
}

void CmsShowMain::registerPhysicsObject(const FWPhysicsObjectDesc&iItem)
{
  m_eiManager->add(iItem);
}

void CmsShowMain::registerDetailView (const std::string &item_name, 
					 FWDetailView *view)
{
  m_guiManager->registerDetailView(item_name, view);
}

//
// const member functions
//
/*
int
CmsShowMain::draw(const fwlite::Event& iEvent)
{
  const CmsShowMain* c = this;
  return c->draw(iEvent);
}

int
CmsShowMain::draw(const fwlite::Event& iEvent) const
{
  TStopwatch stopwatch;
  m_eiManager->setGeom(&m_detIdToGeo);
  m_eiManager->newEvent(&iEvent);
  // m_textView->newEvent(iEvent);
  stopwatch.Stop();
  stopwatch.Print();
  return m_guiManager->allowInteraction();
}

void 
CmsShowMain::writeConfigurationFile(const std::string& iFileName) const
{
  m_configurationManager->writeToFile(iFileName);
}
*/

//
// static member functions
//


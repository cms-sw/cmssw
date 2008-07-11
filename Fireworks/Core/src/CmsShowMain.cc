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
// $Id: CmsShowMain.cc,v 1.20 2008/07/09 06:54:26 jmuelmen Exp $
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
#include "TEveLine.h"

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
bool CmsShowMain::m_autoField = true;
double CmsShowMain::m_magneticField = 3.8;
int CmsShowMain::m_numberOfFieldEstimates = 0;
int CmsShowMain::m_numberOfFieldIsOnEstimates = 0;
double CmsShowMain::m_caloScale = 2;

void CmsShowMain::setMagneticField(double var)
{
   m_magneticField = var;
}

double CmsShowMain::getMagneticField()
{
   if ( m_numberOfFieldIsOnEstimates > m_numberOfFieldEstimates/2 ||
	m_numberOfFieldEstimates == 0 )
     return m_magneticField;
   else
     return 0;
}

void CmsShowMain::guessFieldIsOn(bool isOn)
{
   if ( isOn ) ++m_numberOfFieldIsOnEstimates;
   ++m_numberOfFieldEstimates;
}

//
// constructors and destructor
//
static const char* const kInputFileOpt ="input-file";
static const char* const kInputFileCommandOpt ="input-file,i";
static const char* const kConfigFileOpt = "config-file";
static const char* const kConfigFileCommandOpt = "config-file,c";
static const char* const kGeomFileOpt = "geom-file";
static const char* const kGeomFileCommandOpt = "geom-file,g";
static const char* const kNoConfigFileOpt = "noconfig";
static const char* const kNoConfigFileCommandOpt = "noconfig,n";
static const char* const kFastOpt = "fast";
static const char* const kFastCommandOpt = "fast,f";
static const char* const kDebugOpt = "debug";
static const char* const kDebugCommandOpt = "debug,d";
static const char* const kAdvancedRenderOpt = "shine";
static const char* const kAdvancedRenderCommandOpt = "shine,s";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";

CmsShowMain::CmsShowMain(int argc, char *argv[]) :
  m_configurationManager(new FWConfigurationManager),
  m_changeManager(new FWModelChangeManager),
  m_selectionManager(new FWSelectionManager(m_changeManager.get())),
  m_eiManager(new FWEventItemsManager(m_changeManager.get(),
                                      m_selectionManager.get())),
  m_viewManager( new FWViewManagerManager(m_changeManager.get())),
  m_textView(0)
  //  m_configFileName(iConfigFileName)
{
   try {
      std::string descString(argv[0]);
      descString += " [options]\nAllowed options";
      
      namespace po = boost::program_options;
      po::options_description desc(descString);
      desc.add_options()
      (kInputFileCommandOpt,    po::value<std::string>(), "Input root file")
      (kConfigFileCommandOpt, po::value<std::string>(), "Include configuration file")
      (kGeomFileCommandOpt,   po::value<std::string>(), "Include geometry file")
      (kNoConfigFileCommandOpt,    "Don't load any configuration file")
      (kFastCommandOpt,        "Faster running by not providing tables")
      (kDebugCommandOpt,       "Show Eve browser to help debug problems")
      (kAdvancedRenderCommandOpt,       "Use advance options to improve rendering quality (anti-alias etc)")
      (kHelpCommandOpt, "Display help message");
      po::positional_options_description p;
      p.add(kInputFileOpt, -1);
      
      int newArgc = argc;
      char **newArgv = argv;
      po::variables_map vm;
      //po::store(po::parse_command_line(newArgc, newArgv, desc), vm);
      //po::notify(vm);
      po::store(po::command_line_parser(newArgc, newArgv).
                options(desc).positional(p).run(), vm);
      po::notify(vm);
      if(vm.count(kHelpOpt)) {
         std::cout << desc <<std::endl;
         exit(0);
      }
      
      if (vm.count(kInputFileOpt)) {
         m_inputFileName = vm[kInputFileOpt].as<std::string>();
      } else {
         printf("No data file name.\n");
      }
      if (vm.count(kConfigFileOpt)) {
         m_configFileName = vm[kConfigFileOpt].as<std::string>();
      } else {
         if (vm.count(kNoConfigFileOpt)){
            printf("No configiguration is loaded, show everything.\n");
            m_configFileName = "";
         } else {
            m_configFileName = "src/Fireworks/Core/macros/default.fwc";
         }
      }
      if (vm.count(kGeomFileOpt)) {
         m_geomFileName = vm[kGeomFileOpt].as<std::string>();
      } else {
         printf("No geom file name.  Choosing default.\n");
         m_geomFileName = "cmsGeom10.root";
      }
      bool debugMode = vm.count(kDebugOpt);
      
      //Delay creating guiManager until here so that if we have a 'help' request we don't
      // open any graphics
      m_guiManager = std::auto_ptr<FWGUIManager>(new FWGUIManager(m_selectionManager.get(),
                                                                  m_eiManager.get(),
                                                                  m_changeManager.get(),
                                                                  m_viewManager.get(),
                                                                  false));
      
      if ( vm.count(kAdvancedRenderOpt) ) {
         TEveLine::SetDefaultSmooth(kTRUE);
      }
      
      if ( !vm.count(kFastOpt) ) {
         m_textView = std::auto_ptr<FWTextView>(
                                                new FWTextView(this, &*m_selectionManager, &*m_changeManager,
                                                               &*m_guiManager) );
      }
      
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
                                       "$.pt()>15",
                                       3);
         
         FWPhysicsObjectDesc l1Muons("L1-Muons",
                                     TClass::GetClass("l1extra::L1MuonParticleCollection"),
                                     "L1-Muons",
                                     FWDisplayProperties(kViolet),
                                     "hltL1extraParticles",
                                     "",
                                     "",
                                     "",
                                     3);
         
         FWPhysicsObjectDesc l1MET("L1-MET",
                                   TClass::GetClass("l1extra::L1EtMissParticleCollection"),
                                   "L1-MET",
                                   FWDisplayProperties(kTeal),
                                   "hltL1extraParticles",
                                   "",
                                   "",
                                   "",
                                   3);
         
         FWPhysicsObjectDesc l1Jets("L1-Jets",
                                    TClass::GetClass("l1extra::L1JetParticleCollection"),
                                    "L1-Jets",
                                    FWDisplayProperties(kMagenta),
                                    "hltL1extraParticles",
                                    "Central",
                                    "",
                                    "$.pt()>15",
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
                                          "abs($.pdgId())==11 || abs($.pdgId())==13",
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
         
         FWPhysicsObjectDesc mets("MET",
                                  TClass::GetClass("reco::CaloMETCollection"),
                                  "MET",
                                  FWDisplayProperties(kRed),
                                  "metNoHF",
                                  "",
                                  "",
                                  "",
                                  3);
         
         FWPhysicsObjectDesc dtSegments("DT-segments",
                                        TClass::GetClass("DTRecSegment4DCollection"),
                                        "DT-segments",
                                        FWDisplayProperties(kBlue),
                                        "dt4DSegments",
                                        "",
                                        "",
                                        "",
                                        1);
         
         FWPhysicsObjectDesc cscSegments("CSC-segments",
                                         TClass::GetClass("CSCSegmentCollection"),
                                         "CSC-segments",
                                         FWDisplayProperties(kBlue),
                                         "cscSegments",
                                         "",
                                         "",
                                         "",
                                         1);
         registerPhysicsObject(ecal);
         registerPhysicsObject(hcal);
         registerPhysicsObject(jets);
         registerPhysicsObject(l1EmTrigs);
         registerPhysicsObject(l1Muons);
         registerPhysicsObject(l1MET);
         registerPhysicsObject(l1Jets);
         registerPhysicsObject(tracks);
         registerPhysicsObject(muons);
         registerPhysicsObject(electrons);
         registerPhysicsObject(genParticles);
         registerPhysicsObject(vertices);
         registerPhysicsObject(mets);
         registerPhysicsObject(dtSegments);
         registerPhysicsObject(cscSegments);
         
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
      m_navigator->newFileLoaded.connect(boost::bind(&CmsShowMain::resetInitialization,this));
      m_navigator->newFileLoaded.connect(sigc::mem_fun(*m_guiManager,&FWGUIManager::newFile));
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
      if (CSGAction* action = m_guiManager->getAction("Event Filter")) 
         action->activated.connect(boost::bind(&CmsShowNavigator::filterEvents,m_navigator,action));
      else
         printf("Why?\n\n\n\n\n\n");
      if(m_inputFileName.size()) {
         m_navigator->loadFile(m_inputFileName);
      }
      
      if(debugMode) {
         m_guiManager->openEveBrowserForDebugging();
      }else{
         gSystem->IgnoreSignal(kSigSegmentationViolation, true);
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


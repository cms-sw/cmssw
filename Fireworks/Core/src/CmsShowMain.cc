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
// $Id: CmsShowMain.cc,v 1.150 2010/03/26 20:20:20 matevz Exp $
//

// system include files
#include <sstream>
#include <sigc++/sigc++.h>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>
#include <string.h>

#include "TSystem.h"
#include "TClass.h"
#include "TGLWidget.h"
#include "TTimer.h"
#include "TStopwatch.h"
#include "TFile.h"
#include "TROOT.h"
#include "TGFileDialog.h"
#include "TMonitor.h"
#include "TServerSocket.h"

#include "TEveManager.h"
#include "TEveBrowser.h"
#include "TEveTrackProjected.h"
#include "TEveSelection.h"

//needed to work around a bug
//#include "TApplication.h"

#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/FWRhoPhiZViewManager.h"
#include "Fireworks/Core/interface/FWEveLegoViewManager.h"
#include "Fireworks/Core/interface/FWGlimpseViewManager.h"
#include "Fireworks/Core/interface/FWTableViewManager.h"
#include "Fireworks/Core/interface/FWTriggerTableViewManager.h"
#include "Fireworks/Core/interface/FWL1TriggerTableViewManager.h"
#include "Fireworks/Core/interface/FW3DViewManager.h"

#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"
#include "Fireworks/Core/interface/FWConfigurationManager.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Core/interface/Context.h"

#include "Fireworks/Core/interface/CmsShowNavigator.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/CSGContinuousAction.h"

#include "Fireworks/Core/interface/ActionsList.h"

#include "Fireworks/Core/src/CmsShowTaskExecutor.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"
#include "Fireworks/Core/interface/CmsShowSearchFiles.h"

#include "Fireworks/Core/interface/fwLog.h"

#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "DataFormats/FWLite/interface/Event.h"

//
// constants, enums and typedefs
//

static const char* const kInputFilesOpt        = "input-files";
static const char* const kInputFilesCommandOpt = "input-files,i";
static const char* const kConfigFileOpt        = "config-file";
static const char* const kConfigFileCommandOpt = "config-file,c";
static const char* const kGeomFileOpt          = "geom-file";
static const char* const kGeomFileCommandOpt   = "geom-file,g";
static const char* const kNoConfigFileOpt      = "noconfig";
static const char* const kNoConfigFileCommandOpt = "noconfig,n";
static const char* const kPlayOpt              = "play";
static const char* const kPlayCommandOpt       = "play,p";
static const char* const kLoopOpt              = "loop";
static const char* const kLoopCommandOpt       = "loop";
static const char* const kDebugOpt             = "debug";
static const char* const kDebugCommandOpt      = "debug,d";
static const char* const kLogLevelCommandOpt   = "log";
static const char* const kLogLevelOpt          = "log";
static const char* const kEveOpt               = "eve";
static const char* const kEveCommandOpt        = "eve";
static const char* const kAdvancedRenderOpt        = "shine";
static const char* const kAdvancedRenderCommandOpt = "shine,s";
static const char* const kHelpOpt        = "help";
static const char* const kHelpCommandOpt = "help,h";
static const char* const kSoftCommandOpt = "soft";
static const char* const kPortCommandOpt = "port";
static const char* const kPlainRootCommandOpt = "root";
static const char* const kRootInteractiveCommandOpt = "root-interactive,r";
static const char* const kChainCommandOpt = "chain";
static const char* const kLiveCommandOpt  = "live";
static const char* const kFieldCommandOpt = "field";
static const char* const kFreePaletteCommandOpt = "free-palette";
static const char* const kAutoSaveAllViews = "auto-save-all-views";


//
// constructors and destructor
//
CmsShowMain::CmsShowMain(int argc, char *argv[]) :
   m_configurationManager(new FWConfigurationManager),
   m_changeManager(new FWModelChangeManager),
   m_colorManager( new FWColorManager(m_changeManager.get())),
   m_selectionManager(new FWSelectionManager(m_changeManager.get())),
   m_eiManager(new FWEventItemsManager(m_changeManager.get())),
   m_viewManager( new FWViewManagerManager(m_changeManager.get(), m_colorManager.get())),
   m_context(new fireworks::Context(m_changeManager.get(),
                                    m_selectionManager.get(),
                                    m_eiManager.get(),
                                    m_colorManager.get())),
   m_navigator(new CmsShowNavigator(*this)),

   m_loadedAnyInputFile(false),

   m_autoLoadTimer(0),
   m_autoLoadTimerRunning(kFALSE),
   m_liveTimer(0),
   m_live(0),
   m_isPlaying(false),
   m_forward(true),
   m_loop(false),
   m_playDelay(3.f),
   m_lastPointerPositionX(-999),
   m_lastPointerPositionY(-999),
   m_liveTimeout(600000)
{
   try {
      TGLWidget* w = TGLWidget::Create(gClient->GetDefaultRoot(), kTRUE, kTRUE, 0, 10, 10);
      delete w;
   }
   catch (std::exception& iException) {
      std::cerr <<"Insufficient GL support. " << iException.what() << std::endl;
      throw;
   } 
   
   m_eiManager->setContext(m_context.get());
   
   try {
      std::string descString(argv[0]);
      descString += " [options] <data file>\nAllowed options";

      namespace po = boost::program_options;
      po::options_description desc(descString);
      desc.add_options()
         (kInputFilesCommandOpt, po::value< std::vector<std::string> >(),   "Input root files")
         (kConfigFileCommandOpt, po::value<std::string>(),   "Include configuration file")
         (kGeomFileCommandOpt,   po::value<std::string>(),   "Include geometry file")
         (kNoConfigFileCommandOpt,                           "Don't load any configuration file")
         (kPlayCommandOpt, po::value<float>(),               "Start in play mode with given interval between events in seconds")
         (kPortCommandOpt, po::value<unsigned int>(),        "Listen to port for new data files to open")
         (kEveCommandOpt,                                    "Show Eve browser to help debug problems")
         (kLoopCommandOpt,                                   "Loop events in play mode")
         (kPlainRootCommandOpt,                              "Plain ROOT without event display")
         (kRootInteractiveCommandOpt,                        "Enable root interactive prompt")
         (kDebugCommandOpt,                                  "Start the display from a debugger and producer a crash report")
         (kLogLevelCommandOpt, po::value<unsigned int>(),    "Set log level starting from 0 to 4 : kDebug, kInfo, kWarning, kError")
         (kAdvancedRenderCommandOpt,                         "Use advance options to improve rendering quality       (anti-alias etc)")
         (kSoftCommandOpt,                                   "Try to force software rendering to avoid problems with bad hardware drivers")
         (kChainCommandOpt, po::value<unsigned int>(),       "Chain up to a given number of recently open files. Default is 1 - no chain")
         (kLiveCommandOpt,                                   "Enforce playback mode if a user is not using display")
         (kFieldCommandOpt, po::value<double>(),             "Set magnetic field value explicitly. Default is auto-field estimation")
         (kFreePaletteCommandOpt,                            "Allow free color selection (requires special configuration!)")
         (kAutoSaveAllViews, po::value<std::string>(),       "Auto-save all views with given prefix (run_event_lumi_view.png is appended)")
         (kHelpCommandOpt,                                   "Display help message");
      po::positional_options_description p;
      p.add(kInputFilesOpt, -1);

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
      
      if(vm.count(kLogLevelOpt)) {
         fwlog::LogLevel level = (fwlog::LogLevel)(vm[kLogLevelOpt].as<unsigned int>());
         fwlog::setPresentLogLevel(level);
      }

      if(vm.count(kPlainRootCommandOpt)) {
         std::cout << "Plain ROOT prompt requested" <<std::endl;
         return;
      }

      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }

      // input file
      if (vm.count(kInputFilesOpt)) {
         m_inputFiles = vm[kInputFilesOpt].as< std::vector<std::string> >();
      }

      if (!m_inputFiles.size())
         std::cout << "No data file given." << std::endl;
      else if (m_inputFiles.size() == 1)
         std::cout << "Input: " << m_inputFiles.front() << std::endl;
      else
         std::cout << m_inputFiles.size() << " input files; first: " << m_inputFiles.front() << ", last: " << m_inputFiles.back() << std::endl;

      // configuration file
      if (vm.count(kConfigFileOpt)) {
         m_configFileName = vm[kConfigFileOpt].as<std::string>();
      } else {
         if (vm.count(kNoConfigFileOpt)) {
            printf("No configuration is loaded, show everything.\n");
            m_configFileName = "";
         } else {
            m_configFileName = "src/Fireworks/Core/macros/default.fwc";
         }
      }
      std::cout << "Config: "  <<  m_configFileName.c_str() << std::endl;

      // geometry
      if (vm.count(kGeomFileOpt)) {
         m_geomFileName = vm[kGeomFileOpt].as<std::string>();
      } else {
         printf("No geom file name.  Choosing default.\n");
         // m_geomFileName =cmspath;
         m_geomFileName.append("cmsGeom10.root");
      }
      std::cout << "Geom: " <<  m_geomFileName.c_str() << std::endl;

      // non-restricted palette
      if (vm.count(kFreePaletteCommandOpt)) {
         m_colorManager->initialize(false);
         std::cout << "Palette restriction removed on user request!\n";
      } else {
         m_colorManager->initialize(true);
      }

      bool eveMode = vm.count(kEveOpt);

      
      //Delay creating guiManager and enabling autoloading until here so that if we have a 'help' request we don't
      // open any graphics or build dictionaries
      AutoLibraryLoader::enable();

      m_guiManager = std::auto_ptr<FWGUIManager>(new FWGUIManager(m_selectionManager.get(),
                                                                  m_eiManager.get(),
                                                                  m_changeManager.get(),
                                                                  m_colorManager.get(),
                                                                  m_viewManager.get(),
                                                                  this,
                                                                  false));

      if ( vm.count(kAdvancedRenderOpt) ) {
         TEveLine::SetDefaultSmooth(kTRUE);
      }

      //connect up the managers
      m_eiManager->newItem_.connect(boost::bind(&FWModelChangeManager::newItemSlot,
                                                m_changeManager.get(), _1) );

      m_eiManager->newItem_.connect(boost::bind(&FWViewManagerManager::registerEventItem,
                                                m_viewManager.get(), _1));
      m_configurationManager->add("EventItems",m_eiManager.get());
      m_configurationManager->add("GUI",m_guiManager.get());
      m_configurationManager->add("EventNavigator",m_navigator);
      m_guiManager->writeToConfigurationFile_.connect(boost::bind(&FWConfigurationManager::writeToFile,
                                                                  m_configurationManager.get(),_1));
      //figure out where to find macros
      //tell ROOT where to find our macros
      std::string macPath(cmspath);
      macPath += "/src/Fireworks/Core/macros";
      const char* base = gSystem->Getenv("CMSSW_RELEASE_BASE");
      if(0!=base) {
         macPath+=":";
         macPath +=base;
         macPath +="/src/Fireworks/Core/macros";
      }
      gROOT->SetMacroPath((std::string("./:")+macPath).c_str());

      gEve->GetHighlight()->SetPickToSelect(TEveSelection::kPS_PableCompound);
      TEveTrack::SetDefaultBreakProjectedTracks(kFALSE);

      m_startupTasks = std::auto_ptr<CmsShowTaskExecutor>(new CmsShowTaskExecutor);
      m_startupTasks->tasksCompleted_.connect(boost::bind(&FWGUIManager::clearStatus,
                                                          m_guiManager.get()) );
      CmsShowTaskExecutor::TaskFunctor f;
      // first check if port is not occupied
      if (vm.count(kPortCommandOpt)) { 	 
         f=boost::bind(&CmsShowMain::setupSocket, this, vm[kPortCommandOpt].as<unsigned int>()); 	 
         m_startupTasks->addTask(f); 	 
      }
    
      f=boost::bind(&CmsShowMain::loadGeometry,this);
      m_startupTasks->addTask(f);

      f=boost::bind(&CmsShowMain::setupViewManagers,this);
      m_startupTasks->addTask(f);
      f=boost::bind(&CmsShowMain::setupConfiguration,this);
      m_startupTasks->addTask(f);
      f=boost::bind(&CmsShowMain::setupDataHandling,this);
      m_startupTasks->addTask(f);
      if (vm.count(kLoopOpt))
         setPlayLoop();

      gSystem->IgnoreSignal(kSigSegmentationViolation, true);
      if(eveMode) {
         f=boost::bind(&CmsShowMain::setupDebugSupport,this);
         m_startupTasks->addTask(f);
      }
      if(vm.count(kChainCommandOpt)) {
         f=boost::bind(&CmsShowNavigator::setMaxNumberOfFilesToChain, m_navigator, vm[kChainCommandOpt].as<unsigned int>());
         m_startupTasks->addTask(f);
      }
      if (vm.count(kPlayOpt)) {
         f=boost::bind(&CmsShowMain::setupAutoLoad, this, vm[kPlayOpt].as<float>());
         m_startupTasks->addTask(f);
      }

      if(vm.count(kLiveCommandOpt))
      {
         f=boost::bind(&CmsShowMain::setLiveMode, this);
         m_startupTasks->addTask(f);
      }
      if(vm.count(kFieldCommandOpt)) {
         m_context->getField()->setAutodetect(false);
         m_context->getField()->setUserField(vm[kFieldCommandOpt].as<double>());
      }
      if(vm.count(kAutoSaveAllViews)) {
         m_autoSaveAllViewsFormat  = vm[kAutoSaveAllViews].as<std::string>();
         m_autoSaveAllViewsFormat += "%d_%d_%d_%s.png";
       }
      m_startupTasks->startDoingTasks();
   } catch(std::exception& iException) {
      std::cerr <<"CmsShowMain caught exception "<<iException.what()<<std::endl;
      throw;
   }
}

//
// Destruction
//
CmsShowMain::~CmsShowMain()
{
   //avoids a seg fault from eve which happens if eve is terminated after the GUI is gone
   m_selectionManager->clearSelection();

   delete m_navigator;
   delete m_autoLoadTimer;
}


class DieTimer : public TTimer
{
protected:
   CmsShowMain* fApp;
public:
   DieTimer(CmsShowMain* app) : TTimer(), fApp(app)
   {
      Start(0, kTRUE);
   }

   virtual Bool_t Notify()
   {
      TurnOff();
      fApp->doExit();
      delete this;
      return kFALSE;
   }
};

void CmsShowMain::quit()
{
   new DieTimer(this);
}

void CmsShowMain::doExit()
{
   // fflush(stdout);
   m_guiManager->evePreTerminate();
   // sleep at least 150 ms
   // windows in ROOT GUI are destroyed in 150 ms timeout after
   gSystem->Sleep(151);
   gSystem->ProcessEvents();

   gSystem->ExitLoop();
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

const fwlite::Event* CmsShowMain::getCurrentEvent() const
{
   if (m_navigator)
   {
      return m_navigator->getCurrentEvent();
   }
   return 0;
}

void CmsShowMain::resetInitialization() {
   //printf("Need to reset\n");
}

void CmsShowMain::draw()
{
   m_guiManager->updateStatus("loading event ...");

   m_viewManager->eventBegin();
   m_eiManager->setGeom(&m_detIdToGeo);
   m_eiManager->newEvent(m_navigator->getCurrentEvent());
   m_viewManager->eventEnd();

   if ( ! m_autoSaveAllViewsFormat.empty())
   {
      m_guiManager->updateStatus("auto saving images ...");
      m_guiManager->exportAllViews(m_autoSaveAllViewsFormat);
   }

   m_guiManager->clearStatus();
}

void CmsShowMain::openData()
{
   const char* kRootType[] = {"ROOT files","*.root", 0, 0};
   TGFileInfo fi;
   fi.fFileTypes = kRootType;
   /* this is how things used to be done:
      fi.fIniDir = ".";
      this is bad because the destructor calls delete[] on fIniDir.
    */
   fi.fIniDir = new char[10];
   strcpy(fi.fIniDir, ".");
   new TGFileDialog(gClient->GetDefaultRoot(), m_guiManager->getMainFrame(), kFDOpen, &fi);
   m_guiManager->updateStatus("loading file ...");
   if (fi.fFilename) {
      m_navigator->openFile(fi.fFilename);
      m_loadedAnyInputFile = true;
      m_navigator->firstEvent();
      checkPosition();
      draw();
   }
   m_guiManager->clearStatus();
}

void CmsShowMain::appendData()
{
   const char* kRootType[] = {"ROOT files","*.root", 0, 0};
   TGFileInfo fi;
   fi.fFileTypes = kRootType;
   /* this is how things used to be done:
      fi.fIniDir = ".";
      this is bad because the destructor calls delete[] on fIniDir.
    */
   fi.fIniDir = new char[10];
   strcpy(fi.fIniDir, ".");
   new TGFileDialog(gClient->GetDefaultRoot(), m_guiManager->getMainFrame(), kFDOpen, &fi);
   m_guiManager->updateStatus("loading file ...");
   if (fi.fFilename) {
      m_navigator->appendFile(fi.fFilename, false, false);
      m_loadedAnyInputFile = true;
      checkPosition();
      draw();
   }
   m_guiManager->clearStatus();
}

void
CmsShowMain::openDataViaURL()
{
   if (m_searchFiles.get() == 0) {
      m_searchFiles = std::auto_ptr<CmsShowSearchFiles>( new CmsShowSearchFiles("",
                                                                                "Open Remote Data Files",
                                                                                m_guiManager->getMainFrame(),
                                                                                500, 400));
      m_searchFiles->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   std::string chosenFile = m_searchFiles->chooseFileFromURL();
   if(!chosenFile.empty()) {
      m_guiManager->updateStatus("loading file ...");
      if(m_navigator->openFile(chosenFile.c_str())) {
         m_navigator->firstEvent();
         checkPosition();
         draw();
         m_guiManager->clearStatus();
      } else {
         m_guiManager->updateStatus("failed to load data file");
      }
   }
}


void CmsShowMain::registerPhysicsObject(const FWPhysicsObjectDesc&iItem)
{
   m_eiManager->add(iItem);
}

//
// const member functions
//

//STARTUP TASKS

void
CmsShowMain::loadGeometry()
{      // prepare geometry service
   // ATTN: this should be made configurable
   m_guiManager->updateStatus("Loading geometry...");
   m_detIdToGeo.loadGeometry( m_geomFileName.c_str() );
   m_detIdToGeo.loadMap( m_geomFileName.c_str() );
}

void
CmsShowMain::setupViewManagers()
{
   m_guiManager->updateStatus("Setting up view manager...");
   boost::shared_ptr<FWViewManagerBase> rpzViewManager( new FWRhoPhiZViewManager(m_guiManager.get()) );
   rpzViewManager->setGeom(&m_detIdToGeo);
   m_viewManager->add(rpzViewManager);

   m_viewManager->add( boost::shared_ptr<FWViewManagerBase>( new FWEveLegoViewManager(m_guiManager.get()) ) );

   m_viewManager->add( boost::shared_ptr<FWViewManagerBase>( new FWGlimpseViewManager(m_guiManager.get()) ) );

   boost::shared_ptr<FWTableViewManager> tableViewManager( new FWTableViewManager(m_guiManager.get()) );
   m_configurationManager->add(std::string("Tables"), tableViewManager.get());
   m_viewManager->add( tableViewManager );

   boost::shared_ptr<FWTriggerTableViewManager> triggerTableViewManager( new FWTriggerTableViewManager(m_guiManager.get()) );
   m_configurationManager->add(std::string("TriggerTables"), triggerTableViewManager.get());
   m_viewManager->add( triggerTableViewManager );

   boost::shared_ptr<FWL1TriggerTableViewManager> l1TriggerTableViewManager( new FWL1TriggerTableViewManager(m_guiManager.get()) );
   m_configurationManager->add(std::string("L1TriggerTables"), l1TriggerTableViewManager.get());
   m_viewManager->add( l1TriggerTableViewManager );

   boost::shared_ptr<FWViewManagerBase> plain3DViewManager( new FW3DViewManager(m_guiManager.get()) );
   plain3DViewManager->setGeom(&m_detIdToGeo);
   m_viewManager->add( plain3DViewManager );
}

void
CmsShowMain::setupConfiguration()
{
   m_guiManager->updateStatus("Setting up configuration...");
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
         std::cerr <<"unable to load configuration file '"<<m_configFileName<<"' will load default instead"<<std::endl;
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
}

//______________________________________________________________________________

namespace {
class SignalTimer : public TTimer {
public:
   virtual Bool_t Notify() {
      timeout_();
      return true;
   }
   sigc::signal<void> timeout_;
};
}

//______________________________________________________________________________
void
CmsShowMain::setupAutoLoad(float x)
{
   m_playDelay = x;
   m_guiManager->setDelayBetweenEvents(m_playDelay);
   if (!m_guiManager->playEventsAction()->isEnabled())
      m_guiManager->playEventsAction()->enable();

   m_guiManager->playEventsAction()->switchMode();
}

void CmsShowMain::startAutoLoadTimer()
{
   m_autoLoadTimer->SetTime((Long_t)(m_playDelay*1000));
   m_autoLoadTimer->Reset();
   m_autoLoadTimer->TurnOn();
   m_autoLoadTimerRunning = kTRUE;
}

void CmsShowMain::stopAutoLoadTimer()
{
   m_autoLoadTimer->TurnOff();
   m_autoLoadTimerRunning = kFALSE;
}

//_______________________________________________________________________________
void CmsShowMain::autoLoadNewEvent()
{
   stopAutoLoadTimer();
   
   // case when start with no input file
   if (!m_loadedAnyInputFile)
   {
      if (m_monitor.get()) 
         startAutoLoadTimer();
      return;
   }

   bool reachedEnd = (m_forward && m_navigator->isLastEvent()) || (!m_forward && m_navigator->isFirstEvent());

   if (m_loop && reachedEnd)
   {
      m_forward ? m_navigator->firstEvent() : m_navigator->lastEvent();
      draw();
   }
   else if (!reachedEnd)
   {
      m_forward ? m_navigator->nextEvent() : m_navigator->previousEvent();
      draw();
   }

   // stop loop in case no loop or monitor mode
   if ( reachedEnd && (m_loop || m_monitor.get()) == kFALSE)
   {
      if (m_forward && m_navigator->isLastEvent())
      {
         m_guiManager->enableActions();
         checkPosition();
      }

      if ((!m_forward) && m_navigator->isFirstEvent())
      {
         m_guiManager->enableActions();
         checkPosition();
      }
   }
   else
   {
      startAutoLoadTimer();
   }
}

//______________________________________________________________________________

void CmsShowMain::checkPosition()
{
   if ((m_monitor.get() || m_loop ) && m_isPlaying)
      return;
   
   m_guiManager->getMainFrame()->enableNavigatorControls();

   if (m_navigator->isFirstEvent())
   {
      m_guiManager->disablePrevious();
   }

   if (m_navigator->isLastEvent())
   {
      m_guiManager->disableNext();
      // force enable play events action in --port mode
      if (m_monitor.get() && !m_guiManager->playEventsAction()->isEnabled())
         m_guiManager->playEventsAction()->enable();
   }
}

void CmsShowMain::doFirstEvent()
{
   m_navigator->firstEvent();
   checkPosition();
   draw();
}

void CmsShowMain::doNextEvent()
{
   m_navigator->nextEvent();
   checkPosition();
   draw();
}

void CmsShowMain::doPreviousEvent()
{
   m_navigator->previousEvent();
   checkPosition();
   draw();
}
void CmsShowMain::doLastEvent()
{
   m_navigator->lastEvent();
   checkPosition();
   draw();
}

void CmsShowMain::goToRunEvent(int run, int event)
{
   m_navigator->goToRunEvent(run, event);
   checkPosition();
   draw();
}

//==============================================================================

void
CmsShowMain::setupDataHandling()
{
   m_guiManager->updateStatus("Setting up data handling...");

   m_navigator->newEvent_.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::loadEvent));
   m_navigator->fileChanged_.connect(sigc::mem_fun(*m_guiManager,&FWGUIManager::fileChanged));

   // navigator filtering  ->
   m_navigator->editFiltersExternally_.connect(boost::bind(&FWGUIManager::updateEventFilterEnable, m_guiManager.get(),_1));
   m_navigator->filterStateChanged_.connect(boost::bind(&CmsShowMain::navigatorChangedFilterState,this, _1));
   m_navigator->postFiltering_.connect(boost::bind(&CmsShowMain::postFiltering,this));

   // navigator fitlering <-
   m_guiManager->showEventFilterGUI_.connect(boost::bind(&CmsShowNavigator::showEventFilterGUI, m_navigator,_1));
   m_guiManager->filterButtonClicked_.connect(boost::bind(&CmsShowMain::filterButtonClicked,this));

   if (m_guiManager->getAction(cmsshow::sOpenData)    != 0) m_guiManager->getAction(cmsshow::sOpenData)   ->activated.connect(sigc::mem_fun(*this, &CmsShowMain::openData));
   if (m_guiManager->getAction(cmsshow::sAppendData)  != 0) m_guiManager->getAction(cmsshow::sAppendData) ->activated.connect(sigc::mem_fun(*this, &CmsShowMain::appendData));
   if (m_guiManager->getAction(cmsshow::sSearchFiles) != 0) m_guiManager->getAction(cmsshow::sSearchFiles)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::openDataViaURL));
   if (m_guiManager->getAction(cmsshow::sNextEvent) != 0)
      m_guiManager->getAction(cmsshow::sNextEvent)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::doNextEvent));
   if (m_guiManager->getAction(cmsshow::sPreviousEvent) != 0)
      m_guiManager->getAction(cmsshow::sPreviousEvent)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::doPreviousEvent));
   if (m_guiManager->getAction(cmsshow::sGotoFirstEvent) != 0)
      m_guiManager->getAction(cmsshow::sGotoFirstEvent)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::doFirstEvent));
   if (m_guiManager->getAction(cmsshow::sGotoLastEvent) != 0)
      m_guiManager->getAction(cmsshow::sGotoLastEvent)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::doLastEvent));
 
   m_guiManager->changedEventId_.connect(boost::bind(&CmsShowMain::goToRunEvent,this,_1,_2));


   if (m_guiManager->getAction(cmsshow::sQuit) != 0) m_guiManager->getAction(cmsshow::sQuit)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::quit));
   m_guiManager->playEventsAction()->started_.connect(sigc::mem_fun(*this,&CmsShowMain::playForward));
   m_guiManager->playEventsAction()->stopped_.connect(sigc::mem_fun(*this,&CmsShowMain::stopPlaying));
   m_guiManager->playEventsBackwardsAction()->started_.connect(sigc::mem_fun(*this,&CmsShowMain::playBackward));
   m_guiManager->playEventsBackwardsAction()->stopped_.connect(sigc::mem_fun(*this,&CmsShowMain::stopPlaying));
   m_guiManager->loopAction()->started_.connect(sigc::mem_fun(*this,&CmsShowMain::setPlayLoopImp));
   m_guiManager->loopAction()->stopped_.connect(sigc::mem_fun(*this,&CmsShowMain::unsetPlayLoopImp));

   m_guiManager->setDelayBetweenEvents(m_playDelay);
   m_guiManager->changedDelayBetweenEvents_.connect(boost::bind(&CmsShowMain::setPlayDelay,this,_1));

  
   // init data from  CmsShowNavigator configuration, can do this with signals since there were not connected yet
   m_guiManager->setFilterButtonIcon(m_navigator->getFilterState());
   m_autoLoadTimer = new SignalTimer();
   ((SignalTimer*) m_autoLoadTimer)->timeout_.connect(boost::bind(&CmsShowMain::autoLoadNewEvent,this));

   for (unsigned int ii = 0; ii < m_inputFiles.size(); ++ii)
   {
      const std::string& fname = m_inputFiles[ii];
      if (fname.size())
      {
         m_guiManager->updateStatus("loading data file ...");
         if (!m_navigator->appendFile(fname, false, false))
         {
            m_guiManager->updateStatus("failed to load data file");
            openData();
         }
         else
         {
            m_loadedAnyInputFile = true;
         }
      }
   }

   if (m_loadedAnyInputFile)
   {
      m_navigator->firstEvent();
      checkPosition();
      draw();
   }
   else if (m_monitor.get() == 0)
   {
      openData();
   }
}

void
CmsShowMain::setLiveMode()
{
   m_live = true;
   m_liveTimer = new SignalTimer();
   ((SignalTimer*)m_liveTimer)->timeout_.connect(boost::bind(&CmsShowMain::checkLiveMode,this));


   Window_t rootw, childw;
   Int_t root_x, root_y, win_x, win_y;
   UInt_t mask;
   gVirtualX->QueryPointer(gClient->GetDefaultRoot()->GetId(),
                           rootw, childw,
                           root_x, root_y,
                           win_x, win_y,
                           mask);


   m_liveTimer->SetTime(m_liveTimeout);
   m_liveTimer->Reset();
   m_liveTimer->TurnOn();
}

void
CmsShowMain::setPlayDelay(Float_t val)
{
   m_playDelay = val;
}

void
CmsShowMain::setupDebugSupport()
{
   m_guiManager->updateStatus("Setting up Eve debug window...");
   m_guiManager->openEveBrowserForDebugging();
}

void
CmsShowMain::setupSocket(unsigned int iSocket)
{
   m_monitor = std::auto_ptr<TMonitor>(new TMonitor);
   TServerSocket* server = new TServerSocket(iSocket,kTRUE);
   if (server->GetErrorCode())
   {
      fwLog(fwlog::kError) << "CmsShowMain::setupSocket, can't create socket on port "<< iSocket << "." << std::endl;
      exit(0);
   }
   m_monitor->Connect("Ready(TSocket*)","CmsShowMain",this,"notified(TSocket*)");
   m_monitor->Add(server);
}

void
CmsShowMain::notified(TSocket* iSocket)
{
   TServerSocket* server = dynamic_cast<TServerSocket*> (iSocket);
   if (server)
   {
      TSocket* connection = server->Accept();
      if (connection)
      {
         m_monitor->Add(connection);
         std::stringstream s;
         s << "received connection from "<<iSocket->GetInetAddress().GetHostName();
         m_guiManager->updateStatus(s.str().c_str());
      }
   }
   else
   {
      char buffer[4096];
      memset(buffer,0,sizeof(buffer));
      if (iSocket->RecvRaw(buffer, sizeof(buffer)) <= 0)
      {
         m_monitor->Remove(iSocket);
         //std::stringstream s;
         //s << "closing connection to "<<iSocket->GetInetAddress().GetHostName();
         //m_guiManager->updateStatus(s.str().c_str());
         delete iSocket;
         return;
      }
      std::string fileName(buffer);
      std::string::size_type lastNonSpace = fileName.find_last_not_of(" \n\t");
      if (lastNonSpace != std::string::npos)
      {
         fileName.erase(lastNonSpace+1);
      }

      std::stringstream s;
      s <<"New file notified '"<<fileName<<"'";
      m_guiManager->updateStatus(s.str().c_str());

      bool appended = m_navigator->appendFile(fileName, true, m_live);

      if (appended)
      {
         if (m_live && m_isPlaying && m_forward)
         {
            m_navigator->activateNewFileOnNextEvent();
         }
         else if (!m_isPlaying)
         {
            checkPosition();
         }

         // bootstrap case: --port  and no input file
         if (!m_loadedAnyInputFile)
         {
            m_loadedAnyInputFile = true;
            m_navigator->firstEvent();
            if (!m_isPlaying)
            {
               draw();
            }
         }

         std::stringstream sr;
         sr <<"New file registered '"<<fileName<<"'";
         m_guiManager->updateStatus(sr.str().c_str());
      }
      else
      {
         std::stringstream sr;
         sr <<"New file NOT registered '"<<fileName<<"'";
         m_guiManager->updateStatus(sr.str().c_str());
      }
   }
}

void
CmsShowMain::playForward()
{
   m_forward   = true;
   m_isPlaying = true;
   m_guiManager->enableActions(kFALSE);
   startAutoLoadTimer();
}

void
CmsShowMain::playBackward()
{
   m_forward=false;
   m_isPlaying = true;
   m_guiManager->enableActions(kFALSE);
   startAutoLoadTimer();
}

void
CmsShowMain::stopPlaying()
{
   stopAutoLoadTimer();
   if (m_live)
      m_navigator->resetNewFileOnNextEvent();
   m_isPlaying = false;
   m_guiManager->enableActions();
   checkPosition();
}

void
CmsShowMain::setPlayLoop()
{
   if(!m_loop) {
      setPlayLoopImp();
      m_guiManager->loopAction()->activated();
   }
}

void
CmsShowMain::unsetPlayLoop()
{
   if(m_loop) {
      unsetPlayLoopImp();
      m_guiManager->loopAction()->stop();
   }
}

void
CmsShowMain::setPlayLoopImp()
{
   m_loop = true;
}

void
CmsShowMain::unsetPlayLoopImp()
{
   m_loop = false;
}

void
CmsShowMain::navigatorChangedFilterState(int state)
{
   m_guiManager->setFilterButtonIcon(state);
   if (m_navigator->filesNeedUpdate() == false)
   {
      m_guiManager->setFilterButtonText(m_navigator->filterStatusMessage());
      checkPosition();
   }
}

void
CmsShowMain::filterButtonClicked()
{
   if (m_navigator->getFilterState() == CmsShowNavigator::kWithdrawn )
      m_guiManager->showEventFilterGUI();
   else
      m_navigator->toggleFilterEnable();
}

void
CmsShowMain::preFiltering()
{
   // called only if filter has changed
   m_guiManager->updateStatus("Filtering events");
}

void
CmsShowMain::postFiltering()
{
   // called only filter is changed
   m_guiManager->clearStatus();
   draw();
   checkPosition();
   m_guiManager->setFilterButtonText(m_navigator->filterStatusMessage());
}

void
CmsShowMain::checkLiveMode()
{
   m_liveTimer->TurnOff();

   Window_t rootw, childw;
   Int_t root_x, root_y, win_x, win_y;
   UInt_t mask;
   gVirtualX->QueryPointer(gClient->GetDefaultRoot()->GetId(),
                           rootw, childw,
                           root_x, root_y,
                           win_x, win_y,
                           mask);


   if ( !m_isPlaying &&
        m_lastPointerPositionX == root_x && 
        m_lastPointerPositionY == root_y )
   {
      m_guiManager->playEventsAction()->switchMode();
   }

   m_lastPointerPositionX = root_x;
   m_lastPointerPositionY = root_y;


   m_liveTimer->SetTime((Long_t)(m_liveTimeout));
   m_liveTimer->Reset();
   m_liveTimer->TurnOn();
}


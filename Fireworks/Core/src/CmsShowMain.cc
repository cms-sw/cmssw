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
// $Id: CmsShowMain.cc,v 1.89 2009/08/14 10:23:32 amraktad Exp $
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
#include "TTimer.h"
#include "TColor.h"
//socket
#include "TMonitor.h"
#include "TServerSocket.h"
//memset
#include "string.h"

//geometry
#include "TFile.h"
#include "TROOT.h"

#include "TGButton.h"
#include "TGComboBox.h"
#include "TGLabel.h"
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
#include "Fireworks/Core/interface/FWTableViewManager.h"
#include "Fireworks/Core/interface/FW3DViewManager.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"

#include "DataFormats/FWLite/interface/Event.h"

#include "Fireworks/Core/interface/FWConfigurationManager.h"
#include "Fireworks/Core/interface/Context.h"

#include "Fireworks/Core/interface/CmsShowNavigator.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/CSGContinuousAction.h"

#include "Fireworks/Core/interface/ActionsList.h"

#include "Fireworks/Core/src/CmsShowTaskExecutor.h"


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

void CmsShowMain::resetFieldEstimate()
{
   m_numberOfFieldIsOnEstimates = 0;
   m_numberOfFieldEstimates = 0;
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
static const char* const kPlayOpt = "play";
static const char* const kPlayCommandOpt = "play,p";
static const char* const kLoopOpt = "loop";
static const char* const kLoopCommandOpt = "loop";
static const char* const kDebugOpt = "debug";
static const char* const kDebugCommandOpt = "debug,d";
static const char* const kEveOpt = "eve";
static const char* const kEveCommandOpt = "eve";
static const char* const kAdvancedRenderOpt = "shine";
static const char* const kAdvancedRenderCommandOpt = "shine,s";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
// static char const* const kSoftOpt = "soft";
static char const* const kSoftCommandOpt = "soft";
static const char* const kPortCommandOpt = "port";
static char const* const kPlainRootCommandOpt = "root";
static char const* const kRootInteractiveCommandOpt = "root-interactive,r";

CmsShowMain::CmsShowMain(int argc, char *argv[]) :
   m_configurationManager(new FWConfigurationManager),
   m_changeManager(new FWModelChangeManager),
   m_colorManager( new FWColorManager(m_changeManager.get())),
   m_selectionManager(new FWSelectionManager(m_changeManager.get())),
   m_eiManager(new FWEventItemsManager(m_changeManager.get(),
                                       m_selectionManager.get())),
   m_viewManager( new FWViewManagerManager(m_changeManager.get(), m_colorManager.get())),
   m_context(new fireworks::Context(m_changeManager.get(),
                                    m_selectionManager.get(),
                                    m_eiManager.get(),
                                    m_colorManager.get())),
   m_playTimer(0),
   m_playBackTimer(0),
   m_isPlaying(false),
   m_forward(true),
   m_rewindMode(false),
   m_playDelay(3.f)
   //  m_configFileName(iConfigFileName)
{
   //m_colorManager->setBackgroundColorIndex(FWColorManager::kWhiteIndex);
   m_eiManager->setContext(m_context.get());
   try {
      std::string descString(argv[0]);
      descString += " [options] <data file>\nAllowed options";

      namespace po = boost::program_options;
      po::options_description desc(descString);
      desc.add_options()
              (kInputFileCommandOpt,    po::value<std::string>(), "Input root file")
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
              (kAdvancedRenderCommandOpt,                         "Use advance options to improve rendering quality       (anti-alias etc)")
              (kSoftCommandOpt,                                   "Try to force software rendering to avoid problems with bad hardware drivers")
              (kHelpCommandOpt,                                   "Display help message");
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
      if(vm.count(kPlainRootCommandOpt)) {
         std::cout << "Plain ROOT prompt requested" <<std::endl;
         return;
      }

      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }

      // input file
      if (vm.count(kInputFileOpt)) {
         m_inputFileName = vm[kInputFileOpt].as<std::string>();
      }      

      if (!m_inputFileName.size())
         std::cout << "No data file given." << std::endl;
      else
         std::cout << "Input: " << m_inputFileName.c_str() << std::endl;

      // configuration file
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

      bool eveMode = vm.count(kEveOpt);

      //Delay creating guiManager until here so that if we have a 'help' request we don't
      // open any graphics
      m_guiManager = std::auto_ptr<FWGUIManager>(new FWGUIManager(m_selectionManager.get(),
                                                                  m_eiManager.get(),
                                                                  m_changeManager.get(),
                                                                  m_colorManager.get(),
                                                                  m_viewManager.get(),
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
      f=boost::bind(&CmsShowMain::loadGeometry,this);
      m_startupTasks->addTask(f);

      f=boost::bind(&CmsShowMain::setupViewManagers,this);
      m_startupTasks->addTask(f);
      f=boost::bind(&CmsShowMain::setupConfiguration,this);
      m_startupTasks->addTask(f);
      f=boost::bind(&CmsShowMain::setupDataHandling,this);
      m_startupTasks->addTask(f);
      if (vm.count(kLoopOpt)) 
        m_rewindMode = true;

      gSystem->IgnoreSignal(kSigSegmentationViolation, true);
      if(eveMode) {
         f=boost::bind(&CmsShowMain::setupDebugSupport,this);
         m_startupTasks->addTask(f);
      }

      if(vm.count(kPortCommandOpt)) {
         f=boost::bind(&CmsShowMain::setupSocket, this, vm[kPortCommandOpt].as<unsigned int>());
         m_startupTasks->addTask(f);
      }
      if (vm.count(kPlayOpt)) {
         m_playDelay = vm[kPlayOpt].as<float>();
         f=boost::bind(&CSGContinuousAction::switchMode,m_guiManager->playEventsAction());
         m_startupTasks->addTask(f);
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
   delete m_playTimer;
   delete m_playBackTimer;
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
void CmsShowMain::resetInitialization() {
   //printf("Need to reset\n");
}

void CmsShowMain::draw(const fwlite::Event& event)
{
   // TStopwatch stopwatch;
   m_guiManager->updateStatus("loading event ...");
   m_guiManager->enableActions(false);
   m_eiManager->setGeom(&m_detIdToGeo);
   m_eiManager->newEvent(&event);
   // stopwatch.Stop(); printf("Total event draw time: "); stopwatch.Print("m");
   m_guiManager->clearStatus();
   if(m_isPlaying) {
      if(m_forward) {
         m_playTimer->Start((Long_t)(m_playDelay*1000), kFALSE);
      } else {
         m_playBackTimer->Start((Long_t)(m_playDelay*1000), kFALSE);
      }
   } else {
      m_guiManager->enableActions();
   }
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
   new TGFileDialog(gClient->GetDefaultRoot(), gClient->GetDefaultRoot(), kFDOpen, &fi);
   m_guiManager->updateStatus("loading file ...");
   if (fi.fFilename) m_navigator->loadFile(fi.fFilename);
   m_guiManager->clearStatus();
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

namespace {
   class SignalTimer : public TTimer {
public:
      Bool_t Notify() {
         TurnOff();
         timeout_();
         return true;
      }
      sigc::signal<void> timeout_;
   };
}

void
CmsShowMain::setupDataHandling()
{
   m_guiManager->updateStatus("Setting up data handling...");
   m_navigator = new CmsShowNavigator(*this);
   m_navigator->oldEvent_.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::loadEvent));
   m_navigator->newEvent_.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::loadEvent));
   m_navigator->newEvent_.connect(sigc::mem_fun(*this, &CmsShowMain::draw));
   m_navigator->newFileLoaded_.connect(boost::bind(&CmsShowMain::resetInitialization,this));
   m_navigator->newFileLoaded_.connect(sigc::mem_fun(*m_guiManager,&FWGUIManager::newFile));
   m_navigator->atBeginning_.connect((sigc::mem_fun(*this, &CmsShowMain::reachedBeginning)));
   m_navigator->atEnd_.connect(sigc::mem_fun(*this, &CmsShowMain::reachedEnd));
   m_navigator->preFiltering_.connect(boost::bind(&CmsShowMain::preFiltering,this));
   m_navigator->postFiltering_.connect(boost::bind(&CmsShowMain::postFiltering,this));
   if (m_guiManager->getAction(cmsshow::sOpenData) != 0) m_guiManager->getAction(cmsshow::sOpenData)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::openData));
   if (m_guiManager->getAction(cmsshow::sNextEvent) != 0) m_guiManager->getAction(cmsshow::sNextEvent)->activated.connect(sigc::mem_fun(*m_navigator, &CmsShowNavigator::nextEvent));
   if (m_guiManager->getAction(cmsshow::sPreviousEvent) != 0) m_guiManager->getAction(cmsshow::sPreviousEvent)->activated.connect(sigc::mem_fun(*m_navigator, &CmsShowNavigator::previousEvent));
   if (m_guiManager->getAction(cmsshow::sGotoFirstEvent) != 0) m_guiManager->getAction(cmsshow::sGotoFirstEvent)->activated.connect(sigc::mem_fun(*m_navigator, &CmsShowNavigator::firstEvent));
   if (m_guiManager->getAction(cmsshow::sGotoLastEvent) != 0) m_guiManager->getAction(cmsshow::sGotoLastEvent)->activated.connect(sigc::mem_fun(*m_navigator, &CmsShowNavigator::lastEvent));
   if (m_guiManager->getAction(cmsshow::sQuit) != 0) m_guiManager->getAction(cmsshow::sQuit)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::quit));
   m_guiManager->playEventsAction()->started_.connect(sigc::mem_fun(*this,&CmsShowMain::playForward));
   m_guiManager->playEventsAction()->stopped_.connect(sigc::mem_fun(*this,&CmsShowMain::stopPlaying));
   m_guiManager->playEventsBackwardsAction()->started_.connect(sigc::mem_fun(*this,&CmsShowMain::playBackward));
   m_guiManager->playEventsBackwardsAction()->stopped_.connect(sigc::mem_fun(*this,&CmsShowMain::stopPlaying));

   m_guiManager->setDelayBetweenEvents(m_playDelay);
   m_guiManager->changedDelayBetweenEvents_.connect(boost::bind(&CmsShowMain::setPlayDelay,this,_1));

   m_guiManager->changedRunId_.connect(boost::bind(&CmsShowNavigator::goToRun,m_navigator,_1));
   m_guiManager->changedEventId_.connect(boost::bind(&CmsShowNavigator::goToEvent,m_navigator,_1));
   m_guiManager->changedEventFilter_.connect(boost::bind(&CmsShowNavigator::filterEventsAndReset,m_navigator,_1));

   {
      SignalTimer* timer = new SignalTimer();
      timer->timeout_.connect(m_guiManager->getAction(cmsshow::sNextEvent)->activated);
      m_playTimer=timer;
      timer = new SignalTimer();
      timer->timeout_.connect(m_guiManager->getAction(cmsshow::sPreviousEvent)->activated);
      m_playBackTimer=timer;
   }

   if(m_inputFileName.size()) {
      m_guiManager->updateStatus("loading data file...");
      if (! m_navigator->loadFile(m_inputFileName) ){
	m_guiManager->updateStatus("failed to load data file");
	openData();
      }
   }
   else if (m_monitor.get() != 0) {
      openData();
   }
}

void
CmsShowMain::setPlayDelay(Float_t val)
{
   m_playDelay = val;
   m_playTimer->Reset();
   m_playTimer->SetTime((Long_t)(m_playDelay*1000));
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
   m_monitor->Connect("Ready(TSocket*)","CmsShowMain",this,"notified(TSocket*)");
   m_monitor->Add(server);
}


void
CmsShowMain::notified(TSocket* iSocket)
{
   TServerSocket* server = dynamic_cast<TServerSocket*> (iSocket);
   if(0!=server) {
      TSocket* connection = server->Accept();
      if(0!=connection) {
         m_monitor->Add(connection);
         std::stringstream s;
         s << "received connection from "<<iSocket->GetInetAddress().GetHostName();
         m_guiManager->updateStatus(s.str().c_str());
      }
   } else {
      char buffer[4096];
      memset(buffer,0,sizeof(buffer));
      if( iSocket->RecvRaw(buffer, sizeof(buffer)) <= 0) {
         m_monitor->Remove(iSocket);
         //std::stringstream s;
         //s << "closing connection to "<<iSocket->GetInetAddress().GetHostName();
         //m_guiManager->updateStatus(s.str().c_str());
         delete iSocket;
         return;
      }
      std::string fileName(buffer);
      std::string::size_type lastNonSpace = fileName.find_last_not_of(" \n\t");
      if(lastNonSpace != std::string::npos) {
         fileName.erase(lastNonSpace+1);
      }
      std::stringstream s;
      s <<"Ready to change to new file '"<<fileName<<"'";
      m_guiManager->updateStatus(s.str().c_str());

      // bootstrap case: --port without and no input file
      if (!m_inputFileName.size())
      {
         m_navigator->loadFile(fileName);
      }
      else
      {
         m_navigator->newRemoteFile(fileName);
         if ( !m_isPlaying )
            m_guiManager->playEventsAction()->switchMode();
      }
      m_inputFileName = fileName; 
   }
}

void
CmsShowMain::playForward()
{
   m_isPlaying=true;
   m_forward=true;
   m_guiManager->setPlayMode(m_isPlaying);
   m_guiManager->getAction(cmsshow::sNextEvent)->activated();
}

void
CmsShowMain::playBackward()
{
   m_isPlaying=true;
   m_forward=false;
   m_guiManager->setPlayMode(m_isPlaying);
   m_guiManager->getAction(cmsshow::sPreviousEvent)->activated();
}

void
CmsShowMain::stopPlaying()
{
   m_guiManager->setPlayMode(false);
   if(m_isPlaying) {
      m_isPlaying=false;
      if(m_forward) {
         m_playTimer->TurnOff();
      } else {
         m_playBackTimer->TurnOff();
      }
      m_guiManager->enableActions();
   }
}

void
CmsShowMain::reachedEnd()
{
   if (m_isPlaying)
   {
      if ( m_forward && !m_rewindMode ) {
         stopPlaying();
         m_guiManager->disableNext();
      }
   }
   else m_guiManager->disableNext();
}

void
CmsShowMain::reachedBeginning()
{
   if (m_isPlaying)
   {
      if ( !m_forward && !m_rewindMode) {
         stopPlaying();
         m_guiManager->disablePrevious();
      }
   }
   else m_guiManager->disablePrevious();
}

void 
CmsShowMain::preFiltering()
{
   m_guiManager->enableActions(false);
   m_guiManager->updateStatus("Filtering events");

}
void 
CmsShowMain::postFiltering()
{
   m_guiManager->enableActions(true);
}

//
// static member functions
//


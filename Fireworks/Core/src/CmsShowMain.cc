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
//

// system include files
#include <sstream>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>
#include <string.h>

#include "TSystem.h"
#include "TGLWidget.h"
#include "TTimer.h"
#include "TROOT.h"
#include "TGFileDialog.h"
#include "TGMsgBox.h"
#include "TMonitor.h"
#include "TServerSocket.h"
#include "TEveLine.h"
#include "TEveManager.h"
#include "TFile.h"
#include "TGClient.h"
#include <KeySymbols.h>

#include "Fireworks/Core/src/CmsShowMain.h"

#include "Fireworks/Core/interface/FWEveViewManager.h"

#include "Fireworks/Core/interface/FWTableViewManager.h"
#include "Fireworks/Core/interface/FWTriggerTableViewManager.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWLiteJobMetadataManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWConfigurationManager.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Core/interface/Context.h"

#include "Fireworks/Core/interface/CmsShowNavigator.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/CSGContinuousAction.h"
#include "Fireworks/Core/interface/FWLiteJobMetadataUpdateRequest.h"

#include "Fireworks/Core/interface/ActionsList.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWMagField.h"

#include "Fireworks/Core/src/CmsShowTaskExecutor.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"
#include "Fireworks/Core/interface/CmsShowSearchFiles.h"

#include "Fireworks/Core/interface/fwLog.h"

#include "FWCore/FWLite/interface/FWLiteEnabler.h"

#if defined(R__LINUX)
#include "TGX11.h" // !!!! AMT has to be at the end to pass build
#include "X11/Xlib.h"
#endif
//
// constants, enums and typedefs
//

static const char* const kInputFilesOpt        = "input-files";
static const char* const kInputFilesCommandOpt = "input-files,i";
static const char* const kConfigFileOpt        = "config-file";
static const char* const kConfigFileCommandOpt = "config-file,c";
static const char* const kGeomFileOpt          = "geom-file";
static const char* const kGeomFileCommandOpt   = "geom-file,g";
static const char* const kSimGeomFileOpt       = "sim-geom-file";
static const char* const kSimGeomFileCommandOpt= "sim-geom-file";
static const char* const kNoConfigFileOpt      = "noconfig";
static const char* const kNoConfigFileCommandOpt = "noconfig,n";
static const char* const kPlayOpt              = "play";
static const char* const kPlayCommandOpt       = "play,p";
static const char* const kLoopOpt              = "loop";
static const char* const kLoopCommandOpt       = "loop";
static const char* const kLogLevelCommandOpt   = "log";
static const char* const kLogLevelOpt          = "log";
static const char* const kEveOpt               = "eve";
static const char* const kEveCommandOpt        = "eve";
static const char* const kAdvancedRenderOpt        = "shine";
static const char* const kAdvancedRenderCommandOpt = "shine,s";
static const char* const kHelpOpt        = "help";
static const char* const kHelpCommandOpt = "help,h";
static const char* const kSoftCommandOpt = "soft";
static const char* const kExpertCommandOpt = "expert";
static const char* const kPortCommandOpt = "port";
static const char* const kPlainRootCommandOpt = "prompt";
static const char* const kRootInteractiveCommandOpt = "root-interactive,r";
static const char* const kChainCommandOpt = "chain";
static const char* const kLiveCommandOpt  = "live";
static const char* const kFieldCommandOpt = "field";
static const char* const kFreePaletteCommandOpt = "free-palette";
static const char* const kAutoSaveAllViews = "auto-save-all-views";
static const char* const kAutoSaveType     = "auto-save-type";
static const char* const kAutoSaveHeight   = "auto-save-height";
static const char* const kSyncAllViews     = "sync-all-views";
static const char* const kEnableFPE        = "enable-fpe";
static const char* const kZeroWinOffsets   = "zero-window-offsets";
static const char* const kNoVersionCheck   = "no-version-check";


//
// constructors and destructor
//
CmsShowMain::CmsShowMain(int argc, char *argv[]) 
   : CmsShowMainBase(),
     m_navigator(new CmsShowNavigator(*this)),
     m_metadataManager(new FWLiteJobMetadataManager()),
     m_context(new fireworks::Context(changeManager(),
                                      selectionManager(),
                                      eiManager(),
                                      colorManager(),
                                      m_metadataManager.get())),
     m_loadedAnyInputFile(false),
     m_openFile(0),
     m_live(false),
     m_liveTimer(new SignalTimer()),
     m_liveTimeout(600000),
     m_lastXEventSerial(0),
     m_noVersionCheck(false)
{
   try {
      TGLWidget* w = TGLWidget::Create(gClient->GetDefaultRoot(), kTRUE, kTRUE, 0, 10, 10);
      delete w;
   }
   catch (std::exception& iException) {
      fwLog(fwlog::kError) << "Failed creating an OpenGL window: " << iException.what() << "\n"
         "Things to check:\n"
         "- Is DISPLAY environment variable set?\n"
         "- Are OpenGL libraries installed?\n"
         "- If running remotely, make sure you use 'ssh -X' or 'ssh -Y'.\n"
         "See also: https://twiki.cern.ch/twiki/bin/viewauth/CMS/WorkBookFireworks\n";
      gSystem->Exit(1);
   }

   eiManager()->setContext(m_context.get());

   
   std::string descString(argv[0]);
   descString += " [options] <data file>\nGeneral";

   namespace po = boost::program_options;
   po::options_description desc(descString);
   desc.add_options()
      (kInputFilesCommandOpt, po::value< std::vector<std::string> >(),   "Input root files")
      (kConfigFileCommandOpt, po::value<std::string>(),   "Include configuration file")
      (kNoConfigFileCommandOpt,                           "Empty configuration")
      (kNoVersionCheck,                                   "No file version check")
      (kGeomFileCommandOpt,   po::value<std::string>(),   "Reco geometry file. Default is cmsGeom10.root")
      (kSimGeomFileCommandOpt,po::value<std::string>(),   "Geometry file for browsing in table view. Default is CmsSimGeom-14.root. Can be simulation or reco geometry in TGeo format")
     (kFieldCommandOpt, po::value<double>(),             "Set magnetic field value explicitly. Default is auto-field estimation")
   (kRootInteractiveCommandOpt,                        "Enable root interactive prompt")
     (kSoftCommandOpt,                                   "Try to force software rendering to avoid problems with bad hardware drivers")
   (kExpertCommandOpt,                                   "Enable PF user plugins.")
      (kHelpCommandOpt,                                   "Display help message");

 po::options_description livedesc("Live Event Display");
 livedesc.add_options()
      (kPlayCommandOpt, po::value<float>(),               "Start in play mode with given interval between events in seconds")
      (kPortCommandOpt, po::value<unsigned int>(),        "Listen to port for new data files to open")
      (kLoopCommandOpt,                                   "Loop events in play mode")
      (kChainCommandOpt, po::value<unsigned int>(),       "Chain up to a given number of recently open files. Default is 1 - no chain")
      (kLiveCommandOpt,                                   "Enforce playback mode if a user is not using display")
      (kAutoSaveAllViews, po::value<std::string>(),       "Auto-save all views with given prefix (run_event_lumi_view.<auto-save-type> is appended)")
      (kAutoSaveType,     po::value<std::string>(),       "Image type of auto-saved views, png or jpg (png is default)")
      (kAutoSaveHeight, po::value<int>(),                 "Screenshots height when auto-save-all-views is enabled")
      (kSyncAllViews,                                     "Synchronize all views on new event");

 po::options_description debugdesc("Debug");
   debugdesc.add_options()
   (kLogLevelCommandOpt, po::value<unsigned int>(),    "Set log level starting from 0 to 4 : kDebug, kInfo, kWarning, kError")
   (kEveCommandOpt,                                    "Show TEveBrowser to help debug problems")
   (kEnableFPE,                                        "Enable detection of floating-point exceptions");


 po::options_description rnrdesc("Appearance");
 rnrdesc.add_options()
      (kFreePaletteCommandOpt,                            "Allow free color selection (requires special configuration!)")
   (kZeroWinOffsets,                                   "Disable auto-detection of window position offsets")
   (kAdvancedRenderCommandOpt,                         "Enable line anti-aliasing");
   po::positional_options_description p;
   p.add(kInputFilesOpt, -1);


 po::options_description hiddendesc("hidden");
 hiddendesc.add_options();

   po::options_description all("");
 all.add(desc).add(rnrdesc).add(livedesc).add(debugdesc);


   int newArgc = argc;
   char **newArgv = argv;
   po::variables_map vm;
   try{ 
      po::store(po::command_line_parser(newArgc, newArgv).
                options(all).positional(p).run(), vm);

      po::notify(vm);
   }
   catch ( const std::exception& e)
   {
      // Return with exit status 0 to avoid generating crash reports

      fwLog(fwlog::kError) <<  e.what() << std::endl;
      std::cout << all <<std::endl;
      exit(0); 
   }

   if(vm.count(kHelpOpt)) {
      std::cout << all <<std::endl;
      exit(0);
   }
      
   if(vm.count(kLogLevelOpt)) {
      fwlog::LogLevel level = (fwlog::LogLevel)(vm[kLogLevelOpt].as<unsigned int>());
      fwlog::setPresentLogLevel(level);
   }

   if(vm.count(kPlainRootCommandOpt)) {
      fwLog(fwlog::kInfo) << "Plain ROOT prompt requested" << std::endl;
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
      fwLog(fwlog::kInfo) << "No data file given." << std::endl;
   else if (m_inputFiles.size() == 1)
      fwLog(fwlog::kInfo) << "Input " << m_inputFiles.front() << std::endl;
   else
      fwLog(fwlog::kInfo) << m_inputFiles.size() << " input files; first: " << m_inputFiles.front() << ", last: " << m_inputFiles.back() << std::endl;

   // configuration file
   if (vm.count(kConfigFileOpt)) {
      setConfigFilename(vm[kConfigFileOpt].as<std::string>());
      if (access(configFilename(), R_OK) == -1)
      {
         fwLog(fwlog::kError) << "Specified configuration file does not exist. Quitting.\n";
         exit(1);
      }

      fwLog(fwlog::kInfo) << "Config "  <<  configFilename() << std::endl;
   } else {
      if (vm.count(kNoConfigFileOpt)) {
         fwLog(fwlog::kInfo) << "No configuration is loaded.\n";
         configurationManager()->setIgnore();
      } 
   }

   // geometry
   if (vm.count(kGeomFileOpt)) {
      setGeometryFilename(vm[kGeomFileOpt].as<std::string>());
      fwLog(fwlog::kInfo) << "Geometry file " << geometryFilename() << "\n";
   } else {
      //  fwLog(fwlog::kInfo) << "No geom file name.  Choosing default.\n";
      setGeometryFilename("cmsGeom10.root");
   }

   if (vm.count(kSimGeomFileOpt)) {
      setSimGeometryFilename(vm[kSimGeomFileOpt].as<std::string>());
   } else {
      setSimGeometryFilename("cmsSimGeom-14.root");
   }
   // Free-palette palette
   if (vm.count(kFreePaletteCommandOpt)) {
      FWColorPopup::EnableFreePalette();
      fwLog(fwlog::kInfo) << "Palette restriction removed on user request!\n";
   }
   bool eveMode = vm.count(kEveOpt);

   //Delay creating guiManager and enabling autoloading until here so that if we have a 'help' request we don't
   // open any graphics or build dictionaries
   FWLiteEnabler::enable();

   TEveManager::Create(kFALSE, eveMode ? "FIV" : "FI");
 
   if(vm.count(kExpertCommandOpt)) 
   {
      m_context->setHidePFBuilders(false);
   }
   else {
      m_context->setHidePFBuilders(true);
   }

   if(vm.count(kExpertCommandOpt)) 
   {
      m_context->setHidePFBuilders(false);
   }
   else {
      m_context->setHidePFBuilders(true);
   }

   setup(m_navigator.get(), m_context.get(), m_metadataManager.get());

   if (vm.count(kZeroWinOffsets))
   {
      guiManager()->resetWMOffsets();
      fwLog(fwlog::kInfo) << "Window offsets reset on user request!\n";
   }

   if ( vm.count(kAdvancedRenderOpt) ) 
   {
      TEveLine::SetDefaultSmooth(kTRUE);
   }

   //figure out where to find macros
   //tell ROOT where to find our macros
   CmsShowTaskExecutor::TaskFunctor f;
   // first check if port is not occupied
   if (vm.count(kPortCommandOpt)) { 	 
      f=boost::bind(&CmsShowMain::setupSocket, this, vm[kPortCommandOpt].as<unsigned int>()); 	 
      startupTasks()->addTask(f); 	 
   }
    
   f=boost::bind(&CmsShowMainBase::loadGeometry,this);
   startupTasks()->addTask(f);
   f=boost::bind(&CmsShowMainBase::setupViewManagers,this);
   startupTasks()->addTask(f);

   if(vm.count(kLiveCommandOpt))
   {
      f = boost::bind(&CmsShowMain::setLiveMode, this);
      startupTasks()->addTask(f);
   }
      
   if(vm.count(kFieldCommandOpt)) 
   {
      m_context->getField()->setSource(FWMagField::kUser);
      m_context->getField()->setUserField(vm[kFieldCommandOpt].as<double>());
   }
  
   f=boost::bind(&CmsShowMain::setupDataHandling,this);
   startupTasks()->addTask(f);

  
   if (vm.count(kLoopOpt))
      setPlayLoop();

   if (eveMode) {
      f = boost::bind(&CmsShowMainBase::setupDebugSupport,this);
      startupTasks()->addTask(f);
   }
   if(vm.count(kChainCommandOpt)) {
      f = boost::bind(&CmsShowNavigator::setMaxNumberOfFilesToChain, m_navigator.get(), vm[kChainCommandOpt].as<unsigned int>());
      startupTasks()->addTask(f);
   }
   if (vm.count(kPlayOpt)) {
      f = boost::bind(&CmsShowMainBase::setupAutoLoad, this, vm[kPlayOpt].as<float>());
      startupTasks()->addTask(f);
   }

   if(vm.count(kAutoSaveAllViews)) {
      std::string type = "png";
      if(vm.count(kAutoSaveType)) {
         type = vm[kAutoSaveType].as<std::string>();
         if (type != "png" && type != "jpg")
         {
            fwLog(fwlog::kError) << "Specified auto-save type not supported. Quitting.\n";
            exit(1);
         }
      }
      std::string fmt = vm[kAutoSaveAllViews].as<std::string>();
      fmt += "%u_%u_%llu_%s.";
      fmt += type;
      setAutoSaveAllViewsFormat(fmt);
   }
   if (vm.count(kAutoSaveHeight)) {
      setAutoSaveAllViewsHeight(vm[kAutoSaveHeight].as<int>());
   }
   if (vm.count(kSyncAllViews)) {
       FWEveViewManager::syncAllViews();
   }
   if(vm.count(kNoVersionCheck)) {
      m_noVersionCheck=true;
   }
   if(vm.count(kEnableFPE)) {
      gSystem->SetFPEMask();
   }

   if (vm.count(kPortCommandOpt)) { 	 
      f=boost::bind(&CmsShowMain::connectSocket, this); 	 
      startupTasks()->addTask(f); 	 
   }

   startupTasks()->startDoingTasks();
}

//
// Destruction
//

CmsShowMain::~CmsShowMain()
{
   //avoids a seg fault from eve which happens if eve is terminated after the GUI is gone
   selectionManager()->clearSelection();
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

   virtual Bool_t Notify() override
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
   // pre terminate eve
   m_context->deleteEveElements();
   guiManager()->evePreTerminate();

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

const fwlite::Event* 
CmsShowMain::getCurrentEvent() const
{
   if (m_navigator.get())
     return static_cast<const fwlite::Event*>(m_navigator->getCurrentEvent());
   return 0;
}

void
CmsShowMain::fileChangedSlot(const TFile *file)
{
   m_openFile = file;
   if (file)
      guiManager()->titleChanged(m_navigator->frameTitle());


   if (context()->getField()->getSource() == FWMagField::kNone) {
      context()->getField()->resetFieldEstimate();
   }
   m_metadataManager->update(new FWLiteJobMetadataUpdateRequest(getCurrentEvent(), m_openFile));
}

void
CmsShowMain::eventChangedImp()
{
   CmsShowMainBase::eventChangedImp();
   guiManager()->titleChanged(m_navigator->frameTitle());
   m_metadataManager->update(new FWLiteJobMetadataUpdateRequest(getCurrentEvent(), m_openFile));
}

void CmsShowMain::resetInitialization() {
   //printf("Need to reset\n");
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
   fi.fIniDir = new char[128];
   strncpy(fi.fIniDir, ".", 127);  
   guiManager()->updateStatus("waiting for data file ...");
   new TGFileDialog(gClient->GetDefaultRoot(), guiManager()->getMainFrame(), kFDOpen, &fi);
   guiManager()->updateStatus("loading file ...");
   if (fi.fFilename) {
      m_navigator->openFile(fi.fFilename);

      setLoadedAnyInputFileAfterStartup();
      m_navigator->firstEvent();
      checkPosition();
      draw();
   }
   guiManager()->clearStatus();
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
   fi.fIniDir = new char[128];
   strncpy(fi.fIniDir,  ".", 127);
   guiManager()->updateStatus("waiting for data file ...");
   new TGFileDialog(gClient->GetDefaultRoot(), guiManager()->getMainFrame(), kFDOpen, &fi);
   guiManager()->updateStatus("loading file ...");
   if (fi.fFilename) {
      m_navigator->appendFile(fi.fFilename, false, false);
      setLoadedAnyInputFileAfterStartup();
      checkPosition();
      draw();
      guiManager()->titleChanged(m_navigator->frameTitle());
   }
   guiManager()->clearStatus();
}

void
CmsShowMain::openDataViaURL()
{
   if (m_searchFiles.get() == 0) {
      m_searchFiles = std::auto_ptr<CmsShowSearchFiles>(new CmsShowSearchFiles("",
                                                                               "Open Remote Data Files",
                                                                               guiManager()->getMainFrame(),
                                                                               500, 400));
      m_searchFiles->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   std::string chosenFile = m_searchFiles->chooseFileFromURL();
   if(!chosenFile.empty()) {
      guiManager()->updateStatus("loading file ...");
      if(m_navigator->openFile(chosenFile.c_str())) {
         setLoadedAnyInputFileAfterStartup();
         m_navigator->firstEvent();
         checkPosition();
         draw();
         guiManager()->clearStatus();
      } else {
         guiManager()->updateStatus("failed to load data file");
      }
   }
}

//
// const member functions
//

//_______________________________________________________________________________
void 
CmsShowMain::autoLoadNewEvent()
{
   stopAutoLoadTimer();
   
   // case when start with no input file
   if (!m_loadedAnyInputFile)
   {
      if (m_monitor.get()) 
         startAutoLoadTimer();
      return;
   }

   bool reachedEnd = (forward() && m_navigator->isLastEvent()) || (!forward() && m_navigator->isFirstEvent());

   if (loop() && reachedEnd)
   {
      forward() ? m_navigator->firstEvent() : m_navigator->lastEvent();
      draw();
   }
   else if (!reachedEnd)
   {
      forward() ? m_navigator->nextEvent() : m_navigator->previousEvent();
      draw();
   }

   // stop loop in case no loop or monitor mode
   if (reachedEnd && (loop() || m_monitor.get()) == kFALSE)
   {
      if (forward() && m_navigator->isLastEvent())
      {
         guiManager()->enableActions();
         checkPosition();
      }

      if ((!forward()) && m_navigator->isFirstEvent())
      {
         guiManager()->enableActions();
         checkPosition();
      }
   }
   else
      startAutoLoadTimer();
}

//______________________________________________________________________________

void 
CmsShowMain::checkPosition()
{
   if ((m_monitor.get() || loop() ) && isPlaying())
      return;
   
   guiManager()->getMainFrame()->enableNavigatorControls();

   if (m_navigator->isFirstEvent())
      guiManager()->disablePrevious();

   if (m_navigator->isLastEvent())
   {
      guiManager()->disableNext();
      // force enable play events action in --port mode
      if (m_monitor.get() && !guiManager()->playEventsAction()->isEnabled())
         guiManager()->playEventsAction()->enable();
   }
}

//==============================================================================
void
CmsShowMain::setupDataHandling()
{
   guiManager()->updateStatus("Setting up data handling...");

   // navigator filtering  ->
   m_navigator->fileChanged_.connect(boost::bind(&CmsShowMain::fileChangedSlot, this, _1));
   m_navigator->editFiltersExternally_.connect(boost::bind(&FWGUIManager::updateEventFilterEnable, guiManager(), _1));
   m_navigator->filterStateChanged_.connect(boost::bind(&CmsShowMain::navigatorChangedFilterState, this, _1));
   m_navigator->postFiltering_.connect(boost::bind(&CmsShowMain::postFiltering, this, _1));

   // navigator fitlering <-
   guiManager()->showEventFilterGUI_.connect(boost::bind(&CmsShowNavigator::showEventFilterGUI, m_navigator.get(),_1));
   guiManager()->filterButtonClicked_.connect(boost::bind(&CmsShowMain::filterButtonClicked,this));

   // Data handling. File related and therefore not in the base class.
   if (guiManager()->getAction(cmsshow::sOpenData)    != 0) 
      guiManager()->getAction(cmsshow::sOpenData)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::openData));
   if (guiManager()->getAction(cmsshow::sAppendData)  != 0) 
      guiManager()->getAction(cmsshow::sAppendData)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::appendData));
   if (guiManager()->getAction(cmsshow::sSearchFiles) != 0)
      guiManager()->getAction(cmsshow::sSearchFiles)->activated.connect(sigc::mem_fun(*this, &CmsShowMain::openDataViaURL));

   setupActions();
   // init data from  CmsShowNavigator configuration, can do this with signals since there were not connected yet
   guiManager()->setFilterButtonIcon(m_navigator->getFilterState());

   for (unsigned int ii = 0; ii < m_inputFiles.size(); ++ii)
   {
      const std::string& fname = m_inputFiles[ii];
      if (fname.empty())
         continue;
      guiManager()->updateStatus("loading data file ...");
      if (!m_navigator->appendFile(fname, false, false))
      {
         guiManager()->updateStatus("failed to load data file");
      }
      else
      {
         m_loadedAnyInputFile = true; 
      }
   }

   if (m_loadedAnyInputFile)
   {
      m_navigator->firstEvent();
      checkPosition();
      if (configurationManager()->getIgnore())
         guiManager()->initEmpty();
      else
         setupConfiguration();
   }
   else {
      if (configFilename()[0] == '\0') {
         guiManager()->initEmpty();
      }
      else {
         setupConfiguration();
      }

      if (m_monitor.get() == 0 && (configurationManager()->getIgnore() == false)) {
         if (m_inputFiles.empty())
            openDataViaURL();
         else
            openData();
      }
   }
}

void
CmsShowMain::setLoadedAnyInputFileAfterStartup()
{
   if (m_loadedAnyInputFile == false) {
      m_loadedAnyInputFile = true;
      if ((configFilename()[0] == '\0') && (configurationManager()->getIgnore() == false))
         setupConfiguration();
   }
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
   m_monitor->Add(server);
}

void
CmsShowMain::connectSocket()
{
  m_monitor->Connect("Ready(TSocket*)","CmsShowMain",this,"notified(TSocket*)");

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
         guiManager()->updateStatus(s.str().c_str());
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
      guiManager()->updateStatus(s.str().c_str());

      bool appended = m_navigator->appendFile(fileName, true, m_live);

      if (appended)
      {
         if (m_live && isPlaying() && forward())
            m_navigator->activateNewFileOnNextEvent();
         else if (!isPlaying())
            checkPosition();

         // bootstrap case: --port  and no input file
         if (!m_loadedAnyInputFile)
         {
            m_loadedAnyInputFile = true;
            m_navigator->firstEvent();
            if (!isPlaying())
               draw();
         }

         std::stringstream sr;
         sr <<"New file registered '"<<fileName<<"'";
         guiManager()->updateStatus(sr.str().c_str());
      }
      else
      {
         std::stringstream sr;
         sr <<"New file NOT registered '"<<fileName<<"'";
         guiManager()->updateStatus(sr.str().c_str());
      }
   }
}

void
CmsShowMain::checkKeyBindingsOnPLayEventsStateChanged()
{
    if (m_live) {
        Int_t keycode = gVirtualX->KeysymToKeycode((int)kKey_Space);
        Window_t id = FWGUIManager::getGUIManager()->getMainFrame()->GetId();
        gVirtualX->GrabKey(id, keycode, 0, isPlaying());
    }
}

void
CmsShowMain::stopPlaying()
{
   stopAutoLoadTimer();
   if (m_live)
      m_navigator->resetNewFileOnNextEvent();
   CmsShowMainBase::stopPlaying();
   guiManager()->enableActions();
   checkPosition();
}

void
CmsShowMain::navigatorChangedFilterState(int state)
{
   guiManager()->setFilterButtonIcon(state);
   if (m_navigator->filesNeedUpdate() == false)
   {
      guiManager()->setFilterButtonText(m_navigator->filterStatusMessage());
      checkPosition();
   }
}

void
CmsShowMain::filterButtonClicked()
{
   if (m_navigator->getFilterState() == CmsShowNavigator::kWithdrawn )
      guiManager()->showEventFilterGUI();
   else
      m_navigator->toggleFilterEnable();
}

void
CmsShowMain::preFiltering()
{
   // called only if filter has changed
   guiManager()->updateStatus("Filtering events");
}

void
CmsShowMain::postFiltering(bool doDraw)
{
   // called only filter is changed
   guiManager()->clearStatus();
   if (doDraw) draw();
   checkPosition();
   guiManager()->setFilterButtonText(m_navigator->filterStatusMessage());
}

//______________________________________________________________________________

void
CmsShowMain::setLiveMode()
{
   m_live = true;
   m_liveTimer.reset(new SignalTimer());
   m_liveTimer->timeout_.connect(boost::bind(&CmsShowMain::checkLiveMode,this));

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
CmsShowMain::checkLiveMode()
{
   m_liveTimer->TurnOff();
   
#if defined(R__LINUX)
   TGX11 *x11 = dynamic_cast<TGX11*>(gVirtualX);
   if (x11) {
      XAnyEvent *ev = (XAnyEvent*) x11->GetNativeEvent();
      // printf("serial %d \n",(int)ev->serial );

      if ( !isPlaying() && m_lastXEventSerial == ev->serial )
         guiManager()->playEventsAction()->switchMode();
      m_lastXEventSerial = ev->serial;
   }
#endif
   m_liveTimer->SetTime((Long_t)(m_liveTimeout));
   m_liveTimer->Reset();
   m_liveTimer->TurnOn();
}

#include <iostream>
#include "boost/bind.hpp"

#include "Fireworks/FWInterface/interface/FWFFLooper.h"
#include "Fireworks/FWInterface/src/FWFFNavigator.h"
#include "Fireworks/FWInterface/src/FWFFMetadataManager.h"
#include "Fireworks/FWInterface/src/FWFFMetadataUpdateRequest.h"
#include "Fireworks/FWInterface/src/FWPathsPopup.h"
#include "Fireworks/Core/interface/FWConfigurationManager.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/src/CmsShowTaskExecutor.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/CSGContinuousAction.h"
#include "Fireworks/Core/interface/FWRecoGeom.h"
#include "Fireworks/Core/interface/FWGeometryTableViewManager.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Geometry/interface/FWRecoGeometry.h"
#include "Fireworks/Geometry/interface/FWRecoGeometryRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/Framework/interface/ModuleChanger.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "DataFormats/FWLite/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TRint.h"
#include "TGLWidget.h"

#include "TEveManager.h"
#include "TEveEventManager.h"
#include "TEveTrackPropagator.h"
#include "TEveBrowser.h"
#include "TGeoManager.h"


namespace edm
{
   class StreamContext;
   class ModuleCallingContext;
}

namespace
{
   class CmsEveMagField : public TEveMagField
   {
   private:
      Float_t fField;
      Float_t fFieldMag;

   public:

      CmsEveMagField() : TEveMagField(), fField(-3.8), fFieldMag(3.8) {}
      virtual ~CmsEveMagField() {}

      // set current
      void SetFieldByCurrent(Float_t avg_current)
      {
         fField    = -3.8 * avg_current / 18160.0;
         fFieldMag = TMath::Abs(fField);
      }

      // get field values
      virtual Float_t GetMaxFieldMag() const override
      {
         return fFieldMag;
      }

      virtual TEveVector GetField(Float_t x, Float_t y, Float_t z) const override
      {
         static const Float_t barrelFac = 1.2 / 3.8;
         static const Float_t endcapFac = 2.0 / 3.8;

         const Float_t R    = sqrt(x*x+y*y);
         const Float_t absZ = TMath::Abs(z);

         //barrel
         if (absZ < 724.0f)
         {
            //inside solenoid
            if (R < 300.0f) return TEveVector(0, 0, fField);

            // outside solinoid
            if ((R > 461.0f && R < 490.5f) ||
                (R > 534.5f && R < 597.5f) ||
                (R > 637.0f && R < 700.0f))
            {
               return TEveVector(0, 0, -fField*barrelFac);
            }
         } else {
            if ((absZ > 724.0f && absZ < 786.0f) ||
                (absZ > 850.0f && absZ < 910.0f) ||
                (absZ > 975.0f && absZ < 1003.0f))
            {
               const Float_t fac = (z >= 0 ? fField : -fField) * endcapFac / R;
               return TEveVector(x*fac, y*fac, 0);
            }
         }
         return TEveVector(0, 0, 0);
      }
   };
}

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//==============================================================================
// constructors and destructor
//==============================================================================

FWFFLooper::FWFFLooper(edm::ParameterSet const&ps) 
   : CmsShowMainBase(),
     m_navigator(new FWFFNavigator(*this)), 
     m_metadataManager(new FWFFMetadataManager()),
     m_context(new fireworks::Context(changeManager(),
                                      selectionManager(),
                                      eiManager(),
                                      colorManager(),
                                      m_metadataManager.get())),
     m_Rint(m_appHelper->app()),
     m_AllowStep(true),
     m_ShowEvent(true),
     m_firstTime(true),
     m_pathsGUI(0),
     m_geomWatcher(this, &FWFFLooper::remakeGeometry)
{
   setup(m_navigator.get(), m_context.get(), m_metadataManager.get());

   eiManager()->setContext(m_context.get());

   // By default, we look up geometry and configuration in the workarea, then
   // in the release area then in the local directory.  It is also possible to
   // override those locations by using the displayConfigurationFilename and
   // geometryFilename in the parameterset.
   const char *releaseBase = getenv("CMSSW_RELEASE_BASE");
   const char *workarea = getenv("CMSSW_BASE");
   std::string displayConfigRelFilename = "/src/Fireworks/FWInterface/macros/ffw.fwc";
   std::string geometryRelFilename = "/src/Fireworks/FWInterface/data/cmsGeom10.root";

   std::string displayConfigFilename = "ffw.fwc";
   std::string geometryFilename;

   if (releaseBase && access((releaseBase + displayConfigFilename).c_str(), R_OK) == 0)
      displayConfigFilename = releaseBase + displayConfigRelFilename; 
   if (workarea && access((workarea + displayConfigRelFilename).c_str(), R_OK) == 0)
      displayConfigFilename = workarea + displayConfigRelFilename;   

   if (releaseBase && access((releaseBase + geometryRelFilename).c_str(), R_OK) == 0)
      geometryFilename = releaseBase + geometryRelFilename;
   if (workarea && access((workarea + geometryRelFilename).c_str(), R_OK) == 0)
      geometryFilename = workarea + geometryRelFilename;

   displayConfigFilename = ps.getUntrackedParameter<std::string>("displayConfigFilename", displayConfigFilename);
   geometryFilename = ps.getUntrackedParameter<std::string>("geometryFilename", geometryFilename);
   if( !geometryFilename.empty())
   {
      loadDefaultGeometryFile();
   }
   setGeometryFilename(geometryFilename);
   setConfigFilename(displayConfigFilename);

   if( !geometryFilename.empty())
   {
      loadDefaultGeometryFile();
   }

   m_MagField = new CmsEveMagField();
}

void
FWFFLooper::loadDefaultGeometryFile( void )
{
   CmsShowTaskExecutor::TaskFunctor f;
   f=boost::bind( &CmsShowMainBase::loadGeometry, this );
   startupTasks()->addTask( f );
}

void
FWFFLooper::attachTo(edm::ActivityRegistry &ar)
{
   m_pathsGUI = new FWPathsPopup(this, guiManager());

   ar.watchPostModuleEvent(m_pathsGUI, &FWPathsPopup::postModuleEvent);
   ar.watchPreModuleEvent(m_pathsGUI, &FWPathsPopup::preModuleEvent);
   ar.watchPostEndJob(this, &FWFFLooper::postEndJob);
}

FWFFLooper::~FWFFLooper()
{
   delete m_MagField;
}


//==============================================================================
// Service watchers
//==============================================================================

void
FWFFLooper::startingNewLoop(unsigned int count)
{
   // Initialise on first loop.
   if (count == 0)
   {
      const edm::ScheduleInfo *info = scheduleInfo();
      m_pathsGUI->setup(info);

      // We need to enter the GUI loop in order to 
      // have all the callbacks executed. The last callback will
      // be responsible for returning the control to CMSSW. 
      assert(m_Rint);
      CmsShowTaskExecutor::TaskFunctor f;
      f=boost::bind(&TApplication::Terminate, m_Rint, 0);
      startupTasks()->addTask(f);
      // FIXME: do we really need to delay tasks like this?
      startupTasks()->startDoingTasks(); 
      m_Rint->Run(kTRUE);
      // Show the GUI ...
      gSystem->ProcessEvents();
   }
}

void 
FWFFLooper::postEndJob()
{
   printf("FWFFLooper::postEndJob\n");
   TEveManager::Terminate();
}

void
FWFFLooper::checkPosition()
{
   if (loop() && isPlaying())
      return;
  
   guiManager()->getMainFrame()->enableNavigatorControls();
   guiManager()->getMainFrame()->enableComplexNavigation(false);

   if (m_isFirstEvent)
      guiManager()->disablePrevious();

   if (m_isLastEvent)
   {
      guiManager()->disableNext();
      // force enable play events action in --port mode
      if (!guiManager()->playEventsAction()->isEnabled())
         guiManager()->playEventsAction()->enable();
   }
}

/** This actually needs to be different from the standalone
    case because nextEvent() / previousEvent() will immediately
    interrupt the GUI event loop and fall back to the looper.
  */
void
FWFFLooper::autoLoadNewEvent()
{
   stopAutoLoadTimer();
   bool reachedEnd = (forward() && m_isLastEvent) || (!forward() && m_isFirstEvent);

   if (!reachedEnd || loop())
   {
      // Will exit the loop here!
      m_autoReload = true;
      forward() ? m_navigator->nextEvent() : m_navigator->previousEvent();
   }
   else
   {
      m_autoReload = false;
      CmsShowMainBase::stopPlaying();
      guiManager()->enableActions();
      guiManager()->getMainFrame()->enableComplexNavigation(false);
   }
}

void
FWFFLooper::stopPlaying()
{
   stopAutoLoadTimer();
   m_autoReload = false;
   CmsShowMainBase::stopPlaying();
   guiManager()->enableActions();
   guiManager()->getMainFrame()->enableComplexNavigation(false);
   checkPosition();
}


//------------------------------------------------------------------------------

void
FWFFLooper::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
   // Check DisplayGeomRecord changes.
   try {
      m_geomWatcher.check(iSetup);
   }
   catch (...) {}
   
   // If the geometry was not picked up from a file, we try to get it from the
   // EventSetup!
   // FIXME: we need to check we execute only once because the view managers
   //        depend on geometry and they cannot be initialised more than once.
   //        This should actually be cleaned up so that the various view manager
   //        don't care about geometry.
   // FIXME: we should actually be able to update the geometry when requested.
   //        this is not possible at the moment.
   if (m_firstTime == true)
   {
      if (m_context->getGeom() == 0)
      {
	 try
	 {
	    guiManager()->updateStatus("Loading geometry...");
	    edm::ESHandle<FWRecoGeometry> geoh;
	    iSetup.get<FWRecoGeometryRecord>().get(geoh);
	    getGeom().initMap(geoh.product()->idToName);
	    m_context->setGeom(&(getGeom()));
	 }
	 catch( const cms::Exception& exception )
	 {
            setGeometryFilename("cmsGeom10.root");
	    CmsShowMainBase::loadGeometry();
	 }
      }

      setupViewManagers();
      setupConfiguration();
      setupActions();
      
      guiManager()->showEventFilterGUI_.connect(boost::bind(&FWFFLooper::showPathsGUI, this, _1));
      guiManager()->setFilterButtonText("Show paths / CMSSW configuration editor");
      guiManager()->filterButtonClicked_.connect(boost::bind(&FWGUIManager::showEventFilterGUI, guiManager()));

      m_firstTime = false;
      m_autoReload = false;
   }

   float current = 18160.0f;
   try {
      edm::Handle<edm::ConditionsInRunBlock> runCond;
      // bool res = iRun.getByType(runCond);
      bool res = iRun.getByLabel("conditionsInEdm", runCond);
      if (res && runCond.isValid())
      {
         printf("Got current from conds in edm %f\n", runCond->BAvgCurrent);
         current = runCond->BAvgCurrent;
      }
      else
      {
         printf("Could not extract run-conditions get-result=%d, is-valid=%d\n", res, runCond.isValid());

         const edm::eventsetup::EventSetupRecord* rec = iSetup.find( edm::eventsetup::EventSetupRecordKey::makeKey<RunInfoRcd>());
         if( 0 != rec )
         {
            edm::ESHandle<RunInfo> sum;
            iSetup.get<RunInfoRcd>().get(sum);

            current = sum->m_avg_current;
            printf("Got current from RunInfoRcd %f\n", sum->m_avg_current);
         }
      }
   }
   catch (...) {
      fwLog(fwlog::kInfo) << "ConditionsInRunBlock not available\n";
   }
   static_cast<CmsEveMagField*>(m_MagField)->SetFieldByCurrent(current);
   context()->getField()->setFFFieldMag(m_MagField->GetMaxFieldMag());
}

//------------------------------------------------------------------------------
edm::EDLooperBase::Status
FWFFLooper::duringLoop(const edm::Event &event, 
                       const edm::EventSetup&es, 
                       edm::ProcessingController &controller)
{
   // Check DisplayGeomRecord changes.
   try { 
      m_geomWatcher.check(es);
   } catch (...) {}
   

   m_pathsGUI->postEvent(event);

   m_isLastEvent = controller.forwardState() == edm::ProcessingController::kAtLastEvent;
   m_isFirstEvent = controller.reverseState() == edm::ProcessingController::kAtFirstEvent;
   // If the next event id is valid, set the transition so 
   // that we go to it go to to it.
   if (m_nextEventId != edm::EventID())
   {
      controller.setTransitionToEvent(m_nextEventId);
      m_nextEventId = edm::EventID();
      return kContinue;
   }
   // We handle "last event" by going to the first event and then moving to the
   // previous event.
   if (m_navigator->currentTransition() == FWFFNavigator::kLastEvent)
   {
      m_navigator->resetTransition();
      controller.setTransitionToPreviousEvent();
      return kContinue;
   }

   m_pathsGUI->hasChanges() = false;
   m_metadataManager->update(new FWFFMetadataUpdateRequest(event));
   m_navigator->setCurrentEvent(&event);
   if (m_autoReload == true)
      startAutoLoadTimer();

   checkPosition();
   draw();
      
   m_Rint->Run(kTRUE);
   // If the GUI changed the PSet, save the current event to reload
   // it on next iteration.
   if (m_pathsGUI->hasChanges())
   {
      m_nextEventId = edm::EventID();
      return kStop;
   }
   else if (m_navigator->currentTransition() == FWFFNavigator::kFirstEvent)
   {
      m_nextEventId = m_navigator->getFirstEventID();
      return kStop;
   }
   else if (m_navigator->currentTransition() == FWFFNavigator::kLastEvent)
   {
      m_nextEventId = m_navigator->getFirstEventID();
      return kStop;
   }
   else if (m_navigator->currentTransition() == FWFFNavigator::kNextEvent)
      controller.setTransitionToNextEvent();
   else if (m_navigator->currentTransition() == FWFFNavigator::kPreviousEvent)
      controller.setTransitionToPreviousEvent();
   return kContinue;
}

//------------------------------------------------------------------------------
void 
FWFFLooper::display(const std::string& info)
{
   // Display whatever was registered so far, wait until user presses
   // the "Step" button.

   if (m_AllowStep)
   {
      gEve->Redraw3D();
      m_Rint->Run(kTRUE);
   }
}

//==============================================================================
// Getters for cleints
//==============================================================================

TEveMagField* 
FWFFLooper::getMagField()
{
   return m_MagField;
}

void 
FWFFLooper::setupFieldForPropagator(TEveTrackPropagator* prop)
{
   prop->SetMagFieldObj(m_MagField, kFALSE);
}

void 
FWFFLooper::quit()
{
   gSystem->ExitLoop();

   // Throwing exception here is bad because:
   //   a) it does not work when in a "debug step";
   //   b) does not restore terminal state.
   // So we do exit instead for now.
   // throw cms::Exception("UserTerminationRequest");
   gSystem->Exit(0);
}

/** This is called at the end of looping.
    We always continue because we want the transition
    set in the ProcessingController to happen.
  */
edm::EDLooperBase::Status
FWFFLooper::endOfLoop(const edm::EventSetup&, unsigned int)
{
   // Looks like the module changer is availble only here.
   for (ModuleChanges::iterator i = m_scheduledChanges.begin(),
                                e = m_scheduledChanges.end();
        i != e; ++i)
   {
      try {
         moduleChanger()->changeModule(i->first, i->second);
      }
      catch (cms::Exception const& e)
      { 
         fwLog(fwlog::kError) << "FWFFLooper::endOfLoop caught exception.\n";
         std::cerr << e.what() << std::endl;
      } 
   }
   m_scheduledChanges.clear();
   return kContinue;
}

void
FWFFLooper::showPathsGUI(const TGWindow *)
{
   if (!m_pathsGUI)
      return;
   if (m_pathsGUI->IsMapped())
   {
      guiManager()->setFilterButtonText("Show paths / CMSSW configuration editor");
      m_pathsGUI->UnmapWindow();
   }
   else
   {
      guiManager()->setFilterButtonText("Hide paths / CMSSW configuration editor");
      m_pathsGUI->MapWindow();
   }
}

void
FWFFLooper::requestChanges(const std::string &moduleLabel, const edm::ParameterSet& ps)
{
   m_scheduledChanges[moduleLabel] = ps;
}
 
//______________________________________________________________________________

void
FWFFLooper::remakeGeometry(const DisplayGeomRecord& dgRec)
{
   fwLog(fwlog::kInfo) << "FWFFLooper set TGeo geomtery from DisplayGeomRecord.\n";

   edm::ESHandle<TGeoManager> geom;
   dgRec.get(geom);
   TEveGeoManagerHolder _tgeo(const_cast<TGeoManager*>(geom.product()));
   FWGeometryTableViewManager::setGeoManagerRuntime(gGeoManager);
}

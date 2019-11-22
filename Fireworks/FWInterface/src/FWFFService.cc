#include <iostream>
#include "boost/bind.hpp"
#include "Fireworks/FWInterface/interface/FWFFService.h"
#include "Fireworks/FWInterface/src/FWFFNavigator.h"
#include "Fireworks/FWInterface/src/FWFFMetadataManager.h"
#include "Fireworks/FWInterface/src/FWFFMetadataUpdateRequest.h"
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/src/CmsShowTaskExecutor.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/CSGContinuousAction.h"
#include "Fireworks/Core/interface/FWRecoGeom.h"
#include "Fireworks/Geometry/interface/FWRecoGeometry.h"
#include "Fireworks/Geometry/interface/FWRecoGeometryRecord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TRint.h"
#include "TEveManager.h"
#include "TEveEventManager.h"
#include "TEveTrackPropagator.h"
#include "TGLWidget.h"

#include "TEveBrowser.h"

namespace {
  class CmsEveMagField : public TEveMagField {
  private:
    Float_t fField;
    Float_t fFieldMag;

  public:
    CmsEveMagField() : TEveMagField(), fField(-3.8), fFieldMag(3.8) {}
    ~CmsEveMagField() override {}

    // set current
    void SetFieldByCurrent(Float_t avg_current) {
      fField = -3.8 * avg_current / 18160.0;
      fFieldMag = TMath::Abs(fField);
    }

    // get field values
    Float_t GetMaxFieldMag() const override { return fFieldMag; }

    TEveVector GetField(Float_t x, Float_t y, Float_t z) const override {
      static const Float_t barrelFac = 1.2 / 3.8;
      static const Float_t endcapFac = 2.0 / 3.8;

      const Float_t R = sqrt(x * x + y * y);
      const Float_t absZ = TMath::Abs(z);

      //barrel
      if (absZ < 724.0f) {
        //inside solenoid
        if (R < 300.0f)
          return TEveVector(0, 0, fField);

        // outside solinoid
        if ((R > 461.0f && R < 490.5f) || (R > 534.5f && R < 597.5f) || (R > 637.0f && R < 700.0f)) {
          return TEveVector(0, 0, -fField * barrelFac);
        }
      } else {
        if ((absZ > 724.0f && absZ < 786.0f) || (absZ > 850.0f && absZ < 910.0f) || (absZ > 975.0f && absZ < 1003.0f)) {
          const Float_t fac = (z >= 0 ? fField : -fField) * endcapFac / R;
          return TEveVector(x * fac, y * fac, 0);
        }
      }
      return TEveVector(0, 0, 0);
    }
  };
}  // namespace

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//==============================================================================
// constructors and destructor
//==============================================================================

FWFFService::FWFFService(edm::ParameterSet const& ps, edm::ActivityRegistry& ar)
    : CmsShowMainBase(),
      m_navigator(new FWFFNavigator(*this)),
      m_metadataManager(new FWFFMetadataManager()),
      m_context(new fireworks::Context(
          changeManager(), selectionManager(), eiManager(), colorManager(), m_metadataManager.get())),
      m_appHelper(ps, ar),
      m_Rint(m_appHelper.app()),
      m_AllowStep(true),
      m_ShowEvent(true),
      m_firstTime(true) {
  printf("FWFFService::FWFFService CTOR\n");

  setup(m_navigator.get(), m_context.get(), m_metadataManager.get());

  eiManager()->setContext(m_context.get());

  // By default, we look up geometry and configuration in the workarea, then
  // in the release area then in the local directory.  It is also possible to
  // override those locations by using the displayConfigurationFilename and
  // geometryFilename in the parameterset.
  const char* releaseBase = std::getenv("CMSSW_RELEASE_BASE");
  const char* workarea = std::getenv("CMSSW_BASE");
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

  displayConfigFilename = ps.getUntrackedParameter<std::string>("diplayConfigFilename", displayConfigFilename);
  geometryFilename = ps.getUntrackedParameter<std::string>("geometryFilename", geometryFilename);

  setGeometryFilename(geometryFilename);
  setConfigFilename(displayConfigFilename);

  CmsShowTaskExecutor::TaskFunctor f;

  if (!geometryFilename.empty()) {
    f = boost::bind(&CmsShowMainBase::loadGeometry, this);
    startupTasks()->addTask(f);
  }

  m_MagField = new CmsEveMagField();

  // ----------------------------------------------------------------

  ar.watchPostBeginJob(this, &FWFFService::postBeginJob);
  ar.watchPostEndJob(this, &FWFFService::postEndJob);

  ar.watchPostBeginRun(this, &FWFFService::postBeginRun);

  ar.watchPostProcessEvent(this, &FWFFService::postProcessEvent);
}

FWFFService::~FWFFService() {
  printf("FWFFService::~FWFFService DTOR\n");

  delete m_MagField;
}

//==============================================================================
// Service watchers
//==============================================================================

void FWFFService::postBeginJob() {
  printf("FWFFService::postBeginJob\n");
  // We need to enter the GUI loop in order to
  // have all the callbacks executed. The last callback will
  // be responsible for returning the control to CMSSW.
  assert(m_Rint);
  CmsShowTaskExecutor::TaskFunctor f;
  f = boost::bind(&TApplication::Terminate, m_Rint, 0);
  startupTasks()->addTask(f);
  // FIXME: do we really need to delay tasks like this?
  startupTasks()->startDoingTasks();
  m_Rint->Run(kTRUE);
  // Show the GUI ...
  gSystem->ProcessEvents();
}

void FWFFService::postEndJob() {
  printf("FWFFService::postEndJob\n");

  TEveManager::Terminate();
}

void FWFFService::checkPosition() {
  if (loop() && isPlaying())
    return;

  guiManager()->getMainFrame()->enableNavigatorControls();

  if (m_navigator->isFirstEvent())
    guiManager()->disablePrevious();

  if (m_navigator->isLastEvent()) {
    guiManager()->disableNext();
    // force enable play events action in --port mode
    if (!guiManager()->playEventsAction()->isEnabled())
      guiManager()->playEventsAction()->enable();
  }
}
//------------------------------------------------------------------------------

void FWFFService::postBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  // If the geometry was not picked up from a file, we try to get it from the
  // EventSetup!
  // FIXME: we need to check we execute only once because the view managers
  //        depend on geometry and they cannot be initialised more than once.
  //        This should actually be cleaned up so that the various view manager
  //        don't care about geometry.
  // FIXME: we should actually be able to update the geometry when requested.
  //        this is not possible at the moment.
  if (m_firstTime == true) {
    if (m_context->getGeom() == nullptr) {
      guiManager()->updateStatus("Loading geometry...");
      edm::ESTransientHandle<FWRecoGeometry> geoh;
      iSetup.get<FWRecoGeometryRecord>().get(geoh);
      getGeom().initMap(geoh.product()->idToName);
      m_context->setGeom(&(getGeom()));
    }

    setupViewManagers();
    setupConfiguration();
    setupActions();
    m_firstTime = false;
  }

  float current = 18160.0f;

  edm::Handle<edm::ConditionsInRunBlock> runCond;
  // bool res = iRun.getByType(runCond);
  bool res = iRun.getByLabel("conditionsInEdm", runCond);
  if (res && runCond.isValid()) {
    printf("Got current from conds in edm %f\n", runCond->BAvgCurrent);
    current = runCond->BAvgCurrent;
  } else {
    printf("Could not extract run-conditions get-result=%d, is-valid=%d\n", res, runCond.isValid());

    edm::ESHandle<RunInfo> sum;
    iSetup.get<RunInfoRcd>().get(sum);

    current = sum->m_avg_current;
    printf("Got current from RunInfoRcd %f\n", sum->m_avg_current);
  }

  static_cast<CmsEveMagField*>(m_MagField)->SetFieldByCurrent(current);
}

//------------------------------------------------------------------------------
void FWFFService::postProcessEvent(const edm::Event& event, const edm::EventSetup& es) {
  printf("FWFFService::postProcessEvent: Starting GUI loop.\n");

  m_metadataManager->update(new FWFFMetadataUpdateRequest(event));
  m_navigator->setCurrentEvent(&event);
  checkPosition();
  draw();
  m_Rint->Run(kTRUE);
}

//------------------------------------------------------------------------------

void FWFFService::display(const std::string& info) {
  // Display whatever was registered so far, wait until user presses
  // the "Step" button.

  if (m_AllowStep) {
    gEve->Redraw3D();
    m_Rint->Run(kTRUE);
  }
}

//==============================================================================
// Getters for cleints
//==============================================================================

TEveMagField* FWFFService::getMagField() { return m_MagField; }

void FWFFService::setupFieldForPropagator(TEveTrackPropagator* prop) { prop->SetMagFieldObj(m_MagField, kFALSE); }

void FWFFService::quit() {
  gSystem->ExitLoop();
  printf("FWFFService exiting on user request.\n");

  // Throwing exception here is bad because:
  //   a) it does not work when in a "debug step";
  //   b) does not restore terminal state.
  // So we do exit instead for now.
  // throw cms::Exception("UserTerminationRequest");
  gSystem->Exit(0);
}

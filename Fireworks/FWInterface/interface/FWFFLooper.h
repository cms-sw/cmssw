#ifndef Fireworks_Core_FWFFLooper_h
#define Fireworks_Core_FWFFLooper_h

#include "Fireworks/Core/interface/CmsShowMainBase.h"
#include "Fireworks/FWInterface/interface/FWFFHelper.h"
#include "FWCore/Framework/interface/EDLooperBase.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "Fireworks/Geometry/interface/DisplayGeomRecord.h"
#include <string>
#include <Rtypes.h>
#include <memory>

namespace edm
{
   class ParameterSet;
   class ActivityRegistry;
   class Run;
   class Event;
   class EventSetup;
   class ProcessingController;
   class ModuleChanger;
   class ParameterSet;
   class LuminosityBlockPrincipal;
}

class FWFFNavigator;
class FWFFMetadataManager;
class FWPathsPopup;

namespace fireworks
{
   class Context;
}

class TEveManager;
class TEveElement;
class TEveMagField;
class TEveTrackPropagator;
class TRint;
class TGWindow;

class FWFFLooper : public CmsShowMainBase, public edm::EDLooperBase
{
public:
   FWFFLooper(const edm::ParameterSet&);
   virtual ~FWFFLooper();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   virtual void attachTo(edm::ActivityRegistry &) override;
   void postBeginJob();
   void postEndJob();

   virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

   void display(const std::string& info="");

   TEveMagField* getMagField();
   void          setupFieldForPropagator(TEveTrackPropagator* prop);

   virtual void checkPosition() override;
   virtual void stopPlaying() override ;
   virtual void autoLoadNewEvent() override;

   void showPathsGUI(const TGWindow *p);

   void quit() override;

   virtual void startingNewLoop(unsigned int) override;
   virtual edm::EDLooperBase::Status endOfLoop(const edm::EventSetup&, unsigned int) override;
   virtual edm::EDLooperBase::Status duringLoop(const edm::Event&, const edm::EventSetup&, edm::ProcessingController&) override; 
   void requestChanges(const std::string &, const edm::ParameterSet &);

   void remakeGeometry(const DisplayGeomRecord& dgRec);

private:
   FWFFLooper(const FWFFLooper&);                  // stop default
   const FWFFLooper& operator=(const FWFFLooper&); // stop default

   void loadDefaultGeometryFile( void );

   edm::Service<FWFFHelper>            m_appHelper;
   std::auto_ptr<FWFFNavigator>        m_navigator;
   std::auto_ptr<FWFFMetadataManager>  m_metadataManager;
   std::auto_ptr<fireworks::Context>   m_context;

   TEveManager  *m_EveManager;
   TRint        *m_Rint;

   TEveMagField *m_MagField;
   
   bool          m_AllowStep;
   bool          m_ShowEvent;
   bool          m_firstTime;
   FWPathsPopup  *m_pathsGUI;
   
   typedef std::map<std::string, edm::ParameterSet> ModuleChanges;
   ModuleChanges m_scheduledChanges;
   edm::EventID  m_nextEventId;
   bool          m_autoReload;
   bool          m_isFirstEvent;
   bool          m_isLastEvent;

   edm::ESWatcher<DisplayGeomRecord> m_geomWatcher;
};

#endif

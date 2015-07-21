#ifndef Fireworks_Core_CmsShowMainBase_h
#define Fireworks_Core_CmsShowMainBase_h

#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"
#include "Fireworks/Core/interface/FWGeometry.h"

#include "DataFormats/Provenance/interface/EventID.h"

#include <memory>
#include <string>
#include <cassert>
#include "sigc++/signal.h"

#include "TTimer.h"

class FWEventItemsManager;
class FWGUIManager;
class FWJobMetadataManager;
class FWNavigatorBase;

// replace forfard declaration with include to avoid compilation warnigs
/*
class FWModelChangeManager;
class FWSelectionManager;
class FWViewManagerManager;
class CmsShowTaskExecutor;
class FWChangeManager;
class FWColorManager;
class FWConfigurationManager;
*/
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWConfigurationManager.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/src/CmsShowTaskExecutor.h"

namespace fireworks {
   class Context;
}

/** This is the base class to be used for setting up the main loop and
    navigation. FF and standalone main should derive from this one,
    since it takes care of most of the set up and navigation logic. 
    Concrete implementations are actually supposed to create a concrete
    instance of the Context, FWNavigatorBase and FWJobMetadataManager. 
  */
class CmsShowMainBase
{
public:
   CmsShowMainBase();
   virtual ~CmsShowMainBase();

   FWModelChangeManager       *changeManager() {return m_changeManager.get(); }
   FWColorManager             *colorManager() { return  m_colorManager.get(); }
   FWConfigurationManager     *configurationManager() { return m_configurationManager.get(); }
   FWEventItemsManager        *eiManager() { return m_eiManager.get(); }
   FWModelChangeManager       *modelChangeManager() { return m_changeManager.get(); }
   FWSelectionManager         *selectionManager() { return m_selectionManager.get(); }
   FWViewManagerManager       *viewManager() { return m_viewManager.get(); }
   FWGUIManager               *guiManager() 
   { 
      assert(m_guiManager.get() && "Call CmsShowMainBase::setup first!"); 
      return m_guiManager.get(); 
   }
   
   CmsShowTaskExecutor        *startupTasks() { return m_startupTasks.get(); }
   
   void setup(FWNavigatorBase             *navigator,
              fireworks::Context          *context,
              FWJobMetadataManager        *metadataManager);

   void setupActions();
   void setupViewManagers();

   // Configuration handling.
   void setConfigFilename(const std::string &f) { m_configFileName = f; };
   const char *configFilename() const { return m_configFileName.c_str(); };

   void writeToConfigFile(const std::string &config);
   void writeToCurrentConfigFile();
   void writePartialToConfigFile();
   void reloadConfiguration(const std::string &config);
   void partialWriteToConfigFile(const std::string &config);
   void partialLoadConfiguration(const std::string &config);
   void setupConfiguration();
   
   void registerPhysicsObject(const FWPhysicsObjectDesc&iItem);
   void draw();

   // Geometry handling
   void loadGeometry();
   void setGeometryFilename(const std::string &filename) {m_geometryFilename = filename; }
   const std::string &geometryFilename(void) { return m_geometryFilename; }
   FWGeometry& getGeom() { return m_geom; }

   void setSimGeometryFilename(const std::string &filename) {m_simGeometryFilename = filename; }
   
   // Event navigation.
   void doFirstEvent();
   void doPreviousEvent();
   void doNextEvent();
   void doLastEvent();
   void goToRunEvent(edm::RunNumber_t, edm::LuminosityBlockNumber_t, edm::EventNumber_t);
   virtual void checkPosition() = 0;
   bool forward() const { return m_forward; }
   bool loop() const { return m_loop; }
   virtual void quit() = 0;
   
   void setupAutoLoad(float x);
   void startAutoLoadTimer();
   void stopAutoLoadTimer();
   void setupDebugSupport();
   
   void setPlayDelay(Float_t val);
   void playForward();
   void playBackward();
   bool isPlaying() const { return m_isPlaying; }

   virtual void checkKeyBindingsOnPLayEventsStateChanged() {}
   virtual void stopPlaying();
   virtual void autoLoadNewEvent() = 0;

   void setPlayLoop();
   void unsetPlayLoop();

   void setAutoSaveAllViewsFormat(const std::string& fmt) { m_autoSaveAllViewsFormat = fmt; }
   void setAutoSaveAllViewsHeight(int x) { m_autoSaveAllViewsHeight = x; }

   class SignalTimer : public TTimer {
   public:
      virtual Bool_t Notify() {
         timeout_();
         return true;
      }
      sigc::signal<void> timeout_;
   };

protected: 
   void eventChangedSlot();
   virtual void eventChangedImp();
   void sendVersionInfo();
   fireworks::Context* context() { return m_contextPtr; }

private:
   // The base class is responsible for the destruction of fwlite / FF
   // agnostic managers.
   std::auto_ptr<FWModelChangeManager>   m_changeManager;
   std::auto_ptr<FWColorManager>         m_colorManager;
   std::auto_ptr<FWConfigurationManager> m_configurationManager;
   std::auto_ptr<FWEventItemsManager>    m_eiManager;
   std::auto_ptr<FWGUIManager>           m_guiManager;
   std::auto_ptr<FWSelectionManager>     m_selectionManager;
   std::auto_ptr<CmsShowTaskExecutor>    m_startupTasks;
   std::auto_ptr<FWViewManagerManager>   m_viewManager;

  

   std::auto_ptr<SignalTimer>                 m_autoLoadTimer;
   
   // These are actually set by the concrete implementation via the setup 
   // method.
   FWNavigatorBase                      *m_navigatorPtr;
   FWJobMetadataManager                 *m_metadataManagerPtr;
   fireworks::Context                   *m_contextPtr;

   void setPlayLoopImp();
   void unsetPlayLoopImp();
   
   std::string                           m_autoSaveAllViewsFormat;
   int                                   m_autoSaveAllViewsHeight;
   bool                                  m_autoLoadTimerRunning;             
   bool                                  m_forward;
   bool                                  m_isPlaying;
   bool                                  m_loop;
   Float_t                               m_playDelay;
   std::string                           m_configFileName;
   std::string                           m_geometryFilename;
   FWGeometry                            m_geom;
   std::string                           m_simGeometryFilename;
};

#endif

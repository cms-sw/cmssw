#ifndef Fireworks_Core_CmsShowMainBase_h
#define Fireworks_Core_CmsShowMainBase_h

#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"

#include <memory>
#include <string>
#include <cassert>
#include "sigc++/signal.h"

#include "TTimer.h"

class CmsShowTaskExecutor;
class FWChangeManager;
class FWColorManager;
class FWConfigurationManager;
class FWEventItemsManager;
class FWGUIManager;
class FWJobMetadataManager;
class FWModelChangeManager;
class FWNavigatorBase;
class FWSelectionManager;
class FWViewManagerManager;

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
   // Configuration handling.
   void setConfigFilename(const std::string &f) { m_configFileName = f; };
   const char *configFilename() const { return m_configFileName.c_str(); };

   void reloadConfiguration(const std::string &config);
   void setupConfiguration();
   
   void registerPhysicsObject(const FWPhysicsObjectDesc&iItem);
   void draw();

   // Event navigation.
   void doFirstEvent();
   void doPreviousEvent();
   void doNextEvent();
   void doLastEvent();
   void goToRunEvent(int, int);
   virtual void checkPosition() = 0;
   bool forward() const { return m_forward; }
   bool loop() const { return m_loop; }
   bool live() const { return m_live; }
   virtual void quit() = 0;
   
   void setupAutoLoad(float x);
   void startAutoLoadTimer();
   void stopAutoLoadTimer();
   void setLiveMode();
   void setupDebugSupport();
   void checkLiveMode();
   
   void setPlayDelay(Float_t val);
   void playForward();
   void playBackward();
   bool isPlaying() const { return m_isPlaying; }
   void setIsPlaying(bool value) { m_isPlaying = value; }
   virtual void stopPlaying() = 0;
   virtual void autoLoadNewEvent() = 0;

   void setPlayLoop();
   void unsetPlayLoop();
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

   class SignalTimer : public TTimer {
   public:
      virtual Bool_t Notify() {
         timeout_();
         return true;
      }
      sigc::signal<void> timeout_;
   };

   std::auto_ptr<SignalTimer>                 m_autoLoadTimer;
   std::auto_ptr<SignalTimer>                 m_liveTimer;
   
   // These are actually set by the concrete implementation via the setup 
   // method.
   FWNavigatorBase                      *m_navigator;
   FWJobMetadataManager                 *m_metadataManager;
   fireworks::Context                   *m_context;

   void setPlayLoopImp();
   void unsetPlayLoopImp();
   
   std::string                           m_autoSaveAllViewsFormat;
   bool                                  m_autoLoadTimerRunning;             
   bool                                  m_forward;
   bool                                  m_isPlaying;
   bool                                  m_loop;
   Int_t                                 m_lastPointerPositionX;
   Int_t                                 m_lastPointerPositionY;
   bool                                  m_live;
   int                                   m_liveTimeout;
   Float_t                               m_playDelay;
   std::string                           m_configFileName;
};

#endif

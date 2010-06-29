// -*- C++ -*-
#ifndef Fireworks_Core_CmsShowMain_h
#define Fireworks_Core_CmsShowMain_h
//
// Package:     Core
// Class  :     CmsShowMain
//
/**\class CmsShowMain CmsShowMain.h Fireworks/Core/interface/CmsShowMain.h

   Description: Displays an fwlite::Event in ROOT

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Mon Dec  3 08:34:30 PST 2007
// $Id: CmsShowMain.h,v 1.43 2010/01/21 21:01:32 amraktad Exp $
//

// system include files
#include <vector>
#include <string>
#include <memory>
#include <boost/shared_ptr.hpp>
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/DetIdToMatrix.h"

// forward declarations
class TGPictureButton;
class TGComboBox;
class TGTextButton;
class TGTextEntry;
class FWEventItemsManager;
class FWViewManagerManager;
class FWModelChangeManager;
class FWColorManager;
class FWSelectionManager;
class FWGUIManager;
class FWEventItem;
class FWPhysicsObjectDesc;
class FWConfigurationManager;
class TTimer;
class TMonitor;
class TSocket;
class CmsShowNavigator;
class CmsShowTaskExecutor;
class CSGAction;
class CmsShowSearchFiles;

namespace fireworks {
   class Context;
}

namespace fwlite {
   class Event;
}

class CmsShowMain
{
public:
   CmsShowMain(int argc, char *argv[]);
   virtual ~CmsShowMain();
   void resetInitialization();
   void draw();
   void openData();
   void appendData();
   void openDataViaURL();
   void quit();
   void doExit();

   // ---------- const member functions ---------------------
   const DetIdToMatrix& getIdToGeo() const {
      return m_detIdToGeo;
   }

   //  void writeConfigurationFile(const std::string& iFileName) const;
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   //  int draw(const fwlite::Event& );

   void registerPhysicsObject(const FWPhysicsObjectDesc&);

   void notified(TSocket*);
   const fwlite::Event* getCurrentEvent() const;

private:
   CmsShowMain(const CmsShowMain&); // stop default
   const CmsShowMain& operator=(const CmsShowMain&); // stop default

   void loadGeometry();
   void setupViewManagers();
   void setupConfiguration();
   void setupDataHandling();
   void setupDebugSupport();
   void setupSocket(unsigned int);

   void autoLoadNewEvent();

   void doFirstEvent();
   void doPreviousEvent();
   void doNextEvent();
   void doLastEvent();
   void goToRunEvent(int, int);
   void checkPosition();

   void playForward();
   void playBackward();
   void stopPlaying();
   void reachedEnd();
   void reachedBeginning();
   void setPlayLoop();
   void unsetPlayLoop();

   void setPlayLoopImp();
   void unsetPlayLoopImp();

   void navigatorChangedFilterState(int);
   void filterButtonClicked();
   void preFiltering();
   void postFiltering();

   void setPlayDelay(Float_t);
   void checkLiveMode();

   void setLiveMode();

   void startAutoLoadTimer();
   void stopAutoLoadTimer();
   void setupAutoLoad(float);

   // ---------- member data --------------------------------
   std::auto_ptr<FWConfigurationManager> m_configurationManager;
   std::auto_ptr<FWModelChangeManager>   m_changeManager;
   std::auto_ptr<FWColorManager>         m_colorManager;
   std::auto_ptr<FWSelectionManager>     m_selectionManager;
   std::auto_ptr<FWEventItemsManager>    m_eiManager;
   std::auto_ptr<FWViewManagerManager>   m_viewManager;
   std::auto_ptr<FWGUIManager>           m_guiManager;
   std::auto_ptr<fireworks::Context>     m_context;

   CmsShowNavigator* m_navigator;

   DetIdToMatrix            m_detIdToGeo;
   std::vector<std::string> m_inputFiles;
   bool                     m_loadedAnyInputFile;
   std::string              m_configFileName;
   std::string              m_geomFileName;


   std::auto_ptr<CmsShowTaskExecutor> m_startupTasks;
   std::auto_ptr<CmsShowSearchFiles>  m_searchFiles;

   TTimer* m_autoLoadTimer;
   Bool_t  m_autoLoadTimerRunning;

   TTimer* m_liveTimer;
   bool m_live;
   bool m_isPlaying;
   bool m_forward;
   bool m_loop;
   Float_t m_playDelay;  // delay between events in seconds
   Int_t m_lastPointerPositionX;
   Int_t m_lastPointerPositionY;
   Int_t m_liveTimeout;
   std::string m_autoSaveAllViewsFormat;

   std::auto_ptr<TMonitor> m_monitor;
};


#endif

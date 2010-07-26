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
// $Id: CmsShowMain.h,v 1.50 2010/07/23 08:35:03 eulisse Exp $
//

#include "Fireworks/Core/interface/CmsShowMainBase.h"
// user include files
#include "Fireworks/Core/interface/DetIdToMatrix.h"

// system include files
#include <vector>
#include <string>
#include <memory>
#include <boost/shared_ptr.hpp>
#include "Rtypes.h"


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
class FWLiteJobMetadataManager;
class TTimer;
class TMonitor;
class TSocket;
class CmsShowNavigator;
class CmsShowTaskExecutor;
class CSGAction;
class CmsShowSearchFiles;

namespace fwlite {
   class Event;
}

class CmsShowMain : public CmsShowMainBase
{
public:
   CmsShowMain(int argc, char *argv[]);
   virtual ~CmsShowMain();
   void resetInitialization();
   void openData();
   void appendData();
   void openDataViaURL();
   virtual void quit();
   void doExit();

   //  void writeConfigurationFile(const std::string& iFileName) const;
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   //  int draw(const fwlite::Event& );

   void notified(TSocket*);
   const fwlite::Event* getCurrentEvent() const;
   const fireworks::Context* context() const { return m_context.get(); };

   void eventChangedSlot();
   void fileChangedSlot(const TFile *file);
private:
   CmsShowMain(const CmsShowMain&); // stop default
   const CmsShowMain& operator=(const CmsShowMain&); // stop default

   void loadGeometry();
   void setupViewManagers();
   void setupDataHandling();
   void setupSocket(unsigned int);

   virtual void autoLoadNewEvent();
   virtual void checkPosition();
   virtual void stopPlaying();

   void reachedEnd();
   void reachedBeginning();

   // Filtering bits.
   void navigatorChangedFilterState(int);
   void filterButtonClicked();
   void preFiltering();
   void postFiltering();

   // ---------- member data --------------------------------
   std::auto_ptr<CmsShowNavigator>           m_navigator;
   std::auto_ptr<FWLiteJobMetadataManager>   m_metadataManager;
   std::auto_ptr<fireworks::Context>         m_context;

   std::vector<std::string> m_inputFiles;
   bool                     m_loadedAnyInputFile;
   const TFile             *m_openFile;

   std::auto_ptr<CmsShowSearchFiles>  m_searchFiles;
   Bool_t  m_autoLoadTimerRunning;

   Int_t m_liveTimeout;
   std::string m_autoSaveAllViewsFormat;

   std::auto_ptr<TMonitor> m_monitor;
};

#endif

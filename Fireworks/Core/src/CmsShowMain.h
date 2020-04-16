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
//

#include "Fireworks/Core/interface/CmsShowMainBase.h"
// user include files
#include "Fireworks/Core/interface/FWGeometry.h"

// system include files
#include <vector>
#include <string>
#include <memory>
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

class CmsShowMain : public CmsShowMainBase {
public:
  CmsShowMain(int argc, char* argv[]);
  ~CmsShowMain() override;
  void resetInitialization();
  void openData();
  void appendData();
  void openDataViaURL();
  void quit() override;
  void doExit();

  //  void writeConfigurationFile(const std::string& iFileName) const;
  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  //  int draw(const fwlite::Event& );

  void notified(TSocket*);
  const fwlite::Event* getCurrentEvent() const;
  const fireworks::Context* context() const { return m_context.get(); };
  bool getVersionCheck() const { return !m_noVersionCheck; }
  bool getGlobalTagCheck() const { return m_globalTagCheck; }

  void fileChangedSlot(const TFile* file);

protected:
  void eventChangedImp() override;

private:
  CmsShowMain(const CmsShowMain&);                   // stop default
  const CmsShowMain& operator=(const CmsShowMain&);  // stop default

  void loadGeometry();
  void setupDataHandling();
  void setupSocket(unsigned int);
  void connectSocket();
  void setLoadedAnyInputFileAfterStartup();

  void autoLoadNewEvent() override;
  void checkPosition() override;
  void stopPlaying() override;
  void checkKeyBindingsOnPLayEventsStateChanged() override;

  void reachedEnd();
  void reachedBeginning();

  // Filtering bits.
  void navigatorChangedFilterState(int);
  void filterButtonClicked();
  void preFiltering();
  void postFiltering(bool);

  void setLiveMode();
  void checkLiveMode();

  // ---------- member data --------------------------------
  std::unique_ptr<CmsShowNavigator> m_navigator;
  std::unique_ptr<FWLiteJobMetadataManager> m_metadataManager;
  std::unique_ptr<fireworks::Context> m_context;

  std::vector<std::string> m_inputFiles;
  bool m_loadedAnyInputFile;
  const TFile* m_openFile;

  std::unique_ptr<CmsShowSearchFiles> m_searchFiles;

  // live options
  bool m_live;
  std::unique_ptr<SignalTimer> m_liveTimer;
  int m_liveTimeout;
  UInt_t m_lastXEventSerial;

  bool m_noVersionCheck;
  bool m_globalTagCheck;

  std::unique_ptr<TMonitor> m_monitor;
};

#endif

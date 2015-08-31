// -*- C++ -*-
#ifndef Fireworks_Core_FWGUIManager_h
#define Fireworks_Core_FWGUIManager_h
//
// Package:     Core
// Class  :     FWGUIManager
//
/**\class FWGUIManager FWGUIManager.h Fireworks/Core/interface/FWGUIManager.h

   Description: Manages the GUI

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Mon Feb 11 10:52:24 EST 2008
//

// system include files
#include <map>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <sigc++/sigc++.h>
#include "Rtypes.h"
#include "GuiTypes.h"
#include "TGFileDialog.h"
#include <memory>

// user include files
#include "Fireworks/Core/interface/FWConfigurable.h"
#include "DataFormats/Provenance/interface/EventID.h"

// forward declarations
class TGPictureButton;
class TGComboBox;
class TGTextButton;
class TGTextEntry;
class TGFrame;
class TGSplitFrame;
class TGVerticalFrame;
class TGMainFrame;
class TGPack;
class TGCompositeFrame;
class TGCheckButton;
class TGPopupMenu;

class TGListTreeItem;
class TGListTree;
class TEveElementList;
class TEveElement;
class TEveWindowPack;
class TEveWindowSlot;
class TEveCompositeFrame;
class TEveWindow;

class CmsShowMainFrame;
class FWEventItem;
class FWViewBase;
class FWGUISubviewArea;

class FWSelectionManager;
class FWSummaryManager;
class FWDetailViewManager;
class FWInvMassDialog;
class CSGAction;
class CSGContinuousAction;

class FWGUIEventDataAdder;

class CmsShowTaskExecutor;

class FWNavigatorBase;

class FWModelContextMenuHandler;
class FWViewContextMenuHandlerBase;
class TGWindow;

namespace edm {
   class EventBase;
}

class CmsShowEDI;
class CmsShowModelPopup;
class CmsShowViewPopup;
class FWViewManagerManager;
class CmsShowCommonPopup;
class CmsShowHelpPopup;

namespace fireworks {
   class Context;
}

class FWGUIManager : public FWConfigurable
{
   // typedefs
public:
   typedef boost::function2<FWViewBase*,TEveWindowSlot*, const std::string& > ViewBuildFunctor;
   typedef std::map<std::string, ViewBuildFunctor > NameToViewBuilder;
private:
   typedef std::map<TEveWindow*, FWViewBase*> ViewMap_t;
   typedef ViewMap_t::iterator                ViewMap_i;
   
   
public:
   FWGUIManager(fireworks::Context* ctx,
                const FWViewManagerManager* iVMMgr,
                FWNavigatorBase* navigator);

   virtual ~FWGUIManager();
   void     evePreTerminate();
   
   //configuration management interface
   void addTo(FWConfiguration&) const;
   void setFrom(const FWConfiguration&);
   void setWindowInfoFrom(const FWConfiguration& iFrom, TGMainFrame* iFrame);
   void initEmpty();

   TGVerticalFrame* createList(TGCompositeFrame *p);
   void createViews(TEveWindowSlot *slot);
   void exportImageOfMainView();
   void exportImagesOfAllViews();
   void exportAllViews(const std::string& format, int height);

   void createEDIFrame();
   ///Allowed values are -1 or ones from FWDataCategories enum
   void showEDIFrame(int iInfoToShow=-1);
   
   void open3DRegion();
   void showCommonPopup();
   
   void createModelPopup();
   void showModelPopup();
   void showViewPopup();
   void popupViewClosed();
   
   void showSelectedModelContextMenu(Int_t iGlobalX, Int_t iGlobalY, FWViewContextMenuHandlerBase* iHandler);

   void showInvMassDialog();
   //   void showGeometryBrowser();

   // help
   void createHelpPopup ();
   void createShortcutPopup ();
   void createHelpGLPopup ();
   
   // ---------- const member functions ---------------------
   //      bool waitingForUserAction() const;
   CSGContinuousAction* playEventsAction();
   CSGContinuousAction* playEventsBackwardsAction();
   CSGContinuousAction* loopAction();
   
   // ---------- static member functions --------------------
   static FWGUIManager* getGUIManager();
   static  TGFrame* makeGUIsubview(TEveCompositeFrame* cp, TGCompositeFrame* parent, Int_t height);
   
   // ---------- member functions ---------------------------
   //have to use the portable syntax else the reflex code will not build
   void registerViewBuilder(const std::string& iName,
                            ViewBuildFunctor& iBuilder);
   
   
   ViewMap_i createView(const std::string& iName, TEveWindowSlot* slot = 0);
   void newViewSlot(const std::string& iName);
   
   void connectSubviewAreaSignals(FWGUISubviewArea*);
   void enableActions(bool enable = true);
   void disablePrevious();
   void disableNext();
   void setPlayMode(bool);
   void updateStatus(const char* status);
   void clearStatus();
   void eventChangedCallback();
   
   CSGAction* getAction(const std::string name);
   
   void addData();
   void titleChanged(const char *title);
   
   void openEveBrowserForDebugging() const;
   void setDelayBetweenEvents(Float_t);
   
   void showEventFilterGUI();
   void filterButtonClicked();
   void setFilterButtonText(const char* txt);
   void setFilterButtonIcon(int);
   void updateEventFilterEnable(bool);
   
   void runIdChanged();
   void lumiIdChanged();
   void eventIdChanged();
   void checkSubviewAreaIconState(TEveWindow*);
   void subviewIsBeingDestroyed(FWGUISubviewArea*);
   void subviewDestroy(FWGUISubviewArea*); // timeout funct
   void subviewDestroyAll();
   void subviewInfoSelected(FWGUISubviewArea*);
   void subviewInfoUnselected(FWGUISubviewArea*);
   void subviewSwapped(FWGUISubviewArea*);
   
   CmsShowMainFrame* getMainFrame() const { return m_cmsShowMainFrame; }
   const edm::EventBase* getCurrentEvent() const;

   void resetWMOffsets();

   // signals
   sigc::signal<void> filterButtonClicked_;
   sigc::signal<void, const TGWindow*> showEventFilterGUI_;
   sigc::signal<void, const std::string&> writeToConfigurationFile_;
   sigc::signal<void, const std::string&> writePartialToConfigurationFile_;
   sigc::signal<void, const std::string&> loadFromConfigurationFile_;
   sigc::signal<void, const std::string&> loadPartialFromConfigurationFile_;
   sigc::signal<void, edm::RunNumber_t, edm::LuminosityBlockNumber_t, edm::EventNumber_t> changedEventId_;
   sigc::signal<void> goingToQuit_;
   sigc::signal<void> writeToPresentConfigurationFile_;
   
   sigc::signal<void> changedRunEntry_;
   sigc::signal<void> changedEventEntry_;
   sigc::signal<void, Float_t> changedDelayBetweenEvents_;
   
private:
   FWGUIManager(const FWGUIManager&);    // stop default
   const FWGUIManager& operator=(const FWGUIManager&);    // stop default
      
   TEveWindow* getSwapCandidate();
   
   void newItem(const FWEventItem*);

   bool promptForConfigurationFile(std::string &result, enum EFileDialogMode mode);
   void promptForSaveConfigurationFile();
   void promptForPartialSaveConfigurationFile();
   void promptForLoadConfigurationFile();
   void promptForPartialLoadConfigurationFile();
   void savePartialToConfigurationFile();
   
   void delaySliderChanged(Int_t);
   
   void finishUpColorChange();
   
   void setViewPopup(TEveWindow*);

   void measureWMOffsets();
   
   // ---------- static member data --------------------------------   
   
   static FWGUIManager*  m_guiManager;
   
   // ---------- member data --------------------------------   
   fireworks::Context*   m_context;

   FWSummaryManager*     m_summaryManager;
   
   //views are owned by their individual view managers
   FWDetailViewManager*        m_detailViewManager;
   const FWViewManagerManager* m_viewManagerManager;
   FWModelContextMenuHandler*  m_contextMenuHandler;
   FWNavigatorBase            *m_navigator;
   CmsShowMainFrame*           m_cmsShowMainFrame;
   
   TGPopupMenu* m_fileMenu;
   FWGUIEventDataAdder* m_dataAdder;
   
   // event data inspector
   CmsShowEDI*             m_ediFrame;
   CmsShowModelPopup*      m_modelPopup;
   CmsShowViewPopup*       m_viewPopup;
   CmsShowCommonPopup*     m_commonPopup;
   FWInvMassDialog*        m_invMassDialog;

   // help
   CmsShowHelpPopup *m_helpPopup;
   CmsShowHelpPopup  *m_shortcutPopup;
   CmsShowHelpPopup  *m_helpGLPopup;
   
   // subview memebers
   mutable ViewMap_t m_viewMap;
   NameToViewBuilder m_nameToViewBuilder;
   
   TEveWindowPack    *m_viewPrimPack;
   TEveWindowPack    *m_viewSecPack;

   sigc::connection   m_modelChangeConn;

   std::auto_ptr<CmsShowTaskExecutor> m_tasks;

   int m_WMOffsetX, m_WMOffsetY, m_WMDecorH;
};


#endif

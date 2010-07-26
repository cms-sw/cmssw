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
// $Id: FWGUIManager.h,v 1.106 2010/07/16 14:34:18 eulisse Exp $
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

// forward declarations
class TGPictureButton;
class TGComboBox;
class TGTextButton;
class TGTextEntry;
class TGFrame;
class TGSplitFrame;
class TGVerticalFrame;
class TGMainFrame;
class TGTab;
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
class FWSelectionManager;
class FWEventItemsManager;
class FWEventItem;
class FWViewBase;
class FWGUISubviewArea;

class FWSummaryManager;
class FWDetailViewManager;
class FWModelChangeManager;

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
class FWColorManager;
class CmsShowBrightnessPopup;
class CmsShowHelpPopup;
class FWJobMetadataManager;

class FWGUIManager : public FWConfigurable
{
   // typedefs
public:
   typedef boost::function1<FWViewBase*,TEveWindowSlot*> ViewBuildFunctor;
   typedef std::map<std::string, ViewBuildFunctor > NameToViewBuilder;
private:
   typedef std::map<TEveWindow*, FWViewBase*> ViewMap_t;
   typedef ViewMap_t::iterator                ViewMap_i;
   
   
public:
   FWGUIManager(FWSelectionManager*,
                FWEventItemsManager*,
                FWModelChangeManager*,
                FWColorManager*,
                const FWViewManagerManager*,
                FWJobMetadataManager*,
		          FWNavigatorBase *,
                bool iDebugInterface = false);
   virtual ~FWGUIManager();
   void     evePreTerminate();
   
   //configuration management interface
   void addTo(FWConfiguration&) const;
   void setFrom(const FWConfiguration&);
   void setWindowInfoFrom(const FWConfiguration& iFrom, TGMainFrame* iFrame);

   TGVerticalFrame* createList(TGSplitFrame *p);
   void createViews(TGTab *p);
   void exportImageOfMainView();
   void exportImagesOfAllViews();
   void exportAllViews(const std::string& format);

   void createEDIFrame();
   ///Allowed values are -1 or ones from FWDataCategories enum
   void showEDIFrame(int iInfoToShow=-1);
   
   void showBrightnessPopup();
   
   void createModelPopup();
   void showModelPopup();
   void showViewPopup();
   void popupViewClosed();
   
   void showSelectedModelContextMenu(Int_t iGlobalX, Int_t iGlobalY, FWViewContextMenuHandlerBase* iHandler);
   
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
   void loadEvent();
   
   CSGAction* getAction(const std::string name);
   
   void addData();
   void titleChanged(const char *title);
   
   void processGUIEvents();
   void openEveBrowserForDebugging() const;
   void setDelayBetweenEvents(Float_t);
   
   void showEventFilterGUI();
   void filterButtonClicked();
   void setFilterButtonText(const char* txt);
   void setFilterButtonIcon(int);
   void updateEventFilterEnable(bool);
   
   void runIdChanged();
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
   
   // signals
   sigc::signal<void> filterButtonClicked_;
   sigc::signal<void, const TGWindow*> showEventFilterGUI_;
   sigc::signal<void, const std::string&> writeToConfigurationFile_;
   sigc::signal<void, const std::string&> loadFromConfigurationFile_;
   sigc::signal<void, int, int> changedEventId_;
   sigc::signal<void> goingToQuit_;
   sigc::signal<void> writeToPresentConfigurationFile_;
   
   sigc::signal<void> changedRunEntry_;
   sigc::signal<void> changedEventEntry_;
   sigc::signal<void, Float_t> changedDelayBetweenEvents_;
   
private:
   FWGUIManager(const FWGUIManager&);    // stop default
   const FWGUIManager& operator=(const FWGUIManager&);    // stop default
   
   void selectionChanged(const FWSelectionManager&);
   
   TEveWindow* getSwapCandidate();
   
   void newItem(const FWEventItem*);

   bool promptForConfigurationFile(std::string &result, enum EFileDialogMode mode);
   void promptForSaveConfigurationFile();
   void promptForLoadConfigurationFile();
   
   void delaySliderChanged(Int_t);
   
   void finishUpColorChange();
   
   void setViewPopup(TEveWindow*);

   void measureWMOffsets();
   
   // ---------- static member data --------------------------------   
   
   static FWGUIManager*  m_guiManager;
   
   // ---------- member data --------------------------------   
   FWSelectionManager*   m_selectionManager;
   FWSummaryManager*     m_summaryManager;
   FWEventItemsManager*  m_eiManager;
   FWModelChangeManager* m_changeManager;
   FWColorManager*       m_colorManager;
   
   //views are owned by their individual view managers
   FWDetailViewManager*        m_detailViewManager;
   const FWViewManagerManager* m_viewManagerManager;
   FWJobMetadataManager *      m_metadataManager;
   FWModelContextMenuHandler*  m_contextMenuHandler;
   FWNavigatorBase            *m_navigator;
   CmsShowMainFrame*     m_cmsShowMainFrame;
   
   TGPopupMenu* m_fileMenu;
   FWGUIEventDataAdder* m_dataAdder;
   
   // event data inspector
   CmsShowEDI*             m_ediFrame;
   CmsShowModelPopup*      m_modelPopup;
   CmsShowViewPopup*       m_viewPopup;
   CmsShowBrightnessPopup* m_brightnessPopup;
   
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

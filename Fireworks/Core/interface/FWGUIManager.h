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
// $Id: FWGUIManager.h,v 1.56 2009/04/07 14:01:34 chrjones Exp $
//

// system include files
#include <map>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <sigc++/sigc++.h>
#include "Rtypes.h"
#include "GuiTypes.h"
#include <memory>

// user include files
#include "Fireworks/Core/interface/FWConfigurable.h"
#include "DataFormats/FWLite/interface/Event.h"

// forward declarations
class TGPictureButton;
class TGComboBox;
class TGTextButton;
class TGTextEntry;
class FWSelectionManager;
class TGFrame;
class TGSplitFrame;
class TGVerticalFrame;
class CmsShowMainFrame;
class TGMainFrame;
class TGTab;
class TGCompositeFrame;
class TGCheckButton;
class FWGUISubviewArea;

class FWEventItemsManager;
class FWEventItem;
class FWListEventItem;
class FWViewBase;

class FWListModel;

class TGListTreeItem;
class TGListTree;
//class TEveGedEditor;
class TEveElementList;
class TEveElement;
class TEveWindowPack;
class TEveWindowSlot;
class TEveCompositeFrame;
class TEveWindow;

class FWSummaryManager;
class FWDetailViewManager;
class FWModelChangeManager;

class TGPopupMenu;
class CSGAction;
class CSGContinuousAction;

class TFile;

class FWGUIEventDataAdder;

class CmsShowTaskExecutor;

namespace fwlite {
   class Event;
}

class CmsShowEDI;
class CmsShowModelPopup;
class CmsShowViewPopup;
class FWViewManagerManager;
class FWColorManager;
class CmsShowHelpPopup;

class FWGUIManager : public FWConfigurable
{

public:
   FWGUIManager(FWSelectionManager*,
                FWEventItemsManager*,
                FWModelChangeManager*,
                FWColorManager*,
                const FWViewManagerManager*,
                bool iDebugInterface = false);
   virtual ~FWGUIManager();
   void     evePreTerminate();

   //configuration management interface
   void addTo(FWConfiguration&) const;
   void setFrom(const FWConfiguration&);

   TGVerticalFrame* createList(TGSplitFrame *p);
   void createViews(TGTab *p);
   void createTextView(TGTab *p);

   void createEDIFrame();
   void updateEDI(FWEventItem* iItem);
   void resetEDIFrame();
   ///Allowed values are -1 or ones from FWDataCategories enum
   void showEDIFrame(int iInfoToShow=-1);

   void createModelPopup();
   void updateModel(FWEventItem* iItem);
   void resetModelPopup();
   void showModelPopup();

   void createViewPopup();
   void refillViewPopup(FWViewBase* iView);
   void resetViewPopup();
   void showViewPopup();

   // help
   void createHelpPopup ();
   void resetHelpPopup ();
   void createShortcutPopup ();
   void resetShortcutPopup ();

   // ---------- const member functions ---------------------
   //      bool waitingForUserAction() const;
   CSGContinuousAction* playEventsAction();
   CSGContinuousAction* playEventsBackwardsAction();

   // ---------- static member functions --------------------
   static FWGUIManager* getGUIManager();

   static FWGUISubviewArea* getGUISubviewArea(TEveWindow*);

   // ---------- member functions ---------------------------
   TEveWindowSlot* parentForNextView();

   //have to use the portable syntax else the reflex code will not build
   typedef boost::function1<FWViewBase*,TEveWindowSlot*> ViewBuildFunctor;
   void registerViewBuilder(const std::string& iName,
                            ViewBuildFunctor& iBuilder);

   void createView(const std::string& iName);

   void enableActions(bool enable = true);
   void disablePrevious();
   void disableNext();
   void setPlayMode(bool);
   void updateStatus(const char* status);
   void clearStatus();
   void loadEvent(const fwlite::Event& event);
   void newFile(const TFile*);

   CSGAction* getAction(const std::string name);

   void addData();
   void unselectAll();
   void selectByExpression();

   void processGUIEvents();

   void itemChecked(TObject* obj, Bool_t state);
   void itemClicked(TGListTreeItem *entry, Int_t btn,  UInt_t mask, Int_t x, Int_t y);
   void itemDblClicked(TGListTreeItem* item, Int_t btn);
   void itemKeyPress(TGListTreeItem *entry, UInt_t keysym, UInt_t mask);
   void itemBelowMouse(TGListTreeItem*, UInt_t);

   sigc::signal<void, const std::string&> writeToConfigurationFile_;
   sigc::signal<void, const std::string&> changedEventFilter_;
   sigc::signal<void, int> changedEventId_;
   sigc::signal<void, int> changedRunId_;
   sigc::signal<void> goingToQuit_;
   sigc::signal<void> writeToPresentConfigurationFile_;

   sigc::signal<void> changedRunEntry_;
   sigc::signal<void> changedEventEntry_;
   sigc::signal<void> changedFileterEntry_;

   sigc::signal<void, Float_t> changedDelayBetweenEvents_;

   void openEveBrowserForDebugging() const;
   void setDelayBetweenEvents(Float_t);

   void eventFilterChanged();
   void runIdChanged();
   void eventIdChanged();
   void subviewCurrentChanged(TEveWindow*);
   void subviewIsBeingDestroyed(FWGUISubviewArea*);
   void subviewDestroy(FWGUISubviewArea*); // timeout funct
   void subviewSelected(FWGUISubviewArea*);
   void subviewUnselected(FWGUISubviewArea*);
   void subviewSwapWithCurrent(FWGUISubviewArea*);

   static  TGFrame* makeGUIsubview(TEveCompositeFrame* cp, TGCompositeFrame* parent, Int_t height);

private:
   FWGUIManager(const FWGUIManager&);    // stop default

   const FWGUIManager& operator=(const FWGUIManager&);    // stop default

   void selectionChanged(const FWSelectionManager&);

   void newItem(const FWEventItem*);

   void exportImageOfMainView();
   void promptForConfigurationFile();

   void delaySliderChanged(Int_t);

   // ---------- member data --------------------------------

   static FWGUIManager* m_guiManager;
   FWSelectionManager* m_selectionManager;
   FWEventItemsManager* m_eiManager;
   FWModelChangeManager* m_changeManager;
   FWColorManager* m_colorManager;
   const fwlite::Event* m_presentEvent;
   mutable bool m_continueProcessingEvents;
   mutable bool m_waitForUserAction;
   mutable int m_code;     // respond code for the control loop
   //  1 - move forward
   // -1 - move backward
   //  0 - do nothing
   // -2 - start over
   // -3 - stop event loop


   TGPictureButton* m_homeButton;
   TGPictureButton* m_advanceButton;
   TGPictureButton* m_backwardButton;
   TGPictureButton* m_stopButton;

   TGComboBox* m_selectionItemsComboBox;
   TGTextEntry* m_selectionExpressionEntry;
   TGTextButton* m_selectionRunExpressionButton;
   TGTextButton* m_unselectAllButton;

   TGPopupMenu* m_fileMenu;

   CmsShowMainFrame* m_cmsShowMainFrame;
   TGMainFrame* m_mainFrame;
   TGSplitFrame* m_splitFrame;
   std::vector<TEveWindow*> m_viewWindows;

   typedef std::map<std::string, ViewBuildFunctor > NameToViewBuilder;
   NameToViewBuilder m_nameToViewBuilder;

   TEveElement* m_editableSelected;

   FWSummaryManager* m_summaryManager;

   //views are owned by their individual view managers
   std::vector<FWViewBase*> m_viewBases;

   FWDetailViewManager* m_detailViewManager;
   const FWViewManagerManager* m_viewManagerManager;

   const TFile* m_openFile;
   FWGUIEventDataAdder* m_dataAdder;

   // event data inspector
   CmsShowEDI* m_ediFrame;
   CmsShowModelPopup* m_modelPopup;
   CmsShowViewPopup* m_viewPopup;

   // help
   CmsShowHelpPopup *m_helpPopup, *m_shortcutPopup;

   TGTab             *m_textViewTab;
   TGCompositeFrame  *m_textViewFrame[3];
   TEveWindowPack    *m_viewPrimPack;
   TEveWindowPack    *m_viewSecPack;
   sigc::connection m_modelChangeConn;

   std::auto_ptr<CmsShowTaskExecutor> m_tasks;
};


#endif

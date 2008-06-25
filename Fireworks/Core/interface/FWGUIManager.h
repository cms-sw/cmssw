#ifndef Fireworks_Core_FWGUIManager_h
#define Fireworks_Core_FWGUIManager_h
// -*- C++ -*-
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
// $Id: FWGUIManager.h,v 1.26 2008/06/23 15:51:11 chrjones Exp $
//

// system include files
#include <map>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <sigc++/signal.h>
#include "Rtypes.h"

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
class FWGUISubviewArea;

class FWEventItemsManager;
class FWEventItem;
class FWViewBase;

class TGListTreeItem;
class TGListTree;
class TEveGedEditor;
class TEveElementList;
class TEveElement;

class FWSummaryManager;
class FWDetailViewManager;
class FWDetailView;
class FWModelChangeManager;

class  TGPopupMenu;
class CSGAction;

class FWGUIEventDataAdder;

class FWGUIManager : public FWConfigurable
{

   public:
      FWGUIManager(FWSelectionManager*,
                   FWEventItemsManager*,
                   FWModelChangeManager*,
                   bool iDebugInterface = false);
      virtual ~FWGUIManager();

      //configuration management interface
      void addTo(FWConfiguration&) const;
      void setFrom(const FWConfiguration&);
   
      TGVerticalFrame* createList(TGSplitFrame *p);
      TGMainFrame* createViews(TGCompositeFrame *p);
      TGMainFrame* createTextView(TGTab *p);
      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void addFrameHoldingAView(TGFrame*);
      TGFrame* parentForNextView();
   
      //have to use the portable syntax else the reflex code will not build
      typedef boost::function1<FWViewBase*,TGFrame*> ViewBuildFunctor;
      void registerViewBuilder(const std::string& iName, 
                              ViewBuildFunctor& iBuilder);
   
      void registerDetailView (const std::string &iItemName, 
                               FWDetailView *iView);
      void createView(const std::string& iName);

      void enableActions(bool enable = true);
      void disablePrevious();
      void disableNext();
      void loadEvent(int i);

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
      sigc::signal<void> goingToQuit_;
      sigc::signal<void> writeToPresentConfigurationFile_;
   
      void openEveBrowserForDebugging() const;
   private:
      FWGUIManager(const FWGUIManager&); // stop default

      const FWGUIManager& operator=(const FWGUIManager&); // stop default

      void selectionChanged(const FWSelectionManager&);

      void newItem(const FWEventItem*);

      void subviewWasSwappedToBig(unsigned int);
      void subviewIsBeingDestroyed(unsigned int);
   
      void exportImageOfMainView();
      void promptForConfigurationFile();

      // ---------- member data --------------------------------

      FWSelectionManager* m_selectionManager;
      FWEventItemsManager* m_eiManager;
      FWModelChangeManager* m_changeManager;
      mutable bool m_continueProcessingEvents;
      mutable bool m_waitForUserAction;
      mutable int  m_code; // respond code for the control loop
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
      std::vector<FWGUISubviewArea*> m_viewFrames;
      
      typedef std::map<std::string, ViewBuildFunctor > NameToViewBuilder;
      NameToViewBuilder m_nameToViewBuilder;
  
      TGListTree* m_listTree;
      TEveGedEditor* m_editor;
      TEveElementList* m_views;
   
      TEveElement* m_editableSelected;

      FWSummaryManager* m_summaryManager;

      //views are owned by their individual view managers
      std::vector<FWViewBase* > m_viewBases;

      FWDetailViewManager* m_detailViewManager;
   
      FWGUIEventDataAdder* m_dataAdder;

      TGTab		*m_textViewTab;
      TGCompositeFrame	*m_textViewFrame[3];
};


#endif

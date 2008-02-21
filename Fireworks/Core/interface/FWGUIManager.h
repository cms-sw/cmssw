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
// $Id: FWGUIManager.h,v 1.4 2008/02/21 16:09:54 chrjones Exp $
//

// system include files
#include <map>
#include <boost/function.hpp>

// user include files

// forward declarations
class TGPictureButton;
class TGComboBox;
class TGTextButton;
class TGTextEntry;
class FWSelectionManager;
class TGFrame;
class TGSplitFrame;
class TGMainFrame;
class TGCompositeFrame;

class FWEventItemsManager;
class FWEventItem;
class FWViewBase;

class FWGUIManager
{

   public:
      FWGUIManager(FWSelectionManager*,
                   FWEventItemsManager*);
      virtual ~FWGUIManager();

      // ---------- const member functions ---------------------
      bool waitingForUserAction() const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void addFrameHoldingAView(TGFrame*);
      TGFrame* parentForNextView();
   
      //have to use the portable syntax else the reflex code will not build
      typedef boost::function1<FWViewBase*,TGFrame*> ViewBuildFunctor;
      void registerViewBuilder(const std::string& iName, 
                              ViewBuildFunctor& iBuilder);
   
      void createView(const std::string& iName);
   
      void goForward();
      void goBack();
      void goHome();
      void stop();
      void waitForUserAction();
      void doNotWaitForUserAction();

      void unselectAll();
      void selectByExpression();

      void processGUIEvents();
      int allowInteraction();

   private:
      FWGUIManager(const FWGUIManager&); // stop default

      const FWGUIManager& operator=(const FWGUIManager&); // stop default

      void selectionChanged(const FWSelectionManager&);

      void newItem(const FWEventItem*);

      // ---------- member data --------------------------------

      FWSelectionManager* m_selectionManager;
      FWEventItemsManager* m_eiManager;
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
   
      TGMainFrame* m_mainFrame;
      TGSplitFrame* m_splitFrame;
      std::vector<TGCompositeFrame*> m_viewFrames;
      std::vector<TGCompositeFrame*>::iterator m_nextFrame;
      
      typedef std::map<std::string, ViewBuildFunctor > NameToViewBuilder;
      NameToViewBuilder m_nameToViewBuilder;
};


#endif

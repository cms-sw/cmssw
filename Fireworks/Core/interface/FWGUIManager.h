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
// $Id$
//

// system include files

// user include files

// forward declarations
class TGPictureButton;
class TGComboBox;
class TGTextButton;
class TGTextEntry;
class FWSelectionManager;
class TGFrame;

class FWEventItemsManager;
class FWEventItem;

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
   
};


#endif

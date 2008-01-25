#ifndef Fireworks_Core_FWDisplayEvent_h
#define Fireworks_Core_FWDisplayEvent_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDisplayEvent
// 
/**\class FWDisplayEvent FWDisplayEvent.h Fireworks/Core/interface/FWDisplayEvent.h

 Description: Displays an fwlite::Event in ROOT

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon Dec  3 08:34:30 PST 2007
// $Id: FWDisplayEvent.h,v 1.13 2008/01/24 00:32:32 chrjones Exp $
//

// system include files
#include <vector>
#include <string>
#include <memory>
#include <boost/shared_ptr.hpp>
#include "Rtypes.h"

// user include files
#include "DetIdToMatrix.h"

// forward declarations
class TGPictureButton;
class TGComboBox;
class TGTextButton;
class TGTextEntry;
class FWEventItemsManager;
class FWViewManagerManager;
class FWModelChangeManager;
class FWSelectionManager;
class FWEventItem;
class FWPhysicsObjectDesc;

namespace fwlite {
  class Event;
}


class FWDisplayEvent
{

   public:
      FWDisplayEvent();
      virtual ~FWDisplayEvent();

      // ---------- const member functions ---------------------
      int draw(const fwlite::Event& ) const;

      bool waitingForUserAction() const;
      const DetIdToMatrix& getIdToGeo() const { return m_detIdToGeo; }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void goForward();
      void goBack();
      void goHome();
      void stop();
      void waitForUserAction();
      void doNotWaitForUserAction();
      int draw(const fwlite::Event& );
      void registerProxyBuilder(const std::string&, 
				const std::string&);
      
      void registerPhysicsObject(const FWPhysicsObjectDesc&);
      void unselectAll();
      void selectByExpression();
   
   private:
      FWDisplayEvent(const FWDisplayEvent&); // stop default

      const FWDisplayEvent& operator=(const FWDisplayEvent&); // stop default

      void selectionChanged(const FWSelectionManager&);
      // ---------- member data --------------------------------
      std::auto_ptr<FWModelChangeManager> m_changeManager;
      std::auto_ptr<FWSelectionManager> m_selectionManager;
      std::auto_ptr<FWEventItemsManager> m_eiManager;
      std::auto_ptr<FWViewManagerManager> m_viewManager;

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
   
      DetIdToMatrix    m_detIdToGeo;
};


#endif

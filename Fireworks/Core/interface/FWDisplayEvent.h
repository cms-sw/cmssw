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
// $Id: FWDisplayEvent.h,v 1.7 2008/01/07 05:48:45 chrjones Exp $
//

// system include files
#include <vector>
#include <string>
#include <memory>
#include <boost/shared_ptr.hpp>
#include "Rtypes.h"
// user include files

// forward declarations
class TGPictureButton;
class FWViewManagerBase;
class FWEventItemsManager;
class FWEventItem;

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
      
      void registerEventItem(const FWEventItem&);
   private:
      FWDisplayEvent(const FWDisplayEvent&); // stop default

      const FWDisplayEvent& operator=(const FWDisplayEvent&); // stop default

      // ---------- member data --------------------------------
      std::auto_ptr<FWEventItemsManager> m_eiManager;
      std::vector<boost::shared_ptr<FWViewManagerBase> > m_viewManagers;

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
   
};


#endif

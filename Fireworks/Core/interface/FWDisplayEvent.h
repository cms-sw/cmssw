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
// $Id: FWDisplayEvent.h,v 1.1.1.1 2007/12/06 01:40:00 chrjones Exp $
//

// system include files
#include <vector>
#include <string>

// user include files

// forward declarations
class TEveTrackList;
class TEveProjectionManager;
class TEveElement;
class TEveElementList;
class TGPictureButton;

namespace fwlite {
  class Event;
}

class FWDisplayEvent
{

   public:
      FWDisplayEvent();
      virtual ~FWDisplayEvent();

      // ---------- const member functions ---------------------
      void draw(const fwlite::Event& ) const;

      bool waitingForUserAction() const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void continueProcessingEvents();
      void waitForUserAction();
      void doNotWaitForUserAction();

   private:
      FWDisplayEvent(const FWDisplayEvent&); // stop default

      const FWDisplayEvent& operator=(const FWDisplayEvent&); // stop default

      // ---------- member data --------------------------------
      mutable std::vector<std::string> m_physicsTypes;
      mutable std::vector<TEveElementList*> m_physicsElements;
      //mutable TEveTrackList* m_tracks;

      mutable bool m_continueProcessingEvents;
      mutable bool m_waitForUserAction;

      TEveElement* m_geom;
      TEveProjectionManager* m_rhoPhiProjMgr;

      TGPictureButton* m_advanceButton;
};


#endif

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
// $Id: FWDisplayEvent.h,v 1.4 2007/12/15 21:14:31 dmytro Exp $
//

// system include files
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
// user include files

// forward declarations
class TEveTrackList;
class TEveProjectionManager;
class TEveElement;
class TEveElementList;
class TGPictureButton;
class FWDataProxyBuilder;
class TObject;
class TCanvas;

namespace fwlite {
  class Event;
}

struct FWModelProxy
{
   std::string                             type;
   std::string                             builderName;
   boost::shared_ptr<FWDataProxyBuilder>   builder;
   TObject*                                product; //owned by builder
   FWModelProxy():product(0){}
};


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
      void registerProxyBuilder(std::string, std::string);

   private:
      FWDisplayEvent(const FWDisplayEvent&); // stop default

      const FWDisplayEvent& operator=(const FWDisplayEvent&); // stop default

      // ---------- member data --------------------------------
      mutable std::vector<FWModelProxy> m_modelProxies;

      mutable bool m_continueProcessingEvents;
      mutable bool m_waitForUserAction;
      mutable int  m_code; // respond code for the control loop
      //  1 - move forward
      // -1 - move backward
      //  0 - do nothing
      // -2 - start over
      // -3 - stop event loop 
      
      mutable TEveElement* m_geom;
      TEveProjectionManager* m_rhoPhiProjMgr;

      TGPictureButton* m_homeButton;
      TGPictureButton* m_advanceButton;
      TGPictureButton* m_backwardButton;
      TGPictureButton* m_stopButton;
   
      // stuff for lego plot view
      mutable TCanvas* m_legoCanvas;
};


#endif

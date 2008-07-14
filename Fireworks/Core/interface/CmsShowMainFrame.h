#ifndef Fireworks_Core_CmsShowsMainFrame_h
#define Fireworks_Core_CmsShowMainFrame_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowMainFrame
// 
/**\class CmsShowMainFrame CmsShowMainFrame.h Fireworks/Core/interface/CmsShowMainFrame.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu May 29 18:11:16 CDT 2008
// $Id: CmsShowMainFrame.h,v 1.4 2008/07/08 02:43:46 dmytro Exp $
//

// system include files
#include <TQObject.h>
#include <RQ_OBJECT.h>
#include <sigc++/sigc++.h>
#include <vector>
#include <TGFrame.h>

// user include files

// forward declarations
class TGWindow;
class TGTextButton;
class TGPictureButton;
class TGPopupMenu;
class TTimer;
class CSGAction;
class CSGNumAction;
class FWGUIManager;
class TGPopupMenu;
class TGTextEntry;

namespace fwlite {
  class Event;
}

class CmsShowMainFrame : public TGMainFrame, public sigc::trackable {

public:
   CmsShowMainFrame(const TGWindow *p = 0,UInt_t w = 1,UInt_t h = 1,FWGUIManager *m = 0);
   virtual ~CmsShowMainFrame();
   
   // ---------- const member functions ---------------------
   const std::vector<CSGAction*>& getListOfActions() const;
   CSGNumAction* getRunEntry() const;
   CSGNumAction* getEventEntry() const;
   Long_t getDelay() const;
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void addToActionMap(CSGAction *action);
   Bool_t activateTextButton(TGTextButton *button);
   Bool_t activatePictureButton(TGPictureButton *button);
   Bool_t activateMenuEntry(int entry);
   Bool_t activateToolBarEntry(int entry);
   void defaultAction();
   void loadEvent(const fwlite::Event& event);
   void goForward();
   void goBackward();
   void goToFirst();
   void playEvents();
   void playEventsBack();
   void pause();
   void quit();
   CSGAction* getAction(const std::string& name);
   void enableActions(bool enable = true);
   void enablePrevious(bool enable = true);
   void enableNext(bool enable = true);
   bool previousIsEnabled();
   bool nextIsEnabled();
   void resizeMenu(TGPopupMenu *menu);
   void HandleMenu(Int_t id);
   Bool_t HandleKey(Event_t *event);
   //   Bool_t HandleTimer(TTimer *timer);
   
   CSGAction* createNewViewerAction(const std::string& iActionName);
private:
   CmsShowMainFrame(const CmsShowMainFrame&); // stop default
   
   const CmsShowMainFrame& operator=(const CmsShowMainFrame&); // stop default
   
   // ---------- member data --------------------------------
   std::vector<CSGAction*> m_actionList;
   FWGUIManager *m_manager;
   Long_t m_delay;
   CSGNumAction *m_runEntry;
   CSGNumAction *m_eventEntry;
   CSGAction *m_nextEvent;
   CSGAction *m_previousEvent;
   CSGAction *m_goToFirst;
   Int_t m_playRate;
   Int_t m_playBackRate;
   TTimer *m_playTimer;
   TTimer *m_playBackTimer;   
   
   TGPopupMenu *m_newViewerMenu;
};


#endif

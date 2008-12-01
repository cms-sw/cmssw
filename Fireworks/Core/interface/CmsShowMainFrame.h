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
// $Id: CmsShowMainFrame.h,v 1.14 2008/11/28 22:26:05 amraktad Exp $
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
class TGStatusBar;
class TTimer;
class CSGAction;
class CSGContinuousAction;
class FWGUIManager;
class TGPopupMenu;
class TGTextEntry;
class TGLabel;

namespace fwlite {
  class Event;
}

class CmsShowMainFrame : public TGMainFrame, public sigc::trackable {

public:
   CmsShowMainFrame(const TGWindow *p = 0,UInt_t w = 1,UInt_t h = 1,FWGUIManager *m = 0);
   virtual ~CmsShowMainFrame();

   // ---------- const member functions ---------------------
   const std::vector<CSGAction*>& getListOfActions() const;
   CSGAction* getRunEntry() const;
   CSGAction* getEventEntry() const;
   //delay for tooltips
   Long_t getDelay() const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void addToActionMap(CSGAction *action);
   Bool_t activateMenuEntry(int entry);
   Bool_t activateToolBarEntry(int entry);
   void defaultAction();
   void loadEvent(const fwlite::Event& event);
   void quit();
   CSGAction* getAction(const std::string& name);
   void enableActions(bool enable = true);
   void enablePrevious(bool enable = true);
   void enableNext(bool enable = true);
   bool previousIsEnabled();
   bool nextIsEnabled();
   void updateStatusBar(const char* status);
   void clearStatusBar();
   void resizeMenu(TGPopupMenu *menu);
   void HandleMenu(Int_t id);
   Bool_t HandleKey(Event_t *event);
   CSGContinuousAction* playEventsAction() const { return m_playEvents;}
   CSGContinuousAction* playEventsBackwardsAction() const { return m_playEventsBack;}

   CSGAction* createNewViewerAction(const std::string& iActionName);
private:
   CmsShowMainFrame(const CmsShowMainFrame&); // stop default

   const CmsShowMainFrame& operator=(const CmsShowMainFrame&); // stop default

  void makeFixedSizeLabel(TGHorizontalFrame* p, const char* txt, UInt_t bgCol, UInt_t txtCol);
   // ---------- member data --------------------------------
   std::vector<CSGAction*> m_actionList;
   FWGUIManager *m_manager;
   Long_t m_delay;
   CSGAction *m_runEntry;
   CSGAction *m_eventEntry;
   CSGAction *m_delaySlider;
   TGLabel* m_timeText;
   CSGAction *m_nextEvent;
   CSGAction *m_previousEvent;
   CSGAction *m_goToFirst;
   CSGAction *m_goToLast;
   CSGContinuousAction *m_playEvents;
   CSGContinuousAction *m_playEventsBack;
   TGStatusBar* m_statBar;

   TGPopupMenu *m_newViewerMenu;
};


#endif

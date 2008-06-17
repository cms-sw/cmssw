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
// $Id$
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
class TGNumberEntryField;
class TTimer;
class CSGAction;
class FWGUIManager;

class CmsShowMainFrame : public TGMainFrame, public sigc::trackable {
   RQ_OBJECT("CmsShowMainFrame")

public:
   CmsShowMainFrame(const TGWindow *p = 0,UInt_t w = 1,UInt_t h = 1,FWGUIManager *m = 0);
   virtual ~CmsShowMainFrame();
   
   // ---------- const member functions ---------------------
   const std::vector<CSGAction*>& getListOfActions() const;
   Long_t getDelay() const;
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void addToActionMap(CSGAction *action);
   Bool_t activateTextButton(TGTextButton *button);
   Bool_t activatePictureButton(TGPictureButton *button);
   Bool_t activateMenuEntry(int entry);
   Bool_t activateToolBarEntry(int entry);
   void connect(TQObject *sender, const char *signal, const char *slot);
   void defaultAction();
   void loadEvent(int i);
   void goForward();
   void goBackward();
   void goToFirst();
   void playEvents();
   void playEventsBack();
   void pause();
   void quit();
   CSGAction* getAction(const std::string name);
   void enableActions(bool enable = true);
   void enablePrevious(bool enable = true);
   void enableNext(bool enable = true);
   bool previousIsEnabled();
   bool nextIsEnabled();
   void resizeMenu(TGPopupMenu *menu);
   void HandleMenu(Int_t id);
   Bool_t HandleKey(Event_t *event);
   //   Bool_t HandleTimer(TTimer *timer);
   
private:
   CmsShowMainFrame(const CmsShowMainFrame&); // stop default
   
   const CmsShowMainFrame& operator=(const CmsShowMainFrame&); // stop default
   
   // ---------- member data --------------------------------
   std::vector<CSGAction*> m_actionList;
   FWGUIManager *m_manager;
   Long_t m_delay;
   TGNumberEntryField *m_runEntry;
   TGNumberEntryField *m_eventEntry;
   CSGAction *m_nextEvent;
   CSGAction *m_previousEvent;
   Int_t m_playRate;
   Int_t m_playBackRate;
   TTimer *m_playTimer;
   TTimer *m_playBackTimer;   
};


#endif

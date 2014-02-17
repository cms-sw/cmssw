#ifndef Fireworks_Core_CmsShowMainFrame_h
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
// $Id: CmsShowMainFrame.h,v 1.45 2011/07/14 02:21:31 amraktad Exp $
//

// system include files
#include <TQObject.h>
#include <RQ_OBJECT.h>
#include <vector>
#include <TGFrame.h>

#ifndef __CINT__
#include <sigc++/sigc++.h>
#include "Fireworks/Core/interface/CSGActionSupervisor.h"
#endif

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
class TGSlider;
class FWIntValueListener;
class FWCustomIconsButton;
class FWNumberEntryField;

namespace edm {
   class EventBase;
}

class CmsShowMainFrame : public TGMainFrame
#ifndef __CINT__
                       , public CSGActionSupervisor, public sigc::trackable
#endif
{
   friend class FWGUIManager;
public:
   CmsShowMainFrame(const TGWindow *p = 0,UInt_t w = 1,UInt_t h = 1,FWGUIManager *m = 0);
   virtual ~CmsShowMainFrame();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void CloseWindow();

   void loadEvent(const edm::EventBase& event);
   void enableNavigatorControls();
   void quit();
   virtual void enableActions(bool enable = true);
   void enablePrevious(bool enable = true);
   void enableNext(bool enable = true);
   void enableComplexNavigation(bool enable = true);
   bool previousIsEnabled();
   bool nextIsEnabled();
   void updateStatusBar(const char* status);
   void clearStatusBar();
   void setPlayDelayGUI(Float_t val, Bool_t sliderChanged);
   virtual void HandleMenu(Int_t id);
   Bool_t HandleKey(Event_t *event);
   CSGContinuousAction* playEventsAction() const {
      return m_playEvents;
   }
   CSGContinuousAction* loopAction() const {
      return m_loopAction;
   }
   
   CSGContinuousAction* playEventsBackwardsAction() const {
      return m_playEventsBack;
   }

   CSGAction* createNewViewerAction(const std::string& iActionName, bool seaprator);

   void showFWorksInfo();

   void bindCSGActionKeys(const TGMainFrame* f) const;

   void setSummaryViewWeight(float);
   float getSummaryViewWeight() const;

   ClassDef(CmsShowMainFrame, 0);

protected:
   FWCustomIconsButton* m_filterEnableBtn;
   TGTextButton*        m_filterShowGUIBtn;
   FWNumberEntryField*  m_runEntry;
   FWNumberEntryField*  m_lumiEntry;
   FWNumberEntryField*  m_eventEntry;
   FWIntValueListener*  m_delaySliderListener;
   
   const TGPicture*     m_filterIcons[9];

private:
   CmsShowMainFrame(const CmsShowMainFrame&); // stop default
   const CmsShowMainFrame& operator=(const CmsShowMainFrame&); // stop default

   void makeFixedSizeLabel(TGHorizontalFrame* p, const char* txt,
                           UInt_t bgCol, UInt_t txtCol,
                           Int_t  width, Int_t  height);

   // ---------- member data --------------------------------

   FWGUIManager *m_manager;
   Long_t m_tooltipDelay;
   TGLabel* m_timeText;
   CSGAction *m_nextEvent;
   CSGAction *m_previousEvent;
   CSGAction *m_goToFirst;
   CSGAction *m_goToLast;
   CSGAction *m_playDelay;
   CSGAction *m_fworksInfo;
   CSGContinuousAction *m_playEvents;
   CSGContinuousAction *m_playEventsBack;
   CSGContinuousAction *m_loopAction;
   
   TGMainFrame* m_fworksAbout;

   TGSlider* m_delaySlider;
   TGLabel*  m_delayLabel;

   TGStatusBar* m_statBar;

   TGPopupMenu *m_newViewerMenu;
};


#endif

// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowMainFrame
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu May 29 20:58:23 CDT 2008
// $Id: CmsShowMainFrame.cc,v 1.23 2008/11/14 15:41:44 chrjones Exp $
//

// system include files
#include <sigc++/sigc++.h>
#include <TCollection.h>
#include <TApplication.h>
#include <TGClient.h>
#include <TGResourcePool.h>
#include <TGFrame.h>
#include <TGSplitter.h>
#include <TGSplitFrame.h>
#include <TGLayout.h>
#include <TCanvas.h>
#include <TGButton.h>
#include <TGMenu.h>
#include <TGLabel.h>
#include <TGTab.h>
#include <TGStatusBar.h>
#include <TGNumberEntry.h>
#include <TTimer.h>
#include <KeySymbols.h>
#include "TGTextEntry.h"
#include <TSystem.h>
// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/CSGContinuousAction.h"
#include "Fireworks/Core/interface/CSGNumAction.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"
#include "Fireworks/Core/interface/ActionsList.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "Fireworks/Core/interface/FWGUIManager.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//


//
// constructors and destructor
//
CmsShowMainFrame::CmsShowMainFrame(const TGWindow *p,UInt_t w,UInt_t h,FWGUIManager *m) :
TGMainFrame(p, w, h)
{
   Connect("CloseWindow()","CmsShowMainFrame",this,"quit()");

   m_manager = m;
   CSGAction *openData = new CSGAction(this, cmsshow::sOpenData.c_str());
   CSGAction *loadConfig = new CSGAction(this, cmsshow::sLoadConfig.c_str());
   CSGAction *saveConfig = new CSGAction(this, cmsshow::sSaveConfig.c_str());
   CSGAction *saveConfigAs = new CSGAction(this, cmsshow::sSaveConfigAs.c_str());
   CSGAction *exportImage = new CSGAction(this, cmsshow::sExportImage.c_str());
   CSGAction *quit = new CSGAction(this, cmsshow::sQuit.c_str());
   CSGAction *undo = new CSGAction(this, cmsshow::sUndo.c_str());
   undo->disable();
   CSGAction *redo = new CSGAction(this, cmsshow::sRedo.c_str());
   redo->disable();
   CSGAction *cut = new CSGAction(this, cmsshow::sCut.c_str());
   cut->disable();
   CSGAction *copy = new CSGAction(this, cmsshow::sCopy.c_str());
   copy->disable();
   CSGAction *paste = new CSGAction(this, cmsshow::sPaste.c_str());
   paste->disable();
   CSGAction *goToFirst = new CSGAction(this, cmsshow::sGotoFirstEvent.c_str());
   CSGAction *goToLast = new CSGAction(this, cmsshow::sGotoLastEvent.c_str());
   CSGAction *nextEvent = new CSGAction(this, cmsshow::sNextEvent.c_str());
   CSGAction *previousEvent = new CSGAction(this, cmsshow::sPreviousEvent.c_str());
   CSGContinuousAction *playEvents = new CSGContinuousAction(this, cmsshow::sPlayEvents.c_str());
   CSGContinuousAction *playEventsBack = new CSGContinuousAction(this, cmsshow::sPlayEventsBack.c_str());
   CSGAction *showObjInsp = new CSGAction(this, cmsshow::sShowObjInsp.c_str());
   CSGAction *showEventDisplayInsp = new CSGAction(this, cmsshow::sShowEventDisplayInsp.c_str());
   CSGAction *showMainViewCtl = new CSGAction(this, cmsshow::sShowMainViewCtl.c_str());
   CSGAction *showAddCollection = new CSGAction(this, cmsshow::sShowAddCollection.c_str());
   CSGAction *help = new CSGAction(this, cmsshow::sHelp.c_str());
   CSGAction *keyboardShort = new CSGAction(this, cmsshow::sKeyboardShort.c_str());
   m_runEntry = new CSGAction(this, "Run Entry");
   m_eventEntry = new CSGAction(this, "Event Entry");
   CSGAction *eventFilter = new CSGAction(this, "Event Filter");
   m_nextEvent = nextEvent;
   m_previousEvent = previousEvent;
   m_goToFirst = goToFirst;
   m_goToLast = goToLast;
   m_playEvents = playEvents;
   m_playEventsBack = playEventsBack;

   goToFirst->setToolTip("Goto first event");
   goToLast->setToolTip("Goto last event");
   previousEvent->setToolTip("Goto previous event");
   nextEvent->setToolTip("Goto next event");
   playEvents->setToolTip("Play events");
   playEventsBack->setToolTip("Play events backwards");

   TGMenuBar *menuBar = new TGMenuBar(this, this->GetWidth(), 12);

   TGPopupMenu *fileMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("File", fileMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   m_newViewerMenu = new TGPopupMenu(gClient->GetRoot());

   fileMenu->AddPopup("New Viewer", m_newViewerMenu);
   fileMenu->AddSeparator();

   openData->createMenuEntry(fileMenu);
   loadConfig->createMenuEntry(fileMenu);
   saveConfig->createMenuEntry(fileMenu);
   saveConfigAs->createMenuEntry(fileMenu);
   fileMenu->AddSeparator();

   exportImage->createMenuEntry(fileMenu);
   fileMenu->AddSeparator();

   quit->createMenuEntry(fileMenu);

   openData->createShortcut(kKey_O, "CTRL");
   loadConfig->createShortcut(kKey_L, "CTRL");
   saveConfig->createShortcut(kKey_S, "CTRL");
   saveConfigAs->createShortcut(kKey_S, "CTRL+SHIFT");
   exportImage->createShortcut(kKey_P, "CTRL");
   quit->createShortcut(kKey_Q, "CTRL");

   TGPopupMenu *editMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("Edit", editMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   undo->createMenuEntry(editMenu);
   undo->createShortcut(kKey_Z, "CTRL");
   redo->createMenuEntry(editMenu);
   redo->createShortcut(kKey_Z, "CTRL+SHIFT");
   editMenu->AddSeparator();

   cut->createMenuEntry(editMenu);
   cut->createShortcut(kKey_X, "CTRL");
   copy->createMenuEntry(editMenu);
   copy->createShortcut(kKey_C, "CTRL");
   paste->createMenuEntry(editMenu);
   paste->createShortcut(kKey_V, "CTRL");

   TGPopupMenu *viewMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("View", viewMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   nextEvent->createMenuEntry(viewMenu);
   nextEvent->createShortcut(kKey_Right, "CTRL");
   previousEvent->createMenuEntry(viewMenu);
   previousEvent->createShortcut(kKey_Left, "CTRL");
   goToFirst->createMenuEntry(viewMenu);
   goToLast->createMenuEntry(viewMenu);
   playEvents->createMenuEntry(viewMenu);
   playEvents->createShortcut(kKey_Right, "CTRL+SHIFT");
   playEventsBack->createMenuEntry(viewMenu);
   playEventsBack->createShortcut(kKey_Left, "CTRL+SHIFT");

   TGPopupMenu* windowMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("Window", windowMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));

   showObjInsp->createMenuEntry(windowMenu);
   showObjInsp->createShortcut(kKey_I, "CTRL");
   showEventDisplayInsp->createMenuEntry(windowMenu);
   showMainViewCtl->createMenuEntry(windowMenu);
   showAddCollection->createMenuEntry(windowMenu);

   TGPopupMenu *helpMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("Help", helpMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   help->createMenuEntry(helpMenu);
   keyboardShort->createMenuEntry(helpMenu);

   AddFrame(menuBar, new TGLayoutHints(kLHintsExpandX,2,2,2,2));

   /*DEBUGGING menu
   FontStruct_t font = gClient->GetResourcePool()->GetMenuHiliteFont()->GetFontStruct();
   std::vector<CSGAction*>::iterator it_act;
   printf("Width of <shift> : %d\n", gVirtualX->TextWidth(font, "<shift> ", 8));
   for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
     if ((*it_act)->getMenu() != 0 && (*it_act)->getMenu()->GetEntry((*it_act)->getMenuEntry()) != 0) {
       printf("Menu %s has width %d\n", (*it_act)->getMenu()->GetEntry((*it_act)->getMenuEntry())->GetLabel()->Data(), gVirtualX->TextWidth(font, (*it_act)->getMenu()->GetEntry((*it_act)->getMenuEntry())->GetLabel()->Data(), (*it_act)->getMenu()->GetEntry((*it_act)->getMenuEntry())->GetLabel()->Length()));
     }
   }
    */

   /*
   if(0==gSystem->Getenv("ROOTSYS")) {
     std::cerr<<"environment variable ROOTSYS is not set" <<
       std::endl;
     throw std::runtime_error("ROOTSYS environment variable not set");
   }
   TString icondir(Form("%s/icons/",gSystem->Getenv("ROOTSYS")));
   */

   const unsigned int kBackgroundColor=0x2f2f2f;
   const unsigned int kTextColor= 0xb3b3b3;
   TGHorizontalFrame *fullbar = new TGHorizontalFrame(this, this->GetWidth(), 30,0,kBackgroundColor);
   m_statBar = new TGStatusBar(this, this->GetWidth(), 12);
   AddFrame(m_statBar, new TGLayoutHints(kLHintsBottom | kLHintsExpandX));
   MapSubwindows();
   Layout();
   MapWindow();
   //printf("this->GetWidth(): %d\n", this->GetWidth());
   /*TGToolBar *tools = new TGToolBar(fullbar, 400, 30);
   tools->SetBackgroundColor(kBlack);*/
   TGCompositeFrame* tools = new TGHorizontalFrame(fullbar,10,10,0,kBackgroundColor);
   TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));

   goToFirst->createCustomIconsButton(tools,fClient->GetPicture(coreIcondir+"button-gotofirst.png"),
                                      fClient->GetPicture(coreIcondir+"button-gotofirst-over.png"),
                                      fClient->GetPicture(coreIcondir+"button-gotofirst-disabled.png"),
                                      new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 7,0,5,0));
   playEventsBack->createCustomIconsButton(tools,fClient->GetPicture(coreIcondir+"button-backward.png"),
                                           fClient->GetPicture(coreIcondir+"button-backward-over.png"),
                                           fClient->GetPicture(coreIcondir+"button-backward-disabled.png"),
                                           fClient->GetPicture(coreIcondir+"button-pause.png"),
                                           fClient->GetPicture(coreIcondir+"button-pause-over.png"),
                                          new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 7,0,5,0));
   previousEvent->createCustomIconsButton(tools,fClient->GetPicture(coreIcondir+"button-stepback.png"),
                                      fClient->GetPicture(coreIcondir+"button-stepback-over.png"),
                                      fClient->GetPicture(coreIcondir+"button-stepback-disabled.png"),
                                      new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 7,0,5,0));
   nextEvent->createCustomIconsButton(tools,fClient->GetPicture(coreIcondir+"button-stepforward.png"),
                                      fClient->GetPicture(coreIcondir+"button-stepforward-over.png"),
                                      fClient->GetPicture(coreIcondir+"button-stepforward-disabled.png"),
                                      new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 7,0,5,0));
   playEvents->createCustomIconsButton(tools,fClient->GetPicture(coreIcondir+"button-forward.png"),
                                       fClient->GetPicture(coreIcondir+"button-forward-over.png"),
                                       fClient->GetPicture(coreIcondir+"button-forward-disabled.png"),
                                       fClient->GetPicture(coreIcondir+"button-pause.png"),
                                       fClient->GetPicture(coreIcondir+"button-pause-over.png"),
                                       new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 7,0,5,0));
   goToLast->createCustomIconsButton(tools,fClient->GetPicture(coreIcondir+"button-gotolast.png"),
                                      fClient->GetPicture(coreIcondir+"button-gotolast-over.png"),
                                      fClient->GetPicture(coreIcondir+"button-gotolast-disabled.png"),
                                      new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 7,0,5,0));
   
   fullbar->AddFrame(tools, new TGLayoutHints(kLHintsLeft,2,2,2,2));
   // TGHorizontalFrame *texts = new TGHorizontalFrame(fullbar, fullbar->GetWidth() - tools->GetWidth(), 60);
   TGVerticalFrame *texts   = new TGVerticalFrame(fullbar, fullbar->GetWidth() - tools->GetWidth(), 60,0,kBackgroundColor);

   fullbar->AddFrame(texts, new TGLayoutHints(kLHintsExpandX,20,2,2,2));
   AddFrame(fullbar, new TGLayoutHints(kLHintsExpandX,2,2,2,2));

   TGHorizontalFrame *runInfo   = new TGHorizontalFrame(texts, fullbar->GetWidth() - tools->GetWidth(), 30, 0, kBackgroundColor);
   texts->AddFrame(runInfo, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY,0,0,0,0));
   TGHorizontalFrame *evtFilter = new TGHorizontalFrame(texts, fullbar->GetWidth() - tools->GetWidth(), 30, 0, kBackgroundColor);
   texts->AddFrame(evtFilter, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY,0,0,0,0));

   //printf("text width: %d\n", this->GetWidth()-tools->GetWidth());
   TGLabel *runText = new TGLabel(runInfo, "Run");
   UInt_t maxWidth = runText->GetWidth();
   TGLabel *filterText = new TGLabel(evtFilter, "Filter");
   if(maxWidth < filterText->GetWidth()) {
      maxWidth = filterText->GetWidth();
   }
   runText->SetBackgroundColor(kBackgroundColor);
   runText->SetTextColor(kTextColor);
   runInfo->AddFrame(runText, new TGLayoutHints(kLHintsCenterY|kLHintsLeft,maxWidth-runText->GetWidth(),0,0,0));
   m_runEntry->createNumberEntry(runInfo, new TGLayoutHints(kLHintsCenterY|kLHintsLeft,10,0,0,0));
   TGLabel *eventText = new TGLabel(runInfo, "Event");
   eventText->SetBackgroundColor(kBackgroundColor);
   eventText->SetTextColor(kTextColor);
   runInfo->AddFrame(eventText, new TGLayoutHints(kLHintsCenterY|kLHintsLeft,10,0,0,0));
   m_eventEntry->createNumberEntry(runInfo, new TGLayoutHints(kLHintsCenterY|kLHintsLeft,10,0,0,0));
   TGLabel* timeLabel = new TGLabel(runInfo,"Time");
   timeLabel->SetBackgroundColor(kBackgroundColor);
   timeLabel->SetTextColor(kTextColor);
   runInfo->AddFrame(timeLabel, new TGLayoutHints(kLHintsCenterY|kLHintsLeft,10,0,0,0));
   m_timeText = new TGLabel(runInfo, "");
   m_timeText->SetTextJustify(kTextLeft);
   m_timeText->SetBackgroundColor(kBackgroundColor);
   m_timeText->SetTextColor(kTextColor);
   runInfo->AddFrame(m_timeText, new TGLayoutHints(kLHintsCenterY|kLHintsLeft|kLHintsExpandX,5,0,0,0));

   filterText->SetBackgroundColor(kBackgroundColor);
   filterText->SetTextColor(kTextColor);
   evtFilter->AddFrame(filterText, new TGLayoutHints(kLHintsCenterY|kLHintsLeft,maxWidth-filterText->GetWidth(),0,0,0));
   eventFilter->createTextEntry(evtFilter, new TGLayoutHints(kLHintsCenterY|kLHintsLeft|kLHintsExpandX,0,0,0,0));

   //Start disabled
   goToFirst->disable();
   goToLast->disable();
   previousEvent->disable();
   nextEvent->disable();
   playEvents->disable();
   playEventsBack->disable();

   TGSplitFrame *csArea = new TGSplitFrame(this, this->GetWidth(), this->GetHeight()-42);
   csArea->VSplit(200);
   csArea->GetFirst()->AddFrame(m_manager->createList(csArea->GetFirst()), new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY));
   TGTab *tabFrame = new TGTab(csArea->GetSecond(), csArea->GetSecond()->GetWidth(), csArea->GetSecond()->GetHeight());
   tabFrame->AddTab("Views", m_manager->createViews(tabFrame));
   csArea->GetSecond()->AddFrame(tabFrame, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY));
   m_manager->createTextView(tabFrame);
   AddFrame(csArea,new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY,2,2,0,2));
   SetWindowName("cmsShow");
   MapSubwindows();
   //   printf("Default main frame size: %d, %d\n", this->GetDefaultSize().fWidth, this->GetDefaultSize().fHeight);
   //   printf("Main frame size: %d, %d\n", this->GetWidth(), this->GetHeight());
   //   Resize(this->GetDefaultSize());
   Layout();
   MapWindow();
}

// CmsShowMainFrame::CmsShowMainFrame(const CmsShowMainFrame& rhs)
// {
//    // do actual copying here;
// }

CmsShowMainFrame::~CmsShowMainFrame() {
   Cleanup();
   for(std::vector<CSGAction*>::iterator it= m_actionList.begin(),itEnd = m_actionList.end();
       it != itEnd;
       ++it) {
      delete *it;
   }
   //delete m_statBar;
}

//
// assignment operators
//
// const CmsShowMainFrame& CmsShowMainFrame::operator=(const CmsShowMainFrame& rhs)
// {
//   //An exception safe implementation is
//   CmsShowMainFrame temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void CmsShowMainFrame::addToActionMap(CSGAction *action) {
   m_actionList.push_back(action);
}

CSGAction*
CmsShowMainFrame::createNewViewerAction(const std::string& iActionName)
{
   CSGAction* action(new CSGAction(this, iActionName.c_str()));
   action->createMenuEntry(m_newViewerMenu);
   return action;
}

Bool_t CmsShowMainFrame::activateMenuEntry(int entry) {
   std::vector<CSGAction*>::iterator it_act;
   for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
      if (entry == (*it_act)->getMenuEntry()) {
         (*it_act)->activated.emit();
         return kTRUE;
      }
   }
   return kFALSE;
}

Bool_t CmsShowMainFrame::activateToolBarEntry(int entry) {
   std::vector<CSGAction*>::iterator it_act;
   for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
      if ((*it_act)->getToolBarData() && (*it_act)->getToolBarData()->fId == entry) {
         (*it_act)->activated.emit();
         return kTRUE;
      }
   }
   return kFALSE;
}

Long_t CmsShowMainFrame::getDelay() const {
   return m_delay;
}

void CmsShowMainFrame::defaultAction() {
   printf("Default action!\n");
}

void CmsShowMainFrame::loadEvent(const fwlite::Event& event) {
  m_runEntry->getNumberEntry()->SetIntNumber(event.id().run());
  m_eventEntry->getNumberEntry()->SetIntNumber(event.id().event());
  m_timeText->SetText( fw::getTimeGMT( event ).c_str() );
  // loadEvent gets called before the special cases [at beginning, at end, etc]
  // so we can enable all our event controls here
  m_nextEvent->enable();
  m_previousEvent->enable();
  m_goToFirst->enable();
   m_goToLast->enable();
  m_playEvents->enable();
  m_playEventsBack->enable();
}

void CmsShowMainFrame::quit() {
   getAction(cmsshow::sQuit)->activated();
}

CSGAction*
CmsShowMainFrame::getAction(const std::string& name)
{
  std::vector<CSGAction*>::iterator it_act;
  for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
    if ((*it_act)->getName() == name)
      return *it_act;
  }
   std::cout << "No action is found with name \"" << name << "\"" << std::endl;
  return 0;
}

void
CmsShowMainFrame::enableActions(bool enable)
{

  std::vector<CSGAction*>::iterator it_act;
  for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
    if (enable)
      (*it_act)->globalEnable();
    else
      (*it_act)->globalDisable();
  }
  if (enable) {
    m_runEntry->enable();
    m_eventEntry->enable();
  }
  else {
    m_runEntry->disable();
    m_eventEntry->disable();
  }
}

void
CmsShowMainFrame::enablePrevious(bool enable)
{
  if (m_previousEvent != 0) {
     if (enable) {
        m_previousEvent->enable();
        m_playEventsBack->enable();
     } else {
        m_previousEvent->disable();
        m_playEventsBack->disable();
        m_playEventsBack->stop();
     }
  }
  if (m_goToFirst != 0) {
    if (enable)
      m_goToFirst->enable();
    else
      m_goToFirst->disable();
  }
}

void
CmsShowMainFrame::enableNext(bool enable)
{
  if (m_nextEvent != 0) {
     if (enable) {
        m_nextEvent->enable();
        m_playEvents->enable();
        m_goToLast->enable();
     } else {
        m_nextEvent->disable();
        m_playEvents->disable();
        m_goToLast->disable();
        m_playEvents->stop();
     }
  }
}

bool
CmsShowMainFrame::nextIsEnabled()
{
  return m_nextEvent->isEnabled();
}

bool
CmsShowMainFrame::previousIsEnabled()
{
  return m_previousEvent->isEnabled();
}

void CmsShowMainFrame::updateStatusBar(const char* status) {
  m_statBar->SetText(status, 0);
  //force the status bar to update its image
  gClient->ProcessEventsFor(m_statBar);
}

void CmsShowMainFrame::clearStatusBar()
{
   m_statBar->SetText("", 0);
   //don't process immediately since we want this on the event queue
   // since results of the last action may still be happening
}

void CmsShowMainFrame::HandleMenu(Int_t id) {
   switch(id) {
      case 1:
      {
         gApplication->Terminate(0);
      }
         break;
      default:
         printf("Invalid menu id\n");
         break;
   }
}

Bool_t CmsShowMainFrame::HandleKey(Event_t *event) {
   if (event->fType == kGKeyPress) {
      std::vector<CSGAction*>::iterator it_act;
      Int_t keycode;
      Int_t modcode;
      for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
         keycode = (*it_act)->getKeycode();
         modcode = (*it_act)->getModcode();
         if ((event->fCode == (UInt_t)keycode) &&
             ((event->fState == (UInt_t)modcode) ||
              (event->fState == (UInt_t)(modcode | kKeyMod2Mask)) ||
              (event->fState == (UInt_t)(modcode | kKeyLockMask)) ||
              (event->fState == (UInt_t)(modcode | kKeyMod2Mask | kKeyLockMask)))) {
            (*it_act)->activated.emit();
            return kTRUE;
         }
      }
   }
   return kFALSE;
}

void CmsShowMainFrame::resizeMenu(TGPopupMenu *menu) {
   std::vector<CSGAction*>::iterator it_act;
   for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
      if ((*it_act)->getMenu() == menu && (*it_act)->getKeycode() != 0) {
         (*it_act)->resizeMenuEntry();
      }
   }
}

const std::vector<CSGAction *>& CmsShowMainFrame::getListOfActions() const {
   return m_actionList;
}

CSGAction*
CmsShowMainFrame::getRunEntry() const {
  return m_runEntry;
}

CSGAction*
CmsShowMainFrame::getEventEntry() const {
  return m_eventEntry;
}

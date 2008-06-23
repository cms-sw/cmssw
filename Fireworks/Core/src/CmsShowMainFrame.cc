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
// $Id: CmsShowMainFrame.cc,v 1.2 2008/06/23 01:05:55 chrjones Exp $
//

// system include files
#include <sigc++/sigc++.h>
#include <TCollection.h>
#include <TApplication.h>
#include <TGClient.h>
#include <TGFrame.h>
#include <TGSplitter.h>
#include <TGSplitFrame.h>
#include <TGLayout.h>
#include <TCanvas.h>
#include <TGButton.h>
#include <TGMenu.h>
#include <TGLabel.h>
#include <TGTab.h>
#include <TGNumberEntry.h>
#include <TTimer.h>
#include <KeySymbols.h>

// user include files
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"

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
   m_manager = m;
   m_delay = 250;
   m_playRate = 1000;
   m_playBackRate = 1000;
   m_playTimer = new TTimer(m_playRate);
   m_playBackTimer = new TTimer(m_playBackRate);
   m_playTimer->SetObject(this);
   m_playBackTimer->SetObject(this);
   CSGAction *goToFirst = new CSGAction(this, "Home");
   /*
   CSGAction *addRhoPhi = new CSGAction(this, "Rho Phi");
   CSGAction *addRhoZ = new CSGAction(this, "Rho Z");
   CSGAction *addLego = new CSGAction(this, "Lego");
    */
   CSGAction *openData = new CSGAction(this, "Open Data File...");
   CSGAction *loadConfig = new CSGAction(this, "Load Configuration...");
   CSGAction *saveConfig = new CSGAction(this, "Save Configuration");
   CSGAction *saveConfigAs = new CSGAction(this, "Save Configuration As...");
   CSGAction *exportImage = new CSGAction(this, "Export Main Viewer Image...");
   CSGAction *quit = new CSGAction(this, "Quit");
   CSGAction *undo = new CSGAction(this, "Undo");
   CSGAction *redo = new CSGAction(this, "Redo");
   CSGAction *cut = new CSGAction(this, "Cut");
   CSGAction *copy = new CSGAction(this, "Copy");
   CSGAction *paste = new CSGAction(this, "Paste");
   CSGAction *nextEvent = new CSGAction(this, "Goto Next Event");
   CSGAction *previousEvent = new CSGAction(this, "Goto Previous Event");
   CSGAction *playEvents = new CSGAction(this, "Play Events");
   CSGAction *playEventsBack = new CSGAction(this, "Play Events Backwards");
   CSGAction *pause = new CSGAction(this, "Pause");
   CSGAction *showObjInsp = new CSGAction(this, "Show Object Inspector");
   CSGAction *showMainViewCtl = new CSGAction(this, "Show Main View Controller");
   CSGAction *help = new CSGAction(this, "cmsShow Help...");
   CSGAction *keyboardShort = new CSGAction(this, "Keyboard Shortcuts...");
   //   saveConfigAs->activated.connect(sigc::mem_fun(*m_manager, &FWGUIManager::saveConfigAs));
   //   saveConfig->activated.connect(sigc::mem_fun(*m_manager, &FWGUIManager::saveConfig));
   //   goToFirst->activated.connect(sigc::mem_fun(*m_manager, &FWGUIManager::goToFirst));
   //   nextEvent->activated.connect(sigc::mem_fun(*m_manager, &FWGUIManager::goForward));
   //   previousEvent->activated.connect(sigc::mem_fun(*m_manager, &FWGUIManager::goBack));
   m_nextEvent = nextEvent;
   m_previousEvent = previousEvent;
   //   playEvents->activated.connect(sigc::mem_fun(*this, &CmsShowMainFrame::playEvents));
   //   playEventsBack->activated.connect(sigc::mem_fun(*this, &CmsShowMainFrame::playEventsBack));
   //   pause->activated.connect(sigc::mem_fun(*m_manager, &FWGUIManager::stop));
   //   quit->activated.connect(sigc::mem_fun(*m_manager, &FWGUIManager::quit));
   //    enableNext->activated.connect(sigc::mem_fun(*nextEvent, &CSGAction::enable));

   nextEvent->setToolTip("Load next event");

   TGMenuBar *menuBar = new TGMenuBar(this, this->GetWidth(), 12);

   TGPopupMenu *fileMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("File", fileMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   m_newViewerMenu = new TGPopupMenu(gClient->GetRoot());
   /*
   addRhoPhi->createMenuEntry(newViewerMenu);
   addRhoZ->createMenuEntry(newViewerMenu);
   addLego->createMenuEntry(newViewerMenu);
   addLego->disable();
    */
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
   playEvents->createMenuEntry(viewMenu);
   playEvents->createShortcut(kKey_Right, "CTRL+SHIFT");
   playEventsBack->createMenuEntry(viewMenu);
   playEventsBack->createShortcut(kKey_Left, "CTRL+SHIFT");
   pause->createMenuEntry(viewMenu);
   viewMenu->AddSeparator();
   
   showObjInsp->createMenuEntry(viewMenu);
   showObjInsp->createShortcut(kKey_I, "CTRL");
   showMainViewCtl->createMenuEntry(viewMenu);
   
   TGPopupMenu *helpMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("Help", helpMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   help->createMenuEntry(helpMenu);
   keyboardShort->createMenuEntry(helpMenu);

   //   enableNext->createShortcut(kKey_M, "CTRL");
   
   AddFrame(menuBar, new TGLayoutHints(kLHintsExpandX,2,2,2,2));

   /*
   if(0==gSystem->Getenv("ROOTSYS")) {
     std::cerr<<"environment variable ROOTSYS is not set" <<
       std::endl;
     throw std::runtime_error("ROOTSYS environment variable not set");
   }
   TString icondir(Form("%s/icons/",gSystem->Getenv("ROOTSYS")));
   */

   TGHorizontalFrame *fullbar = new TGHorizontalFrame(this, this->GetWidth(), 30);
   TGToolBar *tools = new TGToolBar(fullbar, 400, 30);
   goToFirst->createToolBarEntry(tools, "first_t.xpm");
   previousEvent->createToolBarEntry(tools, "previous_t.xpm");
   nextEvent->createToolBarEntry(tools, "next_t.xpm");
   pause->createToolBarEntry(tools, "stop_t.xpm");
   fullbar->AddFrame(tools, new TGLayoutHints(kLHintsExpandX,2,2,2,2));
   TGHorizontalFrame *texts = new TGHorizontalFrame(fullbar, fullbar->GetWidth() - tools->GetWidth(), 30);
   TGLabel *runText = new TGLabel(texts, "Run: ");
   texts->AddFrame(runText, new TGLayoutHints(kLHintsCenterY,2,2,2,2));
   m_runEntry = new TGNumberEntryField(texts, -1, 1);
   texts->AddFrame(m_runEntry, new TGLayoutHints(kLHintsCenterY,2,2,2,2));
   TGLabel *eventText = new TGLabel(texts, "Event: ");
   texts->AddFrame(eventText, new TGLayoutHints(kLHintsCenterY,2,2,2,2));
   m_eventEntry = new TGNumberEntryField(texts, -1, 1);
   texts->AddFrame(m_eventEntry, new TGLayoutHints(kLHintsCenterY,2,2,2,2));
   fullbar->AddFrame(texts, new TGLayoutHints(kLHintsExpandX,2,2,2,2));
   AddFrame(fullbar, new TGLayoutHints(kLHintsExpandX,2,2,2,2));
   
   TGSplitFrame *csArea = new TGSplitFrame(this, this->GetWidth(), this->GetHeight()-42);
   //   TGGroupFrame *csList = m_manager->createList(csArea);
   //   TGVerticalFrame *csList = new TGVerticalFrame(csArea, 10, 10);
   //   TGVerticalFrame *csDisplay = new TGVerticalFrame(csArea, 10, 10);
   //   TGCompositeFrame *fFleft = new TGCompositeFrame(csList, 10, 10, kSunkenFrame);
   //   TGCompositeFrame *fFright = new TGCompositeFrame(csDisplay, 10, 10, kSunkenFrame);

   //   TGLabel *fLleft = new TGLabel(fFleft, "List");
   //   TGLabel *fLright = new TGLabel(fFright, "Display");
   //   fFleft->AddFrame(fLleft,new TGLayoutHints(kLHintsLeft | kLHintsCenterY,3,0,0,0));
   //   fFright->AddFrame(fLright,new TGLayoutHints(kLHintsLeft | kLHintsCenterY,
   //                                               3,0,0,0));
   //   csList->AddFrame(fFleft,new TGLayoutHints(kLHintsTop | kLHintsExpandX |
   //                                                kLHintsExpandY,0,0,0,0));
   //   csDisplay->AddFrame(fFright,new TGLayoutHints(kLHintsTop | kLHintsExpandX |
   //						 kLHintsExpandY,0,0,0,0));

   //   csList->Resize(0.2*this->GetWidth(), this->GetHeight()-12);
   csArea->VSplit(200);
   csArea->GetFirst()->AddFrame(m_manager->createList(csArea->GetFirst()), new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY));
   TGTab *tabFrame = new TGTab(csArea->GetSecond(), csArea->GetSecond()->GetWidth(), csArea->GetSecond()->GetHeight());
   tabFrame->AddTab("Views", m_manager->createViews(tabFrame));
   csArea->GetSecond()->AddFrame(tabFrame, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY));
   m_manager->createTextView(tabFrame);
   //   csArea->GetFirst()->AddFrame(csList,new TGLayoutHints(kLHintsLeft | kLHintsExpandY));
   //   TGVSplitter *splitter = new TGVSplitter(csArea,2,584);
   //   splitter->SetFrame(csList, kTRUE);
   //   csArea->AddFrame(splitter,new TGLayoutHints(kLHintsLeft | kLHintsExpandY));
   //   csArea->GetSecond()->AddFrame(csDisplay,new TGLayoutHints(kLHintsRight|kLHintsExpandX|kLHintsExpandY));
   //   csArea->Resize(csArea->GetDefaultSize()); 
   AddFrame(csArea,new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY,2,2,0,2));
   SetWindowName("cmsShow");
   MapSubwindows();
   //   printf("Default main frame size: %d, %d\n", this->GetDefaultSize().fWidth, this->GetDefaultSize().fHeight);
   //   printf("Main frame size: %d, %d\n", this->GetWidth(), this->GetHeight());
   //   Resize(this->GetDefaultSize());
   MapWindow();   
   Layout();
}

// CmsShowMainFrame::CmsShowMainFrame(const CmsShowMainFrame& rhs)
// {
//    // do actual copying here;
// }

CmsShowMainFrame::~CmsShowMainFrame() {
   Cleanup();
   delete m_playTimer;
   delete m_playBackTimer;
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


Bool_t CmsShowMainFrame::activateTextButton(TGTextButton *button) {
   std::vector<CSGAction*>::iterator it_act;
   for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
      if (button == (*it_act)->getTextButton()) {
         (*it_act)->activated.emit();
         return kTRUE;
      }
   }
   return kFALSE;
}

Bool_t CmsShowMainFrame::activatePictureButton(TGPictureButton *button) {
   std::vector<CSGAction*>::iterator it_act;
   for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
      if (button == (*it_act)->getPictureButton()) {
         (*it_act)->activated.emit();
         return kTRUE;
      }
   }
   return kFALSE;
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

void CmsShowMainFrame::loadEvent(int i) {
  m_eventEntry->SetNumber(i);
}

void CmsShowMainFrame::goForward() {
  m_eventEntry->SetNumber(m_eventEntry->GetNumber()+1);
}

void CmsShowMainFrame::goBackward() {
   if (m_eventEntry->GetNumber() > 1) {
      m_eventEntry->SetNumber(m_eventEntry->GetNumber()-1);
   }
   else {
      m_playBackTimer->TurnOff();
   }
}

void CmsShowMainFrame::goToFirst() {
  m_eventEntry->SetNumber(1);
}

void CmsShowMainFrame::playEvents() {
   m_playBackTimer->TurnOff();
   m_playTimer->TurnOn();
}

void CmsShowMainFrame::playEventsBack() {
   m_playTimer->TurnOff();
   m_playBackTimer->TurnOn();
}

void CmsShowMainFrame::pause() {
   m_playTimer->TurnOff();
   m_playBackTimer->TurnOff();
}

void CmsShowMainFrame::quit() {
   gApplication->Terminate(0);
}

CSGAction*
CmsShowMainFrame::getAction(const std::string& name)
{
  std::vector<CSGAction*>::iterator it_act;
  for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
    if ((*it_act)->getName() == name)
      return *it_act;
  }
  printf("None Found!\n");
  return 0;
}

void
CmsShowMainFrame::enableActions(bool enable)
{
  std::vector<CSGAction*>::iterator it_act;
  for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
    if (enable) 
      (*it_act)->enable();
    else
      (*it_act)->disable();
  }
}

void
CmsShowMainFrame::enablePrevious(bool enable)
{
  if (m_previousEvent != 0) {
    if (enable)
      m_previousEvent->enable();
    else
      m_previousEvent->disable();
  }
}

void
CmsShowMainFrame::enableNext(bool enable)
{ 
  if (m_nextEvent != 0) {
    if (enable)
      m_nextEvent->enable();
    else
      m_nextEvent->disable();
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

/*
Bool_t CmsShowMainFrame::HandleTimer(TTimer *timer) {
   m_manager->goForward();
   return kTRUE;
}
*/

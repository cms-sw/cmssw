
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

#include "FWCore/Common/interface/EventBase.h"

// system include files
#include <TCollection.h>
#include <TApplication.h>
#include <TEveWindow.h>
#include <TGClient.h>
#include <TGLayout.h>
#include <TGButton.h>
#include <TGMenu.h>
#include <TGLabel.h>
#include <TGTab.h>
#include <TGPack.h>
#include <TGStatusBar.h>
#include <KeySymbols.h>
#include <TGSlider.h>

#include <TSystem.h>
#include <TImage.h>
// user include files
#include "DataFormats/Provenance/interface/EventID.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/CSGContinuousAction.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"
#include "Fireworks/Core/interface/ActionsList.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"

#include "Fireworks/Core/interface/FWIntValueListener.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"
#include "Fireworks/Core/src/FWNumberEntry.h"

#include "Fireworks/Core/interface/fwPaths.h"

#include <fstream>

//
// constants, enums and typedefs
//

//
// static data member definitions
//


// AMT: temprary workaround until TGPack::ResizeExistingFrames() is public
class FWPack : public TGPack
{
   friend class CmsShowMainFrame;
public:
   FWPack(const TGWindow* w) : TGPack(w, 100, 100) {}
   virtual ~FWPack() {}
};

//
// constructors and destructor
//
CmsShowMainFrame::CmsShowMainFrame(const TGWindow *p,UInt_t w,UInt_t h,FWGUIManager *m) :
   TGMainFrame(p, w, h),
   m_filterEnableBtn(),
   m_filterShowGUIBtn(),
   m_runEntry(0),
   m_lumiEntry(0),
   m_eventEntry(0),
   m_delaySliderListener(0),
   m_manager(m),
   m_fworksAbout(0)
{
   const unsigned int backgroundColor=0x2f2f2f;
   const unsigned int textColor= 0xb3b3b3;
   gClient->SetStyle("classic");

   CSGAction *openData    = new CSGAction(this, cmsshow::sOpenData.c_str());
   CSGAction *appendData  = new CSGAction(this, cmsshow::sAppendData.c_str());
   CSGAction *searchFiles = new CSGAction(this, cmsshow::sSearchFiles.c_str());

   CSGAction *loadConfig   = new CSGAction(this, cmsshow::sLoadConfig.c_str());
   CSGAction *saveConfig   = new CSGAction(this, cmsshow::sSaveConfig.c_str());
   CSGAction *saveConfigAs = new CSGAction(this, cmsshow::sSaveConfigAs.c_str());


   CSGAction *loadPartialConfig   = new CSGAction(this, cmsshow::sLoadPartialConfig.c_str());
   CSGAction *savePartialConfig   = new CSGAction(this, cmsshow::sSavePartialConfig.c_str());
   CSGAction *savePartialConfigAs = new CSGAction(this, cmsshow::sSavePartialConfigAs.c_str());


   CSGAction *exportImage  = new CSGAction(this, cmsshow::sExportImage.c_str());
   CSGAction *exportImages = new CSGAction(this, cmsshow::sExportAllImages.c_str());
   CSGAction *quit = new CSGAction(this, cmsshow::sQuit.c_str());

   CSGAction *undo = new CSGAction(this, cmsshow::sUndo.c_str());
   undo->disable(); //NOTE: All disables happen again later in this routine
   CSGAction *redo  = new CSGAction(this, cmsshow::sRedo.c_str());
   redo->disable(); //NOTE: All disables happen again later in this routine
   CSGAction *cut   = new CSGAction(this, cmsshow::sCut.c_str());
   cut->disable();  //NOTE: All disables happen again later in this routine
   CSGAction *copy  = new CSGAction(this, cmsshow::sCopy.c_str());
   copy->disable(); //NOTE: All disables happen again later in this routine
   CSGAction *paste = new CSGAction(this, cmsshow::sPaste.c_str());
   paste->disable();//NOTE: All disables happen again later in this routine

   CSGAction *goToFirst = new CSGAction(this, cmsshow::sGotoFirstEvent.c_str());
   CSGAction *goToLast = new CSGAction(this, cmsshow::sGotoLastEvent.c_str());

   CSGAction *nextEvent          = new CSGAction(this, cmsshow::sNextEvent.c_str());
   CSGAction *previousEvent      = new CSGAction(this, cmsshow::sPreviousEvent.c_str());

   CSGContinuousAction *playEvents     = new CSGContinuousAction(this, cmsshow::sPlayEvents.c_str());
   CSGContinuousAction *playEventsBack = new CSGContinuousAction(this, cmsshow::sPlayEventsBack.c_str());
   CSGContinuousAction *loop           = new CSGContinuousAction(this, cmsshow::sAutoRewind.c_str());

   CSGAction *showCommonInsp = new CSGAction(this, cmsshow::sShowCommonInsp.c_str());
   CSGAction *colorset       = new CSGAction(this, cmsshow::sBackgroundColor.c_str());

   CSGAction *showObjInsp          = new CSGAction(this, cmsshow::sShowObjInsp.c_str());
   CSGAction *showEventDisplayInsp = new CSGAction(this, cmsshow::sShowEventDisplayInsp.c_str());
   CSGAction *showMainViewCtl      = new CSGAction(this, cmsshow::sShowMainViewCtl.c_str());
   CSGAction *showAddCollection    = new CSGAction(this, cmsshow::sShowAddCollection.c_str());
   CSGAction *showInvMassDialog    = new CSGAction(this, cmsshow::sShowInvMassDialog.c_str());

   CSGAction *help               = new CSGAction(this, cmsshow::sHelp.c_str());
   CSGAction *keyboardShort      = new CSGAction(this, cmsshow::sKeyboardShort.c_str());
   CSGAction *helpGL             = new CSGAction(this, cmsshow::sHelpGL.c_str());

   m_nextEvent = nextEvent;
   m_previousEvent = previousEvent;
   m_goToFirst = goToFirst;
   m_goToLast = goToLast;
   m_playEvents = playEvents;
   m_playEventsBack = playEventsBack;
   m_loopAction = loop;

   goToFirst->setToolTip("Goto first event");
   goToLast->setToolTip("Goto last event");
   previousEvent->setToolTip("Goto previous event");
   nextEvent->setToolTip("Goto next event");
   playEvents->setToolTip("Play events");
   playEventsBack->setToolTip("Play events backwards");

   TGCompositeFrame *menuTopFrame = new TGCompositeFrame(this, 1, 1, kHorizontalFrame, backgroundColor);

   TGMenuBar *menuBar = new TGMenuBar(menuTopFrame, this->GetWidth(), 28, kHorizontalFrame);

   TGPopupMenu *fileMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("File", fileMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 2, 0));
   
   openData->createMenuEntry(fileMenu);
   appendData->createMenuEntry(fileMenu);
   searchFiles->createMenuEntry(fileMenu);
   //searchFiles->disable();

   fileMenu->AddSeparator();
   loadConfig->createMenuEntry(fileMenu);
   saveConfig->createMenuEntry(fileMenu);
   saveConfigAs->createMenuEntry(fileMenu);

 
   TGPopupMenu*  partialSaveMenu = new TGPopupMenu(gClient->GetRoot());
   fileMenu->AddPopup("Advanced Configuration", partialSaveMenu);

   loadPartialConfig->createMenuEntry(partialSaveMenu);
   savePartialConfig->createMenuEntry(partialSaveMenu);
   savePartialConfigAs->createMenuEntry(partialSaveMenu);
   fileMenu->AddSeparator();
    
   exportImage->createMenuEntry(fileMenu);
   exportImages->createMenuEntry(fileMenu);
   fileMenu->AddSeparator();

   quit->createMenuEntry(fileMenu);

   openData->createShortcut(kKey_O, "CTRL", GetId());
   loadConfig->createShortcut(kKey_L, "CTRL", GetId());
   saveConfig->createShortcut(kKey_S, "CTRL", GetId());
   saveConfigAs->createShortcut(kKey_S, "CTRL+SHIFT", GetId());
   exportImage->createShortcut(kKey_P, "CTRL", GetId());
   // comment out the followinf one, seems to get double open file dialog events on OSX
   // exportImages->createShortcut(kKey_P, "CTRL+SHIFT", GetId());
   quit->createShortcut(kKey_Q, "CTRL", GetId());

   TGPopupMenu *editMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("Edit", editMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 2, 0));

   showCommonInsp->createMenuEntry(editMenu);
   showCommonInsp->createShortcut(kKey_A, "CTRL+SHIFT", GetId());
   colorset->createMenuEntry(editMenu);
   colorset->createShortcut(kKey_B, "CTRL", GetId());
   editMenu->AddSeparator();

   undo->createMenuEntry(editMenu);
   undo->createShortcut(kKey_Z, "CTRL", GetId());
   redo->createMenuEntry(editMenu);
   redo->createShortcut(kKey_Z, "CTRL+SHIFT", GetId());
   editMenu->AddSeparator();

   cut->createMenuEntry(editMenu);
   cut->createShortcut(kKey_X, "CTRL", GetId());
   copy->createMenuEntry(editMenu);
   copy->createShortcut(kKey_C, "CTRL", GetId());
   paste->createMenuEntry(editMenu);
   paste->createShortcut(kKey_V, "CTRL", GetId());

   TGPopupMenu *viewMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("View", viewMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 2, 0));  

   m_newViewerMenu = new TGPopupMenu(gClient->GetRoot());
   viewMenu->AddPopup("New Viewer", m_newViewerMenu);

   viewMenu->AddSeparator();

   nextEvent->createMenuEntry(viewMenu);
   nextEvent->createShortcut(kKey_Right, "CTRL", GetId());
   previousEvent->createMenuEntry(viewMenu);
   previousEvent->createShortcut(kKey_Left, "CTRL", GetId());
   goToFirst->createMenuEntry(viewMenu);
   goToLast->createMenuEntry(viewMenu);
   playEvents->createMenuEntry(viewMenu);
   playEvents->createShortcut(kKey_Space, "CTRL", GetId());
   playEventsBack->createMenuEntry(viewMenu);
   playEventsBack->createShortcut(kKey_Space, "CTRL+SHIFT", GetId());
   loop->createMenuEntry(viewMenu);

   TGPopupMenu* windowMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("Window", windowMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 2, 0));

   showCommonInsp->createMenuEntry(windowMenu);
   showObjInsp->createMenuEntry(windowMenu);
   showEventDisplayInsp->createShortcut(kKey_I, "CTRL", GetId());
   showEventDisplayInsp->createMenuEntry(windowMenu);
   showAddCollection->createMenuEntry(windowMenu);
   showMainViewCtl->createMenuEntry(windowMenu);
   showInvMassDialog->createMenuEntry(windowMenu);

   TGPopupMenu *helpMenu = new TGPopupMenu(gClient->GetRoot());
   menuBar->AddPopup("Help", helpMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 2, 0));
   help->createMenuEntry(helpMenu);
   keyboardShort->createMenuEntry(helpMenu);
   helpMenu->AddSeparator();
   helpGL->createMenuEntry(helpMenu);

   // colors
   menuBar->SetBackgroundColor(backgroundColor);
   TIter next(menuBar->GetTitles());
   TGMenuTitle *title;
   while ((title = (TGMenuTitle *)next()))
      title->SetTextColor(textColor);

   menuTopFrame->AddFrame(menuBar, new TGLayoutHints(kLHintsExpandX, 0, 0, 0, 0));
   AddFrame(menuTopFrame, new TGLayoutHints(kLHintsExpandX, 0, 0, 0, 0));

   // !!!! MT Line separating menu from other window components.
   // I would even remove it and squeeze the navigation buttons up.
   AddFrame(new TGFrame(this, 1, 1, kChildFrame, 0x503020),
            new TGLayoutHints(kLHintsExpandX, 0, 0, 0, 0));

   m_statBar = new TGStatusBar(this, this->GetWidth(), 12);
   AddFrame(m_statBar, new TGLayoutHints(kLHintsBottom | kLHintsExpandX));

   TGHorizontalFrame *fullbar = new TGHorizontalFrame(this, this->GetWidth(), 30,0, backgroundColor);

   /**************************************************************************/
   // controls

   TGCompositeFrame* controlFrame = new TGVerticalFrame(fullbar, 10, 20, 0, backgroundColor);

   TGCompositeFrame* buttonFrame = new TGHorizontalFrame(controlFrame, 10, 10, 0, backgroundColor);
   TImage *imgBtn  = TImage::Open(FWCheckBoxIcon::coreIcondir()+"slider-bg-up.png");
   buttonFrame->SetBackgroundPixmap(imgBtn->GetPixmap());


   goToFirst->createCustomIconsButton(buttonFrame,
                                      fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-gotofirst.png"),
                                      fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-gotofirst-over.png"),
                                      fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-gotofirst-disabled.png"),
                                      new TGLayoutHints(kLHintsCenterY| kLHintsLeft, 4, 3, 10, 0));

   playEventsBack->createCustomIconsButton(buttonFrame,
                                           fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-backward.png"),
                                           fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-backward-over.png"),
                                           fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-backward-disabled.png"),
                                           fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-pause.png"),
                                           fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-pause-over.png"),
                                           new TGLayoutHints(kLHintsCenterY| kLHintsLeft, 2, 3, 10, 0));

   previousEvent->createCustomIconsButton(buttonFrame,
                                          fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-stepback.png"),
                                          fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-stepback-over.png"),
                                          fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-stepback-disabled.png"),
                                          new TGLayoutHints(kLHintsCenterY| kLHintsLeft, 2, 3, 10, 0));

   nextEvent->createCustomIconsButton(buttonFrame,
                                      fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-stepforward.png"),
                                      fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-stepforward-over.png"),
                                      fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-stepforward-disabled.png"),
                                      new TGLayoutHints(kLHintsCenterY| kLHintsLeft, 2, 3, 10, 0));


   playEvents->createCustomIconsButton(buttonFrame,
                                       fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-forward.png"),
                                       fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-forward-over.png"),
                                       fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-forward-disabled.png"),
                                       fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-pause.png"),
                                       fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-pause-over.png"),
                                       new TGLayoutHints(kLHintsCenterY| kLHintsLeft, 2, 3, 10, 0));

   goToLast->createCustomIconsButton(buttonFrame,
                                     fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-gotolast.png"),
                                     fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-gotolast-over.png"),
                                     fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"button-gotolast-disabled.png"),
                                     new TGLayoutHints(kLHintsCenterY| kLHintsLeft, 2, 3, 10, 0));

   controlFrame->AddFrame(buttonFrame, new TGLayoutHints(kLHintsTop, 10, 0, 0, 0));

   /**************************************************************************/

   TGHorizontalFrame* sliderFrame = new TGHorizontalFrame(controlFrame, 10, 10, 0, backgroundColor);
   TImage *imgSld  = TImage::Open(FWCheckBoxIcon::coreIcondir()+"slider-bg-down.png");
   sliderFrame->SetBackgroundPixmap(imgSld->GetPixmap());
   TString sldBtn = FWCheckBoxIcon::coreIcondir() +"slider-button.png";

   m_delaySlider = new TGHSlider(sliderFrame, 109, kSlider1 | kScaleNo);
   sliderFrame->AddFrame(m_delaySlider, new TGLayoutHints(kLHintsTop | kLHintsLeft, 39, 8, 1, 3));
   m_delaySlider->SetRange(0, 10000);
   m_delaySlider->SetPosition(0);
   m_delaySlider->SetBackgroundColor(0x1a1a1a);
   m_delaySlider->ChangeSliderPic(sldBtn);

   controlFrame->AddFrame(sliderFrame, new TGLayoutHints(kLHintsTop, 10, 0, 0, 0));

   fullbar->AddFrame(controlFrame, new TGLayoutHints(kLHintsLeft, 2, 2, 5, 8));

   m_delaySliderListener =  new FWIntValueListener();
   TQObject::Connect(m_delaySlider, "PositionChanged(Int_t)", "FWIntValueListenerBase",  m_delaySliderListener, "setValue(Int_t)");

   //==============================================================================

   // delay label
   {
      TGVerticalFrame* delayFrame = new TGVerticalFrame(fullbar, 60, 10, 0, backgroundColor);

      TGLabel *label = new TGLabel(delayFrame, "Delay");
      label->SetTextJustify(kTextCenterX);
      label->SetTextColor(0xb3b3b3);
      label->SetBackgroundColor(backgroundColor);
      delayFrame->AddFrame(label, new TGLayoutHints(kLHintsTop | kLHintsCenterX, 0, 0, 22, 0));

      TGHorizontalFrame *labFixed = new TGHorizontalFrame(delayFrame, 70, 20, kFixedSize, backgroundColor);
      m_delayLabel = new TGLabel(labFixed, "0.0s");
      m_delayLabel->SetBackgroundColor(backgroundColor);
      m_delayLabel->SetTextJustify(kTextCenterX);
      m_delayLabel->SetTextColor(0xffffff);
      labFixed->AddFrame(m_delayLabel, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 0, 0, 0, 0));
      delayFrame->AddFrame(labFixed, new TGLayoutHints(kLHintsLeft, 0, 4, 0, 0));

      fullbar->AddFrame(delayFrame, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   }

   //==============================================================================

   // text/num entries

   Int_t entryHeight = 22;
   TGVerticalFrame *texts = new TGVerticalFrame(fullbar, 400, 10, 0, backgroundColor);

   // upper row
   {
      TGPack *runInfo = new TGPack(texts, 400, entryHeight, kFixedHeight);
      runInfo->SetVertical(kFALSE);
      runInfo->SetUseSplitters(kFALSE);
      runInfo->SetBackgroundColor(backgroundColor);

      TGHorizontalFrame *rLeft = new TGHorizontalFrame(runInfo, 1, entryHeight);
      makeFixedSizeLabel(rLeft, "Run", backgroundColor, 0xffffff, 26, entryHeight);
      m_runEntry = new FWNumberEntryField(rLeft, -1, 0, TGNumberFormat::kNESInteger, TGNumberFormat::kNEAPositive);
      rLeft->AddFrame(m_runEntry, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 0,8,0,0));
      runInfo->AddFrameWithWeight(rLeft, 0, 0.28);

      TGHorizontalFrame *rMid = new TGHorizontalFrame(runInfo, 1, entryHeight);
      makeFixedSizeLabel(rMid, "Lumi", backgroundColor, 0xffffff, 36, entryHeight);
      m_lumiEntry = new FWNumberEntryField(rMid, -1, 0, TGNumberFormat::kNESInteger, TGNumberFormat::kNEAPositive);
      rMid->AddFrame(m_lumiEntry, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 0,8,0,0));
      runInfo->AddFrameWithWeight(rMid, 0, 0.32);

      TGHorizontalFrame *rRight = new TGHorizontalFrame(runInfo, 1, entryHeight);
      makeFixedSizeLabel(rRight, "Event", backgroundColor, 0xffffff, 42, entryHeight);
      m_eventEntry = new FWNumberEntryField(rRight, -1, 0, TGNumberFormat::kNESInteger, TGNumberFormat::kNEAPositive);
      rRight->AddFrame(m_eventEntry, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 0,0,0,0));
      runInfo->AddFrameWithWeight(rRight, 0, 0.4);

      texts->AddFrame(runInfo, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,0,4));
   }

   // lower row
   {
      TGHorizontalFrame *filterFrame = new TGHorizontalFrame(texts, 400, entryHeight, 0, backgroundColor);
   
      // filter state Off
      m_filterIcons[0] = fClient->GetPicture("unchecked_t.xpm");
      m_filterIcons[1] = fClient->GetPicture("unchecked_t.xpm");
      m_filterIcons[2] = fClient->GetPicture("unchecked_dis_t.xpm");
   
      // filter state On
      m_filterIcons[3] = fClient->GetPicture("checked_t.xpm");
      m_filterIcons[4] = fClient->GetPicture("checked_t.xpm");
      m_filterIcons[5] = fClient->GetPicture("checked_dis_t.xpm");
   
      // filter withdrawn
      m_filterIcons[6] = fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "icon-alert-ltgraybg.png");
      m_filterIcons[7] = fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "icon-alert-ltgraybg-over.png");
      m_filterIcons[8] = fClient->GetPicture(FWCheckBoxIcon::coreIcondir() + "icon-alert-ltgraybg.png");
   
      m_filterEnableBtn = new FWCustomIconsButton(filterFrame, m_filterIcons[0], m_filterIcons[1], m_filterIcons[2]);
      m_filterEnableBtn->SetBackgroundColor(backgroundColor);
      m_filterEnableBtn->SetToolTipText("Enable/disable event filtering");
      filterFrame->AddFrame(m_filterEnableBtn, new TGLayoutHints(kLHintsLeft, 4,0,3,0));

      m_filterShowGUIBtn = new TGTextButton(filterFrame,"Event filtering is OFF");
      m_filterShowGUIBtn->ChangeOptions(kRaisedFrame);
      m_filterShowGUIBtn->SetBackgroundColor(backgroundColor);
      m_filterShowGUIBtn->SetTextColor(0xFFFFFF);
      m_filterShowGUIBtn->SetToolTipText("Edit filters");
      filterFrame->AddFrame(m_filterShowGUIBtn, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 6,7,0,0));

      texts->AddFrame(filterFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,4,0));
   }

   fullbar->AddFrame(texts, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 5, 5, 12, 0));

   //==============================================================================

   TGVerticalFrame *texts2 = new TGVerticalFrame(fullbar, 200, 44, kFixedSize, backgroundColor);

   // time
   m_timeText = new TGLabel(texts2, "...");
   m_timeText->SetTextJustify(kTextLeft);
   m_timeText->SetTextColor(0xffffff);
   m_timeText->SetBackgroundColor(backgroundColor);
   texts2->AddFrame(m_timeText, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,0,1));

   fullbar->AddFrame(texts2, new TGLayoutHints(kLHintsLeft, 5, 5, 16, 5));

   //==============================================================================

   //  logo
   {
      TGVerticalFrame* parentLogoFrame = new TGVerticalFrame(fullbar, 70, 53, kFixedSize); 
      parentLogoFrame->SetBackgroundColor(backgroundColor);
      fullbar->AddFrame(parentLogoFrame, new TGLayoutHints(kLHintsRight | kLHintsCenterY));

      TGVerticalFrame* logoFrame = new TGVerticalFrame(parentLogoFrame, 53, 53, kFixedSize);
      TImage *logoImg  = TImage::Open(FWCheckBoxIcon::coreIcondir() + "CMSRedOnBlackThick.png");
      logoFrame->SetBackgroundPixmap(logoImg->GetPixmap());
      parentLogoFrame->AddFrame(logoFrame, new TGLayoutHints(kLHintsRight | kLHintsCenterY, 0, 14, 0, 0));
   }
   {
      TGCompositeFrame *logoFrame = new TGCompositeFrame(this, 61, 23, kFixedSize | kHorizontalFrame, backgroundColor);
      FWCustomIconsButton *infoBut =
         new FWCustomIconsButton(logoFrame, fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"fireworksSmallGray.png"),
                                 fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"fireworksSmallGray-green.png"),
                                 fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"fireworksSmallGray-red.png"),
                                 fClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"fireworksSmallGray-red.png"));
      logoFrame->AddFrame(infoBut);
      infoBut->Connect("Clicked()", "CmsShowMainFrame", this, "showFWorksInfo()");
      //TImage *logoImg  = TImage::Open( FWCheckBoxIcon::coreIcondir() + "fireworksSmallGray.png");
      //logoFrame->SetBackgroundPixmap(logoImg->GetPixmap());
      menuTopFrame->AddFrame(logoFrame, new TGLayoutHints(kLHintsRight | kLHintsBottom, 0, 13, 3, 1));
   }
  
   //==============================================================================

   AddFrame(fullbar, new TGLayoutHints(kLHintsExpandX, 0, 0, 0, 0));

   //Start disabled
   goToFirst->disable();
   goToLast->disable();
   previousEvent->disable();
   nextEvent->disable();
   playEvents->disable();
   playEventsBack->disable();
   loop->disable();
   
   //NOTE: There appears to be a bug in ROOT such that creating a menu item and setting it as
   // disabled immediately is ignored.  Therefore we have to wait till here to actually get ROOT
   // to disable these menu items
   undo->disable();
   redo->disable();
   cut->disable();
   copy->disable();
   paste->disable();
   
   //==============================================================================

   FWPack *csArea = new FWPack(this);
   csArea->SetVertical(kFALSE);

   TGCompositeFrame *cf = m_manager->createList(csArea);
   csArea->AddFrameWithWeight(cf, 0, 20);

   TEveCompositeFrameInPack *slot = new TEveCompositeFrameInPack(csArea, 0, csArea);
   csArea->AddFrameWithWeight(slot, 0, 80);
   TEveWindowSlot *ew_slot = TEveWindow::CreateDefaultWindowSlot();
   ew_slot->PopulateEmptyFrame(slot);
   m_manager->createViews(ew_slot);

   AddFrame(csArea,new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY, 0, 0, 0, 2));
   csArea->MapSubwindows();

   SetWindowName("cmsShow");
}

// CmsShowMainFrame::CmsShowMainFrame(const CmsShowMainFrame& rhs)
// {
//    // do actual copying here;
// }

CmsShowMainFrame::~CmsShowMainFrame() {
   Cleanup();
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

CSGAction*
CmsShowMainFrame::createNewViewerAction(const std::string& iActionName, bool separator)
{
   CSGAction* action(new CSGAction(this, iActionName.c_str()));
   action->createMenuEntry(m_newViewerMenu);
   if (separator) m_newViewerMenu->AddSeparator();
   return action;
}

void CmsShowMainFrame::loadEvent(const edm::EventBase& event)
{
   m_runEntry  ->SetUIntNumber(event.id().run());
   m_lumiEntry ->SetUIntNumber(event.id().luminosityBlock());
   m_eventEntry->SetULong64Number(event.id().event());

   m_timeText->SetText( fireworks::getLocalTime( event ).c_str() );
}

void CmsShowMainFrame::enableNavigatorControls()
{
   m_nextEvent->enable();
   m_previousEvent->enable();
   m_goToFirst->enable();
   m_goToLast->enable();
   m_playEvents->enable();
   m_playEventsBack->enable();
   m_loopAction->enable();
}

void  CmsShowMainFrame::CloseWindow()
{
   getAction(cmsshow::sQuit)->activated();
}

void CmsShowMainFrame::quit() {
   getAction(cmsshow::sQuit)->activated();
}

void
CmsShowMainFrame::enableActions(bool enable)
{
   CSGActionSupervisor::enableActions(enable);

   m_runEntry->SetEnabled(enable);
   m_lumiEntry->SetEnabled(enable);
   m_eventEntry->SetEnabled(enable);
   m_filterEnableBtn->SetEnabled(enable);
   m_filterShowGUIBtn->SetEnabled(enable);
}

void
CmsShowMainFrame::enablePrevious(bool enable)
{
   if (m_previousEvent != 0) {
      if (enable) {
         m_previousEvent->enable();
         m_goToFirst->enable();
         m_playEventsBack->enable();
      } else {
         m_previousEvent->disable();
         m_goToFirst->disable();
         m_playEventsBack->disable();
         m_playEventsBack->stop();
      }
   }
}

void
CmsShowMainFrame::enableNext(bool enable)
{
   if (m_nextEvent != 0) {
      if (enable) {
         m_nextEvent->enable();
         m_goToLast->enable();
         m_playEvents->enable();
      } else {
         m_nextEvent->disable();
         m_goToLast->disable();
         m_playEvents->disable();
         m_playEvents->stop();
      }
   }
}

/** To disable GUI to jump from event to another,
    when this is not possible (i.e. when in full framework mode).
  */
void
CmsShowMainFrame::enableComplexNavigation(bool enable)
{
   if (enable)
      m_goToLast->enable();
   else
      m_goToLast->disable();
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
         fwLog(fwlog::kInfo) << "Invalid menu id\n";
         break;
   }
}

Bool_t CmsShowMainFrame::HandleKey(Event_t *event) {

   if (event->fType == kGKeyPress) {
      const std::vector<CSGAction*>& alist = getListOfActions();
      std::vector<CSGAction*>::const_iterator it_act;
      Int_t keycode;
      Int_t modcode;
      for (it_act = alist.begin(); it_act != alist.end(); ++it_act) {
         keycode = (*it_act)->getKeycode();
         modcode = (*it_act)->getModcode();
         if ((event->fCode == (UInt_t)keycode) &&
             ((event->fState == (UInt_t)modcode) ||
              (event->fState == (UInt_t)(modcode | kKeyMod2Mask)) ||
              (event->fState == (UInt_t)(modcode | kKeyLockMask)) ||
              (event->fState == (UInt_t)(modcode | kKeyMod2Mask | kKeyLockMask)))) {
            (*it_act)->activated.emit();
            //  return kTRUE;
            return false;
         }
      }

      // special case is --live option where Space key is grabbed
      static UInt_t spacecode =  gVirtualX->KeysymToKeycode((int)kKey_Space);
      if (event->fCode == spacecode && event->fState == 0 ) {
          if (playEventsAction()->isRunning() )
              playEventsAction()->switchMode();
          else if (playEventsBackwardsAction()->isRunning() )
              playEventsBackwardsAction()->switchMode();
      }
   }
   return kFALSE;
}

void
CmsShowMainFrame::setPlayDelayGUI(Float_t val, Bool_t sliderChanged)
{
   m_delayLabel->SetText(Form("%.1fs", val));
   if (sliderChanged)
      m_delaySlider->SetPosition(Int_t(val*1000));
}

void
CmsShowMainFrame::makeFixedSizeLabel(TGHorizontalFrame* p, const char* txt,
                                     UInt_t bgCol, UInt_t txtCol,
                                     Int_t  width, Int_t  height)
{
   // Utility function.


   p->SetBackgroundColor(bgCol);
   TGCompositeFrame *lframe = new TGHorizontalFrame(p, width, height, kFixedSize, bgCol);
   TGLabel* label = new TGLabel(lframe, txt);
   label->SetBackgroundColor(bgCol);
   label->SetTextColor(txtCol);
   lframe->AddFrame(label, new TGLayoutHints(kLHintsRight | kLHintsTop, 0, 4));
   p->AddFrame(lframe, new TGLayoutHints(kLHintsLeft, 0, 0, 3, 0));
}

class InfoFrame : public TGMainFrame {
public:
   InfoFrame(const TGWindow* p, UInt_t w, UInt_t h, UInt_t opts) : TGMainFrame(p, w, h, opts) {}
   virtual ~InfoFrame() {}
   
   virtual void CloseWindow() override
   {
      UnmapWindow();  
   }
};

void
CmsShowMainFrame::showFWorksInfo()
{
   if (m_fworksAbout == 0)
   {
      UInt_t ww = 280, hh = 190;
      int number_of_lines = 0;
      int fontSize = 8;
      TString infoText;
      if (gSystem->Getenv("CMSSW_VERSION"))
      {
         infoText = "Version ";
         infoText += gSystem->Getenv("CMSSW_VERSION");
         infoText +="\n";
         number_of_lines += 1;
      }
      else
      {
         TString infoFileName("/data/version.txt");
         fireworks::setPath(infoFileName);
         std::string line;
         std::ifstream infoFile(infoFileName);
         while (std::getline(infoFile, line))
         {
            ++number_of_lines;
            infoText += line.c_str();
            infoText += "\n";
         }
         infoFile.close();
      }
      infoText += "\nIt works or we fix it for free!\nhn-cms-visualization@cern.ch\n";

      hh = 130 + 2* fontSize*(number_of_lines + 1);
      
      m_fworksAbout = new InfoFrame(gClient->GetRoot(), ww, hh, kVerticalFrame | kFixedSize);
      m_fworksAbout->SetWMSizeHints(ww, hh, ww, hh, 0, 0);
      m_fworksAbout->SetBackgroundColor(0x2f2f2f);
      
      TGFrame* logoFrame = new TGFrame(m_fworksAbout, 140, 48, kFixedSize);
      TImage *logoImg  = TImage::Open(FWCheckBoxIcon::coreIcondir()+"logo-fireworks.png");
      logoFrame->SetBackgroundPixmap(logoImg->GetPixmap());
      m_fworksAbout->AddFrame(logoFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX, 0, 0, 16, 0));
      
      TGLabel* label = new TGLabel(m_fworksAbout, infoText);
      label->SetBackgroundColor(0x2f2f2f);
      label->SetForegroundColor(0xffffff);

      FontStruct_t defaultFontStruct = label->GetDefaultFontStruct();
      try
      {
         TGFontPool *pool = gClient->GetFontPool();
         TGFont* defaultFont = pool->GetFont(defaultFontStruct);
         FontAttributes_t attributes = defaultFont->GetFontAttributes();
         label->SetTextFont(pool->GetFont(attributes.fFamily, fontSize, 
                                          attributes.fWeight, attributes.fSlant));
      } 
      catch(...)
      {
      }

      m_fworksAbout->AddFrame(label, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY, 0, 0, 12, 0));      
            
      TGTextButton* btn = new TGTextButton(m_fworksAbout, "  OK  ");
      btn->SetBackgroundColor(0x2f2f2f);
      btn->SetForegroundColor(0xffffff);
      m_fworksAbout->AddFrame(btn, new TGLayoutHints(kLHintsBottom | kLHintsCenterX, 0, 0, 0, 12));
      btn->Connect("Clicked()", "TGMainFrame", m_fworksAbout, "CloseWindow()");

      m_fworksAbout->MapSubwindows();
      m_fworksAbout->Layout();
   }
   
   m_fworksAbout->MapRaised();
}


void
CmsShowMainFrame::bindCSGActionKeys(const TGMainFrame* f) const
{
   for (std::vector<CSGAction*>::const_iterator i = m_actionList.begin(); i != m_actionList.end(); ++i)
   {
      if ((*i)-> getKeycode())
         f->BindKey(this, (*i)->getKeycode(), (*i)->getModcode()); 
   }
}

void
CmsShowMainFrame::setSummaryViewWeight(float x)
{

   TGFrameElement* fe = (TGFrameElement*) GetList()->Last();
   FWPack* pack = (FWPack*)(fe->fFrame);

   TGFrameElementPack* fep;
   fep  = (TGFrameElementPack*)pack->GetList()->At(1);
   fep->fWeight = x;

   fep  = (TGFrameElementPack*)pack->GetList()->At(3);
   fep->fWeight = 100 -x;

   pack->ResizeExistingFrames();
   pack->Layout();
}

float
CmsShowMainFrame::getSummaryViewWeight() const
{
   TGFrameElement* fe = (TGFrameElement*)GetList()->Last();
   TGPack* pack = (TGPack*)(fe->fFrame);

   TGFrameElementPack* fep = (TGFrameElementPack*)pack->GetList()->At(1);
   return fep->fWeight;
      
}

// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGUIManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb 11 11:06:40 EST 2008
// $Id: FWGUIManager.cc,v 1.112 2009/04/12 19:55:21 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>
#include <stdexcept>
#include <iostream>
#include <sstream>

#include "TGButton.h"
#include "TGLabel.h"
#include "TGComboBox.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
#include "TApplication.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TGSplitFrame.h"
#include "TGTab.h"
#include "TG3DLine.h"
#include "TGListTree.h"
#include "TEveBrowser.h"
#include "TBrowser.h"
#include "TGMenu.h"
#include "TEveManager.h"
#include "TGPack.h"
//#include "TEveGedEditor.h"
#include "TEveWindow.h"
#include "TEveWindowManager.h"
#include "TEveSelection.h"
#include "TGFileDialog.h"
#include "TColor.h"
#include "TVirtualX.h"

// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWGUISubviewArea.h"

#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWSummaryManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWViewManagerManager.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/src/FWListEventItem.h"

#include "Fireworks/Core/src/FWListViewObject.h"
#include "Fireworks/Core/src/FWListModel.h"
#include "Fireworks/Core/src/FWListMultipleModels.h"

#include "Fireworks/Core/interface/FWConfiguration.h"

#include "Fireworks/Core/src/accessMenuBar.h"

#include "Fireworks/Core/interface/CmsShowMainFrame.h"

#include "Fireworks/Core/src/FWGUIEventDataAdder.h"

#include "Fireworks/Core/interface/CSGAction.h"

#include "Fireworks/Core/interface/ActionsList.h"

#include "Fireworks/Core/interface/CmsShowEDI.h"
#include "Fireworks/Core/interface/CmsShowModelPopup.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"

#include "Fireworks/Core/interface/CmsShowHelpPopup.h"

#include "Fireworks/Core/src/FWListWidget.h"

#include "Fireworks/Core/src/CmsShowTaskExecutor.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"

#include "Fireworks/Core/interface/FWTypeToRepresentations.h"
#include "Fireworks/Core/interface/FWIntValueListener.h"

//
// constants, enums and typedefs
//
enum {kSaveConfiguration,
      kSaveConfigurationAs,
      kExportImage,
      kQuit};

//
// static data member definitions
//
FWGUIManager* FWGUIManager::m_guiManager = 0;

//
// constructors and destructor
//
FWGUIManager::FWGUIManager(FWSelectionManager* iSelMgr,
                           FWEventItemsManager* iEIMgr,
                           FWModelChangeManager* iCMgr,
                           FWColorManager* iColorMgr,
                           const FWViewManagerManager* iVMMgr,
                           bool iDebugInterface
                           ) :
   m_selectionManager(iSelMgr),
   m_eiManager(iEIMgr),
   m_changeManager(iCMgr),
   m_colorManager(iColorMgr),
   m_presentEvent(0),
   m_continueProcessingEvents(false),
   m_waitForUserAction(true),
   m_code(0),
   m_editableSelected(0),
   m_detailViewManager(new FWDetailViewManager),
   m_viewManagerManager(iVMMgr),
   m_dataAdder(0),
   m_ediFrame(0),
   m_modelPopup(0),
   m_viewPopup(0),
   m_helpPopup(0),
   m_shortcutPopup(0),
   m_tasks(new CmsShowTaskExecutor)
{
   m_guiManager = this;
   m_selectionManager->selectionChanged_.connect(boost::bind(&FWGUIManager::selectionChanged,this,_1));
   m_eiManager->newItem_.connect(boost::bind(&FWGUIManager::newItem,
                                             this, _1) );

   m_colorManager->colorsHaveChangedFinished_.connect(boost::bind(&FWGUIManager::finishUpColorChange,this));
   // These are only needed temporarilty to work around a problem which
   // Matevz has patched in a later version of the code
   TApplication::NeedGraphicsLibs();
   gApplication->InitializeGraphics();

   TEveCompositeFrame::IconBarCreator_foo foo =  &FWGUIManager::makeGUIsubview;
   TEveCompositeFrame::SetupFrameMarkup(foo, 20, 4, false);

   TEveManager::Create(kFALSE, "FI");

   {
      //NOTE: by making sure we defaultly open to a fraction of the full screen size we avoid
      // causing the program to go into full screen mode under default SL4 window manager
      UInt_t width = gClient->GetDisplayWidth();
      UInt_t height = static_cast<UInt_t>(gClient->GetDisplayHeight()*.8);
      //try to deal with multiple horizontally placed monitors.  Since present monitors usually
      // have less than 2000 pixels horizontally, when we see more it is a good indicator that
      // we are dealing with more than one monitor.
      while(width > 2000) {
         width /= 2;
      }
      width = static_cast<UInt_t>(width*.8);
      m_cmsShowMainFrame = new CmsShowMainFrame(gClient->GetRoot(),
                                                width,
                                                height,
                                                this);
      m_cmsShowMainFrame->SetWindowName("CmsShow");
      m_cmsShowMainFrame->SetCleanup(kDeepCleanup);

      getAction(cmsshow::sExportImage)->activated.connect(sigc::mem_fun(*this, &FWGUIManager::exportImageOfMainView));
      getAction(cmsshow::sSaveConfig)->activated.connect(writeToPresentConfigurationFile_);
      getAction(cmsshow::sSaveConfigAs)->activated.connect(sigc::mem_fun(*this,&FWGUIManager::promptForConfigurationFile));
      getAction(cmsshow::sShowEventDisplayInsp)->activated.connect(boost::bind( &FWGUIManager::showEDIFrame,this,-1));
      getAction(cmsshow::sShowMainViewCtl)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::showViewPopup));
      getAction(cmsshow::sShowObjInsp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::showModelPopup));
      getAction(cmsshow::sShowAddCollection)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::addData));
      assert(getAction(cmsshow::sHelp) != 0);
      getAction(cmsshow::sHelp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createHelpPopup));
      assert(getAction(cmsshow::sKeyboardShort) != 0);
      getAction(cmsshow::sKeyboardShort)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createShortcutPopup));
      getAction(cmsshow::sBackgroundColor)->activated.connect(sigc::mem_fun(*this, &FWGUIManager::changeBackgroundColor));
      
      // toolbar special widget with non-void actions
      m_cmsShowMainFrame->m_delaySliderListener->valueChanged_.connect(boost::bind(&FWGUIManager::delaySliderChanged,this,_1));

      TQObject::Connect(m_cmsShowMainFrame->m_runEntry,"ReturnPressed()", "FWGUIManager", this, "runIdChanged()");
      TQObject::Connect(m_cmsShowMainFrame->m_eventEntry, "ReturnPressed()", "FWGUIManager", this, "eventIdChanged()");
      TQObject::Connect(m_cmsShowMainFrame->m_filterEntry, "ReturnPressed()", "FWGUIManager", this, "eventFilterChanged()");

      TQObject::Connect(gEve->GetWindowManager(), "WindowSelected(TEveWindow*)", "FWGUIManager", this, "subviewCurrentChanged(TEveWindow*)");
   }
}

//
// Destructor
//
FWGUIManager::~FWGUIManager()
{
   delete m_summaryManager;
   delete m_detailViewManager;
   delete m_editableSelected;
   delete m_cmsShowMainFrame;
   delete m_ediFrame;
}

void
FWGUIManager::evePreTerminate()
{
   //gDebug = 1;
   gEve->GetWindowManager()->Disconnect("WindowSelected(TEveWindow*)", this, "subviewCurrentChanged(TEveWindow*)");
   m_cmsShowMainFrame->UnmapWindow();

   for(std::vector<FWViewBase* >::iterator it = m_viewBases.begin(), itEnd = m_viewBases.end();
        it != itEnd;
        ++it) {
       (*it)->destroy();
    }
}

//______________________________________________________________________________
// subviews construction
//

TGFrame*
FWGUIManager::makeGUIsubview(TEveCompositeFrame* cp, TGCompositeFrame* parent, Int_t height)
{
   TGFrame* frame = new FWGUISubviewArea(cp, parent, height);
   return frame;
}

void
FWGUIManager::registerViewBuilder(const std::string& iName,
                                  ViewBuildFunctor& iBuilder)
{
   m_nameToViewBuilder[iName]=iBuilder;
   CSGAction* action=m_cmsShowMainFrame->createNewViewerAction(iName);
   action->activated.connect(boost::bind(&FWGUIManager::createView,this,iName));
}

TEveWindowSlot*
FWGUIManager::parentForNextView()
{
   TEveWindowSlot* slot = 0;
   if (m_viewSecPack == 0) {
      slot = m_viewPrimPack->NewSlot();
      getGUISubviewArea(slot)->configurePrimaryView();
      m_viewSecPack = m_viewPrimPack->NewSlot()->MakePack();
      m_viewSecPack->SetElementName("VerticalPack");
      m_viewSecPack->SetVertical();
      m_viewSecPack->SetShowTitleBar(kFALSE);
   } else {
      slot = m_viewSecPack->NewSlot();
   }
   return slot;
}

void
FWGUIManager::createView(const std::string& iName)
{
   NameToViewBuilder::iterator itFind = m_nameToViewBuilder.find(iName);
   assert (itFind != m_nameToViewBuilder.end());
   if(itFind == m_nameToViewBuilder.end()) {
      throw std::runtime_error(std::string("Unable to create view named ")+iName+" because it is unknown");
   }

   TEveWindowSlot* slot = parentForNextView();
   TEveCompositeFrame *ef = slot->GetEveFrame();

   FWViewBase* view = itFind->second(slot);

   TEveWindow *ew = ef->GetEveWindow();
   ew->SetElementName(iName.c_str());
   ew->SetUserData(view);

   m_viewWindows.push_back(ew);
   m_viewBases.push_back(view);
}


//
// actions
//

void
FWGUIManager::enableActions(bool enable)
{
   m_cmsShowMainFrame->enableActions(enable);
}

void
FWGUIManager::newFile(const TFile* iFile)
{
   m_openFile = iFile;
}

void
FWGUIManager::loadEvent(const fwlite::Event& event) {
   // To be replaced when we can get index from fwlite::Event
   m_cmsShowMainFrame->loadEvent(event);
   m_presentEvent=&event;
   if(m_dataAdder) {
      m_dataAdder->update(m_openFile, &event);
   }
}

CSGAction*
FWGUIManager::getAction(const std::string name)
{
   return m_cmsShowMainFrame->getAction(name);
}

CSGContinuousAction*
FWGUIManager::playEventsAction()
{
   return m_cmsShowMainFrame->playEventsAction();
}

CSGContinuousAction*
FWGUIManager::playEventsBackwardsAction()
{
   return m_cmsShowMainFrame->playEventsBackwardsAction();
}

void
FWGUIManager::disablePrevious()
{
   m_cmsShowMainFrame->enablePrevious(false);
}

void
FWGUIManager::disableNext()
{
   m_cmsShowMainFrame->enableNext(false);
}

void
FWGUIManager::setPlayMode(bool play)
{
   m_cmsShowMainFrame->m_runEntry->SetState(!play);
   m_cmsShowMainFrame->m_eventEntry->SetState(!play);
   m_cmsShowMainFrame->m_filterEntry->SetState(!play);
}

void
FWGUIManager::updateStatus(const char* status) {
   m_cmsShowMainFrame->updateStatusBar(status);
}

void
FWGUIManager::clearStatus()
{
   m_cmsShowMainFrame->clearStatusBar();
}

void
FWGUIManager::selectByExpression()
{
   FWModelExpressionSelector selector;
   selector.select(*(m_eiManager->begin()+m_selectionItemsComboBox->GetSelected()),
                   m_selectionExpressionEntry->GetText());
}

void
FWGUIManager::unselectAll()
{
   m_selectionManager->clearSelection();
}

void
FWGUIManager::selectionChanged(const FWSelectionManager& iSM)
{
   //open the modify window the first time someone selects something
   if (m_modelPopup == 0) createModelPopup();
}

void
FWGUIManager::processGUIEvents()
{
   gSystem->ProcessEvents();
}

void
FWGUIManager::newItem(const FWEventItem* iItem)
{
#if defined(THIS_WILL_NEVER_BE_DEFINED)
   m_selectionItemsComboBox->AddEntry(iItem->name().c_str(),iItem->id());
   if(iItem->id()==0) {
      m_selectionItemsComboBox->Select(0);
   }
#endif
}

void
FWGUIManager::addData()
{
   if(0==m_dataAdder) {
      m_dataAdder = new FWGUIEventDataAdder(100,100,
                                            m_eiManager,
                                            m_cmsShowMainFrame,
                                            m_presentEvent,
                                            m_openFile,
                                            m_viewManagerManager->supportedTypesAndRepresentations());
   }
   m_dataAdder->show();
}


//  subview actions
//
void
FWGUIManager::subviewCurrentChanged(TEveWindow*)
{
   for (std::vector<TEveWindow*>::iterator it = m_viewWindows.begin(); it != m_viewWindows.end(); it++)
   {
      FWGUISubviewArea* ar = getGUISubviewArea(*it);
      if (ar) ar->currentWindowChanged();
   }
}

void
FWGUIManager::subviewIsBeingDestroyed(FWGUISubviewArea* sva)
{
   if(sva->isSelected()) {
      if(0!= m_viewPopup) {refillViewPopup(0);}
   }

   CmsShowTaskExecutor::TaskFunctor f;
   f = boost::bind(&FWGUIManager::subviewDestroy, this, sva);
   m_tasks->addTask(f);
   m_tasks->startDoingTasks();
}

void
FWGUIManager::subviewDestroy(FWGUISubviewArea* sva)
{
   std::vector<TEveWindow*>::iterator itw = 
      std::find(m_viewWindows.begin(), m_viewWindows.end(), sva->getEveWindow());
   if (itw == m_viewWindows.end()) 
   {
      return;
   }
   m_viewWindows.erase(itw);
   


   FWViewBase* v = sva->getFWView(); // get view base from user data
   std::vector<FWViewBase*>::iterator itFind = std::find(m_viewBases.begin(), m_viewBases.end(), v);
   m_viewBases.erase(itFind);
   v->destroy();

}

void
FWGUIManager::subviewSelected(FWGUISubviewArea* sva)
{
   showViewPopup();
   refillViewPopup(sva->getFWView());
}

void
FWGUIManager::subviewUnselected(FWGUISubviewArea* /*sva*/)
{
   if(m_viewPopup) {refillViewPopup(0);}
}

void
FWGUIManager::subviewSwapped(FWGUISubviewArea* sva)
{
   // if current selected swap with current
   if (gEve->GetWindowManager()->GetCurrentWindow())
   {
      sva->getEveWindow()->SwapWindowWithCurrent();
      subviewCurrentChanged(sva->getEveWindow());
   }
   else
   {
      // swap with big view
      TGPack* pp = m_viewPrimPack->GetPack();
      TGFrameElement *pel = (TGFrameElement*) pp->GetList()->First();
      TEveCompositeFrame* pef = dynamic_cast<TEveCompositeFrame*>(pel->fFrame);
      TEveWindow::SwapWindows(sva->getEveWindow(), pef->GetEveWindow());
   }
}


TGVerticalFrame*
FWGUIManager::createList(TGSplitFrame *p)
{
   TGVerticalFrame *listFrame = new TGVerticalFrame(p, p->GetWidth(), p->GetHeight());

   TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));

   TGHorizontalFrame* addFrame = new TGHorizontalFrame(p,p->GetWidth(), 10);
   TGLabel* addLabel = new TGLabel(addFrame,"Summary View");
   addLabel->SetTextJustify(kTextLeft);

   addFrame->AddFrame(addLabel, new TGLayoutHints(kLHintsCenterY|kLHintsLeft|kLHintsExpandX,2,2,2,2));
   FWCustomIconsButton* addDataButton = new FWCustomIconsButton(addFrame,
                                                                gClient->GetPicture(coreIcondir+"plus-sign.png"),
                                                                gClient->GetPicture(coreIcondir+"plus-sign-over.png"),
                                                                gClient->GetPicture(coreIcondir+"plus-sign-disabled.png"));
   addDataButton->SetToolTipText("Show additional collections");
   addDataButton->Connect("Clicked()", "FWGUIManager", this, "addData()");
   addFrame->AddFrame(addDataButton, new TGLayoutHints(kLHintsCenterY|kLHintsLeft,2,2,2,2));
   listFrame->AddFrame(addFrame, new TGLayoutHints(kLHintsExpandX|kLHintsLeft|kLHintsTop,2,2,2,2));


   m_summaryManager = new FWSummaryManager(listFrame,
                                           m_selectionManager,
                                           m_eiManager,
                                           this,
                                           m_changeManager,
                                           m_colorManager);
   
   listFrame->AddFrame(m_summaryManager->widget(), new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));

   return listFrame;
}

void
FWGUIManager::createViews(TGTab *tab)
{
   m_viewPrimPack = TEveWindow::CreateWindowInTab(tab)->MakePack();
   m_viewPrimPack->SetHorizontal();
   m_viewPrimPack->SetElementName("Views");
   m_viewPrimPack->SetShowTitleBar(kFALSE);
   m_viewSecPack = 0;

   // debug
   gEve->AddElement( m_viewPrimPack);
}

void
FWGUIManager::createEDIFrame() {
   if (m_ediFrame == 0) {
      m_ediFrame = new CmsShowEDI(m_cmsShowMainFrame, 200, 200, m_selectionManager,m_colorManager);
      m_ediFrame->Connect("CloseWindow()", "FWGUIManager", this, "resetEDIFrame()");
      m_ediFrame->CenterOnParent(kTRUE,TGTransientFrame::kTopRight);
   }
}

void
FWGUIManager::updateEDI(FWEventItem* iItem) {
   createEDIFrame();
   m_ediFrame->fillEDIFrame(iItem);
}

void
FWGUIManager::resetEDIFrame() {
   m_ediFrame->DontCallClose();
   m_ediFrame->UnmapWindow();
}

void
FWGUIManager::showEDIFrame(int iToShow)
{
   createEDIFrame();
   if(-1 != iToShow) {
      m_ediFrame->show(static_cast<FWDataCategories> (iToShow));
   }
   m_ediFrame->MapWindow();
}


void
FWGUIManager::createModelPopup()
{
   m_modelPopup = new CmsShowModelPopup(m_detailViewManager,m_selectionManager, m_colorManager, m_cmsShowMainFrame, 200, 200);
   m_modelPopup->Connect("CloseWindow()", "FWGUIManager", this, "resetModelPopup()");
   m_modelPopup->CenterOnParent(kTRUE,TGTransientFrame::kRight);
}

void
FWGUIManager::showModelPopup()
{
   if (!m_modelPopup) createModelPopup();
   m_modelPopup->MapWindow();
}

void
FWGUIManager::resetModelPopup() {
   m_modelPopup->DontCallClose();
   m_modelPopup->UnmapWindow();
}

void
FWGUIManager::createViewPopup() {
   if (m_viewPopup == 0) {
      m_viewPopup = new CmsShowViewPopup(m_cmsShowMainFrame, 200, 200, m_colorManager, m_viewBases[0]);
      m_viewPopup->Connect("CloseWindow()", "FWGUIManager", this, "resetViewPopup()");
      m_viewPopup->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   /* seems to work but a small scale test caused seg faults
      Int_t x,y;
      UInt_t w,h;
      gVirtualX->GetWindowSize(m_viewPopup->GetId(),
                          x,y,w,h);
      m_viewPopup->SetWMPosition(x,y);
      std::cout <<x<<" "<< y<<std::endl;
    */
}

void
FWGUIManager::refillViewPopup(FWViewBase* iView) {
   m_viewPopup->reset(iView);
}

void
FWGUIManager::resetViewPopup() {
   m_viewPopup->DontCallClose();
   m_viewPopup->UnmapWindow();
}

void
FWGUIManager::showViewPopup() {
   createViewPopup();
   m_viewPopup->MapWindow();
}

void FWGUIManager::createHelpPopup ()
{
   if (m_helpPopup == 0) {
      m_helpPopup = new CmsShowHelpPopup("help.html", "CmsShow Help",
                                         m_cmsShowMainFrame,
                                         800, 600);
      m_helpPopup->Connect("CloseWindow()", "FWGUIManager", this,
                           "resetHelpPopup()");
      m_helpPopup->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   m_helpPopup->MapWindow();
}

void FWGUIManager::resetHelpPopup ()
{
   m_helpPopup->DontCallClose();
   m_helpPopup->UnmapWindow();
}

void FWGUIManager::createShortcutPopup ()
{
   if (m_shortcutPopup == 0) {
      m_shortcutPopup = new CmsShowHelpPopup("shortcuts.html",
                                             "Keyboard Shortcuts",
                                             m_cmsShowMainFrame, 800, 600);
      m_shortcutPopup->Connect("CloseWindow()", "FWGUIManager", this,
                               "resetShortcutPopup()");
      m_shortcutPopup->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   m_shortcutPopup->MapWindow();
}

void FWGUIManager::resetShortcutPopup ()
{
   m_shortcutPopup->DontCallClose();
   m_shortcutPopup->UnmapWindow();
}

void FWGUIManager::createTextView (TGTab *p)
{
   m_textViewTab = p;
   m_textViewFrame[0] = p->AddTab("Physics objects");
   //printf("current tab: %d\n", p->GetCurrent());
   m_textViewFrame[1] = p->AddTab("Triggers");
   //printf("current tab: %d\n", p->GetCurrent());
   m_textViewFrame[2] = p->AddTab("Tracking");
   //printf("current tab: %d\n", p->GetCurrent());

   const unsigned int kTabColor=0x5f5f5f;
   TGTabElement *tabel = 0;
   tabel = p->GetTabTab("Physics objects");
   tabel->SetBackgroundColor(kTabColor);
   tabel = p->GetTabTab("Triggers");
   tabel->SetBackgroundColor(kTabColor);
   tabel = p->GetTabTab("Tracking");
   tabel->SetBackgroundColor(kTabColor);
   tabel = p->GetTabTab("Views");
   tabel->SetBackgroundColor(kTabColor);

   p->MapSubwindows();
   p->MapWindow();
}

//
// const member functions
//

FWGUIManager*
FWGUIManager::getGUIManager()
{
   return m_guiManager;
}

FWGUISubviewArea*
FWGUIManager::getGUISubviewArea(TEveWindow* w)
{
   // horizontal frame
   TGFrameElement *el = (TGFrameElement*) w->GetEveFrame()->GetList()->First();
   TGCompositeFrame* hf = (TGCompositeFrame*)el->fFrame;
   // subview last in the horizontal frame
   el = (TGFrameElement*)hf->GetList()->Last();
   FWGUISubviewArea* ar = dynamic_cast<FWGUISubviewArea*>(el->fFrame);
   return ar;
}

void
FWGUIManager::itemChecked(TObject* obj, Bool_t state)
{
}
void
FWGUIManager::itemClicked(TGListTreeItem *item, Int_t btn,  UInt_t mask, Int_t x, Int_t y)
{
   TEveElement* el = static_cast<TEveElement*>(item->GetUserData());
   FWListItemBase* lib = dynamic_cast<FWListItemBase*>(el);
   //assert(0!=lib);
   if(3==btn) {
      //open the appropriate controller window
   }
   //NOTE: the return of doSelection is 'true' if this is a collection, else it returns false
   if(1==btn || 3==btn) {
      if(lib) {
         bool isCollection =lib->doSelection(mask&kKeyControlMask);
         if(3==btn) {
            if(isCollection) {
               showEDIFrame();
            } else {
               showModelPopup();
            }
         }
         if(isCollection) {
            gEve->GetSelection()->UserPickedElement(el,mask&kKeyControlMask);
         }
      }
   }
}
void
FWGUIManager::itemDblClicked(TGListTreeItem* item, Int_t btn)
{
}
void
FWGUIManager::itemKeyPress(TGListTreeItem *entry, UInt_t keysym, UInt_t mask)
{
}

void
FWGUIManager::itemBelowMouse(TGListTreeItem* item, UInt_t)
{
}

void
FWGUIManager::promptForConfigurationFile()
{
   static const char* kSaveFileTypes[] = {"Fireworks Configuration files","*.fwc",
                                          "All Files","*",
                                          0,0};

   static TString dir(".");

   TGFileInfo fi;
   fi.fFileTypes = kSaveFileTypes;
   fi.fIniDir    = StrDup(dir);
   new TGFileDialog(gClient->GetDefaultRoot(), m_cmsShowMainFrame,
                    kFDSave,&fi);
   dir = fi.fIniDir;
   if (fi.fFilename != 0) { // to handle "cancel" button properly
      std::string name = fi.fFilename;
      // if the extension isn't already specified by hand, specify it now
      std::string ext = kSaveFileTypes[fi.fFileTypeIdx + 1] + 1;
      if (ext.size() != 0 && name.find(ext) == name.npos)
         name += ext;
      writeToConfigurationFile_(name);
   }
}

void
FWGUIManager::exportImageOfMainView()
{
   if(m_viewBases.size()) {
      m_viewBases[0]->promptForSaveImageTo(m_cmsShowMainFrame);
   }
}


static const std::string kMainWindow("main window");
static const std::string kViews("views");
static const std::string kViewArea("view area");
static const std::string kUndocked("undocked views");
static const std::string kControllers("controllers");
static const std::string kCollectionController("collection");
static const std::string kViewController("view");
static const std::string kObjectController("object");
static const std::string kBackgroundColor("background color");
static const std::string kColorControl("color control");

static
void
addWindowInfoTo(const TGFrame* iMain,
                FWConfiguration& oTo)
{
   Window_t wdummy;
   Int_t ax,ay;
   gVirtualX->TranslateCoordinates(iMain->GetId(),
                                   gClient->GetDefaultRoot()->GetId(),
                                   0,0, //0,0 in local coordinates
                                   ax,ay, //coordinates of screen
                                   wdummy);
   {
      std::stringstream s;
      s<<ax;
      oTo.addKeyValue("x",FWConfiguration(s.str()));
   }
   {
      std::stringstream s;
      s<<ay;
      oTo.addKeyValue("y",FWConfiguration(s.str()));
   }
   {
      std::stringstream s;
      s<<iMain->GetWidth();
      oTo.addKeyValue("width",FWConfiguration(s.str()));
   }
   {
      std::stringstream s;
      s<<iMain->GetHeight();
      oTo.addKeyValue("height",FWConfiguration(s.str()));
   }
}

void
FWGUIManager::addTo(FWConfiguration& oTo) const
{
   FWConfiguration mainWindow(1);
   {
      std::stringstream s;
      s << static_cast<int>(m_cmsShowMainFrame->GetWidth());
      mainWindow.addKeyValue("width",FWConfiguration(s.str()));
   }
   {
      std::stringstream s;
      s << static_cast<int>(m_cmsShowMainFrame->GetHeight());
      mainWindow.addKeyValue("height",FWConfiguration(s.str()));
   }
   Window_t wdummy;
   Int_t ax,ay;
   gVirtualX->TranslateCoordinates(m_cmsShowMainFrame->GetId(),
                                   gClient->GetDefaultRoot()->GetId(),
                                   0,0, //0,0 in local coordinates
                                   ax,ay, //coordinates of screen
                                   wdummy);
   {
      std::stringstream s;
      s<<ax;
      mainWindow.addKeyValue("x",FWConfiguration(s.str()));
   }
   {
      std::stringstream s;
      s<<ay;
      mainWindow.addKeyValue("y",FWConfiguration(s.str()));
   }
   printf("Set main coordinates (%d %d) \n", ax, ay);
   oTo.addKeyValue(kMainWindow,mainWindow,true);


   // sort list of TEveWindows reading frame list from TGCompositeFrame
   // becuse TEveElement list is not ordered
   std::vector<TEveWindow*> wpacked;
   {
      // primary pack
      TGPack* pp = m_viewPrimPack->GetPack();
      TGFrameElement *pel = (TGFrameElement*) pp->GetList()->First();
      TEveCompositeFrame* pef = dynamic_cast<TEveCompositeFrame*>(pel->fFrame);
      if (pef)
      {
         //  printf("eve window %s \n", pef->GetEveWindow()->GetElementName());
         wpacked.push_back( pef->GetEveWindow());
      }

      // secondary pack
      TGPack* sp = m_viewSecPack->GetPack();
      Int_t nf = sp->GetList()->GetSize();
      TIter frame_iterator(sp->GetList());
      for (Int_t i=0; i<nf; ++i)
      {
         TGFrameElement *sel = (TGFrameElement*)frame_iterator();
         TEveCompositeFrame *sef = dynamic_cast<TEveCompositeFrame*>(sel->fFrame);
         if (sef)
         {
            // printf("eve window %s \n", sef->GetEveWindow()->GetElementName());
            wpacked.push_back( sef->GetEveWindow());
         }
      }
   }

   // use sorted list to weite view area and FW-views configuration
   FWConfiguration views(1);
   FWConfiguration viewArea(1);
   for(std::vector<TEveWindow*>::const_iterator it = wpacked.begin(); it != wpacked.end(); ++it)
   {
      // FW-view-base
      FWConfiguration tempWiew(1);
      FWViewBase* wb = (FWViewBase*)((*it)->GetUserData());
      wb->addTo(tempWiew);
      views.addKeyValue(wb->typeName(), tempWiew, true);
      // printf("AddTo viewes @@@ wpacked view %s \n", wb->typeName().c_str());

      // view area
      std::stringstream s;
      TGFrame* frame = (*it)->GetGUIFrame();
      int dim;
      if (it ==  wpacked.begin())
         dim = frame->GetWidth();
      else
         dim = frame->GetHeight();
      s<< static_cast<int>(dim);
      viewArea.addValue(s.str());
      // printf("AddTo varea @@@ wpacked view %d \n", dim);
   }
   oTo.addKeyValue(kViews, views, true);
   oTo.addKeyValue(kViewArea, viewArea, true);

   //remember undocked
   FWConfiguration undocked(1);
   {
      
      for(std::vector<TEveWindow*>::const_iterator it = m_viewWindows.begin(); it != m_viewWindows.end(); ++it)
      {
         TEveWindow* ew = (*it);
         TEveCompositeFrameInMainFrame* mainFrame = dynamic_cast<TEveCompositeFrameInMainFrame*>(ew->GetEveFrame());
         if (mainFrame)
         {
            FWConfiguration tempFrame(1);
            addWindowInfoTo(mainFrame,tempFrame);
            undocked.addKeyValue(ew->GetName(), tempFrame, true);
            // printf("AddTo @@@ undocked %s \n", ew->GetElementName());
         }
      }
   }
   oTo.addKeyValue(kUndocked,undocked,true);


   //Remember where controllers were placed if they are open
   FWConfiguration controllers(1);
   {
      if(0!=m_ediFrame && m_ediFrame->IsMapped()) {
         FWConfiguration temp(1);
         addWindowInfoTo(m_ediFrame, temp);
         controllers.addKeyValue(kCollectionController,temp,true);
      }
      if(0!=m_viewPopup && m_viewPopup->IsMapped()) {
         FWConfiguration temp(1);
         addWindowInfoTo(m_viewPopup, temp);
         controllers.addKeyValue(kViewController,temp,true);
      }
      if(0!=m_modelPopup && m_modelPopup->IsMapped()) {
         FWConfiguration temp(1);
         addWindowInfoTo(m_modelPopup, temp);
         controllers.addKeyValue(kObjectController,temp,true);
      }
   }
   oTo.addKeyValue(kControllers,controllers,true);
   
   //Remember what color is the background
   FWConfiguration colorControl(1);
   {
      if(FWColorManager::kBlackIndex==m_colorManager->backgroundColorIndex()) {
         colorControl.addKeyValue(kBackgroundColor,FWConfiguration("black"));
      } else {
         colorControl.addKeyValue(kBackgroundColor,FWConfiguration("white"));
      }
   }
   oTo.addKeyValue(kColorControl,colorControl,true);
}

static
void
setWindowInfoFrom(const FWConfiguration& iFrom,
                  TGFrame* iFrame)
{
   int x = atoi(iFrom.valueForKey("x")->value().c_str());
   int y = atoi(iFrom.valueForKey("y")->value().c_str());
   int width = atoi(iFrom.valueForKey("width")->value().c_str());
   int height = atoi(iFrom.valueForKey("height")->value().c_str());
   iFrame->MoveResize(x,y,width,height);
}

void
FWGUIManager::setFrom(const FWConfiguration& iFrom)
{
   // main window
   {
      const FWConfiguration* mw = iFrom.valueForKey(kMainWindow);
      assert(mw != 0);
      int width,height;
      int x=0;
      int y=0;

      const FWConfiguration* cWidth = mw->valueForKey("width");
      assert(0 != cWidth);
      std::stringstream sw(cWidth->value());
      sw >> width;
      const FWConfiguration* c = mw->valueForKey("height");
      assert(0 != c);
      std::stringstream sh(c->value());
      sh >> height;

      const FWConfiguration* cX = mw->valueForKey("x");
      if( 0!=cX ) {
         std::stringstream sx(cX->value());
         sx >> x;
      }  
      const FWConfiguration* cY = mw->valueForKey("y");
      if(0 != cY) {
         std::stringstream sy(cY->value());
         sy >> y;
      }

      m_cmsShowMainFrame->MoveResize(x, y, width, height);
      m_cmsShowMainFrame->SetWMPosition(x, y);
   }

   // !! when position and size is clear map window
   m_viewPrimPack->GetGUIFrame()->Layout();
   m_cmsShowMainFrame->MapWindow();

   // configure the views
   const FWConfiguration* views = iFrom.valueForKey(kViews);
   assert(0!=views);
   const FWConfiguration::KeyValues* keyVals = views->keyValues();
   assert(0!=keyVals);
   for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it!= keyVals->end(); ++it)
   {
      size_t n = m_viewBases.size();
      createView(it->first);
      // printf("SetFrom @@@  view %s \n", (it->first).c_str());
      assert(n+1 == m_viewBases.size());
      m_viewBases.back()->setFrom(it->second);
   }


   // view Area
   // currently not supported, print info for debug
   if (1)  {
      const FWConfiguration* viewArea = iFrom.valueForKey(kViewArea);
      // assert(0!=viewArea);
      if (viewArea)
      {
         const FWConfiguration::StringValues* sVals = viewArea->stringValues();
         int idx = 0;
         int dim;
         for(FWConfiguration::StringValuesIt it = sVals->begin(); it != sVals->end(); ++it)
         { 
            std::stringstream s(*it);
            s >> dim;
            // printf("setFrom [%d] dim %d \n", idx, dim);
            idx ++;
         }
      }
   }

   // undocked windows
   const FWConfiguration* undocked = iFrom.valueForKey(kUndocked);
   if(0!=undocked) {
      const FWConfiguration::KeyValues* keyVals = undocked->keyValues();
      if(0!=keyVals) {
         for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it != keyVals->end(); ++it)
         {
            int x = atoi(it->second.valueForKey("x")->value().c_str());
            int y = atoi(it->second.valueForKey("y")->value().c_str());
            int width = atoi(it->second.valueForKey("width")->value().c_str());
            int height = atoi(it->second.valueForKey("height")->value().c_str());

            createView(it->first);
            TEveWindow* myw = m_viewWindows.back();
            myw->UndockWindowDestroySlot();
            TEveCompositeFrameInMainFrame* emf = dynamic_cast<TEveCompositeFrameInMainFrame*>(myw->GetEveFrame());
            const TGMainFrame* mf =  dynamic_cast<const TGMainFrame*>(emf->GetParent());
            TGMainFrame* mfp = (TGMainFrame*)mf;
            mfp->MoveResize(x, y, width, height);

            // printf("setFrom (%d, %d)  (%d, %d) \n", x, y, width, height);
            // printf("SetFrom @@@ undock %s %p\n", myw->GetElementName(), mf);
         }
      }
   }

   //handle controllers
   const FWConfiguration* controllers = iFrom.valueForKey(kControllers);
   if(0!=controllers) {
      const FWConfiguration::KeyValues* keyVals = controllers->keyValues();
      if(0!=keyVals) {
         //we have open controllers
         for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it != keyVals->end(); ++it)
         {
            const std::string& controllerName = it->first;
            std::cout <<"found controller "<<controllerName<<std::endl;
            if(controllerName == kCollectionController) {
               showEDIFrame();
               setWindowInfoFrom(it->second,m_ediFrame);
            } else if (controllerName == kViewController) {
               showViewPopup();
               setWindowInfoFrom(it->second, m_viewPopup);
            } else if (controllerName == kObjectController) {
               showModelPopup();
               setWindowInfoFrom(it->second, m_modelPopup);
            }
         }
      }
   }
   const FWConfiguration* colorControl = iFrom.valueForKey(kColorControl);
   if(0!=colorControl) {
      if("black" == colorControl->valueForKey(kBackgroundColor)->value()) {
         m_colorManager->setBackgroundColorIndex( FWColorManager::kBlackIndex);
      } else {
         m_colorManager->setBackgroundColorIndex( FWColorManager::kWhiteIndex);
      }
   }
}

void
FWGUIManager::openEveBrowserForDebugging() const
{
   gEve->GetBrowser()->MapWindow();
}

//
// toolbar widgets callbacks
//
void
FWGUIManager::delaySliderChanged(Int_t val)
{
   Float_t sec = val*0.001;
   m_cmsShowMainFrame->setPlayDelayGUI(sec, kFALSE);
   changedDelayBetweenEvents_.emit(sec);
}

void
FWGUIManager::setDelayBetweenEvents(Float_t val)
{
   m_cmsShowMainFrame->setPlayDelayGUI(val, kTRUE);
}

void FWGUIManager::runIdChanged()
{
   changedRunId_.emit(m_cmsShowMainFrame->m_runEntry->GetIntNumber());
}

void FWGUIManager::eventIdChanged()
{
   changedEventId_.emit(m_cmsShowMainFrame->m_eventEntry->GetIntNumber());
}

void FWGUIManager::eventFilterChanged()
{
   changedEventFilter_.emit(m_cmsShowMainFrame->m_filterEntry->GetText());
}

void 
FWGUIManager::changeBackgroundColor()
{
   if(FWColorManager::kBlackIndex == m_colorManager->backgroundColorIndex()) {
      m_colorManager->setBackgroundColorIndex(FWColorManager::kWhiteIndex);
   } else {
      m_colorManager->setBackgroundColorIndex(FWColorManager::kBlackIndex);
   }
}

void 
FWGUIManager::finishUpColorChange()
{
   gEve->FullRedraw3D();
}

//
// static member functions
//


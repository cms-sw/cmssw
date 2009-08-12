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
// $Id: FWGUIManager.cc,v 1.138 2009/07/28 17:25:55 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>
#include <stdexcept>
#include <iostream>
#include <cstdio>
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
#include "TGSlider.h"
#include "TColor.h"
#include "TVirtualX.h"
#include "TFile.h"

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
#include "Fireworks/Core/interface/CmsShowColorPopup.h"
#include "Fireworks/Core/interface/CmsShowModelPopup.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"

#include "Fireworks/Core/interface/CmsShowHelpPopup.h"

#include "Fireworks/Core/src/FWListWidget.h"

#include "Fireworks/Core/src/CmsShowTaskExecutor.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"

#include "Fireworks/Core/interface/FWTypeToRepresentations.h"
#include "Fireworks/Core/interface/FWIntValueListener.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"

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
   m_detailViewManager(0),
   m_viewManagerManager(iVMMgr),
   m_dataAdder(0),
   m_ediFrame(0),
   m_modelPopup(0),
   m_viewPopup(0),
   m_colorPopup(0),
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

      m_detailViewManager = new FWDetailViewManager(m_cmsShowMainFrame);

      getAction(cmsshow::sExportImage)->activated.connect(sigc::mem_fun(*this, &FWGUIManager::exportImageOfMainView));
      getAction(cmsshow::sSaveConfig)->activated.connect(writeToPresentConfigurationFile_);
      getAction(cmsshow::sSaveConfigAs)->activated.connect(sigc::mem_fun(*this,&FWGUIManager::promptForConfigurationFile));
      getAction(cmsshow::sShowEventDisplayInsp)->activated.connect(boost::bind( &FWGUIManager::showEDIFrame,this,-1));
      getAction(cmsshow::sShowMainViewCtl)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::showViewPopup));
      getAction(cmsshow::sShowObjInsp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::showModelPopup));
      getAction(cmsshow::sShowColorInsp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::showColorPopup));

      getAction(cmsshow::sShowAddCollection)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::addData));
      assert(getAction(cmsshow::sHelp) != 0);
      getAction(cmsshow::sHelp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createHelpPopup));
      assert(getAction(cmsshow::sKeyboardShort) != 0);
      getAction(cmsshow::sKeyboardShort)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createShortcutPopup));

      // toolbar special widget with non-void actions
      m_cmsShowMainFrame->m_delaySliderListener->valueChanged_.connect(boost::bind(&FWGUIManager::delaySliderChanged,this,_1));

      TQObject::Connect(m_cmsShowMainFrame->m_runEntry,"ReturnPressed()", "FWGUIManager", this, "runIdChanged()");
      TQObject::Connect(m_cmsShowMainFrame->m_eventEntry, "ReturnPressed()", "FWGUIManager", this, "eventIdChanged()");
      TQObject::Connect(m_cmsShowMainFrame->m_filterEntry, "ReturnPressed()", "FWGUIManager", this, "eventFilterChanged()");

      TQObject::Connect(gEve->GetWindowManager(), "WindowSelected(TEveWindow*)", "FWGUIManager", this, "checkSubviewAreaIconState(TEveWindow*)");
      TQObject::Connect(gEve->GetWindowManager(), "WindowDocked(TEveWindow*)", "FWGUIManager", this, "checkSubviewAreaIconState(TEveWindow*)");
      TQObject::Connect(gEve->GetWindowManager(), "WindowUndocked(TEveWindow*)", "FWGUIManager", this, "checkSubviewAreaIconState(TEveWindow*)");
   }
}

void FWGUIManager::connectSubviewAreaSignals(FWGUISubviewArea* a)
{
   a->goingToBeDestroyed_.connect(boost::bind(&FWGUIManager::subviewIsBeingDestroyed, this, _1));
   a->selected_.connect(boost::bind(&FWGUIManager::subviewInfoSelected, this, _1));
   a->unselected_.connect(boost::bind(&FWGUIManager::subviewInfoUnselected, this, _1));
   a->swap_.connect(boost::bind(&FWGUIManager::subviewSwapped, this, _1));
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
   //gDebug = 1;// ROOT debug

   gEve->GetWindowManager()->Disconnect("WindowSelected(TEveWindow*)", this, "checkSubviewAreaIconState(TEveWindow*)");
   gEve->GetWindowManager()->Disconnect("WindowDocked(TEveWindow*)", this, "checkSubviewAreaIconState(TEveWindow*)");
   gEve->GetWindowManager()->Disconnect("WindowUndocked(TEveWindow*)", this, "checkSubviewAreaIconState(TEveWindow*)");

   // unmap main frames not to watch dying
   m_cmsShowMainFrame->UnmapWindow();
   for(std::vector<TEveWindow*>::const_iterator wIt = m_viewWindows.begin(); wIt != m_viewWindows.end(); ++wIt)
   {
      TEveCompositeFrameInMainFrame* mainFrame = dynamic_cast<TEveCompositeFrameInMainFrame*>((*wIt)->GetEveFrame());
      if (mainFrame) mainFrame->UnmapWindow();
   }

   // avoid emit signals at end
   gEve->GetSelection()->RemoveElements();

   // destroy views
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
   TEveWindowSlot *slot = 0;
   action->activated.connect(boost::bind(&FWGUIManager::createView,this,iName, slot));
}

void
FWGUIManager::createView(const std::string& iName, TEveWindowSlot* slot)
{
   NameToViewBuilder::iterator itFind = m_nameToViewBuilder.find(iName);
   assert (itFind != m_nameToViewBuilder.end());
   if(itFind == m_nameToViewBuilder.end()) {
      throw std::runtime_error(std::string("Unable to create view named ")+iName+" because it is unknown");
   }
   if (slot == 0) slot =  m_viewSecPack->NewSlotWithWeight(1);
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
   m_cmsShowMainFrame->newFile(m_openFile->GetName());
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

TEveWindow*
FWGUIManager::getSwapCandidate()
{
   TEveWindow* swapCandidate =0;

   if ( gEve->GetWindowManager()->GetCurrentWindow())
   {
      swapCandidate = gEve->GetWindowManager()->GetCurrentWindow();
   }
   else
   {
      // swap with first docked view
      TEveCompositeFrame* pef;
      TGFrameElementPack *pel;

      // check if there is view in prim pack
      TGPack* pp = m_viewPrimPack->GetPack();
      if ( pp->GetList()->GetSize() > 2)
      {
         pel = (TGFrameElementPack*) pp->GetList()->At(1);
         if (pel->fState) // is first undocked
         {
            pef = dynamic_cast<TEveCompositeFrame*>(pel->fFrame);
            if ( pef && pef->GetEveWindow())
               swapCandidate = pef->GetEveWindow();
         }
      }
      if (swapCandidate == 0)
      {
         // no eve window found in primary check secondary
         TGPack* sp = m_viewSecPack->GetPack();
         Int_t nf = sp->GetList()->GetSize();
         TIter frame_iterator(sp->GetList());
         for (Int_t i=0; i<nf; ++i) {
            pel = (TGFrameElementPack*)frame_iterator();
            pef = dynamic_cast<TEveCompositeFrame*>(pel->fFrame);
            if ( pef && pef->GetEveWindow())
            {
               swapCandidate =  pef->GetEveWindow() ;
               break;
            }
         }
      }
   }

   return swapCandidate;
}

void
FWGUIManager::checkSubviewAreaIconState(TEveWindow* /*ew*/)
{
   // First argumet is needed for signals/slot symetry

   // disable swap on the first left TEveCompositeFrame
   // check info button
   TEveWindow* current = getSwapCandidate();
   bool checkInfoBtn =  m_viewPopup ? m_viewPopup->mapped() : 0;
   TEveWindow* selected = m_viewPopup ?  m_viewPopup->GetEveWindow() : 0;

   for (std::vector<TEveWindow*>::iterator it = m_viewWindows.begin(); it != m_viewWindows.end(); it++)
   {
      FWGUISubviewArea* ar = getGUISubviewArea(*it);
      ar->setSwapIcon(current != (*it));
      if (checkInfoBtn && selected)
         ar->setInfoButton(selected == (*it));
   }
}

void
FWGUIManager::subviewIsBeingDestroyed(FWGUISubviewArea* sva)
{
   if(sva->isSelected()) {
      setViewPopup(0);
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
FWGUIManager::subviewInfoSelected(FWGUISubviewArea* sva)
{
   // release button on previously selected
   if (m_viewPopup && m_viewPopup->GetEveWindow())
   {
      FWGUISubviewArea* ar = getGUISubviewArea(m_viewPopup->GetEveWindow());
      ar->setInfoButton(kFALSE);
   }

   setViewPopup(sva->getEveWindow());
}

void
FWGUIManager::subviewInfoUnselected(FWGUISubviewArea* sva)
{
   setViewPopup(0);
}

void
FWGUIManager::subviewSwapped(FWGUISubviewArea* sva)
{
   TEveWindow* curr = getSwapCandidate();
   TEveWindow* swap = sva->getEveWindow();
   if (curr) swap->SwapWindow(curr);

   checkSubviewAreaIconState(0);
}

TGVerticalFrame*
FWGUIManager::createList(TGSplitFrame *p)
{
   TGVerticalFrame *listFrame = new TGVerticalFrame(p, p->GetWidth(), p->GetHeight());

   TGHorizontalFrame* addFrame = new TGHorizontalFrame(p,p->GetWidth(), 10);
   TGLabel* addLabel = new TGLabel(addFrame,"Summary View");
   addLabel->SetTextJustify(kTextLeft);

   addFrame->AddFrame(addLabel, new TGLayoutHints(kLHintsCenterY|kLHintsLeft|kLHintsExpandX,2,2,2,2));
   FWCustomIconsButton* addDataButton = new FWCustomIconsButton(addFrame,
                                                                gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"plus-sign.png"),
                                                                gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"plus-sign-over.png"),
                                                                gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"plus-sign-disabled.png"));
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
      m_ediFrame->CenterOnParent(kTRUE,TGTransientFrame::kTopRight);
   }
}

void
FWGUIManager::updateEDI(FWEventItem* iItem) {
   createEDIFrame();
   m_ediFrame->fillEDIFrame(iItem);
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
FWGUIManager::showColorPopup()
{
  if (! m_colorPopup)
  {
      m_colorPopup = new CmsShowColorPopup(m_cmsShowMainFrame, 200, 200);
  }
  m_colorPopup->MapWindow();
  m_colorPopup->setModel(m_colorManager);
}

void
FWGUIManager::createModelPopup()
{
   m_modelPopup = new CmsShowModelPopup(m_detailViewManager,m_selectionManager, m_colorManager, m_cmsShowMainFrame, 200, 200);
   m_modelPopup->CenterOnParent(kTRUE,TGTransientFrame::kRight);
}

void
FWGUIManager::showModelPopup()
{
   if (!m_modelPopup) createModelPopup();
   m_modelPopup->MapWindow();
}

void
FWGUIManager::popupViewClosed()
{
   if (m_viewPopup->GetEveWindow())
   {
      FWGUISubviewArea* sa = getGUISubviewArea(m_viewPopup->GetEveWindow());
      sa->setInfoButton(kFALSE);
   }
}

void
FWGUIManager::showViewPopup() {
   // CSG action .
   setViewPopup(0);
}

void
FWGUIManager::setViewPopup(TEveWindow* ew) {
   //create if not exist
   if (m_viewPopup == 0)
   {
      m_viewPopup = new CmsShowViewPopup(m_cmsShowMainFrame, 200, 200, m_colorManager, ew);
      m_viewPopup->closed_.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::popupViewClosed));
      m_viewPopup->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   else
   {
      m_viewPopup->reset(ew);
   }

   m_viewPopup->MapWindow();
}


void FWGUIManager::createHelpPopup ()
{
   if (m_helpPopup == 0) {
      m_helpPopup = new CmsShowHelpPopup("help.html", "CmsShow Help",
                                         m_cmsShowMainFrame,
                                         800, 600);
      m_helpPopup->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   m_helpPopup->MapWindow();
}

void FWGUIManager::createShortcutPopup ()
{
   if (m_shortcutPopup == 0) {
      m_shortcutPopup = new CmsShowHelpPopup("shortcuts.html",
                                             "Keyboard Shortcuts",
                                             m_cmsShowMainFrame, 800, 600);

      m_shortcutPopup->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   m_shortcutPopup->MapWindow();
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
static const std::string kBrightness("brightness");
static const std::string kColorControl("color control");

static
void
addWindowInfoTo(const TGFrame* iMain,
                FWConfiguration& oTo)
{
   WindowAttributes_t attr;
   Window_t wdummy;
   Int_t ax,ay;
   gVirtualX->TranslateCoordinates(iMain->GetId(),
                                   gClient->GetDefaultRoot()->GetId(),
                                   0,0, //0,0 in local coordinates
                                   ax,ay, //coordinates of screen
                                   wdummy);
   gVirtualX->GetWindowAttributes(iMain->GetId(), attr);
   ay -=  attr.fY; // move up for decoration height
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

class areaInfo
{
   // helper class to save and restore view area
public:
   areaInfo (TGFrameElementPack* frameElement)
   {
      viewBase          = 0;
      originalSlot      = 0;
      undockedMainFrame = 0;
      weight = frameElement->fWeight;
      undocked = !frameElement->fState;

      TEveCompositeFrame *eveFrame = dynamic_cast<TEveCompositeFrame*>(frameElement->fFrame);
      assert(eveFrame);

      if (undocked)
         originalSlot = eveFrame->GetEveWindow();
      else
         viewBase = (FWViewBase*)((eveFrame->GetEveWindow())->GetUserData());
   }

   areaInfo () {}

   Float_t     weight;
   Bool_t      undocked;
   FWViewBase *viewBase;
   TGMainFrame *undockedMainFrame;// cached to help find original slot for undocked windows
   TEveWindow  *originalSlot;
};

static
void
addAreaInfoTo(areaInfo& pInfo,
              FWConfiguration& oTo)
{
   {
      std::stringstream s;
      s << pInfo.weight;
      oTo.addKeyValue("weight", FWConfiguration(s.str()));
   }
   {
      std::stringstream s;
      s<< pInfo.undocked;
      oTo.addKeyValue("undocked", FWConfiguration(s.str()));
   }

   if (pInfo.undockedMainFrame)
   {
      FWConfiguration temp(oTo);
      addWindowInfoTo(pInfo.undockedMainFrame, temp);
      oTo.addKeyValue("UndockedWindowPos", temp);
   }
}

//______________________________________________________________________________
void
FWGUIManager::addTo(FWConfiguration& oTo) const
{
   Int_t cfgVersion=2;

   FWConfiguration mainWindow(cfgVersion);
   float leftWeight, rightWeight;
   addWindowInfoTo(m_cmsShowMainFrame, mainWindow);
   {
      // write proportions of horizontal pack (can be standalone item outside main frame)
      if ( m_viewPrimPack->GetPack()->GetList()->GetSize() > 2)
      {
         TGFrameElementPack *frameEL;
         frameEL = (TGFrameElementPack*) m_viewPrimPack->GetPack()->GetList()->At(1); // read every second  element, first on is splitter
         leftWeight = frameEL->fWeight;
         frameEL = (TGFrameElementPack*)  m_viewPrimPack->GetPack()->GetList()->At(3);
         rightWeight = frameEL->fWeight;
      }
      else
      {
         leftWeight = 0;
         rightWeight = 1;
      }
      std::stringstream sL;
      sL<<leftWeight;
      mainWindow.addKeyValue("leftWeight",FWConfiguration(sL.str()));
      std::stringstream sR;
      sR<<rightWeight;
      mainWindow.addKeyValue("rightWeight",FWConfiguration(sR.str()));
   }
   oTo.addKeyValue(kMainWindow, mainWindow, true);
   fflush(stdout);
   //------------------------------------------------------------
   // organize info about all docked frames includding hidden, which point to undocked
   std::vector<areaInfo> wpacked;
   if (leftWeight > 0)
   {
      TGPack* pp = m_viewPrimPack->GetPack();
      TGFrameElementPack *frameEL = (TGFrameElementPack*) pp->GetList()->At(1);
      if (frameEL->fWeight > 0)
         wpacked.push_back(areaInfo(frameEL));
   }
   TGPack* sp = m_viewSecPack->GetPack();
   Int_t nf = sp->GetList()->GetSize();
   TIter frame_iterator(sp->GetList());
   for (Int_t i=0; i<nf; ++i) {
      TGFrameElementPack *seFE = (TGFrameElementPack*)frame_iterator();
      if (seFE->fWeight)
         wpacked.push_back(areaInfo(seFE));
   }

   //  add info about undocked
   TEveWindow* ew = 0;
   for(std::vector<TEveWindow*>::const_iterator wIt = m_viewWindows.begin(); wIt != m_viewWindows.end(); ++wIt)
   {
      ew = *wIt;
      TEveCompositeFrameInMainFrame* mainFrame = dynamic_cast<TEveCompositeFrameInMainFrame*>(ew->GetEveFrame());
      if (mainFrame)
      {
         // search for undocked in packed view
         for(std::vector<areaInfo>::iterator pIt = wpacked.begin(); pIt != wpacked.end(); ++pIt)
         {
            if ((*pIt).originalSlot && mainFrame->GetOriginalSlot() == (*pIt).originalSlot)
            {
               (*pIt).viewBase = (FWViewBase*)(ew->GetUserData());
               (*pIt).undockedMainFrame = (TGMainFrame*)mainFrame;
               // printf("found original slot for docked view %s\n", pInfo->viewBase->typeName().c_str());
               break;
            }// found match
         }
      }// end main frames
   }

   //------------------------------------------------------------
   // add sorted list in view area and FW-views configuration
   FWConfiguration views(1);
   FWConfiguration viewArea(cfgVersion);
   FWViewBase* wb;
   for(std::vector<areaInfo>::iterator it = wpacked.begin(); it != wpacked.end(); ++it)
   {
      wb = (*it).viewBase;
      if (wb) {
         FWConfiguration tempWiew(1);
         wb->addTo(tempWiew);
         views.addKeyValue(wb->typeName(), tempWiew, true);
         FWConfiguration tempArea(cfgVersion);
         addAreaInfoTo((*it), tempArea);
         viewArea.addKeyValue(wb->typeName(), tempArea, true);
      }
   }
   oTo.addKeyValue(kViews, views, true);
   oTo.addKeyValue(kViewArea, viewArea, true);

   //------------------------------------------------------------
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

   FWConfiguration colorControl(1);
   {
      // background
      FWConfiguration cbg(1);
      if(FWColorManager::kBlackIndex==m_colorManager->backgroundColorIndex()) {
         colorControl.addKeyValue(kBackgroundColor,FWConfiguration("black"));
      } else {
         colorControl.addKeyValue(kBackgroundColor,FWConfiguration("white"));
      }

      // brightness
      FWConfiguration cbr(1);
      std::stringstream s;
      s << static_cast<int>(m_colorManager->brightness());
      colorControl.addKeyValue(kBrightness,FWConfiguration(s.str()));
   }
   oTo.addKeyValue(kColorControl,colorControl,true);
}

//----------------------------------------------------------------
static void
setWindowInfoFrom(const FWConfiguration& iFrom,
                  TGMainFrame* iFrame)
{
   int x = atoi(iFrom.valueForKey("x")->value().c_str());
   int y = atoi(iFrom.valueForKey("y")->value().c_str());
   int width = atoi(iFrom.valueForKey("width")->value().c_str());
   int height = atoi(iFrom.valueForKey("height")->value().c_str());
   iFrame->MoveResize(x,y,width,height);
   iFrame->SetWMPosition(x, y);
}

void
FWGUIManager::setFrom(const FWConfiguration& iFrom) {
   // main window
   const FWConfiguration* mw = iFrom.valueForKey(kMainWindow);
   assert(mw != 0);
   setWindowInfoFrom(*mw, m_cmsShowMainFrame);

   // !! when position and size is clear map window
   m_cmsShowMainFrame->MapSubwindows();
   m_cmsShowMainFrame->Layout();
   m_cmsShowMainFrame->MapWindow();

   // set from view reading area info nd view info
   float_t leftWeight =1;
   float_t rightWeight=1;
   if ( mw->version() == 2 ) {
      leftWeight = atof(mw->valueForKey("leftWeight")->value().c_str());
      rightWeight = atof(mw->valueForKey("rightWeight")->value().c_str());
   }

   TEveWindowSlot* primSlot = (leftWeight > 0) ? m_viewPrimPack->NewSlotWithWeight(leftWeight) : 0;
   m_viewSecPack = m_viewPrimPack->NewSlotWithWeight(rightWeight)->MakePack();
   m_viewSecPack->SetVertical();
   m_viewSecPack->SetShowTitleBar(kFALSE);

   // views list
   const FWConfiguration* views = iFrom.valueForKey(kViews); assert(0!=views);
   const FWConfiguration::KeyValues* keyVals = views->keyValues();
   const FWConfiguration* viewArea = iFrom.valueForKey(kViewArea);

   // area list (ignored in older version)
   if ( viewArea->version() > 1)
   {
      const FWConfiguration::KeyValues* akv = viewArea->keyValues();
      FWConfiguration::KeyValuesIt areaIt = akv->begin();

      for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it!= keyVals->end(); ++it)
      {
         float weight = atof((areaIt->second).valueForKey("weight")->value().c_str());
         createView(it->first, (m_viewBases.size()|| (primSlot == 0)) ? m_viewSecPack->NewSlotWithWeight(weight) : primSlot);
         m_viewBases.back()->setFrom(it->second);

         bool  undocked = atof((areaIt->second).valueForKey("undocked")->value().c_str());
         TEveWindow* myw = m_viewWindows.back();
         if (undocked)
         {
            myw->UndockWindow();
            TEveCompositeFrameInMainFrame* emf = dynamic_cast<TEveCompositeFrameInMainFrame*>(myw->GetEveFrame());
            const TGMainFrame* mf =  dynamic_cast<const TGMainFrame*>(emf->GetParent());
            TGMainFrame* mfp = (TGMainFrame*)mf;
            const FWConfiguration* mwc = (areaIt->second).valueForKey("UndockedWindowPos");
            setWindowInfoFrom(*mwc, mfp);
         }
         areaIt++;
      }
   }
   else
   {  // create views with same weight in old version
      for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it!= keyVals->end(); ++it) {
         createView(it->first, m_viewBases.size() ? m_viewSecPack->NewSlot() : primSlot);
         m_viewBases.back()->setFrom(it->second);
      }
      // handle undocked windows in old version
      const FWConfiguration* undocked = iFrom.valueForKey(kUndocked);
      if(0!=undocked) {
         const FWConfiguration::KeyValues* keyVals = undocked->keyValues();
         if(0!=keyVals) {

            int idx = m_viewBases.size() -keyVals->size();
            for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it != keyVals->end(); ++it)
            {
               TEveWindow* myw = m_viewWindows[idx];
               idx++;
               myw->UndockWindow();
               TEveCompositeFrameInMainFrame* emf = dynamic_cast<TEveCompositeFrameInMainFrame*>(myw->GetEveFrame());
               const TGMainFrame* mf =  dynamic_cast<const TGMainFrame*>(emf->GetParent());
               TGMainFrame* mfp = (TGMainFrame*)mf;
               setWindowInfoFrom(it->second, mfp);
            }
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
            // std::cout <<"found controller "<<controllerName<<std::endl;
            if(controllerName == kCollectionController) {
               showEDIFrame();
               setWindowInfoFrom(it->second,m_ediFrame);
            } else if (controllerName == kViewController) {
               setViewPopup(0);
               setWindowInfoFrom(it->second, m_viewPopup);
            } else if (controllerName == kObjectController) {
               showModelPopup();
               setWindowInfoFrom(it->second, m_modelPopup);
            }
         }
      }
   }
   // disable fist docked view
   checkSubviewAreaIconState(0);

   // display colors
   const FWConfiguration* colorControl = iFrom.valueForKey(kColorControl);
   if( 0!=colorControl)
   {
      int brightness = 0;
      const FWConfiguration* cbr = colorControl->valueForKey(kBrightness);
      if (cbr)
      {
         std::stringstream sw(cbr->value());
         sw >> brightness;
         // printf("set brightness %d \n", brightness);
         m_colorManager->setBrightness(brightness);
      }

      if("black" == colorControl->valueForKey(kBackgroundColor)->value()) {
         m_colorManager->setBackgroundAndBrightness( FWColorManager::kBlackIndex, brightness);
      } else {
         m_colorManager->setBackgroundAndBrightness( FWColorManager::kWhiteIndex, brightness);
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
FWGUIManager::finishUpColorChange()
{
   gEve->FullRedraw3D(kFALSE,kTRUE);
}



//
// static member functions
//


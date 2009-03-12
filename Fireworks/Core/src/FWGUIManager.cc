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
// $Id: FWGUIManager.cc,v 1.95 2009/03/11 21:16:19 amraktad Exp $
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
//#include "TEveGedEditor.h"
#include "TEveWindow.h"
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
                           const FWViewManagerManager* iVMMgr,
                           bool iDebugInterface
                           ) :
   m_selectionManager(iSelMgr),
   m_eiManager(iEIMgr),
   m_changeManager(iCMgr),
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

   // These are only needed temporarilty to work around a problem which
   // Matevz has patched in a later version of the code
   TApplication::NeedGraphicsLibs();
   gApplication->InitializeGraphics();

   TEveManager::Create(kFALSE, "FI");

   // gEve->SetUseOrphanage(kTRUE);
   // TGFrame* f = (TGFrame*) gClient->GetDefaultRoot();
   // browser->MoveResize(f->GetX(), f->GetY(), f->GetWidth(), f->GetHeight());
   // browser->Resize( gClient->GetDisplayWidth(), gClient->GetDisplayHeight() );


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

      // toolbar special widget with non-void actions
      m_cmsShowMainFrame->m_delaySliderListener->valueChanged_.connect(boost::bind(&FWGUIManager::delaySliderChanged,this,_1));

      TQObject::Connect(m_cmsShowMainFrame->m_runEntry,"ReturnPressed()", "FWGUIManager", this, "runIdChanged()");
      TQObject::Connect(m_cmsShowMainFrame->m_eventEntry, "ReturnPressed()", "FWGUIManager", this, "eventIdChanged()");
      TQObject::Connect(m_cmsShowMainFrame->m_filterEntry, "ReturnPressed()", "FWGUIManager", this, "eventFilterChanged()");
   }
   {
      // createEDIFrame();
      // createModelPopup();
   }
}

// FWGUIManager::FWGUIManager(const FWGUIManager& rhs)
// {
//    // do actual copying here;
// }

FWGUIManager::~FWGUIManager()
{
   for(std::vector<FWViewBase* >::iterator it = m_viewBases.begin(), itEnd = m_viewBases.end();
       it != itEnd;
       ++it) {
      (*it)->destroy();
   }

   delete m_summaryManager;
   delete m_detailViewManager;
   delete m_editableSelected;
   delete m_cmsShowMainFrame;
   delete m_ediFrame;
}

//
// assignment operators
//
// const FWGUIManager& FWGUIManager::operator=(const FWGUIManager& rhs)
// {
//   //An exception safe implementation is
//   FWGUIManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

TEveWindowSlot*
FWGUIManager::parentForNextView()
{
   TEveWindowSlot* slot = 0;
   if (m_viewSecPack == 0) {
      slot = m_viewPrimPack->NewSlot();
      m_viewSecPack = m_viewPrimPack->NewSlot()->MakePack();
      m_viewSecPack->SetVertical();
      m_viewSecPack->SetShowTitleBar(kFALSE);
   } else {
      slot = m_viewSecPack->NewSlot();
   }

   FWGUISubviewArea* hf = new FWGUISubviewArea(m_viewFrames.size(), slot->GetEveFrame());
   m_viewFrames.push_back(hf);
   hf->MapSubwindows();
   slot->GetEveFrame()->ReplaceIconBox(hf);
   hf->goingToBeDestroyed_.connect(boost::bind(&FWGUIManager::subviewIsBeingDestroyed,this,_1));
   hf->selected_.connect(boost::bind(&FWGUIManager::subviewSelected,this,_1));
   hf->unselected_.connect(boost::bind(&FWGUIManager::subviewUnselected,this,_1));
   hf->swapWithCurrentView_.connect(boost::bind(&FWGUIManager::subviewSwapWithCurrent,this,_1));

   return slot;
}

void
FWGUIManager::registerViewBuilder(const std::string& iName,
                                  ViewBuildFunctor& iBuilder)
{
   m_nameToViewBuilder[iName]=iBuilder;
   CSGAction* action=m_cmsShowMainFrame->createNewViewerAction(iName);
   action->activated.connect(boost::bind(&FWGUIManager::createView,this,iName));
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
   slot->SetElementName(iName.c_str());
   FWViewBase* view = itFind->second(slot);
   m_viewBases.push_back(view);
}

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


#if defined(THIS_WILL_NEVER_BE_DEFINED)
   if(1 ==iSM.selected().size() ) {
      delete m_editableSelected;
      FWListModel* model = new FWListModel(*(iSM.selected().begin()), m_detailViewManager);
      const FWEventItem::ModelInfo& info =iSM.selected().begin()->item()->modelInfo(iSM.selected().begin()->index());
      model->SetMainColor(info.displayProperties().color());
      model->SetRnrState(info.displayProperties().isVisible());
      m_editableSelected = model;
      //m_editor->DisplayElement(m_editableSelected);
   } else if(1<iSM.selected().size()) {
      delete m_editableSelected;
      m_editableSelected = new FWListMultipleModels(iSM.selected());
      //m_editor->DisplayElement(m_editableSelected);
   } else {
      /*
         if(m_editor->GetEveElement() == m_editableSelected) {
         //m_editor->DisplayElement(0);
         }
       */
      delete m_editableSelected;
      m_editableSelected=0;
   }
   m_unselectAllButton->SetEnabled( 0 !=iSM.selected().size() );
#endif
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


//______________________________________________________________________________
//  subview area
//

void
FWGUIManager::subviewIsBeingDestroyed(unsigned int iIndex)
{
   assert(iIndex < m_viewFrames.size());

   if(m_viewFrames[iIndex]->isSelected()) {
      if(0!= m_viewPopup) {refillViewPopup(0);}
   }

   m_viewFrames.erase(m_viewFrames.begin()+iIndex);
   m_viewBases.erase(m_viewBases.begin()+iIndex);
}

void
FWGUIManager::subviewSelected(unsigned int iSelIndex)
{
   unsigned int index=0;
   for(std::vector<FWGUISubviewArea*>::iterator it = m_viewFrames.begin(), itEnd=m_viewFrames.end();
       it != itEnd; ++it,++index) {
      if(index != iSelIndex) {
         (*it)->unselect();
      }
   }
   showViewPopup();
   refillViewPopup(m_viewBases[iSelIndex]);
}

void
FWGUIManager::subviewUnselected(unsigned int iSelIndex)
{
   if(m_viewPopup) {refillViewPopup(0);}
}

void
FWGUIManager::subviewSwapWithCurrent(unsigned int index)
{
   TEveWindow* cv = 0;
   int ci = 0;
   for(std::vector<FWGUISubviewArea*>::iterator it = m_viewFrames.begin(); it !=m_viewFrames.end(); ++it) {
      TEveWindow*  ev = (*it)->getEveWindow();
      if (ev->IsCurrent())
      {
         cv = ev;
         break;
      }
      ci++;
   }

   if (cv)
   {
      TEveWindow::SwapWindows(cv, m_viewFrames[index]->getEveWindow());
      std::swap(m_viewBases[ci], m_viewBases[index]);

   }
}

//______________________________________________________________________________

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
   //TGTextButton* addDataButton = new TGTextButton(listFrame,"+");
   addDataButton->SetToolTipText("Show additional collections");
   addDataButton->Connect("Clicked()", "FWGUIManager", this, "addData()");
   addFrame->AddFrame(addDataButton, new TGLayoutHints(kLHintsCenterY|kLHintsLeft,2,2,2,2));
   listFrame->AddFrame(addFrame, new TGLayoutHints(kLHintsExpandX|kLHintsLeft|kLHintsTop,2,2,2,2));


   //  p->Resize(listFrame->GetWidth(), listFrame->GetHeight());
   m_summaryManager = new FWSummaryManager(listFrame,
                                           m_selectionManager,
                                           m_eiManager,
                                           this,
                                           m_changeManager);
   
   listFrame->AddFrame(m_summaryManager->widget(), new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   //m_views =  new TEveElementList("Views");
   //m_views->AddIntoListTree(m_listTree,reinterpret_cast<TGListTreeItem*>(0));
   //m_editor = ltf->GetEditor();
   //m_editor->DisplayElement(0);
   /*
   {
      //m_listTree->Connect("mouseOver(TGListTreeItem*, UInt_t)", "FWGUIManager",
      //                 this, "itemBelowMouse(TGListTreeItem*, UInt_t)");
      m_listTree->Connect("Clicked(TGListTreeItem*, Int_t, UInt_t, Int_t, Int_t)", "FWGUIManager",
                          this, "itemClicked(TGListTreeItem*, Int_t, UInt_t, Int_t, Int_t)");
      m_listTree->Connect("DoubleClicked(TGListTreeItem*, Int_t)", "FWGUIManager",
                          this, "itemDblClicked(TGListTreeItem*, Int_t)");
      m_listTree->Connect("KeyPressed(TGListTreeItem*, ULong_t, ULong_t)", "FWGUIManager",
                          this, "itemKeyPress(TGListTreeItem*, UInt_t, UInt_t)");
   }
    */
   /*
      TGGroupFrame* vf = new TGGroupFrame(listFrame,"Selection",kVerticalFrame);
      {

      TGGroupFrame* vf2 = new TGGroupFrame(vf,"Expression");
      m_selectionItemsComboBox = new TGComboBox(vf2,200);
      m_selectionItemsComboBox->Resize(200,20);
      vf2->AddFrame(m_selectionItemsComboBox, new TGLayoutHints(kLHintsTop | kLHintsLeft,0,5,5,5));
      m_selectionExpressionEntry = new TGTextEntry(vf2,"$.pt() > 10");
      vf2->AddFrame(m_selectionExpressionEntry, new TGLayoutHints(kLHintsExpandX,0,5,5,5));
      m_selectionRunExpressionButton = new TGTextButton(vf2,"Select by Expression");
      vf2->AddFrame(m_selectionRunExpressionButton);
      m_selectionRunExpressionButton->Connect("Clicked()","FWGUIManager",this,"selectByExpression()");
      vf->AddFrame(vf2);

      m_unselectAllButton = new TGTextButton(vf,"Unselect All");
      m_unselectAllButton->Connect("Clicked()", "FWGUIManager",this,"unselectAll()");
      vf->AddFrame(m_unselectAllButton);
      m_unselectAllButton->SetEnabled(kFALSE);

      }
      listFrame->AddFrame(vf);
    */

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
}

void
FWGUIManager::createEDIFrame() {
   if (m_ediFrame == 0) {
      m_ediFrame = new CmsShowEDI(m_cmsShowMainFrame, 200, 200, m_selectionManager);
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
FWGUIManager::createModelPopup() {
   if (m_modelPopup == 0) {
      m_modelPopup = new CmsShowModelPopup(m_detailViewManager,m_selectionManager, m_cmsShowMainFrame, 200, 200);
      m_modelPopup->Connect("CloseWindow()", "FWGUIManager", this, "resetModelPopup()");
      m_modelPopup->CenterOnParent(kTRUE,TGTransientFrame::kRight);
   }
}

void
FWGUIManager::showModelPopup()
{
   m_modelPopup->MapWindow();
}


void
FWGUIManager::updateModel(FWEventItem* iItem) {
   createModelPopup();
   //  m_modelPopup->fillModelPopup(iItem);
}

void
FWGUIManager::resetModelPopup() {
   //  m_modelChangeConn.disconnect();
   m_modelPopup->DontCallClose();
   m_modelPopup->UnmapWindow();
}

void
FWGUIManager::createViewPopup() {
   if (m_viewPopup == 0) {
      m_viewPopup = new CmsShowViewPopup(m_cmsShowMainFrame, 200, 200, m_viewBases[0]);
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
         //NOTE: editor should be decided by looking at FWSelectionManager and NOT directly from clicking
         // in the list
         //m_editor->DisplayElement(el);
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

namespace {
   template<class Op>
   void recursivelyApplyToFrame(TGSplitFrame* iParent, Op& iOp) {
      if(0==iParent) { return;}
      //if it holds something OR is completely empty (because of undocking)
      if(iParent->GetFrame() ||
         (!iParent->GetFirst() && !iParent->GetSecond()) ) {
         iOp(iParent);
      } else {
         recursivelyApplyToFrame(iParent->GetFirst(),iOp);
         recursivelyApplyToFrame(iParent->GetSecond(),iOp);
      }
   }

   struct FrameAddTo {
      FWConfiguration* m_config;
      bool m_isFirst;
      FrameAddTo(FWConfiguration& iConfig) :
         m_config(&iConfig),
         m_isFirst(true) {
      }

      void operator()(TGFrame* iFrame) {
         std::stringstream s;
         if(m_isFirst) {
            m_isFirst = false;
            s<< static_cast<int>(iFrame->GetWidth());
         } else {
            s<< static_cast<int>(iFrame->GetHeight());
         }
         m_config->addValue(s.str());
      }
   };

   struct FrameSetFrom {
      const FWConfiguration* m_config;
      int m_index;
      FrameSetFrom(const FWConfiguration* iConfig) :
         m_config(iConfig),
         m_index(0) {
      }

      void operator()(TGFrame* iFrame) {
         if(0==iFrame) {return;}
         int width=0,height=0;
         if(0==m_index) {
            // top (main) split frame
            width = iFrame->GetWidth();
            std::stringstream s(m_config->value(m_index));
            s >> width;
         } else {
            // bottom left split frame
            height = iFrame->GetHeight();
            std::stringstream s(m_config->value(m_index));
            s >> height;
         }
         iFrame->Resize(width, height);
         ++m_index;
      }
   };
}

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

   oTo.addKeyValue(kMainWindow,mainWindow,true);

   FWConfiguration views(1);
   for(std::vector<FWViewBase* >::const_iterator it = m_viewBases.begin(),
                                                 itEnd = m_viewBases.end();
       it != itEnd;
       ++it) {
      FWConfiguration temp(1);
      (*it)->addTo(temp);
      views.addKeyValue((*it)->typeName(), temp, true);
   }
   oTo.addKeyValue(kViews,views,true);

   //remember the sizes in the view area

   FWConfiguration viewArea(1);
   FrameAddTo frameAddTo(viewArea);
   recursivelyApplyToFrame(m_splitFrame,frameAddTo);
   oTo.addKeyValue(kViewArea,viewArea,true);

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
   //first is main window
   const FWConfiguration* mw = iFrom.valueForKey(kMainWindow);
   assert(mw != 0);
   int width,height;
   {
      const FWConfiguration* cWidth = mw->valueForKey("width");
      assert(0 != cWidth);
      std::stringstream s(cWidth->value());
      s >> width;
   }
   {
      const FWConfiguration* c = mw->valueForKey("height");
      assert(0 != c);
      std::stringstream s(c->value());
      s >> height;
   }
   m_cmsShowMainFrame->Resize(width,height);
   {
      int x=0;
      int y=0;
      {
         const FWConfiguration* cX = mw->valueForKey("x");
         if( 0!=cX ) {
            std::stringstream s(cX->value());
            s >> x;
         }
      }
      {
         const FWConfiguration* cY = mw->valueForKey("y");
         if(0 != cY) {
            std::stringstream s(cY->value());
            s >> y;
         }
      }
      m_cmsShowMainFrame->Move(x,y);
   }

   //now configure the views
   const FWConfiguration* views = iFrom.valueForKey(kViews);
   assert(0!=views);
   const FWConfiguration::KeyValues* keyVals = views->keyValues();
   assert(0!=keyVals);
   for(FWConfiguration::KeyValues::const_iterator it = keyVals->begin(),
                                                  itEnd = keyVals->end();
       it!=itEnd;
       ++it) {
      size_t n = m_viewBases.size();
      createView(it->first);

      assert(n+1 == m_viewBases.size());
      m_viewBases.back()->setFrom(it->second);
   }

   //now configure the view area
   const FWConfiguration* viewArea = iFrom.valueForKey(kViewArea);
   assert(0!=viewArea);
   m_viewPrimPack->GetGUIFrame()->Layout();


   //handle controllers
   const FWConfiguration* controllers = iFrom.valueForKey(kControllers);
   if(0!=controllers) {
      const FWConfiguration::KeyValues* keyVals = controllers->keyValues();
      if(0!=keyVals) {
         //we have open controllers
         for(FWConfiguration::KeyValues::const_iterator it = keyVals->begin(),
                                                        itEnd = keyVals->end();
             it!=itEnd;
             ++it) {
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

//
// static member functions
//


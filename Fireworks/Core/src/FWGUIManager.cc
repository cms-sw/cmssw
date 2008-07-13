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
// $Id: FWGUIManager.cc,v 1.53 2008/07/12 13:50:36 chrjones Exp $
//

// system include files
#include <boost/bind.hpp>
#include <stdexcept>
#include <iostream>
#include <sstream>

#include "TGButton.h"
#include "TGComboBox.h"
#include "TGTextEntry.h"
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
#include "TEveSelection.h"
#include "TGFileDialog.h"
#include "TStopwatch.h"
#include "TColor.h"

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
#include "Fireworks/Core/interface/CSGNumAction.h"

#include "Fireworks/Core/interface/ActionsList.h"

#include "Fireworks/Core/interface/CmsShowEDI.h"
#include "Fireworks/Core/interface/CmsShowModelPopup.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"

#include "Fireworks/Core/src/FWListWidget.h"

#include "Fireworks/Core/src/CmsShowTaskExecutor.h"
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
):
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
   
   TEveManager::Create(kFALSE);

   // gEve->SetUseOrphanage(kTRUE);
   // TGFrame* f = (TGFrame*) gClient->GetDefaultRoot();
   // browser->MoveResize(f->GetX(), f->GetY(), f->GetWidth(), f->GetHeight());
   // browser->Resize( gClient->GetDisplayWidth(), gClient->GetDisplayHeight() );


   {
     m_cmsShowMainFrame = new CmsShowMainFrame(gClient->GetRoot(),
					       1024,
					       768,
					       this);
     m_cmsShowMainFrame->SetWindowName("CmsShow");
     m_cmsShowMainFrame->SetCleanup(kDeepCleanup);
      
     getAction(cmsshow::sExportImage)->activated.connect(sigc::mem_fun(*this, &FWGUIManager::exportImageOfMainView));
     getAction(cmsshow::sSaveConfig)->activated.connect(writeToPresentConfigurationFile_);
     getAction(cmsshow::sSaveConfigAs)->activated.connect(sigc::mem_fun(*this,&FWGUIManager::promptForConfigurationFile));
     getAction(cmsshow::sShowEventDisplayInsp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createEDIFrame));
     getAction(cmsshow::sShowMainViewCtl)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createViewPopup));
      getAction(cmsshow::sShowObjInsp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createModelPopup));
      
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
void 
FWGUIManager::addFrameHoldingAView(TGFrame* iChild)
{
   (m_viewFrames.back())->AddFrame(iChild,new TGLayoutHints(kLHintsExpandX | 
                                                            kLHintsExpandY) );
   
   m_mainFrame->MapSubwindows();
   m_mainFrame->Resize();
   iChild->Resize();
   m_mainFrame->MapWindow();
   
}

TGFrame* 
FWGUIManager::parentForNextView()
{
   
   TGSplitFrame* splitParent=m_splitFrame;
   while( splitParent->GetFrame() || splitParent->GetSecond()) {
      if(! splitParent->GetSecond()) {
         if(splitParent == m_splitFrame) {
            //want to split vertically
            splitParent->SplitHorizontal();
            //need to do a reasonable sizing 
            //TODO CDJ: how do I determine the true size if layout hasn't run yet?
            int width = m_splitFrame->GetWidth();
            int height = m_splitFrame->GetHeight();
            m_splitFrame->GetFirst()->Resize(width,static_cast<int>(0.8*height));
         } else {
            splitParent->SplitVertical();
         }
      }
      splitParent = splitParent->GetSecond();
   }
   
   FWGUISubviewArea* hf = new FWGUISubviewArea(m_viewFrames.size(),splitParent,m_splitFrame);
   hf->swappedToBigView_.connect(boost::bind(&FWGUIManager::subviewWasSwappedToBig,this,_1));
   hf->goingToBeDestroyed_.connect(boost::bind(&FWGUIManager::subviewIsBeingDestroyed,this,_1));
   if(!m_viewFrames.empty()) {
      m_viewFrames.back()->enableDestructionButton(false);
   }
   m_viewFrames.push_back(hf);
   hf->enableDestructionButton(true);
   (splitParent)->AddFrame(hf,new TGLayoutHints(kLHintsExpandX | kLHintsExpandY) );
   
   return m_viewFrames.back();
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
FWGUIManager::registerDetailView (const std::string &iItemName, 
                                  FWDetailView *iView)
{
   m_detailViewManager->registerDetailView(iItemName,iView);
}


void 
FWGUIManager::createView(const std::string& iName)
{
   NameToViewBuilder::iterator itFind = m_nameToViewBuilder.find(iName);
   assert (itFind != m_nameToViewBuilder.end());
   if(itFind == m_nameToViewBuilder.end()) {
      throw std::runtime_error(std::string("Unable to create view named ")+iName+" because it is unknown");
   }
   FWViewBase* view(itFind->second(parentForNextView()));
   addFrameHoldingAView(view->frame());
   
   /*
   FWListViewObject* lst = new FWListViewObject(iName.c_str(),view);
   lst->AddIntoListTree(m_listTree,m_views);
   //TODO: HACK should keep a hold of 'lst' and keep it so that if view is removed this goes as well
   lst->IncDenyDestroy();
    */
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

CSGNumAction*
FWGUIManager::getRunEntry()
{
  return m_cmsShowMainFrame->getRunEntry();
}

CSGNumAction*
FWGUIManager::getEventEntry()
{
  return m_cmsShowMainFrame->getEventEntry();
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
}

void 
FWGUIManager::processGUIEvents()
{
   gSystem->ProcessEvents();
}

void
FWGUIManager::newItem(const FWEventItem* iItem)
{
   m_selectionItemsComboBox->AddEntry(iItem->name().c_str(),iItem->id());
   if(iItem->id()==0) {
      m_selectionItemsComboBox->Select(0);
   }
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
                                            m_viewManagerManager->supportedTypesAndPurpose());
   }
   m_dataAdder->show();
}


void 
FWGUIManager::subviewWasSwappedToBig(unsigned int iIndex)
{
   m_viewFrames[0]->setIndex(iIndex);
   m_viewFrames[iIndex]->setIndex(0);
   std::swap(m_viewBases[0], m_viewBases[iIndex]);
   std::swap(m_viewFrames[0],m_viewFrames[iIndex]);
   //if swapped with last one then toggle destruction button
   if(m_viewFrames.size() == iIndex+1) {
      m_viewFrames[0]->enableDestructionButton(false);
      m_viewFrames[iIndex]->enableDestructionButton(true);
   }
   if (m_viewPopup != 0) refillViewPopup(m_viewBases[0]);
}

void
FWGUIManager::subviewIsBeingDestroyed(unsigned int iIndex)
{
   //We need to delay actually removing the window until the next 'iteration' of the GUI event loop because we need the
   // Button to return from its 'Clicked()' function before we delete the button
   CmsShowTaskExecutor::TaskFunctor f;
   //We know the parent is a TGSplitFrame because the constructor requires it to be so
   TGSplitFrame* p = const_cast<TGSplitFrame*>(static_cast<const TGSplitFrame*>(m_viewFrames[iIndex]->GetParent()));

   f= boost::bind(&TGSplitFrame::CloseAndCollapse,p);
   m_tasks->addTask(f);
   m_tasks->startDoingTasks();
   
   m_viewFrames.erase(m_viewFrames.begin()+iIndex);
   (*(m_viewBases.begin()+iIndex))->destroy();
   m_viewBases.erase(m_viewBases.begin()+iIndex);
   if(!m_viewFrames.empty()) {
      m_viewFrames.back()->enableDestructionButton(true);
   }
}

TGVerticalFrame* 
FWGUIManager::createList(TGSplitFrame *p) 
{
  TGVerticalFrame *listFrame = new TGVerticalFrame(p, p->GetWidth(), p->GetHeight());


  TGTextButton* addDataButton = new TGTextButton(listFrame,"+");
  addDataButton->SetToolTipText("Show additional event data");
  addDataButton->Connect("Clicked()", "FWGUIManager", this, "addData()");
  listFrame->AddFrame(addDataButton);

  FWListWidget* ltf = new FWListWidget(listFrame);
  listFrame->SetEditable(kFALSE);
  listFrame->AddFrame(ltf, new TGLayoutHints(kLHintsExpandX));
  //  p->Resize(listFrame->GetWidth(), listFrame->GetHeight());
  m_listTree = ltf->GetListTree();
  m_summaryManager = new FWSummaryManager(m_listTree,
					  m_selectionManager,
					  m_eiManager,
					  m_detailViewManager,
                                          m_changeManager);
  //m_views =  new TEveElementList("Views");
  //m_views->AddIntoListTree(m_listTree,reinterpret_cast<TGListTreeItem*>(0));
  //m_editor = ltf->GetEditor();
  //m_editor->DisplayElement(0);
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

  return listFrame;
}

TGMainFrame* 
FWGUIManager::createViews(TGCompositeFrame *p) 
{
  m_mainFrame = new TGMainFrame(p,600,450);
  m_splitFrame = new TGSplitFrame(m_mainFrame, 800, 600);
  m_mainFrame->AddFrame(m_splitFrame, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

  p->Resize(m_mainFrame->GetWidth(), m_mainFrame->GetHeight());
  p->MapSubwindows();
  p->MapWindow();
  return m_mainFrame;
}  

void
FWGUIManager::createEDIFrame() {
  if (m_ediFrame == 0) {
    m_ediFrame = new CmsShowEDI(m_cmsShowMainFrame, 200, 200, m_selectionManager);
    m_ediFrame->Connect("CloseWindow()", "FWGUIManager", this, "resetEDIFrame()");
  }
}

void
FWGUIManager::updateEDI(FWEventItem* iItem) {
  createEDIFrame();
  m_ediFrame->fillEDIFrame(iItem);
}

void
FWGUIManager::resetEDIFrame() {
   std::cout <<"resetEDIFrame()"<<std::endl;
   m_ediFrame = 0;
}

void
FWGUIManager::createModelPopup() {
  if (m_modelPopup == 0) {
    m_modelPopup = new CmsShowModelPopup(m_detailViewManager,m_selectionManager, m_cmsShowMainFrame, 200, 200);
    m_modelPopup->Connect("CloseWindow()", "FWGUIManager", this, "resetModelPopup()");
    //m_selectionManager->selectionChanged_.connect(boost::bind(&CmsShowModelPopup::fillModelPopup, m_modelPopup, _1));
    //    m_modelChangeConn = m_changeManager->changeSignalsAreDone_.connect(boost::bind(&CmsShowModelPopup::updateDisplay, m_modelPopup));
  }
}

void
FWGUIManager::updateModel(FWEventItem* iItem) {
  if (m_modelPopup == 0) createModelPopup();
  //  m_modelPopup->fillModelPopup(iItem);
}

void
FWGUIManager::resetModelPopup() {
   //  m_modelChangeConn.disconnect();
   delete m_modelPopup;
   m_modelPopup = 0;
}

void
FWGUIManager::createViewPopup() {
  if (m_viewPopup == 0) {
    m_viewPopup = new CmsShowViewPopup(m_cmsShowMainFrame, 200, 200, m_viewBases[0]);
    m_viewPopup->Connect("CloseWindow()", "FWGUIManager", this, "resetViewPopup()");
  }
}

void
FWGUIManager::refillViewPopup(FWViewBase* iView) {
  m_viewPopup->reset(iView);
}

void
FWGUIManager::resetViewPopup() {
  delete m_viewPopup;
  m_viewPopup = 0;
}


TGMainFrame *FWGUIManager::createTextView (TGTab *p) 
{
     m_textViewTab = p;
     p->Resize(m_mainFrame->GetWidth(), m_mainFrame->GetHeight());
     m_textViewFrame[0] = p->AddTab("Physics objects");
     printf("current tab: %d\n", p->GetCurrent());
     m_textViewFrame[1] = p->AddTab("Triggers");
     printf("current tab: %d\n", p->GetCurrent());
     m_textViewFrame[2] = p->AddTab("Tracking");
     printf("current tab: %d\n", p->GetCurrent());
     
     p->MapSubwindows();
     p->MapWindow();
     return m_mainFrame;
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
   if(1==btn) {
      if(lib && lib->doSelection(mask&kKeyControlMask) ) {
         gEve->GetSelection()->UserPickedElement(el,mask&kKeyControlMask);
         
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
   writeToConfigurationFile_(fi.fFilename);   
}

void 
FWGUIManager::exportImageOfMainView()
{
   static TString dir(".");
   const char *  kImageExportTypes[] = {"Encapsulated PostScript", "*.eps",
      "PDF",                     "*.pdf",
      "GIF",                     "*.gif",
      "JPEG",                    "*.jpg",
      "PNG",                     "*.png",
   0, 0};
   
   TGFileInfo fi;
   fi.fFileTypes = kImageExportTypes;
   fi.fIniDir    = StrDup(dir);
   new TGFileDialog(gClient->GetDefaultRoot(), m_cmsShowMainFrame,
                    kFDSave,&fi);
   dir = fi.fIniDir;
   m_viewBases[0]->saveImageTo(fi.fFilename);
}

static const std::string kMainWindow("main window");
static const std::string kViews("views");
static const std::string kViewArea("view area");

namespace {
   template<class Op>
   void recursivelyApplyToFrame(TGSplitFrame* iParent, Op& iOp) {
      std::cout <<"recursivelyApplyToFrame "<<iParent<<std::endl;
      if(0==iParent) { return;}
      if(iParent->GetFrame()) {
         std::cout <<"   Frame"<<std::endl;
         iOp(iParent);
      } else {
         std::cout <<"   First"<<std::endl;
         recursivelyApplyToFrame(iParent->GetFirst(),iOp);
         std::cout <<"   Second"<<std::endl;
         recursivelyApplyToFrame(iParent->GetSecond(),iOp);         
      }
   }
   
   struct FrameAddTo {
      FWConfiguration* m_config;
      bool m_isFirst;
      FrameAddTo(FWConfiguration& iConfig) :
      m_config(&iConfig),
      m_isFirst(true) {}
      
      void operator()(TGFrame* iFrame) {
         std::stringstream s;
         if(m_isFirst) {
            m_isFirst = false;
            s<< static_cast<int>(iFrame->GetHeight());
            std::cout <<"height "<<s.str()<<std::endl;
         }else {
            s<< static_cast<int>(iFrame->GetWidth());
            std::cout <<"width "<<s.str()<<std::endl;
         }
         m_config->addValue(s.str());
      }         
   };

   struct FrameSetFrom {
      const FWConfiguration* m_config;
      int m_index;
      FrameSetFrom(const FWConfiguration* iConfig) :
      m_config(iConfig),
      m_index(0) {}
      
      void operator()(TGFrame* iFrame) {
         if(0==iFrame) {return;}
         int width=0,height=0;
         if(0==m_index) {
            // top (main) split frame
            width = iFrame->GetWidth();
            std::stringstream s(m_config->value(m_index));
            s >> height;
            std::cout <<"height "<<height<<std::endl;
         } else {
         // bottom left split frame
            height = iFrame->GetHeight();
            std::stringstream s(m_config->value(m_index));
            s >> width;
            std::cout <<"width "<<width<<std::endl;
         }
         iFrame->Resize(width, height);
         ++m_index;
      }
   };
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
   
   FrameSetFrom frameSetFrom(viewArea);
   recursivelyApplyToFrame(m_splitFrame, frameSetFrom);

   m_splitFrame->Layout();
 
   {
      int width = ((TGCompositeFrame *)m_splitFrame->GetParent()->GetParent())->GetWidth();
      int height;
      std::stringstream s(viewArea->value(viewArea->stringValues()->size()-1));
      s >> height;
      ((TGCompositeFrame *)m_splitFrame->GetParent()->GetParent())->Resize(width, height);
   }
   m_cmsShowMainFrame->Layout();
}

void 
FWGUIManager::openEveBrowserForDebugging() const
{
   gEve->GetBrowser()->MapWindow();
}


//
// static member functions
//


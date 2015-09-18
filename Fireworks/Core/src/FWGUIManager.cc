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
// system include files
#include <boost/bind.hpp>
#include <stdexcept>
#include <iostream>
#include <cstdio>
#include <sstream>
#include <thread>
#include <future>

#include "TGButton.h"
#include "TGLabel.h"
#include "TSystem.h"
#include "TGLIncludes.h"
#include "TGLViewer.h"
#include "TEveBrowser.h"
#include "TEveManager.h"
#include "TGPack.h"
#include "TEveWindow.h"
#include "TEveViewer.h"
#include "TEveWindowManager.h"
#include "TEveSelection.h"
#include "TVirtualX.h"
#include "TFile.h"

// user include files
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWGUISubviewArea.h"
#include "Fireworks/Core/interface/FWTEveViewer.h"

#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWSummaryManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/FWGeometryTableViewBase.h"
#include "Fireworks/Core/interface/FWJobMetadataManager.h"
#include "Fireworks/Core/interface/FWInvMassDialog.h"

#include "Fireworks/Core/interface/FWConfiguration.h"

#include "Fireworks/Core/interface/CmsShowMainFrame.h"
#include "Fireworks/Core/interface/FWNavigatorBase.h"

#include "Fireworks/Core/src/FWGUIEventDataAdder.h"
#include "Fireworks/Core/src/FWNumberEntry.h"

#include "Fireworks/Core/interface/CSGAction.h"

#include "Fireworks/Core/interface/ActionsList.h"

#include "Fireworks/Core/interface/CmsShowEDI.h"
#include "Fireworks/Core/interface/CmsShowCommonPopup.h"
#include "Fireworks/Core/interface/CmsShowModelPopup.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"

#include "Fireworks/Core/interface/CmsShowHelpPopup.h"

#include "Fireworks/Core/src/CmsShowTaskExecutor.h"

#include "Fireworks/Core/interface/FWTypeToRepresentations.h"
#include "Fireworks/Core/interface/FWIntValueListener.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"

#include "Fireworks/Core/src/FWModelContextMenuHandler.h"

#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FW3DViewBase.h"

#include "FWCore/Common/interface/EventBase.h"

#include "CommonTools/Utils/src/Grammar.h"
#include "CommonTools/Utils/interface/Exception.h"



// constants, enums and typedefs
//
//
// static data member definitions
//
FWGUIManager* FWGUIManager::m_guiManager = 0;

//
// constructors and destructor
//


FWGUIManager::FWGUIManager(fireworks::Context* ctx,
                           const FWViewManagerManager* iVMMgr,
                           FWNavigatorBase* navigator):
   m_context(ctx),
   m_summaryManager(0),
   m_detailViewManager(0),
   m_viewManagerManager(iVMMgr),
   m_contextMenuHandler(0),
   m_navigator(navigator),
   m_dataAdder(0),
   m_ediFrame(0),
   m_modelPopup(0),
   m_viewPopup(0),
   m_commonPopup(0),
   m_invMassDialog(0),
   m_helpPopup(0),
   m_shortcutPopup(0),
   m_helpGLPopup(0),
   m_tasks(new CmsShowTaskExecutor),
   m_WMOffsetX(0), m_WMOffsetY(0), m_WMDecorH(0)
{
   m_guiManager = this;

   measureWMOffsets();

   FWEventItemsManager* im = (FWEventItemsManager*) m_context->eventItemsManager();
   im->newItem_.connect(boost::bind(&FWGUIManager::newItem, this, _1) );

   m_context->colorManager()->colorsHaveChangedFinished_.connect(boost::bind(&FWGUIManager::finishUpColorChange,this));

  
   TEveCompositeFrame::IconBarCreator_foo foo =  &FWGUIManager::makeGUIsubview;
   TEveCompositeFrame::SetupFrameMarkup(foo, 20, 4, false);

   {
      m_cmsShowMainFrame = new CmsShowMainFrame(gClient->GetRoot(),
                                                950,
                                                750,
                                                this);
     
      m_cmsShowMainFrame->SetCleanup(kDeepCleanup);
    
      /*
        int mlist[FWViewType::kTypeSize] = {FWViewType::kRhoPhi, FWViewType::kRhoZ, FWViewType::k3D, FWViewType::kISpy, FWViewType::kLego, FWViewType::kLegoHF, FWViewType::kGlimpse, 
        FWViewType::kTable, FWViewType::kTableL1, FWViewType::kTableHLT,
        FWViewType::kGeometryTable,
        FWViewType::kRhoPhiPF, FWViewType::kLegoPFECAL}; */

      for (int i = 0 ; i < FWViewType::kTypeSize; ++i)
      {
         if (m_context->getHidePFBuilders() && (i == FWViewType::kLegoPFECAL || i == FWViewType::kRhoPhiPF))
             continue;

         bool separator = (i == FWViewType::kGlimpse || i == FWViewType::kTableHLT || i ==  FWViewType::kLegoPFECAL);
         CSGAction* action = m_cmsShowMainFrame->createNewViewerAction(FWViewType::idToName(i), separator);
         action->activated.connect(boost::bind(&FWGUIManager::newViewSlot, this, FWViewType::idToName(i)));
      }

      m_detailViewManager  = new FWDetailViewManager(m_context);
      m_contextMenuHandler = new FWModelContextMenuHandler(m_context->selectionManager(), m_detailViewManager, m_context->colorManager(), this);


      getAction(cmsshow::sExportImage)->activated.connect(sigc::mem_fun(*this, &FWGUIManager::exportImageOfMainView));
      getAction(cmsshow::sExportAllImages)->activated.connect(sigc::mem_fun(*this, &FWGUIManager::exportImagesOfAllViews));
      getAction(cmsshow::sLoadConfig)->activated.connect(sigc::mem_fun(*this, &FWGUIManager::promptForLoadConfigurationFile));
      getAction(cmsshow::sLoadPartialConfig)->activated.connect(sigc::mem_fun(*this, &FWGUIManager::promptForPartialLoadConfigurationFile));
      getAction(cmsshow::sSaveConfig)->activated.connect(writeToPresentConfigurationFile_);
      getAction(cmsshow::sSavePartialConfig)->activated.connect(sigc::mem_fun(this, &FWGUIManager::savePartialToConfigurationFile));
      getAction(cmsshow::sSaveConfigAs)->activated.connect(sigc::mem_fun(*this,&FWGUIManager::promptForSaveConfigurationFile));
      getAction(cmsshow::sSavePartialConfigAs)->activated.connect(sigc::mem_fun(*this,&FWGUIManager::promptForPartialSaveConfigurationFile));
      getAction(cmsshow::sShowEventDisplayInsp)->activated.connect(boost::bind( &FWGUIManager::showEDIFrame,this,-1));
      getAction(cmsshow::sShowMainViewCtl)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::showViewPopup));
      getAction(cmsshow::sShowObjInsp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::showModelPopup));

      getAction(cmsshow::sBackgroundColor)->activated.connect(sigc::mem_fun(m_context->colorManager(), &FWColorManager::switchBackground));
      getAction(cmsshow::sShowCommonInsp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::showCommonPopup));

      getAction(cmsshow::sShowInvMassDialog)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::showInvMassDialog));

      getAction(cmsshow::sShowAddCollection)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::addData));
      assert(getAction(cmsshow::sHelp) != 0);
      getAction(cmsshow::sHelp)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createHelpPopup));
      assert(getAction(cmsshow::sKeyboardShort) != 0);
      getAction(cmsshow::sKeyboardShort)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createShortcutPopup));
      getAction(cmsshow::sHelpGL)->activated.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::createHelpGLPopup));

      // toolbar special widget with non-void actions
      m_cmsShowMainFrame->m_delaySliderListener->valueChanged_.connect(boost::bind(&FWGUIManager::delaySliderChanged,this,_1));

      TQObject::Connect(m_cmsShowMainFrame->m_runEntry,   "ReturnPressed()", "FWGUIManager", this, "runIdChanged()");
      TQObject::Connect(m_cmsShowMainFrame->m_lumiEntry,  "ReturnPressed()", "FWGUIManager", this, "lumiIdChanged()");
      TQObject::Connect(m_cmsShowMainFrame->m_eventEntry, "ReturnPressed()", "FWGUIManager", this, "eventIdChanged()");

      TQObject::Connect(m_cmsShowMainFrame->m_filterShowGUIBtn, "Clicked()", "FWGUIManager", this, "showEventFilterGUI()");
      TQObject::Connect(m_cmsShowMainFrame->m_filterEnableBtn,  "Clicked()", "FWGUIManager", this, "filterButtonClicked()"); 

      TQObject::Connect(gEve->GetWindowManager(), "WindowSelected(TEveWindow*)", "FWGUIManager", this, "checkSubviewAreaIconState(TEveWindow*)");
      TQObject::Connect(gEve->GetWindowManager(), "WindowDocked(TEveWindow*)"  , "FWGUIManager", this, "checkSubviewAreaIconState(TEveWindow*)");
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
   delete m_invMassDialog;
   delete m_summaryManager;
   delete m_detailViewManager;
   delete m_cmsShowMainFrame;
   delete m_viewPopup;
   delete m_ediFrame;
   delete m_contextMenuHandler;

}

void
FWGUIManager::evePreTerminate()
{
   gEve->GetWindowManager()->Disconnect("WindowSelected(TEveWindow*)", this, "checkSubviewAreaIconState(TEveWindow*)");
   gEve->GetWindowManager()->Disconnect("WindowDocked(TEveWindow*)", this, "checkSubviewAreaIconState(TEveWindow*)");
   gEve->GetWindowManager()->Disconnect("WindowUndocked(TEveWindow*)", this, "checkSubviewAreaIconState(TEveWindow*)");

   // avoid emit signals at end
   gEve->GetSelection()->Disconnect();
   gEve->GetHighlight()->Disconnect();
   gEve->GetSelection()->RemoveElements();
   gEve->GetHighlight()->RemoveElements();
    
   m_cmsShowMainFrame->UnmapWindow();
   for(ViewMap_i wIt = m_viewMap.begin(); wIt != m_viewMap.end(); ++wIt)
   {
     TEveCompositeFrameInMainFrame* mainFrame = dynamic_cast<TEveCompositeFrameInMainFrame*>((*wIt).first->GetEveFrame());
     //  main frames not to watch dying
      if (mainFrame) mainFrame->UnmapWindow();
     // destroy
      (*wIt).second->destroy();
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
}


void
FWGUIManager::newViewSlot(const std::string& iName)
{
   // this function have to exist, becuse CSGAction binds to void functions 
   createView(iName);
}

FWGUIManager::ViewMap_i
FWGUIManager::createView(const std::string& iName, TEveWindowSlot* slot)
{
   NameToViewBuilder::iterator itFind = m_nameToViewBuilder.find(iName);
   assert (itFind != m_nameToViewBuilder.end());
   if(itFind == m_nameToViewBuilder.end()) {
      throw std::runtime_error(std::string("Unable to create view named ")+iName+" because it is unknown");
   }
   
   if (!slot)
   {
      if (m_viewSecPack)
      {
         slot = m_viewSecPack->NewSlot();
      }
      else
      {
         slot = m_viewPrimPack->NewSlot();
         m_viewSecPack = m_viewPrimPack->NewSlot()->MakePack();
         m_viewSecPack->SetShowTitleBar(kFALSE);
      }
   }
   TEveCompositeFrame *ef = slot->GetEveFrame();
   FWViewBase* viewBase = itFind->second(slot, iName);
   //in future, get context from 'view'
   FWViewContextMenuHandlerBase* base= viewBase->contextMenuHandler();
   viewBase->openSelectedModelContextMenu_.connect(boost::bind(&FWGUIManager::showSelectedModelContextMenu ,m_guiManager,_1,_2,base));
   
   TEveWindow *eveWindow = ef->GetEveWindow();
   eveWindow->SetElementName(iName.c_str());

   std::pair<ViewMap_i,bool> insertPair = m_viewMap.insert(std::make_pair(eveWindow, viewBase));
   return insertPair.first;
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
FWGUIManager::titleChanged(const char *subtitle)
{
   char title[128];
   snprintf(title,127,"cmsShow: %s", subtitle);
   m_cmsShowMainFrame->SetWindowName(title);
}

void
FWGUIManager::eventChangedCallback() {
   // To be replaced when we can get index from fwlite::Event
   
   TEveViewerList* viewers = gEve->GetViewers();
   for (TEveElement::List_i i=viewers->BeginChildren(); i!= viewers->EndChildren(); ++i)
   {
      TEveViewer* ev = dynamic_cast<TEveViewer*>(*i);
      if (ev)
         ev->GetGLViewer()->DeleteOverlayAnnotations();
   }
   
   for (auto reg : m_regionViews)
   {
       for(ViewMap_i it = m_viewMap.begin(); it != m_viewMap.end(); ++it)
       {
           if (it->second == reg) {
               m_viewMap.erase(it);
               reg->destroy();
               break;
           }
       }
   }

   m_cmsShowMainFrame->loadEvent(*getCurrentEvent());
   m_detailViewManager->newEventCallback();
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

CSGContinuousAction*
FWGUIManager::loopAction()
{
   return m_cmsShowMainFrame->loopAction();
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
   m_cmsShowMainFrame->m_runEntry->SetEnabled(!play);
   m_cmsShowMainFrame->m_eventEntry->SetEnabled(!play);
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
   if (0==m_dataAdder) {
      m_dataAdder = new FWGUIEventDataAdder(100,100,
                                            (FWEventItemsManager*) m_context->eventItemsManager(),
                                            m_cmsShowMainFrame,
                                            m_context->metadataManager());
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
         // no eve window found in primary, check secondary
         TGPack* sp = m_viewSecPack->GetPack();
         TIter frame_iterator(sp->GetList());
         while ((pel = (TGFrameElementPack*)frame_iterator())) 
         {
            pef = dynamic_cast<TEveCompositeFrame*>(pel->fFrame);
            if ( pef && pef->GetEveWindow() && pel->fState)
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
   TEveWindow* current  = getSwapCandidate();
   bool checkInfoBtn    = m_viewPopup ? m_viewPopup->mapped() : 0;
   TEveWindow* selected = m_viewPopup ? m_viewPopup->getEveWindow() : 0;

   for (ViewMap_i it = m_viewMap.begin(); it != m_viewMap.end(); it++)
   {
      FWGUISubviewArea* ar = FWGUISubviewArea::getToolBarFromWindow(it->first);
      ar->setSwapIcon(current != it->first);
      if (checkInfoBtn && selected)
         ar->setInfoButton(selected == it->first);
   }
}

void
FWGUIManager::subviewIsBeingDestroyed(FWGUISubviewArea* sva)
{
   if (sva->isSelected())
      setViewPopup(0);

   CmsShowTaskExecutor::TaskFunctor f;
   f = boost::bind(&FWGUIManager::subviewDestroy, this, sva);
   m_tasks->addTask(f);
   m_tasks->startDoingTasks();
}

void
FWGUIManager::subviewDestroy(FWGUISubviewArea* sva)
{  
   TEveWindow* ew       = sva->getEveWindow();
   FWViewBase* viewBase = m_viewMap[ew];
   m_viewMap.erase(ew);
   viewBase->destroy();
}

void
FWGUIManager::subviewDestroyAll()
{
   std::vector<FWGUISubviewArea*> sd;
   for(ViewMap_i wIt = m_viewMap.begin(); wIt != m_viewMap.end(); ++wIt)
   {
      FWGUISubviewArea* ar = FWGUISubviewArea::getToolBarFromWindow(wIt->first);
      sd.push_back(ar);
   }

   for (std::vector<FWGUISubviewArea*>::iterator i= sd.begin(); i !=sd.end(); ++i)
   {
      if ((*i)->isSelected())
         setViewPopup(0);
      subviewDestroy(*i);
   }

   gSystem->ProcessEvents();
   gSystem->Sleep(200);


   
   while (m_viewPrimPack->HasChildren())
   {
      TEveWindow* w = dynamic_cast<TEveWindow*>(m_viewPrimPack->FirstChild());
      if (w) w->DestroyWindowAndSlot();
   }

   gSystem->Sleep(200);
   m_viewSecPack = 0;
   gSystem->ProcessEvents();

}

void
FWGUIManager::subviewInfoSelected(FWGUISubviewArea* sva)
{
   // release button on previously selected
   TEveWindow* ew = sva->getEveWindow();
   for(ViewMap_i wIt = m_viewMap.begin(); wIt != m_viewMap.end(); ++wIt)
   {
      if (wIt->first != ew)
         FWGUISubviewArea::getToolBarFromWindow(wIt->first)->setInfoButton(kFALSE);
   }
   setViewPopup(sva->getEveWindow());
}

void
FWGUIManager::subviewInfoUnselected(FWGUISubviewArea* sva)
{
   m_viewPopup->UnmapWindow();
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
FWGUIManager::createList(TGCompositeFrame *p)
{
   TGVerticalFrame *listFrame = new TGVerticalFrame(p, p->GetWidth(), p->GetHeight());

   TGHorizontalFrame* addFrame = new TGHorizontalFrame(listFrame, p->GetWidth(), 10, kRaisedFrame);
   TGLabel* addLabel = new TGLabel(addFrame,"Summary View");
   addFrame->AddFrame(addLabel, new TGLayoutHints(kLHintsCenterX, 0,0,2,2));
   listFrame->AddFrame(addFrame, new TGLayoutHints(kLHintsExpandX | kLHintsTop));

   m_summaryManager = new FWSummaryManager(listFrame,
                                           m_context->selectionManager(),
                                           (FWEventItemsManager*) m_context->eventItemsManager(),
                                           this,
                                           m_context->modelChangeManager(),
                                           m_context->colorManager());

   const unsigned int backgroundColor=0x2f2f2f;
   TGTextButton* addDataButton = new TGTextButton(m_summaryManager->widget(), "Add Collection");
   addDataButton->ChangeOptions(kRaisedFrame);
   addDataButton->SetBackgroundColor(backgroundColor);
   addDataButton->SetTextColor(0xFFFFFF);
   addDataButton->SetToolTipText("Show additional collections");
   addDataButton->Connect("Clicked()", "FWGUIManager", this, "addData()");
   m_summaryManager->widget()->AddFrame(addDataButton, new TGLayoutHints(kLHintsExpandX|kLHintsLeft|kLHintsTop));
   listFrame->AddFrame(m_summaryManager->widget(), new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));

   return listFrame;
}

void
FWGUIManager::createViews(TEveWindowSlot *slot)
{
   m_viewPrimPack = slot->MakePack();
   m_viewPrimPack->SetHorizontal();
   m_viewPrimPack->SetElementName("Views");
   m_viewPrimPack->SetShowTitleBar(kFALSE);
   m_viewSecPack = 0;
}

void
FWGUIManager::createEDIFrame()
{
   if (m_ediFrame == 0)
   {
      m_ediFrame = new CmsShowEDI(m_cmsShowMainFrame, 200, 200, m_context->selectionManager(),m_context->colorManager());
      m_ediFrame->CenterOnParent(kTRUE,TGTransientFrame::kTopRight);
      m_cmsShowMainFrame->bindCSGActionKeys(m_ediFrame);
   }
}

void
FWGUIManager::showEDIFrame(int iToShow)
{
   createEDIFrame();
   if (-1 != iToShow)
   {
      m_ediFrame->show(static_cast<FWDataCategories>(iToShow));
   }
   m_ediFrame->MapRaised();
}


void
FWGUIManager::open3DRegion()
{
   FWModelId id =  *(m_context->selectionManager()->selected().begin());
   float theta =0, phi = 0;
   {
      edm::TypeWithDict type = edm::TypeWithDict((TClass*)id.item()->modelType());
      using namespace boost::spirit::classic;
      reco::parser::ExpressionPtr tmpPtr;
      reco::parser::Grammar grammar(tmpPtr,type);
      edm::ObjectWithDict o(type, (void*)id.item()->modelData(id.index()));
      try {
         parse("theta()", grammar.use_parser<1>() >> end_p, space_p).full;
         theta =  tmpPtr->value(o);
         parse("phi()", grammar.use_parser<1>() >> end_p, space_p).full;
         phi =  tmpPtr->value(o);

         ViewMap_i it = createView( "3D Tower", m_viewSecPack->NewSlot());
         FW3DViewBase* v = static_cast<FW3DViewBase*>(it->second);
         v->setClip(theta, phi);
         it->first->UndockWindow();
         m_regionViews.push_back(v);
      }
      catch(const reco::parser::BaseException& e)
      {
         std::cout <<" FWModelFilter failed to base "<< e.what() << std::endl;
      }
   }
}

void
FWGUIManager::showCommonPopup()
{
   if (! m_commonPopup)
   {
      m_commonPopup = new CmsShowCommonPopup(m_context->commonPrefs(), m_cmsShowMainFrame, 200, 200);
      m_cmsShowMainFrame->bindCSGActionKeys(m_commonPopup);
   }
   m_commonPopup->MapRaised();
}

void
FWGUIManager::createModelPopup()
{
   m_modelPopup = new CmsShowModelPopup(m_detailViewManager,m_context->selectionManager(), m_context->colorManager(), m_cmsShowMainFrame, 200, 200);
   m_modelPopup->CenterOnParent(kTRUE,TGTransientFrame::kRight);
   m_cmsShowMainFrame->bindCSGActionKeys(m_modelPopup);
}

void
FWGUIManager::showModelPopup()
{
   if (! m_modelPopup) createModelPopup();
   m_modelPopup->MapRaised();
}

void
FWGUIManager::popupViewClosed()
{
   if (m_viewPopup->getEveWindow())
   {
      FWGUISubviewArea* sa = FWGUISubviewArea::getToolBarFromWindow(m_viewPopup->getEveWindow());
      sa->setInfoButton(kFALSE);
   }
}

void
FWGUIManager::showViewPopup()
{
   // CSG action.
   setViewPopup(0);
}

void
FWGUIManager::setViewPopup(TEveWindow* ew)
{
   FWViewBase* vb = ew ? m_viewMap[ew] : 0;
   if (m_viewPopup == 0)
   {
      m_viewPopup = new CmsShowViewPopup(0, 200, 200, m_context->colorManager(), vb, ew);
      m_viewPopup->closed_.connect(sigc::mem_fun(*m_guiManager, &FWGUIManager::popupViewClosed));
   }
   else
   {
      m_viewPopup->UnmapWindow();
   }
   m_viewPopup->reset(vb, ew);
   m_viewPopup->MapRaised();
}

void
FWGUIManager::showInvMassDialog()
{
   if (! m_invMassDialog)
   {
      m_invMassDialog = new FWInvMassDialog(m_context->selectionManager());
      m_cmsShowMainFrame->bindCSGActionKeys(m_invMassDialog);
   }
   m_invMassDialog->MapRaised();
}

void
FWGUIManager::createHelpPopup ()
{
   if (m_helpPopup == 0)
   {
      m_helpPopup = new CmsShowHelpPopup("help.html", "CmsShow Help",
                                         m_cmsShowMainFrame,
                                         800, 600);
      m_helpPopup->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   m_helpPopup->MapWindow();
}


void
FWGUIManager::createShortcutPopup ()
{
   if (m_shortcutPopup == 0)
   {
      m_shortcutPopup = new CmsShowHelpPopup("shortcuts.html",
                                             getAction(cmsshow::sKeyboardShort)->getName().c_str(),
                                              m_cmsShowMainFrame, 800, 600);

      m_shortcutPopup->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   m_shortcutPopup->MapWindow();
}

void FWGUIManager::createHelpGLPopup ()
{
   if (m_helpGLPopup == 0)
   {
      m_helpGLPopup = new CmsShowHelpPopup("helpGL.html",
                                            getAction(cmsshow::sHelpGL)->getName().c_str(),
                                            m_cmsShowMainFrame, 800, 600);

      m_helpGLPopup->CenterOnParent(kTRUE,TGTransientFrame::kBottomRight);
   }
   m_helpGLPopup->MapWindow();
}

void 
FWGUIManager::showSelectedModelContextMenu(Int_t iGlobalX, Int_t iGlobalY, FWViewContextMenuHandlerBase* iHandler)
{
   if (! m_context->selectionManager()->selected().empty())
   {
      m_contextMenuHandler->showSelectedModelContext(iGlobalX,iGlobalY, iHandler);
   }
}

//
// const member functions
//

FWGUIManager*
FWGUIManager::getGUIManager()
{
   return m_guiManager;
}

const edm::EventBase*
FWGUIManager::getCurrentEvent() const
{
   return m_navigator->getCurrentEvent();  
}

/** Helper method for a load / save configuration dialog.

    @a result where the picked file is stored.
    
    @a mode the mode for the dialog (i.e. Load / Save).
    
    @return true if a file was successfully picked, false otherwise.
  */
bool
FWGUIManager::promptForConfigurationFile(std::string &result, enum EFileDialogMode mode)
{
   
   const static char* kFileTypes[] = {"Fireworks Configuration files","*.fwc",
                                       "All Files","*",
                                       0,0};

   static TString dir(".");

   TGFileInfo fi;
   fi.fFileTypes = kFileTypes;
   fi.fIniDir    = StrDup(dir);
   new TGFileDialog(gClient->GetDefaultRoot(), m_cmsShowMainFrame, mode, &fi);
   dir = fi.fIniDir;
   if (fi.fFilename == 0) // to handle "cancel" button properly
      return false;
   std::string name = fi.fFilename;
   // if the extension isn't already specified by hand, specify it now
   std::string ext = kFileTypes[fi.fFileTypeIdx + 1] + 1;
   if (ext.size() != 0 && name.find(ext) == name.npos)
      name += ext;
   result = name;
   return true;
}


/** Emits the signal which request to load the configuration file picked up 
    in a dialog.
  */
void
FWGUIManager::promptForLoadConfigurationFile()
{
   std::string name;
   if (!promptForConfigurationFile(name, kFDOpen))
      return;
  
   
   loadFromConfigurationFile_(name);
}


void
FWGUIManager::promptForPartialLoadConfigurationFile()
{
   std::string name;
   if (!promptForConfigurationFile(name, kFDOpen))
      return;
  
   
   loadPartialFromConfigurationFile_(name);
   //
}


/** Emits the signal which requests to save the current configuration in the 
    file picked up in the dialog.
  */
void
FWGUIManager::promptForSaveConfigurationFile()
{
   std::string name;
   if (!promptForConfigurationFile(name, kFDSave))
      return;

   writeToConfigurationFile_(name);
}

void
FWGUIManager::promptForPartialSaveConfigurationFile()
{
   std::string name;
   if (!promptForConfigurationFile(name, kFDSave))
      return;

   writePartialToConfigurationFile_(name);
}

void
FWGUIManager::savePartialToConfigurationFile()
{
   writePartialToConfigurationFile_("current");
}

void
FWGUIManager::exportImageOfMainView()
{
   if (m_viewPrimPack->GetPack()->GetList()->GetSize() > 2)
   {
      TGFrameElementPack* frameEL = (TGFrameElementPack*) m_viewPrimPack->GetPack()->GetList()->At(1);
      TEveCompositeFrame* ef = dynamic_cast<TEveCompositeFrame*>(frameEL->fFrame);
      m_viewMap[ef->GetEveWindow()]->promptForSaveImageTo(m_cmsShowMainFrame);
   }
   else
   {
      fwLog(fwlog::kError) << "Main view has been destroyed." << std::endl; 
   }
}

void
FWGUIManager::exportImagesOfAllViews()
{
   try {
      static TString dir(".");
      const char *  kImageExportTypes[] = {"PNG",                     "*.png",
                                           "GIF",                     "*.gif",
                                           "JPEG",                    "*.jpg",
                                           "PDF",                     "*.pdf",
                                           "Encapsulated PostScript", "*.eps",
                                           0, 0};

      TGFileInfo fi;
      fi.fFileTypes = kImageExportTypes;
      fi.fIniDir    = StrDup(dir);
      new TGFileDialog(gClient->GetDefaultRoot(), m_cmsShowMainFrame,
                       kFDSave,&fi);
      dir = fi.fIniDir;
      if (fi.fFilename != 0) {
         std::string name = fi.fFilename;
         // fi.fFileTypeIdx points to the name of the file type
         // selected in the drop-down menu, so fi.fFileTypeIdx gives us
         // the extension
         std::string ext = kImageExportTypes[fi.fFileTypeIdx + 1] + 1;
         if (name.find(ext) == name.npos)
            name += ext;
         // now add format trailing before the extension
         name.insert(name.rfind('.'), "-%u_%u_%u_%s");
         exportAllViews(name, -1);
      }
   }
   catch (std::runtime_error &e) { std::cout << e.what() << std::endl; }
}

void
FWGUIManager::exportAllViews(const std::string& format, int height)
{
   // Save all GL views.
   // Expects format to have "%u %u %llu %s" which are replaced with
   //   run-number, event number, lumi block and view-name.
   // Blanks in view-name are removed.
   // If several views shave the same name, they are post-fixed
   // with "_%d". They are sorted by view diagonal.

   typedef std::list<FWTEveViewer*>           viewer_list_t;
   typedef viewer_list_t::iterator            viewer_list_i;

   typedef std::map<TString, viewer_list_t>   name_map_t;
   typedef name_map_t::iterator               name_map_i;

   name_map_t vls;

   for (ViewMap_i i = m_viewMap.begin(); i != m_viewMap.end(); ++i)
   {
      FWTEveViewer *ev = dynamic_cast<FWTEveViewer*>(i->first);
      if (ev)
      {
         TString name(ev->GetElementName());
         name.ReplaceAll(" ", "");
         viewer_list_t &l  = vls[name];
         viewer_list_i  li = l.begin();
         while (li != l.end() && (*li)->GetGLViewer()->ViewportDiagonal() < ev->GetGLViewer()->ViewportDiagonal())
            ++li;
         l.insert(li, ev);
      }
   }

   std::vector<std::future<int>> futures;
   
   const edm::EventBase *event = getCurrentEvent();
   for (name_map_i i = vls.begin(); i != vls.end(); ++i)
   {
      bool multi_p    = (i->second.size() > 1);
      int  view_count = 1;
      for (viewer_list_i j = i->second.begin(); j != i->second.end(); ++j, ++view_count)
      {
         TString view_name(i->first);
         if (multi_p)
         {
            view_name += "_";
            view_name += view_count;
         }
         TString file;
         file.Form(format.c_str(), event->id().run(), event->id().event(),
                   event->luminosityBlock(), view_name.Data());

         if (GLEW_EXT_framebuffer_object)
         {
            // Multi-threaded save
            futures.push_back((*j)->CaptureAndSaveImage(file, height));
         }
         else
         {
            // Single-threaded save
            if (height == -1)
               (*j)->GetGLViewer()->SavePicture(file);
            else 
               (*j)->GetGLViewer()->SavePictureHeight(file, height);
         }
      }
   }

   for (auto &f : futures)
   {
      f.get();
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
static const std::string kCommonController("common");

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

class areaInfo
{
   // helper class to save and restore view area
public:
   areaInfo (TGFrameElementPack* frameElement)
   {
      eveWindow         = 0;
      originalSlot      = 0;
      undockedMainFrame = 0;
      weight = frameElement->fWeight;
      undocked = !frameElement->fState;

      TEveCompositeFrame *eveFrame = dynamic_cast<TEveCompositeFrame*>(frameElement->fFrame);
      assert(eveFrame);

      if (frameElement->fState)
         eveWindow    =  eveFrame->GetEveWindow();
      else
         originalSlot = eveFrame->GetEveWindow();
   }

  areaInfo () : weight(0), undocked(false) {}

   Float_t      weight;
   Bool_t       undocked;
   TEveWindow  *eveWindow;
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
   Int_t cfgVersion=3;

   FWConfiguration mainWindow(cfgVersion);
   float leftWeight, rightWeight;
   addWindowInfoTo(m_cmsShowMainFrame, mainWindow);
   {
      // write summary view weight
      {
         std::stringstream ss;
         ss << m_cmsShowMainFrame->getSummaryViewWeight();
         mainWindow.addKeyValue("summaryWeight",FWConfiguration(ss.str()));
      }

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
   TGFrameElementPack *seFE;
   TIter frame_iterator(sp->GetList());
   while ((seFE = (TGFrameElementPack*)frame_iterator() ))
   {
      if (seFE->fWeight)
         wpacked.push_back(areaInfo(seFE));
   }

   //  undocked info
   
   for(ViewMap_i wIt = m_viewMap.begin(); wIt != m_viewMap.end(); ++wIt)
   {
      TEveWindow* ew = wIt->first;
      TEveCompositeFrameInMainFrame* mainFrame = dynamic_cast<TEveCompositeFrameInMainFrame*>(ew->GetEveFrame());
      if (mainFrame)
      {
         for(std::vector<areaInfo>::iterator pIt = wpacked.begin(); pIt != wpacked.end(); ++pIt)
         {
            if ((*pIt).originalSlot && mainFrame->GetOriginalSlot() == (*pIt).originalSlot)
            {
               (*pIt).eveWindow = wIt->first;
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
   for(std::vector<areaInfo>::iterator it = wpacked.begin(); it != wpacked.end(); ++it)
   {
      TEveWindow* ew = (*it).eveWindow;
      if (ew) {
         FWViewBase* wb = m_viewMap[ew];
         FWConfiguration tempWiew(wb->version());
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
      if(0!=m_commonPopup && m_commonPopup->IsMapped()) {
         FWConfiguration temp(1);
         addWindowInfoTo(m_commonPopup, temp);
         controllers.addKeyValue(kCommonController,temp,true);
      }
   }
   oTo.addKeyValue(kControllers,controllers,true);
}

//----------------------------------------------------------------
void
FWGUIManager::setWindowInfoFrom(const FWConfiguration& iFrom,
                                TGMainFrame* iFrame)
{
   int x = atoi(iFrom.valueForKey("x")->value().c_str()) + m_WMOffsetX;
   int y = atoi(iFrom.valueForKey("y")->value().c_str()) + m_WMOffsetY;
   if (y < m_WMDecorH) y = m_WMDecorH;
   int width = atoi(iFrom.valueForKey("width")->value().c_str());
   int height = atoi(iFrom.valueForKey("height")->value().c_str());
   iFrame->MoveResize(x,y,width,height);
   iFrame->SetWMPosition(x, y);
}

void
FWGUIManager::setFrom(const FWConfiguration& iFrom) {
   // main window
   if (m_viewSecPack) subviewDestroyAll();

   const FWConfiguration* mw = iFrom.valueForKey(kMainWindow);
   assert(mw != 0);
   // Window needs to mapped before moving, otherwise move can lead
   // to wrong results on some window managers.
   m_cmsShowMainFrame->MapWindow();
   setWindowInfoFrom(*mw, m_cmsShowMainFrame);
   m_cmsShowMainFrame->MapSubwindows();
   m_cmsShowMainFrame->Layout();
   m_cmsShowMainFrame->MapRaised();

   // set from view reading area info nd view info
   float_t leftWeight =1;
   float_t rightWeight=1;
   if ( mw->version() >= 2 ) {
      leftWeight = atof(mw->valueForKey("leftWeight")->value().c_str());
      rightWeight = atof(mw->valueForKey("rightWeight")->value().c_str());
   }

   if ( mw->version() >= 3 ) {
      float summaryWeight = atof(mw->valueForKey("summaryWeight")->value().c_str());
      m_cmsShowMainFrame->setSummaryViewWeight(summaryWeight);       
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
         TEveWindowSlot* slot = ( m_viewMap.size() || (primSlot == 0) ) ? m_viewSecPack->NewSlotWithWeight(weight) : primSlot;
         std::string name = FWViewType::checkNameWithViewVersion(it->first, it->second.version());
         ViewMap_i lastViewIt = createView(name, slot);
         lastViewIt->second->setFrom(it->second);

         bool  undocked = atof((areaIt->second).valueForKey("undocked")->value().c_str());
         if (undocked)
         {
            TEveWindow* lastWindow = lastViewIt->first;
            lastWindow->UndockWindow();
            TEveCompositeFrameInMainFrame* emf = dynamic_cast<TEveCompositeFrameInMainFrame*>(lastWindow->GetEveFrame());
            if (emf ) {
               const TGMainFrame* mf =  dynamic_cast<const TGMainFrame*>(emf->GetParent());
               if (mf) {
                  m_cmsShowMainFrame->bindCSGActionKeys(mf);
                  TGMainFrame* mfp = (TGMainFrame*)mf; // have to cast in non-const
                  const FWConfiguration* mwc = (areaIt->second).valueForKey("UndockedWindowPos");
                  setWindowInfoFrom(*mwc, mfp);
               }
            }
         }
         areaIt++;
      }
   }
   else
   {  // create views with same weight in old version
      for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it!= keyVals->end(); ++it) {
         std::string name = FWViewType::checkNameWithViewVersion(it->first, it->second.version());       
         createView(name, m_viewMap.size() ? m_viewSecPack->NewSlot() : primSlot); 	 

         ViewMap_i lastViewIt = m_viewMap.end(); lastViewIt--;
         lastViewIt->second->setFrom(it->second);
      }
      // handle undocked windows in old version
      const FWConfiguration* undocked = iFrom.valueForKey(kUndocked);
      if(0!=undocked) {
         fwLog(fwlog::kWarning) << "Restrore of undocked windows with old window management not supported." << std::endl;
      }
   }

   //handle controllers
   const FWConfiguration* controllers = iFrom.valueForKey(kControllers);
   if (0 != controllers)
   {
      const FWConfiguration::KeyValues* keyVals = controllers->keyValues();
      if (0 != keyVals)
      {
         //we have open controllers
         for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it != keyVals->end(); ++it)
         {
            const std::string& controllerName = it->first;
            // std::cout <<"found controller "<<controllerName<<std::endl;
            if (controllerName == kCollectionController) {
               showEDIFrame();
               setWindowInfoFrom(it->second,m_ediFrame);
            } else if (controllerName == kViewController) {
               setViewPopup(0);
               setWindowInfoFrom(it->second, m_viewPopup);
            } else if (controllerName == kObjectController) {
               showModelPopup();
               setWindowInfoFrom(it->second, m_modelPopup);
            } else if (controllerName == kCommonController) {
               showCommonPopup();
               setWindowInfoFrom(it->second, m_commonPopup);
            }
         }
      }
   }


   for(ViewMap_i it = m_viewMap.begin(); it != m_viewMap.end(); ++it)
   {
      if (it->second->typeId() >= FWViewType::kGeometryTable)
      {
         FWGeometryTableViewBase* gv = ( FWGeometryTableViewBase*)it->second;
         gv->populate3DViewsFromConfig();
      }
   }

   // disable first docked view
   checkSubviewAreaIconState(0);
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
   if (m_cmsShowMainFrame->m_runEntry->GetUIntNumber() == edm::invalidRunNumber)
   {
      m_cmsShowMainFrame->m_runEntry->SetUIntNumber(1);
   }

   m_cmsShowMainFrame->m_lumiEntry->SetText("", kFALSE);
   m_cmsShowMainFrame->m_lumiEntry->SetFocus();
}

void FWGUIManager::lumiIdChanged()
{
   if (m_cmsShowMainFrame->m_lumiEntry->GetUIntNumber() == edm::invalidLuminosityBlockNumber)
   {
      m_cmsShowMainFrame->m_lumiEntry->SetUIntNumber(1);
   }

   m_cmsShowMainFrame->m_eventEntry->SetText("", kFALSE);
   m_cmsShowMainFrame->m_eventEntry->SetFocus();
}

void FWGUIManager::eventIdChanged()
{
   if (m_cmsShowMainFrame->m_eventEntry->GetUIntNumber() == edm::invalidEventNumber)
   {
      m_cmsShowMainFrame->m_eventEntry->SetULong64Number(1);
   }

   changedEventId_.emit(m_cmsShowMainFrame->m_runEntry->GetUIntNumber(),
                        m_cmsShowMainFrame->m_lumiEntry->GetUIntNumber(),
                        m_cmsShowMainFrame->m_eventEntry->GetULong64Number());
}

void
FWGUIManager::finishUpColorChange()
{
   if (m_commonPopup) m_commonPopup->colorSetChanged();
   if (m_modelPopup)  m_modelPopup->colorSetChanged();
   if (m_ediFrame)    m_ediFrame->colorSetChanged();

   gEve->FullRedraw3D();
}
//______________________________________________________________________________

void
FWGUIManager::showEventFilterGUI()
{
   showEventFilterGUI_.emit(m_cmsShowMainFrame);
}

void
FWGUIManager::filterButtonClicked()
{
   filterButtonClicked_.emit();
}

void
FWGUIManager::setFilterButtonText(const char* txt)
{
   m_cmsShowMainFrame->m_filterShowGUIBtn->SetText(txt);
}

void
FWGUIManager::setFilterButtonIcon(int state)
{
   int i = state*3;
   m_cmsShowMainFrame->m_filterEnableBtn->setIcons(m_cmsShowMainFrame->m_filterIcons[i],
                                                   m_cmsShowMainFrame->m_filterIcons[i+1],
                                                   m_cmsShowMainFrame->m_filterIcons[i+2]);
}

void
FWGUIManager::updateEventFilterEnable(bool btnEnabled)
{
   m_cmsShowMainFrame->m_filterEnableBtn->SetEnabled(btnEnabled);
}

void
FWGUIManager::measureWMOffsets()
{
  const Int_t x = 100, y = 100;

  TGMainFrame *mf1 = new TGMainFrame(0, 0, 0);
  mf1->MapWindow();
  mf1->Move(x, y);

  // This seems to be the only reliable way to make sure Move() has been processed.
  {
    TGMainFrame *mf2 = new TGMainFrame(0, 0, 0);
    mf2->MapWindow();
    while (!mf2->IsMapped()) gClient->HandleInput();
    delete mf2;
  }
  {
    Int_t    xm, ym;
    Window_t childdum;
    WindowAttributes_t attr;
    gVirtualX->TranslateCoordinates(mf1->GetId(), gClient->GetDefaultRoot()->GetId(),
                                    0, 0, xm, ym, childdum);
    gVirtualX->GetWindowAttributes(mf1->GetId(), attr);
    m_WMOffsetX = x - xm;
    m_WMOffsetY = y - ym;
    m_WMDecorH  = attr.fY;
    fwLog(fwlog::kDebug) << Form("FWGUIManager::measureWMOffsets: required (%d,%d), measured(%d, %d) => dx=%d, dy=%d; decor_h=%d.\n",
                                 x, y, xm, ym, m_WMOffsetX, m_WMOffsetY, m_WMDecorH);
  }
  delete mf1;
}

void
FWGUIManager::resetWMOffsets()
{
   m_WMOffsetX = m_WMOffsetY = m_WMDecorH = 0;
}

void
FWGUIManager::initEmpty()
{   
   int x = 150 + m_WMOffsetX ;
   int y = 50 +  m_WMOffsetY;
   m_cmsShowMainFrame->Move(x, y);
   m_cmsShowMainFrame->SetWMPosition(x, y < m_WMDecorH ? m_WMDecorH : y);
  
   createView("Rho Phi"); 
   createView("Rho Z"); 

   m_cmsShowMainFrame->MapSubwindows();
   m_cmsShowMainFrame->Layout();
   m_cmsShowMainFrame->MapRaised();
}

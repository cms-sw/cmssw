// -*- C++ -*-
//
// Package:     Core
// Class  :     FWSummaryManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Mar  4 09:35:32 EST 2008
// $Id: FWSummaryManager.cc,v 1.22 2010/06/18 10:17:16 yana Exp $
//

// system include files
#include <boost/bind.hpp>

#include "TGFrame.h"

// user include files
#include "Fireworks/Core/interface/FWSummaryManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "Fireworks/Core/src/FWCollectionSummaryWidget.h"
#include "Fireworks/Core/interface/FWDataCategories.h"

#include "Fireworks/Core/src/FWCompactVerticalLayout.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWSummaryManager::FWSummaryManager(TGFrame* iParent,
                                   FWSelectionManager* sm,
                                   FWEventItemsManager* eim,
                                   FWGUIManager* gm,
                                   FWModelChangeManager* cm,
                                   FWColorManager* colorm
) :
   m_guiManager(gm),
   m_colorManager(colorm),
m_itemChanged(false)
{
   colorm->colorsHaveChanged_.connect(boost::bind(&FWSummaryManager::colorsChanged,this));
   sm->selectionChanged_.connect(boost::bind(&FWSummaryManager::selectionChanged,this,_1));
   eim->newItem_.connect(boost::bind(&FWSummaryManager::newItem,
                                     this, _1) );
   eim->goingToClearItems_.connect(boost::bind(&FWSummaryManager::removeAllItems, this));
   eim->goingToClearItems_.connect(boost::bind(&FWModelChangeManager::itemsGoingToBeClearedSlot, cm));

   m_pack = new TGVerticalFrame(iParent);
   m_pack->SetLayoutManager( new FWCompactVerticalLayout(m_pack));
   const unsigned int backgroundColor=0x2f2f2f;
   m_pack->SetBackgroundColor(backgroundColor);
   cm->changeSignalsAreDone_.connect(boost::bind(&FWSummaryManager::changesDone,this));
}

// FWSummaryManager::FWSummaryManager(const FWSummaryManager& rhs)
// {
//    // do actual copying here;
// }

FWSummaryManager::~FWSummaryManager()
{
}

//
// assignment operators
//
// const FWSummaryManager& FWSummaryManager::operator=(const FWSummaryManager& rhs)
// {
//   //An exception safe implementation is
//   FWSummaryManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWSummaryManager::newItem(FWEventItem* iItem)
{
   TGLayoutHints* hints = new TGLayoutHints(kLHintsExpandX);
   FWCollectionSummaryWidget* lst = new FWCollectionSummaryWidget(m_pack,*iItem,hints);
   m_pack->AddFrame(lst, hints);
   m_collectionWidgets.push_back(lst);
   bool backgroundIsWhite = m_colorManager->backgroundColorIndex()==FWColorManager::kWhiteIndex;
   lst->setBackgroundToWhite(backgroundIsWhite);
   iItem->goingToBeDestroyed_.connect(boost::bind(&FWSummaryManager::itemDestroyed,this,_1));
   iItem->itemChanged_.connect(boost::bind(&FWSummaryManager::itemChanged,this,_1));
   lst->Connect("requestForInfo(FWEventItem*)","FWSummaryManager",this,"requestForInfo(FWEventItem*)");
   lst->Connect("requestForFilter(FWEventItem*)","FWSummaryManager",this,"requestForFilter(FWEventItem*)");
   lst->Connect("requestForErrorInfo(FWEventItem*)","FWSummaryManager",this,"requestForError(FWEventItem*)");
   lst->Connect("requestForController(FWEventItem*)","FWSummaryManager",this,"requestForController(FWEventItem*)");
   lst->Connect("requestForModelContextMenu(Int_t,Int_t)","FWSummaryManager",this,"requestForSelectedModelContextMenu(Int_t,Int_t)");
}

void 
FWSummaryManager::itemDestroyed(const FWEventItem* iItem)
{
   m_pack->HideFrame(m_collectionWidgets[iItem->id()]);
   m_pack->RemoveFrame(m_collectionWidgets[iItem->id()]);
   delete m_collectionWidgets[iItem->id()];
   m_collectionWidgets[iItem->id()]=0;
   m_pack->Layout();
   gClient->NeedRedraw(m_pack);
}

void
FWSummaryManager::itemChanged(const FWEventItem*)
{
   m_itemChanged = true;
}

void
FWSummaryManager::removeAllItems()
{
   for(std::vector<FWCollectionSummaryWidget*>::iterator it = m_collectionWidgets.begin(), 
       itEnd = m_collectionWidgets.end();
       it != itEnd;
       ++it) {
      if(0!=*it) {
         m_pack->HideFrame(*it);
         m_pack->RemoveFrame(*it);
         delete *it;
         *it=0;
      }
   }
   m_collectionWidgets.clear();
   m_pack->Layout();
   gClient->NeedRedraw(m_pack);
}

void
FWSummaryManager::selectionChanged(const FWSelectionManager& iSM)
{
}

void
FWSummaryManager::changesDone()
{
   if(m_itemChanged) {
      m_pack->Layout();
      m_itemChanged=false;
   }
}

void
FWSummaryManager::colorsChanged()
{
   bool backgroundIsWhite = m_colorManager->backgroundColorIndex()==FWColorManager::kWhiteIndex;
   
   if(m_colorManager->isColorSetLight()) {
      m_pack->SetBackgroundColor(TGFrame::GetDefaultFrameBackground());
   } else {
      const unsigned int backgroundColor=0x2f2f2f;
      m_pack->SetBackgroundColor(backgroundColor);
   }
   gClient->NeedRedraw(m_pack);
   for(std::vector<FWCollectionSummaryWidget*>::iterator it = m_collectionWidgets.begin(), 
          itEnd = m_collectionWidgets.end();
       it != itEnd;
       ++it) {
      if(0!=*it) {
         (*it)->setBackgroundToWhite(backgroundIsWhite);
      }
   }
}

void 
FWSummaryManager::requestForInfo(FWEventItem* iItem)
{
   m_guiManager->showEDIFrame(kData);
}
void 
FWSummaryManager::requestForFilter(FWEventItem* iItem)
{
   m_guiManager->showEDIFrame(kFilter);
}
void 
FWSummaryManager::requestForError(FWEventItem* iItem)
{
   m_guiManager->showEDIFrame();
}

void 
FWSummaryManager::requestForController(FWEventItem* iItem)
{
   m_guiManager->showEDIFrame();
}

void 
FWSummaryManager::requestForSelectedModelContextMenu(Int_t iGlobalX, Int_t iGlobalY)
{
   m_guiManager->showSelectedModelContextMenu(iGlobalX,iGlobalY,0);
}

//
// const member functions
//
TGCompositeFrame*
FWSummaryManager::widget() const
{
   return m_pack;
}

//
// static member functions
//

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
// $Id: FWSummaryManager.cc,v 1.3 2008/06/23 15:50:16 chrjones Exp $
//

// system include files
#include "TGListTree.h"
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/FWSummaryManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/src/FWListEventItem.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWSummaryManager::FWSummaryManager(TGListTree* iListTree,
                                   FWSelectionManager* sm,
                                   FWEventItemsManager* eim,
                                   FWDetailViewManager* dv,
                                   FWModelChangeManager* cm):
m_detailViewManager(dv)
{
   sm->selectionChanged_.connect(boost::bind(&FWSummaryManager::selectionChanged,this,_1));
   eim->newItem_.connect(boost::bind(&FWSummaryManager::newItem,
                                             this, _1) );
   eim->goingToClearItems_.connect(boost::bind(&FWSummaryManager::removeAllItems,this));

   
   m_listTree = iListTree;
   /*m_eventObjects =  new TEveElementList("Physics Objects");
   m_listTree->OpenItem(m_eventObjects->AddIntoListTree(m_listTree,
                                                        reinterpret_cast<TGListTreeItem*>(0))
                        );*/
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
FWSummaryManager::newItem(const FWEventItem* iItem)
{
   //TEveElementList* lst = new TEveElementList(iItem->name().c_str(),"",kTRUE);
   //lst->SetMainColor(iItem->defaultDisplayProperties().color());
   //NEED TO CHANGE THE SIGNATURE OF THE SIGNAL
   TEveElementList* lst = new FWListEventItem( const_cast<FWEventItem*>(iItem),
                                              m_detailViewManager);
  lst->AddIntoListTree(m_listTree,
                       reinterpret_cast<TGListTreeItem*>(0));
                     
   //lst->AddIntoListTree(m_listTree,m_eventObjects);
   //NOTE: Why didn't I call AddElement of m_eventObjects???  Because it will go in the wrong list tree?
   lst->IncDenyDestroy();
   //Need to hand this lst into some container so we can get rid of it, since it doesn't seem to go into
   // m_eventObjects
}

void
FWSummaryManager::removeAllItems()
{
   //m_eventObjects->DestroyElements();
}

void 
FWSummaryManager::selectionChanged(const FWSelectionManager& iSM)
{
   //m_unselectAllButton->SetEnabled( 0 !=iSM.selected().size() );
}

void 
FWSummaryManager::changesDone()
{
   m_listTree->ClearViewPort();
}

//
// const member functions
//

//
// static member functions
//

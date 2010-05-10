// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWFromTEveCaloDataSelector
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri Oct 23 14:44:33 CDT 2009
// $Id: FWFromTEveCaloDataSelector.cc,v 1.8 2010/05/10 11:49:40 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>
#include <algorithm>
#include "TH2.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

// user include files
#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"


//
// constants, enums and typedefs
//

FWFromSliceSelector::FWFromSliceSelector(TH2F* iHist,
                                         const FWEventItem* iItem) :
m_hist(iHist),
m_item(iItem)
{
}

void
FWFromSliceSelector::doSelect(const TEveCaloData::CellId_t& iCell)
{
   if (!m_item) return;

   const CaloTowerCollection* towers=0;
   m_item->get(towers);
   assert(0!=towers);
   int index = 0;
   FWChangeSentry sentry(*(m_item->changeManager()));
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower,++index) {
      if (m_hist->FindBin(tower->eta(),tower->phi()) == iCell.fTower && 
          m_item->modelInfo(index).m_displayProperties.isVisible() &&
          !m_item->modelInfo(index).isSelected()) {
         //std::cout <<"  doSelect "<<index<<std::endl;
         m_item->select(index);
      }
   }
}

void 
FWFromSliceSelector::clear()
{
   if (!m_item) return;

   const CaloTowerCollection* towers=0;
   m_item->get(towers);
   
   int index = 0;
   
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower,++index) {
      if( m_item->modelInfo(index).m_displayProperties.isVisible() &&
         m_item->modelInfo(index).isSelected()) {
         m_item->unselect(index);
      }
   }
}

FWModelChangeManager* 
FWFromSliceSelector::changeManager() const {
   return m_item->changeManager();
}


void
FWFromSliceSelector::doUnselect(const TEveCaloData::CellId_t& iCell)
{
   if (!m_item) return;

   const CaloTowerCollection* towers=0;
   m_item->get(towers);
   assert(0!=towers);
   int index = 0;
   FWChangeSentry sentry(*(m_item->changeManager()));
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower,++index) {
      if (m_hist->FindBin(tower->eta(),tower->phi()) == iCell.fTower && 
          m_item->modelInfo(index).m_displayProperties.isVisible() &&
          m_item->modelInfo(index).isSelected()) {
         //std::cout <<"  doUnselect "<<index<<std::endl;
         m_item->unselect(index);
      }
   }
}

void
FWFromSliceSelector::reset()
{
   m_item = 0;
   m_hist = 0;
}

//
// static data member definitions
//

//
// constructors and destructor
//
FWFromTEveCaloDataSelector::FWFromTEveCaloDataSelector(TEveCaloData* iData):
m_data(iData),
m_changeManager(0)
{
   // reserve 3 , first slice is background
   m_sliceSelectors.reserve(3);
   m_sliceSelectors.push_back(FWFromSliceSelector(0, 0));
}

// FWFromTEveCaloDataSelector::FWFromTEveCaloDataSelector(const FWFromTEveCaloDataSelector& rhs)
// {
//    // do actual copying here;
// }

//FWFromTEveCaloDataSelector::~FWFromTEveCaloDataSelector()
//{
//}

//
// assignment operators
//
// const FWFromTEveCaloDataSelector& FWFromTEveCaloDataSelector::operator=(const FWFromTEveCaloDataSelector& rhs)
// {
//   //An exception safe implementation is
//   FWFromTEveCaloDataSelector temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWFromTEveCaloDataSelector::doSelect()
{
   assert(m_changeManager);
   FWChangeSentry sentry(*m_changeManager);
   std::for_each(m_sliceSelectors.begin(),
                 m_sliceSelectors.end(),
                 boost::bind(&FWFromSliceSelector::clear,_1));
   const TEveCaloData::vCellId_t& cellIds = m_data->GetCellsSelected();
   for(TEveCaloData::vCellId_t::const_iterator it = cellIds.begin(),itEnd=cellIds.end();
       it != itEnd;
       ++it) {
      assert(it->fSlice < static_cast<int>(m_sliceSelectors.size()));
      m_sliceSelectors[it->fSlice].doSelect(*it);
   }
}

void 
FWFromTEveCaloDataSelector::doUnselect()
{
   assert(m_changeManager);
   //std::cout <<"FWFromTEveCaloDataSelector::doUnselect()"<<std::endl;
   
   FWChangeSentry sentry(*m_changeManager);
   const TEveCaloData::vCellId_t& cellIds = m_data->GetCellsSelected();
   for(TEveCaloData::vCellId_t::const_iterator it = cellIds.begin(),itEnd=cellIds.end();
       it != itEnd;
       ++it) {
      assert(it->fSlice < static_cast<int>(m_sliceSelectors.size()));
      m_sliceSelectors[it->fSlice].doUnselect(*it);
   }
}

void 
FWFromTEveCaloDataSelector::addSliceSelector(int iSlice, const FWFromSliceSelector& iSelector)
{
   assert(iSlice>0 && (iSlice <= static_cast<int>(m_sliceSelectors.size())));

   if(0==m_changeManager) {
      m_changeManager = iSelector.changeManager();
   }

   if(iSlice ==static_cast<int>(m_sliceSelectors.size())) {
      m_sliceSelectors.push_back(iSelector);
   } else {
      assert(iSlice<static_cast<int>(m_sliceSelectors.size()));
      m_sliceSelectors[iSlice]=iSelector;
   }
}

void 
FWFromTEveCaloDataSelector::resetSliceSelector(int iSlice)
{
   m_sliceSelectors[iSlice].reset();
}
//
// const member functions
//

//
// static member functions
//

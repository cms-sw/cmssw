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
// $Id$
//

// system include files
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
   const CaloTowerCollection* towers=0;
   m_item->get(towers);
   assert(0!=towers);
   int index = 0;
   FWChangeSentry(*(m_item->changeManager()));
   for(CaloTowerCollection::const_iterator tower = towers->begin(); tower != towers->end(); ++tower,++index) {
      if (m_hist->FindBin(tower->eta(),tower->phi()) == iCell.fTower) {
         m_item->select(index);
      }
   }
}

//
// static data member definitions
//

//
// constructors and destructor
//
FWFromTEveCaloDataSelector::FWFromTEveCaloDataSelector(TEveCaloData* iData):
m_data(iData)
{
   m_sliceSelectors.reserve(2);
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
   const TEveCaloData::vCellId_t& cellIds = m_data->GetCellsSelected();
   for(TEveCaloData::vCellId_t::const_iterator it = cellIds.begin(),itEnd=cellIds.end();
       it != itEnd;
       ++it) {
      assert(it->fSlice < static_cast<int>(m_sliceSelectors.size()));
      m_sliceSelectors[it->fSlice].doSelect(*it);
   }
}

void 
FWFromTEveCaloDataSelector::addSliceSelector(int iSlice, const FWFromSliceSelector& iSelector)
{
   assert(iSlice ==static_cast<int>(m_sliceSelectors.size()));
   m_sliceSelectors.push_back(iSelector);
}

//
// const member functions
//

//
// static member functions
//

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
//

// system include files
#include <algorithm>
#include <cassert>
#include <functional>

// user include files
#include "Fireworks/Calo/src/FWFromTEveCaloDataSelector.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"

//
// static data member definitions
//

//
// constructors and destructor
//
FWFromTEveCaloDataSelector::FWFromTEveCaloDataSelector(TEveCaloData* iData) : m_data(iData), m_changeManager(nullptr) {
  // reserve 3 , first slice is background
  m_sliceSelectors.reserve(3);
  m_sliceSelectors.push_back(new FWFromSliceSelector(nullptr));
}

// FWFromTEveCaloDataSelector::FWFromTEveCaloDataSelector(const FWFromTEveCaloDataSelector& rhs)
// {
//    // do actual copying here;
// }

FWFromTEveCaloDataSelector::~FWFromTEveCaloDataSelector() {
  for (std::vector<FWFromSliceSelector*>::iterator i = m_sliceSelectors.begin(); i != m_sliceSelectors.end(); ++i) {
    delete *i;
  }
  m_sliceSelectors.clear();
}

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
void FWFromTEveCaloDataSelector::doSelect() {
  assert(m_changeManager);
  FWChangeSentry sentry(*m_changeManager);
  std::for_each(
      m_sliceSelectors.begin(), m_sliceSelectors.end(), std::bind(&FWFromSliceSelector::clear, std::placeholders::_1));
  const TEveCaloData::vCellId_t& cellIds = m_data->GetCellsSelected();
  for (TEveCaloData::vCellId_t::const_iterator it = cellIds.begin(), itEnd = cellIds.end(); it != itEnd; ++it) {
    assert(it->fSlice < static_cast<int>(m_sliceSelectors.size()));
    m_sliceSelectors[it->fSlice]->doSelect(*it);
  }
}

void FWFromTEveCaloDataSelector::doUnselect() {
  assert(m_changeManager);
  //std::cout <<"FWFromTEveCaloDataSelector::doUnselect()"<<std::endl;

  FWChangeSentry sentry(*m_changeManager);
  const TEveCaloData::vCellId_t& cellIds = m_data->GetCellsSelected();
  for (TEveCaloData::vCellId_t::const_iterator it = cellIds.begin(), itEnd = cellIds.end(); it != itEnd; ++it) {
    assert(it->fSlice < static_cast<int>(m_sliceSelectors.size()));
    m_sliceSelectors[it->fSlice]->doUnselect(*it);
  }
}

void FWFromTEveCaloDataSelector::addSliceSelector(int iSlice, FWFromSliceSelector* iSelector) {
  assert(iSlice > 0 && (iSlice <= static_cast<int>(m_sliceSelectors.size())));

  if (nullptr == m_changeManager) {
    m_changeManager = iSelector->changeManager();
  }

  if (iSlice == static_cast<int>(m_sliceSelectors.size())) {
    m_sliceSelectors.push_back(iSelector);
  } else {
    assert(iSlice < static_cast<int>(m_sliceSelectors.size()));
    m_sliceSelectors[iSlice] = iSelector;
  }
}

void FWFromTEveCaloDataSelector::resetSliceSelector(int iSlice) { m_sliceSelectors[iSlice]->reset(); }
//
// const member functions
//

//
// static member functions
//

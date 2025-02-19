// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWFromSliceSelectorBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Wed Jun  2 17:30:49 CEST 2010
// $Id: FWFromSliceSelector.cc,v 1.4 2010/06/18 10:17:51 yana Exp $
//

// system include files

// user include files
#include "Fireworks/Calo/src/FWFromSliceSelector.h"
#include "Fireworks/Core/interface/FWEventItem.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWFromSliceSelector::FWFromSliceSelector(const FWEventItem* iItem) :
m_item(iItem)
{
}

FWFromSliceSelector::~FWFromSliceSelector()
{
}

//
// member functions
//

void 
FWFromSliceSelector::clear()
{
   if (!m_item) return;

   int size =  static_cast<int>(m_item->size());
   for(int index=0; index < size; ++index)
   {
      if( m_item->modelInfo(index).m_displayProperties.isVisible() &&
          m_item->modelInfo(index).isSelected()) {
         m_item->unselect(index);
      }
   }
}

void
FWFromSliceSelector::reset()
{
   m_item = 0;
}

//
// const member functions
//
FWModelChangeManager* 
FWFromSliceSelector::changeManager() const {
   return m_item->changeManager();
}

//
// static member functions
//

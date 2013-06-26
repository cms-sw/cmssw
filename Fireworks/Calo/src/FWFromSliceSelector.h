#ifndef Fireworks_Calo_FWFromSliceSelector_h
#define Fireworks_Calo_FWFromSliceSelector_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWFromSliceSelector
// 
/**\class FWFromSliceSelector FWFromSliceSelector.h Fireworks/Calo/interface/FWFromSliceSelector.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Jun  2 17:30:39 CEST 2010
// $Id: FWFromSliceSelector.h,v 1.2 2010/06/07 17:54:00 amraktad Exp $
//

// system include files

// user include files
#include "TEveCaloData.h"

// forward declarations
class FWEventItem;
class FWModelChangeManager;


class FWFromSliceSelector
{
public:
   FWFromSliceSelector( const FWEventItem*);
   virtual ~FWFromSliceSelector();

   virtual void doSelect(const TEveCaloData::CellId_t&) {}
   virtual void doUnselect(const TEveCaloData::CellId_t&) {}

   void clear();
   void reset();

   // ---------- const member functions ---------------------

   FWModelChangeManager* changeManager() const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
protected:
   const FWEventItem* m_item;
};


#endif

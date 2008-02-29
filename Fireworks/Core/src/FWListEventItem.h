#ifndef Fireworks_Core_FWListEventItem_h
#define Fireworks_Core_FWListEventItem_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListEventItem
// 
/**\class FWListEventItem FWListEventItem.h Fireworks/Core/interface/FWListEventItem.h

 Description: Adapter between the list view and a FWEventItem

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 28 11:05:02 PST 2008
// $Id$
//

// system include files
#include "TEveElement.h"

// user include files

// forward declarations
class FWEventItem;

class FWListEventItem : public TEveElementList
{

   public:
      FWListEventItem(FWEventItem*);
      virtual ~FWListEventItem();

      // ---------- const member functions ---------------------
      FWEventItem* eventItem() const;
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void SetMainColor(Color_t);
   
   private:
      FWListEventItem(const FWListEventItem&); // stop default

      const FWListEventItem& operator=(const FWListEventItem&); // stop default

      // ---------- member data --------------------------------
      FWEventItem* m_item;
   
};


#endif

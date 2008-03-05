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
// $Id: FWListEventItem.h,v 1.2 2008/03/05 15:13:50 chrjones Exp $
//

// system include files
#include <set>
#include "TEveElement.h"

// user include files

// forward declarations
class FWEventItem;
class FWDetailViewManager;
class FWModelId;

class FWListEventItem : public TEveElementList
{

   public:
      FWListEventItem(FWEventItem*,
                      FWDetailViewManager*);
      virtual ~FWListEventItem();

      // ---------- const member functions ---------------------
      FWEventItem* eventItem() const;
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void SetMainColor(Color_t);
      virtual void SetRnrChildren(Bool_t rnr);
   ClassDef(FWListEventItem,0);

      void openDetailViewFor(int index) const;
   private:
      void itemChanged(const FWEventItem*);
      void modelsChanged( const std::set<FWModelId>& );

      FWListEventItem(const FWListEventItem&); // stop default

      const FWListEventItem& operator=(const FWListEventItem&); // stop default

      // ---------- member data --------------------------------
      FWEventItem* m_item;
      FWDetailViewManager* m_detailViewManager;
};


#endif

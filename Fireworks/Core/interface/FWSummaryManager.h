#ifndef Fireworks_Core_FWSummaryManager_h
#define Fireworks_Core_FWSummaryManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWSummaryManager
// 
/**\class FWSummaryManager FWSummaryManager.h Fireworks/Core/interface/FWSummaryManager.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Mar  4 09:35:58 EST 2008
// $Id: FWSummaryManager.h,v 1.1 2008/03/05 15:07:32 chrjones Exp $
//

// system include files
#include "Rtypes.h"

// user include files

// forward declarations
class TGListTree;
class TGListTreeItem;
class TObject;
class TEveElementList;
class FWEventItem;

class FWSelectionManager;
class FWEventItemsManager;
class FWDetailViewManager;
class FWModelChangeManager;

class FWSummaryManager
{

   public:
      FWSummaryManager(TGListTree* iParent,
                       FWSelectionManager*,
                       FWEventItemsManager*,
                       FWDetailViewManager*,
                       FWModelChangeManager*);
      virtual ~FWSummaryManager();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   
   private:
      FWSummaryManager(const FWSummaryManager&); // stop default

      const FWSummaryManager& operator=(const FWSummaryManager&); // stop default

      void selectionChanged(const FWSelectionManager&);
      void newItem(const FWEventItem* iItem);
      void removeAllItems();
      void changesDone();

      // ---------- member data --------------------------------
      TGListTree* m_listTree;
      TEveElementList* m_eventObjects;
      FWDetailViewManager* m_detailViewManager;
};


#endif

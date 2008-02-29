#ifndef Fireworks_Core_FWModelId_h
#define Fireworks_Core_FWModelId_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelId
// 
/**\class FWModelId FWModelId.h Fireworks/Core/interface/FWModelId.h

 Description: identifies a particular model within an FWEventItem

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jan 18 12:44:26 EST 2008
// $Id: FWModelId.h,v 1.1 2008/01/21 01:17:12 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
class FWEventItem;

// forward declarations

class FWModelId
{

   public:
      enum { kContainerIndex=-1};
      FWModelId(const FWEventItem* iItem=0,
                int iIndex=kContainerIndex): m_item(iItem),m_index(iIndex) {}
      //virtual ~FWModelId();

      // ---------- const member functions ---------------------
      bool operator<(const FWModelId& iRHS) const
      {
         return m_item == iRHS.m_item ? m_index<iRHS.m_index : m_item<iRHS.m_item;
      }
   
      const FWEventItem* item() const 
      {
         return m_item;
      }
   
      int index() const 
      {
         return m_index;
      } 
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void unselect() const { if(m_item) {m_item->unselect(m_index);}}
      void select() const { if(m_item) {m_item->select(m_index);}}
      void toggleSelect() const {}
      void setIndex(int iIndex) { m_index=iIndex;}
   private:
      //FWModelId(const FWModelId&); // stop default

      //const FWModelId& operator=(const FWModelId&); // stop default

      // ---------- member data --------------------------------
      const FWEventItem* m_item;
      int m_index;

};


#endif

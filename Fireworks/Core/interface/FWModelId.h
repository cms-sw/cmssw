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
// $Id: FWModelId.h,v 1.6 2009/01/23 21:35:41 amraktad Exp $
//

// system include files

// user include files
class FWEventItem;

// forward declarations

class FWModelId
{

public:
   enum { kContainerIndex=-1};
   FWModelId(const FWEventItem* iItem=0,
             int iIndex=kContainerIndex) : m_item(iItem),m_index(iIndex) {
   }
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
   void unselect() const ;
   void select() const;
   void toggleSelect() const;
   void setIndex(int iIndex) {
      m_index=iIndex;
   }
private:
   //FWModelId(const FWModelId&); // stop default

   //const FWModelId& operator=(const FWModelId&); // stop default

   // ---------- member data --------------------------------
   const FWEventItem* m_item;
   int m_index;

};


#endif

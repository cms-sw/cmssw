#ifndef Fireworks_Core_FWItemSingleAccessor_h
#define Fireworks_Core_FWItemSingleAccessor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemSingleAccessor
//
/**\class FWItemSingleAccessor FWItemSingleAccessor.h Fireworks/Core/interface/FWItemSingleAccessor.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 18 11:36:41 EDT 2008
// $Id: FWItemSingleAccessor.h,v 1.4 2010/07/23 16:02:54 eulisse Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWItemAccessorBase.h"

// forward declarations

class FWItemSingleAccessor : public FWItemAccessorBase {

public:
   FWItemSingleAccessor(const TClass*);
   virtual ~FWItemSingleAccessor();

   // ---------- const member functions ---------------------
   const void* modelData(int iIndex) const;
   const void* data() const;
   unsigned int size() const;
   const TClass* modelType() const;
   const TClass* type() const;

   bool isCollection() const;

   // ---------- member functions ---------------------------
   void setData(const Reflex::Object& );
   virtual void reset();

private:
   FWItemSingleAccessor(const FWItemSingleAccessor&); // stop default

   const FWItemSingleAccessor& operator=(const FWItemSingleAccessor&); // stop default

   // ---------- member data --------------------------------
   const TClass* m_type;
   const void* m_data;

};


#endif

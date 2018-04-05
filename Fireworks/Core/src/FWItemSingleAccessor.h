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
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWItemAccessorBase.h"

// forward declarations

class FWItemSingleAccessor : public FWItemAccessorBase {

public:
   FWItemSingleAccessor(const TClass*);
   ~FWItemSingleAccessor() override;

   // ---------- const member functions ---------------------
   const void* modelData(int iIndex) const override;
   const void* data() const override;
   unsigned int size() const override;
   const TClass* modelType() const override;
   const TClass* type() const override;

   bool isCollection() const override;

   // ---------- member functions ---------------------------
   void setData(const edm::ObjectWithDict& ) override;
   void reset() override;

private:
   FWItemSingleAccessor(const FWItemSingleAccessor&) = delete; // stop default

   const FWItemSingleAccessor& operator=(const FWItemSingleAccessor&) = delete; // stop default

   // ---------- member data --------------------------------
   const TClass* m_type;
   const void* m_data;

};


#endif

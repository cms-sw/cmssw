#ifndef Fireworks_Core_FWItemAccessorBase_h
#define Fireworks_Core_FWItemAccessorBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemAccessorBase
//
/**\class FWItemAccessorBase FWItemAccessorBase.h Fireworks/Core/interface/FWItemAccessorBase.h

   Description: Base class used to access data stored in an edm::EDProduct

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 18 08:14:21 EDT 2008
// $Id: FWItemAccessorBase.h,v 1.6 2012/08/03 18:20:27 wmtan Exp $
//

// system include files
#include <typeinfo>

// user include files

// forward declarations
class TClass;
namespace edm {
   class EDProduct;
   class ObjectWithDict;
}

class FWItemAccessorBase {

public:
   FWItemAccessorBase();
   virtual ~FWItemAccessorBase();

   // ---------- const member functions ---------------------
   virtual const void* modelData(int iIndex) const = 0;
   virtual const void* data() const = 0;
   virtual unsigned int size() const = 0;
   virtual const TClass* modelType() const = 0;
   virtual const TClass* type() const = 0;

   virtual bool isCollection() const = 0;

   ///override if id of an object should be different than the index
   //virtual std::string idForIndex(int iIndex) const;
   // ---------- member functions ---------------------------
   virtual void setData(const edm::ObjectWithDict& )=0;
   virtual void reset() = 0;

private:
   //FWItemAccessorBase(const FWItemAccessorBase&); // stop default

   //const FWItemAccessorBase& operator=(const FWItemAccessorBase&); // stop default

   // ---------- member data --------------------------------

};


#endif

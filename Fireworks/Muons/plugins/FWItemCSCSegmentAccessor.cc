// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemCSCSegmentAccessor
//
// Implementation:
//     An example of how to write a plugin based FWItemAccessorBase derived class.
//
// Original Author:  Giulio Eulisse
//         Created:  Thu Feb 18 15:19:44 EDT 2008
// $Id$
//

// system include files
#include <assert.h>
#include "Reflex/Object.h"
#include "TClass.h"

// user include files
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "Fireworks/Core/interface/FWItemAccessorBase.h"
#include "Fireworks/Core/interface/FWItemAccessorRegistry.h"

// forward declarations
class FWItemCSCSegmentAccessor : public FWItemAccessorBase 
{

public:
   FWItemCSCSegmentAccessor(const TClass *iClass);
   virtual ~FWItemCSCSegmentAccessor();

   REGISTER_FWITEMACCESSOR_METHODS();

   // ---------- const member functions ---------------------
   const void* modelData(int iIndex) const;
   const void* data() const;
   unsigned int size() const;
   const TClass* modelType() const;
   const TClass* type() const;

   bool isCollection() const;

   // ---------- member functions ---------------------------
   void setWrapper(const ROOT::Reflex::Object& );
   virtual void reset();

private:
   FWItemCSCSegmentAccessor(const FWItemCSCSegmentAccessor&); // stop default

   const FWItemCSCSegmentAccessor& operator=(const FWItemCSCSegmentAccessor&); // stop default

   // ---------- member data --------------------------------
   const TClass* m_type;
   mutable void* m_data;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
// FIXME: plugin manager does not support 
FWItemCSCSegmentAccessor::FWItemCSCSegmentAccessor(const TClass *iClass)
: m_type(iClass), m_data(0)
{
}

// FWItemCSCSegmentAccessor::FWItemCSCSegmentAccessor(const FWItemCSCSegmentAccessor& rhs)
// {
//    // do actual copying here;
// }

FWItemCSCSegmentAccessor::~FWItemCSCSegmentAccessor()
{
}

//
// assignment operators
//
// const FWItemCSCSegmentAccessor& FWItemCSCSegmentAccessor::operator=(const FWItemCSCSegmentAccessor& rhs)
// {
//   //An exception safe implementation is
//   FWItemCSCSegmentAccessor temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWItemCSCSegmentAccessor::setWrapper(const ROOT::Reflex::Object& iWrapper)
{
   if(0!=iWrapper.Address()) {
      using ROOT::Reflex::Object;
      //get the Event data from the wrapper
      Object product(iWrapper.Get("obj"));
      if(product.TypeOf().IsTypedef()) {
         product = Object(product.TypeOf().ToType(),product.Address());
      }
      m_data = product.Address();
      assert(0!=m_data);
   } else {
      reset();
   }
}

void
FWItemCSCSegmentAccessor::reset()
{
   m_data = 0;
}

//
// const member functions
//
const void*
FWItemCSCSegmentAccessor::modelData(int iIndex) const
{
   if (!m_data)
      return 0;
   return &(reinterpret_cast<CSCSegmentCollection *>(m_data)->operator[](iIndex));
}

const void*
FWItemCSCSegmentAccessor::data() const
{
   return m_data;
}

unsigned int
FWItemCSCSegmentAccessor::size() const
{
   return reinterpret_cast<const CSCSegmentCollection *>(m_data)->size();
}

const TClass*
FWItemCSCSegmentAccessor::type() const
{
   return m_type;
}

const TClass*
FWItemCSCSegmentAccessor::modelType() const
{
   TClass *type = TClass::GetClass(typeid(CSCSegmentCollection::value_type));
   assert(type);
   return type; 
}

bool
FWItemCSCSegmentAccessor::isCollection() const
{
   return true;
}
//
// static member functions
//

REGISTER_FWITEMACCESSOR(FWItemCSCSegmentAccessor,CSCSegmentCollection,"CSCSegmentCollectionAccessor");

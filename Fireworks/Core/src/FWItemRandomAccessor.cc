// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemRandomAccessor
//
// Implementation:
//    A generic helper class which can be used to create
//    a specialized FWItemAccessorBase plugin for
//    all the classes that expose a std::vector like interface.
//
// Original Author:  Giulio Eulisse
//         Created:  Thu Feb 18 15:19:44 EDT 2008
// $Id: FWItemRandomAccessor.cc,v 1.6 2013/02/10 22:12:04 wmtan Exp $
//

// system include files
#include <assert.h>
#include "TClass.h"

// user include files
#include "Fireworks/Core/interface/FWItemRandomAccessor.h"

// forward declarations

FWItemRandomAccessorBase::FWItemRandomAccessorBase(const TClass *type, const type_info &modelTypeName)
:m_type(type),
 m_modelType(TClass::GetClass(modelTypeName)),
 m_data(0)
{
}

// FWItemRandomAccessor::FWItemRandomAccessor(const FWItemRandomAccessor& rhs)
// {
//    // do actual copying here;
// }

FWItemRandomAccessorBase::~FWItemRandomAccessorBase()
{
}

//
// assignment operators
//
// const FWItemRandomAccessor& FWItemRandomAccessor::operator=(const FWItemRandomAccessor& rhs)
// {
//   //An exception safe implementation is
//   FWItemRandomAccessor temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWItemRandomAccessorBase::setData(const edm::ObjectWithDict& product)
{
   if (product.address() == 0)
   {
      reset();
      return;
   }
   
   m_data = product.address();
   assert(0!=m_data);
}

void
FWItemRandomAccessorBase::reset()
{
   m_data = 0;
}

//
// const member functions
//
const void*
FWItemRandomAccessorBase::data() const
{
   return m_data;
}

void*
FWItemRandomAccessorBase::getDataPtr() const
{
   return m_data;
}

const TClass*
FWItemRandomAccessorBase::type() const
{
   return m_type;
}

const TClass*
FWItemRandomAccessorBase::modelType() const
{
   assert(m_modelType);
   return m_modelType; 
}

bool
FWItemRandomAccessorBase::isCollection() const
{
   return true;
}
//
// static member functions
//

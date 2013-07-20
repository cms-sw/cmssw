// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemTVirtualCollectionProxyAccessor
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 18 08:43:47 EDT 2008
// $Id: FWItemTVirtualCollectionProxyAccessor.cc,v 1.11 2013/02/10 22:12:04 wmtan Exp $
//

// system include files
#include <cassert>
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "TVirtualCollectionProxy.h"

// user include files
#include "Fireworks/Core/src/FWItemTVirtualCollectionProxyAccessor.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWItemTVirtualCollectionProxyAccessor::FWItemTVirtualCollectionProxyAccessor(
   const TClass* iType,
   boost::shared_ptr<TVirtualCollectionProxy> iProxy,
   size_t iOffset)
   : m_type(iType),
     m_colProxy(iProxy),
     m_data(0),
     m_offset(iOffset)
{
}

// FWItemTVirtualCollectionProxyAccessor::FWItemTVirtualCollectionProxyAccessor(const FWItemTVirtualCollectionProxyAccessor& rhs)
// {
//    // do actual copying here;
// }

FWItemTVirtualCollectionProxyAccessor::~FWItemTVirtualCollectionProxyAccessor()
{
}

//
// assignment operators
//
// const FWItemTVirtualCollectionProxyAccessor& FWItemTVirtualCollectionProxyAccessor::operator=(const FWItemTVirtualCollectionProxyAccessor& rhs)
// {
//   //An exception safe implementation is
//   FWItemTVirtualCollectionProxyAccessor temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWItemTVirtualCollectionProxyAccessor::setData(const edm::ObjectWithDict& product)
{
   if (product.address() == 0)
   {
      reset();
      return;
   }

   m_data = product.address();
   assert(0!=m_data);
   m_colProxy->PushProxy(static_cast<char*>(const_cast<void*>(m_data))+m_offset);
}

void
FWItemTVirtualCollectionProxyAccessor::reset()
{
   if (0 != m_data)
   {
      m_data=0;
      m_colProxy->PopProxy();
   }
}

//
// const member functions
//
const void*
FWItemTVirtualCollectionProxyAccessor::modelData(int iIndex) const
{
   if ( 0 == m_data) { return m_data; }
   return m_colProxy->At(iIndex);
}

const void*
FWItemTVirtualCollectionProxyAccessor::data() const
{
   return m_data;
}

unsigned int
FWItemTVirtualCollectionProxyAccessor::size() const
{
   if(m_data==0) {
      return 0;
   }
   return m_colProxy->Size();
}

const TClass*
FWItemTVirtualCollectionProxyAccessor::modelType() const
{
   return m_colProxy->GetValueClass();
}

const TClass*
FWItemTVirtualCollectionProxyAccessor::type() const
{
   return m_type;
}

bool
FWItemTVirtualCollectionProxyAccessor::isCollection() const
{
   return true;
}

//
// static member functions
//

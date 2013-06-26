// -*- C++ -*-
//
// Package:     Core
// Class  :     FWSimpleProxyHelper
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 15:13:22 EST 2008
// $Id: FWSimpleProxyHelper.cc,v 1.3 2010/08/18 10:30:14 amraktad Exp $
//

// system include files
#include <sstream>

#include "Reflex/Object.h"
#include "Reflex/Type.h"
#include "TClass.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"
#include "Fireworks/Core/interface/FWEventItem.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWSimpleProxyHelper::FWSimpleProxyHelper(const std::type_info& iType) :
   m_itemType(&iType),
   m_objectOffset(0)
{
}

// FWSimpleProxyHelper::FWSimpleProxyHelper(const FWSimpleProxyHelper& rhs)
// {
//    // do actual copying here;
// }

//FWSimpleProxyHelper::~FWSimpleProxyHelper()
//{
//}

//
// assignment operators
//
// const FWSimpleProxyHelper& FWSimpleProxyHelper::operator=(const FWSimpleProxyHelper& rhs)
// {
//   //An exception safe implementation is
//   FWSimpleProxyHelper temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWSimpleProxyHelper::itemChanged(const FWEventItem* iItem)
{
   if(0!=iItem) {
      Reflex::Type myType = Reflex::Type::ByTypeInfo(*m_itemType);
      Reflex::Object dummy(Reflex::Type::ByTypeInfo(*(iItem->modelType()->GetTypeInfo())),
                   reinterpret_cast<void*>(0xFFFF));
      Reflex::Object castTo = dummy.CastObject(myType);
      assert(0!=castTo.Address());
      m_objectOffset=static_cast<char*>(dummy.Address())-static_cast<char*>(castTo.Address());
   }
}

//
// static member functions
//

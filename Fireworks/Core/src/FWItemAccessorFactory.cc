// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemAccessorFactory
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 18 14:48:14 EDT 2008
// $Id: FWItemAccessorFactory.cc,v 1.3 2009/01/23 21:35:43 amraktad Exp $
//

// system include files
#include <iostream>
#include "TClass.h"
#include "TVirtualCollectionProxy.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"

// user include files
#include "Fireworks/Core/interface/FWItemAccessorFactory.h"
#include "Fireworks/Core/src/FWItemTVirtualCollectionProxyAccessor.h"
#include "Fireworks/Core/src/FWItemSingleAccessor.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWItemAccessorFactory::FWItemAccessorFactory()
{
}

// FWItemAccessorFactory::FWItemAccessorFactory(const FWItemAccessorFactory& rhs)
// {
//    // do actual copying here;
// }

FWItemAccessorFactory::~FWItemAccessorFactory()
{
}

//
// assignment operators
//
// const FWItemAccessorFactory& FWItemAccessorFactory::operator=(const FWItemAccessorFactory& rhs)
// {
//   //An exception safe implementation is
//   FWItemAccessorFactory temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//
boost::shared_ptr<FWItemAccessorBase>
FWItemAccessorFactory::accessorFor(const TClass* iClass) const
{
  //std::cout <<"accessFor "<<iClass->GetName()<<std::endl;
   if(iClass->GetCollectionProxy()) {
      return boost::shared_ptr<FWItemAccessorBase>(new FWItemTVirtualCollectionProxyAccessor(iClass,
                                                                                             boost::shared_ptr<TVirtualCollectionProxy>(iClass->GetCollectionProxy()->Generate())));
   } else {
      assert(iClass->GetTypeInfo());
      ROOT::Reflex::Type dataType( ROOT::Reflex::Type::ByTypeInfo(*(iClass->GetTypeInfo())));
      assert(dataType != ROOT::Reflex::Type() );

      //is this an object which has only one member item and that item is a container?
      if(dataType.DataMemberSize()==1) {
         ROOT::Reflex::Type memType( dataType.DataMemberAt(0).TypeOf() );
         assert(memType != ROOT::Reflex::Type());
	 //std::cout <<"    memType:"<<memType.Name()<<std::endl;
	 //make sure this is the real type and not a typedef
	 memType = memType.FinalType();
         const TClass* rootMemType = TClass::GetClass(memType.TypeInfo());
         assert(rootMemType != 0);
         if(rootMemType->GetCollectionProxy()) {
            //std::cout <<"  reaching inside object data member"<<std::endl;
            return boost::shared_ptr<FWItemAccessorBase>(
                      new FWItemTVirtualCollectionProxyAccessor(iClass,
                                                                boost::shared_ptr<TVirtualCollectionProxy>(rootMemType->GetCollectionProxy()->Generate())));
         }
      }
   }
   //std::cout <<"  single"<<std::endl;
   return boost::shared_ptr<FWItemAccessorBase>(new FWItemSingleAccessor(iClass));
}

//
// static member functions
//

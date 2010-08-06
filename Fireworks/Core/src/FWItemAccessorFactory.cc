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
// $Id: FWItemAccessorFactory.cc,v 1.6 2010/02/26 09:41:02 eulisse Exp $
//

// system include files
#include <iostream>
#include "TClass.h"
#include "TVirtualCollectionProxy.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"

// user include files
#include "Fireworks/Core/interface/FWItemAccessorFactory.h"
#include "Fireworks/Core/interface/FWItemAccessorRegistry.h"
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
/** Create an accessor for a given type @a iClass.
   
    @a iClass the type for which we need an accessor.

    If the type is known to ROOT to be some sort of collection,
    we return the a FWItemTVirtualCollectionProxyAccessor 
    constructed using the associated TVirtualCollectionProxy.
  
    If the type is not a collection but it contains only
    one element which is a collection, we return a 
    FWItemTVirtualCollectionProxyAccessor using the 
    TVirtualCollectionProxy of that element.

    If none of the above is true, we lookup the plugin based
    FWItemAccessorRegistry for a plugin that can handle the
    given type.

    Failing that, we return a FWItemSingleAccessor which threats
    the object as if it was not a collection. 
 */
boost::shared_ptr<FWItemAccessorBase>
FWItemAccessorFactory::accessorFor(const TClass* iClass) const
{
  //std::cout <<"accessFor "<<iClass->GetName()<<std::endl;
  
  //check if this is a collection known by ROOT but also that the item held by the colletion actually has a dictionary  
   if(iClass->GetCollectionProxy() && 
      iClass->GetCollectionProxy()->GetValueClass() &&
      iClass->GetCollectionProxy()->GetValueClass()->IsLoaded()) {
      return boost::shared_ptr<FWItemAccessorBase>(new FWItemTVirtualCollectionProxyAccessor(iClass,
                                                                                             boost::shared_ptr<TVirtualCollectionProxy>(iClass->GetCollectionProxy()->Generate())));
   } else {
      assert(iClass->GetTypeInfo());
      ROOT::Reflex::Type dataType(ROOT::Reflex::Type::ByTypeInfo(*(iClass->GetTypeInfo())));
      assert(dataType != ROOT::Reflex::Type());

      //is this an object which has only one member item and that item is a container?
      if(dataType.DataMemberSize()==1) {
         ROOT::Reflex::Type memType( dataType.DataMemberAt(0).TypeOf() );
         assert(memType != ROOT::Reflex::Type());
	 //std::cout <<"    memType:"<<memType.Name()<<std::endl;
	 //make sure this is the real type and not a typedef
	 memType = memType.FinalType();
         const TClass* rootMemType = TClass::GetClass(memType.TypeInfo());
         //check if this is a collection known by ROOT but also that the item held by the colletion actually has a dictionary  
         if(rootMemType &&
            rootMemType->GetCollectionProxy() &&
            rootMemType->GetCollectionProxy()->GetValueClass() &&
            rootMemType->GetCollectionProxy()->GetValueClass()->IsLoaded() ) {
            //std::cout <<"  reaching inside object data member"<<std::endl;
            return boost::shared_ptr<FWItemAccessorBase>(
                      new FWItemTVirtualCollectionProxyAccessor(iClass,
                                                                boost::shared_ptr<TVirtualCollectionProxy>(rootMemType->GetCollectionProxy()->Generate())));
         }
      }
   }
   
   //std::cout <<"  single"<<std::endl;
 
   // Iterate on the available plugins and use the one which handles 
   // the iClass type. 
   // NOTE: This is done only a few times, not really performance critical.
   // If you want this to be fast, the loop can be moved in the 
   // constructor. Notice that this will require constructing FWEventItemsManager 
   // after the plugin manager (i.e. invoking AutoLibraryLoader::enable()) is configured
   // (i.e. invoking AutoLibraryLoader::enable()) in CmsShowMain.
   const std::vector<edmplugin::PluginInfo> &available = FWItemAccessorRegistry::get()->available();
   for (size_t i = 0, e = available.size(); i != e; ++i)
   {
      std::string name = available[i].name_;
      std::string type = name.substr(0, name.find_first_of('@'));
      if (iClass->GetTypeInfo()->name() == type)
        return boost::shared_ptr<FWItemAccessorBase>(FWItemAccessorRegistry::get()->create(name, iClass));
   } 

   return boost::shared_ptr<FWItemAccessorBase>(new FWItemSingleAccessor(iClass));
}

//
// static member functions
//

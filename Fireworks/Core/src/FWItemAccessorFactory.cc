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
// $Id: FWItemAccessorFactory.cc,v 1.15 2013/02/10 22:12:04 wmtan Exp $
//

// system include files
#include <iostream>
#include "TClass.h"
#include "TVirtualCollectionProxy.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"

// user include files
#include "Fireworks/Core/interface/FWItemAccessorFactory.h"
#include "Fireworks/Core/interface/FWItemAccessorRegistry.h"
#include "Fireworks/Core/src/FWItemTVirtualCollectionProxyAccessor.h"
#include "Fireworks/Core/src/FWItemSingleAccessor.h"
#include "Fireworks/Core/interface/fwLog.h"

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
    the object as if it was not a collection. Notice that this also will
    mean that the product associated to @a iClass will not show up in the
    "Add Collection" table.
 */
boost::shared_ptr<FWItemAccessorBase>
FWItemAccessorFactory::accessorFor(const TClass* iClass) const
{
   static const bool debug = false;

   TClass *member = 0;
   size_t offset=0;

   if(hasTVirtualCollectionProxy(iClass)) 
   {
      if (debug)
         fwLog(fwlog::kDebug) << "class " << iClass->GetName()
                              << " uses FWItemTVirtualCollectionProxyAccessor." << std::endl;
      return boost::shared_ptr<FWItemAccessorBase>(
         new FWItemTVirtualCollectionProxyAccessor(iClass,
            boost::shared_ptr<TVirtualCollectionProxy>(iClass->GetCollectionProxy()->Generate())));
   } 
   else if (hasMemberTVirtualCollectionProxy(iClass, member,offset)) 
   {
      if (debug)
         fwLog(fwlog::kDebug) << "class "<< iClass->GetName()
                              << " only contains data member " << member->GetName()
                              << " which uses FWItemTVirtualCollectionProxyAccessor."
                              << std::endl;
   	   
      return boost::shared_ptr<FWItemAccessorBase>(
         new FWItemTVirtualCollectionProxyAccessor(iClass,
            boost::shared_ptr<TVirtualCollectionProxy>(member->GetCollectionProxy()->Generate()),
                                                   offset));
   }
   
   // Iterate on the available plugins and use the one which handles 
   // the iClass type. 
   // NOTE: This is done only a few times, not really performance critical.
   // If you want this to be fast, the loop can be moved in the 
   // constructor. Notice that this will require constructing FWEventItemsManager 
   // after the plugin manager (i.e. invoking AutoLibraryLoader::enable()) is configured
   // (i.e. invoking AutoLibraryLoader::enable()) in CmsShowMain.
   std::string accessorName;
   if (hasAccessor(iClass, accessorName))
   {
      if (debug)
         fwLog(fwlog::kDebug) << "class " << iClass->GetName() << " uses " 
                              << accessorName << "." << std::endl;
      return boost::shared_ptr<FWItemAccessorBase>(FWItemAccessorRegistry::get()->create(accessorName, iClass));
   }
   
   return boost::shared_ptr<FWItemAccessorBase>(new FWItemSingleAccessor(iClass));
}

/** Helper method which @return true if the passes @a iClass can be accessed via
    TVirtualCollectionProxy.
  */
bool
FWItemAccessorFactory::hasTVirtualCollectionProxy(const TClass *iClass)
{
   // Check if this is a collection known by ROOT but also that the item held by
   // the colletion actually has a dictionary  
   return iClass &&
          iClass->GetCollectionProxy() && 
          iClass->GetCollectionProxy()->GetValueClass() &&
          iClass->GetCollectionProxy()->GetValueClass()->IsLoaded();
}

/** Helper method which checks if the object has only one data member and 
    if that data memeber can be accessed via a TVirtualCollectionProxy.
    
    @a oMember a reference to the pointer which will hold the actual TClass
     of the datamember to be used to build the TVirtualCollectionProxy.
 
    @oOffset a reference which will hold the offset of the member relative
     to the beginning address of a class instance.
    
    @return true if this is the case, false otherwise.
*/
bool
FWItemAccessorFactory::hasMemberTVirtualCollectionProxy(const TClass *iClass,
                                                        TClass *&oMember,
                                                        size_t& oOffset)
{
   assert(iClass->GetTypeInfo());
   edm::TypeWithDict dataType(*(iClass->GetTypeInfo()));
   assert(bool(dataType));

   // If the object has more than one data member, we avoid guessing. 
   edm::TypeDataMembers members(dataType);
   if (members.size() != 1)
      return false;
   
   edm::MemberWithDict member(*members.begin());
   edm::TypeWithDict memType(member.typeOf());
   assert(bool(memType));
   oMember = TClass::GetClass(memType.typeInfo());
   oOffset = member.offset();
   
   // Check if this is a collection known by ROOT but also that the item held by
   // the colletion actually has a dictionary  
            
   if (!hasTVirtualCollectionProxy(oMember))
      return false;

   return true;
}

/** Helper method which can be used to retrieve the name of the accessor 
    plugin which has to be created for a object of type @a iClass.
    
    The result is stored in the passed reference @a result.
    
    @return true if the plugin coul be found, false otherwise.
 */
bool
FWItemAccessorFactory::hasAccessor(const TClass *iClass, std::string &result)
{
   const std::vector<edmplugin::PluginInfo> &available 
      = FWItemAccessorRegistry::get()->available();
   
   for (size_t i = 0, e = available.size(); i != e; ++i)
   {
      std::string name = available[i].name_;
      std::string type = name.substr(0, name.find_first_of('@'));
      if (iClass->GetTypeInfo()->name() == type)
      {
         result.swap(name);
         return true;
      }
   }
   return false; 
}

/** Helper method which checks if the object will be treated as a collection.
 
 @return true if this is the case, false otherwise.
 */

bool FWItemAccessorFactory::classAccessedAsCollection(const TClass* iClass)
{
   std::string accessorName;
   TClass *member = 0;
   size_t offset=0;
   
   // This is pretty much the same thing that happens 
   return (FWItemAccessorFactory::hasTVirtualCollectionProxy(iClass) 
           || FWItemAccessorFactory::hasMemberTVirtualCollectionProxy(iClass, member,offset)
           || FWItemAccessorFactory::hasAccessor(iClass, accessorName));
}

//
// static member functions
//

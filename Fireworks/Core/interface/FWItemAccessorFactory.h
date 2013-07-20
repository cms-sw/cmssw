#ifndef Fireworks_Core_FWItemAccessorFactory_h
#define Fireworks_Core_FWItemAccessorFactory_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemAccessorFactory
//
/**\class FWItemAccessorFactory FWItemAccessorFactory.h Fireworks/Core/interface/FWItemAccessorFactory.h

   Description: Factory for constructing FWItemAccessorBases appropriate to a certain type

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 18 14:47:03 EDT 2008
// $Id: FWItemAccessorFactory.h,v 1.6 2010/06/03 19:38:31 chrjones Exp $
//

// system include files
#include <boost/shared_ptr.hpp>
#include <string>

// user include files

// forward declarations
class FWItemAccessorBase;
class TClass;

class FWItemAccessorFactory {

public:
   FWItemAccessorFactory();
   virtual ~FWItemAccessorFactory();

   // ---------- const member functions ---------------------
   boost::shared_ptr<FWItemAccessorBase> accessorFor(const TClass*) const;
   static bool hasAccessor(const TClass *iClass, std::string &result);
   static bool hasTVirtualCollectionProxy(const TClass *iClass);
   static bool hasMemberTVirtualCollectionProxy(const TClass *iClass,
                                                TClass *&oMember,
                                                size_t& oOffset);
   
   static bool classAccessedAsCollection(const TClass*);
   
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWItemAccessorFactory(const FWItemAccessorFactory&); // stop default

   const FWItemAccessorFactory& operator=(const FWItemAccessorFactory&); // stop default

   // ---------- member data --------------------------------

};

#endif

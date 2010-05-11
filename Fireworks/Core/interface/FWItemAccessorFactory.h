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
// $Id: FWItemAccessorFactory.h,v 1.3 2009/01/23 21:35:41 amraktad Exp $
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
                                                TClass *&member);
   
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWItemAccessorFactory(const FWItemAccessorFactory&); // stop default

   const FWItemAccessorFactory& operator=(const FWItemAccessorFactory&); // stop default

   // ---------- member data --------------------------------

};

#endif

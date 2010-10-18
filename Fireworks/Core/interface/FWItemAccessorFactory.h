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
// $Id: FWItemAccessorFactory.h,v 1.2 2008/11/06 22:05:23 amraktad Exp $
//

// system include files
#include <boost/shared_ptr.hpp>

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

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWItemAccessorFactory(const FWItemAccessorFactory&); // stop default

   const FWItemAccessorFactory& operator=(const FWItemAccessorFactory&); // stop default

   // ---------- member data --------------------------------

};


#endif

#ifndef Fireworks_Core_FWItemTVirtualCollectionProxyAccessor_h
#define Fireworks_Core_FWItemTVirtualCollectionProxyAccessor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemTVirtualCollectionProxyAccessor
//
/**\class FWItemTVirtualCollectionProxyAccessor FWItemTVirtualCollectionProxyAccessor.h Fireworks/Core/interface/FWItemTVirtualCollectionProxyAccessor.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sat Oct 18 08:43:45 EDT 2008
// $Id: FWItemTVirtualCollectionProxyAccessor.h,v 1.7 2012/08/03 18:20:28 wmtan Exp $
//

// system include files
#include "boost/shared_ptr.hpp"

// user include files
#include "Fireworks/Core/interface/FWItemAccessorBase.h"

// forward declarations
class TVirtualCollectionProxy;

class FWItemTVirtualCollectionProxyAccessor : public FWItemAccessorBase {

public:
   FWItemTVirtualCollectionProxyAccessor(const TClass* iType,
                                         boost::shared_ptr<TVirtualCollectionProxy> iProxy,
                                         size_t iOffset=0);
   virtual ~FWItemTVirtualCollectionProxyAccessor();

   // ---------- const member functions ---------------------
   virtual const void* modelData(int iIndex) const ;
   virtual const void* data() const;
   virtual unsigned int size() const;
   const TClass* modelType() const;
   const TClass* type() const;

   bool isCollection() const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setData(const edm::ObjectWithDict& );
   void reset();

private:
   FWItemTVirtualCollectionProxyAccessor(const FWItemTVirtualCollectionProxyAccessor&); // stop default

   const FWItemTVirtualCollectionProxyAccessor& operator=(const FWItemTVirtualCollectionProxyAccessor&); // stop default

   // ---------- member data --------------------------------
   const TClass* m_type;
   boost::shared_ptr<TVirtualCollectionProxy> m_colProxy; //should be something other than shared_ptr
   mutable const void * m_data;
   size_t m_offset;
};


#endif

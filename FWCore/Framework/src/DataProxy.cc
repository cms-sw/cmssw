// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxy
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:49:19 EST 2005
// $Id: DataProxy.cc,v 1.5 2007/12/21 04:36:26 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/ComponentDescription.h"


//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {
//
// static data member definitions
//
static
const ComponentDescription*
dummyDescription()
{
   static ComponentDescription s_desc;
   return &s_desc;
}     
//
// constructors and destructor
//
DataProxy::DataProxy() :
   cacheIsValid_(false),
   nonTransientAccessRequested_(false),
   description_(dummyDescription())
{
}

// DataProxy::DataProxy(const DataProxy& rhs)
// {
//    // do actual copying here;
// }

DataProxy::~DataProxy()
{
}

//
// assignment operators
//
// const DataProxy& DataProxy::operator=(const DataProxy& rhs)
// {
//   //An exception safe implementation is
//   DataProxy temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
DataProxy::setCacheIsValidAndAccessType(bool iTransientAccessOnly) { 
   cacheIsValid_ = true;
   if(!iTransientAccessOnly) {
      nonTransientAccessRequested_ = true;
   }
}
      
void DataProxy::clearCacheIsValid() { 
   cacheIsValid_ = false;
   nonTransientAccessRequested_ = false;
}
      
void 
DataProxy::resetIfTransient() {
   if (!nonTransientAccessRequested_) {
      invalidate();
   }
}
      
//
// const member functions
//

//
// static member functions
//
   }
}

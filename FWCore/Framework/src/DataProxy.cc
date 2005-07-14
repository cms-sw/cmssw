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
// $Id: DataProxy.cc,v 1.2 2005/06/23 19:59:48 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/DataProxy.h"


//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {
//
// static data member definitions
//

//
// constructors and destructor
//
DataProxy::DataProxy() :
   cacheIsValid_(false)
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

//
// const member functions
//

//
// static member functions
//
   }
}

// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProductDeletedException
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu Jan 12 13:32:41 CST 2012
// $Id: ProductDeletedException.cc,v 1.1 2012/02/09 22:12:56 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/ProductDeletedException.h"


using namespace edm;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ProductDeletedException::ProductDeletedException():
cms::Exception("ProductDeleted")
{
}

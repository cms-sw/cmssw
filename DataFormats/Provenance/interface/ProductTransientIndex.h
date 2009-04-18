#ifndef DataFormats_Provenance_ProductTransientIndex_h
#define DataFormats_Provenance_ProductTransientIndex_h
// -*- C++ -*-
//
// Package:     Provenance
// Class  :     ProductTransientIndex
// 
/**\class ProductTransientIndex ProductTransientIndex.h DataFormats/Provenance/interface/ProductTransientIndex.h

 Description: A 'run-time' only index used to uniquely identify an EDProduct

 Usage:
    Used internally by the Pricipal and ProductRegistry

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  1 10:27:31 CDT 2009
//

namespace edm {
   typedef unsigned int ProductTransientIndex;
}

#endif

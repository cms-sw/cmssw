#ifndef FWLite_stdNamespaceAdder_h
#define FWLite_stdNamespaceAdder_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     stdNamespaceAdder
// 
/**\class stdNamespaceAdder stdNamespaceAdder.h FWCore/FWLite/interface/stdNamespaceAdder.h

 Description: Adds back the 'std::' namespace prefix to standard classes

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Tue Dec  6 09:18:09 EST 2005
// $Id: stdNamespaceAdder.h,v 1.1 2006/05/17 16:43:44 wmtan Exp $
//

// system include files
#include <string>

// user include files

// forward declarations
namespace fwlite {
   std::string stdNamespaceAdder(const std::string&);
}
#endif

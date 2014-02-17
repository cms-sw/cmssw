// -*- C++ -*-
//
// Package:     RootAutoLibraryLoader
// Class  :     stdNamespaceAdder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Tue Dec  6 09:18:05 EST 2005
// $Id: stdNamespaceAdder.cc,v 1.2 2006/12/20 00:23:40 wmtan Exp $
//

// system include files
#include "boost/regex.hpp"

// user include files
#include "FWCore/RootAutoLibraryLoader/src/stdNamespaceAdder.h"


//
// constants, enums and typedefs
//
namespace edm {
  namespace root {
    std::string stdNamespaceAdder(const std::string& iClassName)
    {
      //adds the std:: prefix to vector, string, map, list or deque if it is not
      // already there
      static const boost::regex e("(^|[^[:alnum:]_:])((?:vector)|(?:string)|(?:map)|(?:list)|(?:deque))");
      const std::string format("\\1std::\\2");
      
      return regex_replace(iClassName, e, format, boost::match_default | boost::format_sed);
    }
  }
}

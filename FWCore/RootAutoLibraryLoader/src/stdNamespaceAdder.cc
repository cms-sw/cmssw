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
//

#include "FWCore/RootAutoLibraryLoader/src/stdNamespaceAdder.h"

#include "boost/regex.hpp"

namespace edm {
namespace root {

std::string stdNamespaceAdder(const std::string& iClassName)
{
  //adds the std:: prefix to vector, string, map, list or deque if it is not
  // already there
  static const boost::regex
  e("(^|[^[:alnum:]_:])((?:vector)|(?:string)|(?:map)|(?:list)|(?:deque))");
  const std::string format("\\1std::\\2");
  return regex_replace(iClassName, e, format,
                       boost::match_default | boost::format_sed);
}

} // namespace root
} // namespace edm


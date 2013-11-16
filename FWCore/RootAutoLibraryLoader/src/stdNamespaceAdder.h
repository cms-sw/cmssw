#ifndef FWLite_stdNamespaceAdder_h
#define FWLite_stdNamespaceAdder_h

//
// Package:     FWLite
// Class  :     stdNamespaceAdder
//
//
// Description: Adds back the 'std::' namespace prefix to standard classes
//
// Original Author:
//         Created:  Tue Dec  6 09:18:09 EST 2005
//

#include <string>

namespace edm {
namespace root {
std::string stdNamespaceAdder(const std::string&);
}
}

#endif // FWLite_stdNamespaceAdder_h

#ifndef FWCore_Utilities_resolveSymbolicLinks_h
#define FWCore_Utilities_resolveSymbolicLinks_h

#include <string>
namespace edm {
  // Resolves symlinks anywhere in fullPath recursively.
  // If there are no symlinks, or if fullPath does not begin with '/',
  // fullPath will not be modified.

  void resolveSymbolicLinks(std::string& fullPath);
}

#endif

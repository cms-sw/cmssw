#include "Alignment/CommonAlignment/interface/AlignableMap.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//_____________________________________________________________________________
align::Alignables& AlignableMap::get(const std::string& name) { return theStore[name]; }

//_____________________________________________________________________________
align::Alignables& AlignableMap::find(const std::string& name) {
  typename Container::iterator o = theStore.find(name);

  if (theStore.end() == o) {
    std::ostringstream knownKeys;
    for (auto it = theStore.begin(); it != theStore.end(); ++it) {
      knownKeys << (it != theStore.begin() ? ", " : "") << it->first;
    }

    throw cms::Exception("AlignableMapError") << "Cannot find an object of name " << name << " in AlignableMap, "
                                              << "know only " << knownKeys.str() << ".";
  }

  return o->second;
}

//_____________________________________________________________________________
void AlignableMap::dump(void) const {
  edm::LogInfo("AlignableMap") << "Printing out AlignSetup: ";
  for (typename Container::const_iterator it = theStore.begin(); it != theStore.end(); ++it) {
    edm::LogVerbatim("AlignableMap") << it->first << std::endl;
  }
}

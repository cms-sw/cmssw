#ifndef FWCore_Framework_throwIfImproperDependencies_h
#define FWCore_Framework_throwIfImproperDependencies_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Function:    throwIfImproperDependencies
// 
/**\function throwIfImproperDependencies throwIfImproperDependencies.h "throwIfImproperDependencies.h"

 Description: Function which uses the graph of dependencies to determine if there are any cycles

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 06 Sep 2016 16:04:26 GMT
//

// system include files
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <limits>

// user include files

// forward declarations

namespace edm {
  namespace graph {
    constexpr auto kInvalidIndex = std::numeric_limits<unsigned int>::max();
    //This index is used as the Path index for the case where we are
    // describing a data dependency and not a dependency on a Path
    constexpr auto kDataDependencyIndex = std::numeric_limits<unsigned int>::max();
    
    using SimpleEdge =  std::pair<unsigned int, unsigned int>;
    using EdgeToPathMap = std::map<SimpleEdge, std::vector<unsigned int>>;
  
    void throwIfImproperDependencies(EdgeToPathMap const&,
                                     std::vector<std::vector<unsigned int>> const& iPathIndexToModuleIndexOrder,
                                     std::vector<std::string> const& iPathNames,
                                     std::unordered_map<unsigned int,std::string> const& iModuleIndexToNames);
  }
};

#endif

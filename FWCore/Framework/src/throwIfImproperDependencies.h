#ifndef FWCore_Framework_throwIfImproperDependencies_h
#define FWCore_Framework_throwIfImproperDependencies_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     DependencyCycleDetector
// 
/**\function throwIfImproperDependencies throwIfImproperDependencies.h "throwIfImproperDependencies.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Tue, 06 Sep 2016 16:04:26 GMT
//

// system include files
#include <map>
#include <string>
#include <list>
#include <vector>

#include "boost/graph/graph_traits.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/visitors.hpp"


// user include files

// forward declarations

namespace edm {
  namespace graph {
    using SimpleEdge =  std::pair<unsigned int, unsigned int>;
    using EdgeToPathMap = std::map<SimpleEdge, std::vector<unsigned int>>;
  
    void throwIfImproperDependencies(EdgeToPathMap const&,
                                     std::vector<std::string> const& iPathNames,
                                     std::map<std::string,unsigned int> const& iModuleNamesToIndex);
  }
};

#endif

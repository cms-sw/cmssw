#ifndef ParameterSet_ProcessDesc_h
#define ParameterSet_ProcessDesc_h

#include "boost/shared_ptr.hpp"
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Nodes.h"

namespace edm
{
  
  typedef boost::shared_ptr<pset::WrapperNode> WrapperNodePtr ;

  struct ProcessDesc
  {
    //Path and sequence information
    typedef std::vector< WrapperNodePtr > PathContainer;
    PathContainer pathFragments_;
    ParameterSet pset_;
    std::vector< ParameterSet> services_;
  };
}

#endif

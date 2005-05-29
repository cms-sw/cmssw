#ifndef EDM_PARAMETERSET_PROCESS_DESC_INC
#define EDM_PARAMETERSET_PROCESS_DESC_INC

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
  };
}

#endif

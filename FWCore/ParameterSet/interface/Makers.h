#ifndef PARAMETERSET_MAKERS_HPP
#define PARAMETERSET_MAKERS_HPP

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/Nodes.h"

#include "boost/shared_ptr.hpp"

#include <memory>
#include <map>
#include <string>

namespace edm {
   namespace pset {
      typedef std::map<std::string, boost::shared_ptr<edm::ParameterSet> > NamedPSets;
      boost::shared_ptr<edm::ParameterSet> makePSet(const NodePtrList& nodes,
                                            const NamedPSets& blocks =NamedPSets(),
                                            const NamedPSets& psets=NamedPSets());
      boost::shared_ptr<edm::ProcessDesc> makeProcess(NodePtrListPtr nodes);
   }
}
#endif

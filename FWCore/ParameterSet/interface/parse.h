#ifndef PARAMETERSET_PARSE_H
#define PARAMETERSET_PARSE_H

#include "boost/shared_ptr.hpp"

#include "FWCore/ParameterSet/interface/Nodes.h"
typedef boost::shared_ptr<edm::pset::NodePtrList> ParseResults;

namespace edm {
   namespace pset {
      ParseResults parse(char const* spec);
   }
}

#endif

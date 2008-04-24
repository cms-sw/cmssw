#ifndef FWCore_ParameterSet_parse_h
#define FWCore_ParameterSet_parse_h

#include "boost/shared_ptr.hpp"

#include "FWCore/Utilities/interface/Parse.h"
#include "FWCore/ParameterSet/interface/Node.h"
typedef boost::shared_ptr<edm::pset::NodePtrList> ParseResults;

namespace edm {
   namespace pset {
      /// only does the yacc interpretation
      ParseResults parse(char const* spec);
   }
}
#endif


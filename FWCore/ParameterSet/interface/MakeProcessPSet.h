#ifndef ParameterSet_MakeProcessPSet_h
#define ParameterSet_MakeProcessPSet_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/shared_ptr.hpp"
#include <string>
#include <iosfwd>

namespace edm {
  boost::shared_ptr<edm::ParameterSet> makeProcessPSet(const std::string& config);

  // Write the parse tree hanging from this top-level node, in Python format
  void write_python_form(edm::pset::Node const& topnode, std::ostream& os) ;
}

#endif


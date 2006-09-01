#ifndef PythonParseTreeSummary_h
#define PythonParseTreeSummary_h

#include <string>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python.hpp>
#include "FWCore/ParameterSet/src/ParseResultsTweaker.h"

class PythonParseTreeSummary
{
public:
  PythonParseTreeSummary(const std::string & filename);

  /// the names of all modules in this parse tree
  boost::python::list modules() const;

  /// names of all modules of type, e.g., service or es_source
  boost::python::list modulesOfType(const std::string & type) const;

  boost::python::list outputModules() const;




private:

  edm::pset::ParseResultsTweaker theTweaker;

};

#endif


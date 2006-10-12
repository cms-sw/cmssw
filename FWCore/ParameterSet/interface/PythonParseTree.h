#ifndef PythonParseTree_h
#define PythonParseTree_h

#include <string>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python.hpp>
#include "FWCore/ParameterSet/interface/ParseTree.h"
#include "FWCore/Utilities/interface/EDMException.h"

class PythonParseTree
{
public:
  PythonParseTree(const std::string & filename);

  /// the names of all modules in this parse tree
  boost::python::list modules() const;

  /// names of all modules of type, e.g., service or es_source
  boost::python::list modulesOfType(const std::string & type) const;

  boost::python::list outputModules() const;

  /// processes all includes, renames, etc.
  void process();

  /// replaces the value of an entry
  void replaceValue(const std::string & dotDelimitedNode,
                    const std::string & value);

  /// WARNING the input list gets destroyed by pop()s
  void replaceValues(const std::string & dotDelimitedNode,
                     boost::python::list & values);

  void dump(const std::string & dotDelimitedNode) const;

  /// only works for EntryNodes inside modules.  Hope to include top-level PSets soon
  std::string value(const std::string & dotDelimitedNode) const;

  /// only works for VEntryNodes
  boost::python::list values(const std::string & dotDelimitedNode) const;

  /// names of the nodes below this one.  Includes are transparent
  boost::python::list children(const std::string & dotDelimitedNode) const;

  /// dump the tree to a string, to be written to a file.
  std::string dumpTree() const;

  /// changes edm::Exceptions to python exceptions
  static void exceptionTranslator(const edm::Exception & e);

private:

  edm::pset::ParseTree theTree;

};

#endif


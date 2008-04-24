#include "FWCore/ParameterSet/interface/PythonParseTree.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/PSetNode.h"
#include "FWCore/ParameterSet/src/PythonWrapper.h"


PythonParseTree::PythonParseTree(const std::string & filename)
:  theTree(edm::read_whole_file(filename))
{

  theTree.process();
}


boost::python::list 
PythonParseTree::modules() const
{
  return edm::toPythonList<std::string>(theTree.modules());
}


boost::python::list 
PythonParseTree::modulesOfType(const std::string & type) const
{
  return edm::toPythonList<std::string>(theTree.modulesOfType(type));
}


void PythonParseTree::process()
{
  theTree.process();
}


void PythonParseTree::replaceValue(const std::string & dotDelimitedNode,
                 const std::string & value)
{
  theTree.replace(dotDelimitedNode, value);
}


void PythonParseTree::replaceValues(const std::string & dotDelimitedNode,
                                           boost::python::list & values)
{
  theTree.replace(dotDelimitedNode, edm::toVector<std::string>(values));
}


void PythonParseTree::dump(const std::string & dotDelimitedNode) const
{
  theTree.print(dotDelimitedNode);
}


std::string PythonParseTree::typeOf(const std::string & dotDelimitedNode) const
{
  return theTree.typeOf(dotDelimitedNode);
}


std::string PythonParseTree::value(const std::string & dotDelimitedNode) const
{
  return theTree.value(dotDelimitedNode);
}


boost::python::list PythonParseTree::values(const std::string & dotDelimitedNode) const
{
  return edm::toPythonList<std::string>(theTree.values(dotDelimitedNode));
}


boost::python::list PythonParseTree::children(const std::string & dotDelimitedNode) const
{
  return edm::toPythonList<std::string>(theTree.children(dotDelimitedNode));
}


std::string PythonParseTree::dumpTree() const
{
  std::ostringstream ost;
  theTree.top()->print(ost, edm::pset::Node::COMPRESSED);
  return ost.str();
}


void PythonParseTree::exceptionTranslator(const edm::Exception & e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}


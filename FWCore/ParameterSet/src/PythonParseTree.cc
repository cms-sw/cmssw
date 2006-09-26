#include "FWCore/ParameterSet/interface/PythonParseTree.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/PSetNode.h"

using namespace boost::python;
using std::string;
using std::vector;

// utility to translate from an STL vector of strings to
// a Python list
boost::python::list toPythonList(const std::vector<std::string> & v)
{
  boost::python::list result;
  for(std::vector<std::string>::const_iterator vItr = v.begin();
      vItr != v.end(); ++vItr)
  {
    result.append(*vItr);
  }
  return result;
}

// and back.  Destroys the input via pop()s
std::vector<std::string> toVector(boost::python::list & l)
{
  std::vector<std::string> result;
  bool is_ok = true;
  try 
  {
    while( is_ok ) {
      boost::python::extract<std::string>  x( l.pop( 0 ));

      if( x.check()) {
        result.push_back( x());
      } else {
        is_ok = false;
      }
    }
  }
  // the pop will end in an exception
  catch(...)
  {
  }

  return result;
}


PythonParseTree::PythonParseTree(const string & filename)
:  theTree(edm::pset::read_whole_file(filename))
{

  theTree.process();
}


boost::python::list 
PythonParseTree::modules() const
{
  return toPythonList(theTree.modules());
}


boost::python::list 
PythonParseTree::modulesOfType(const std::string & type) const
{
  return toPythonList(theTree.modulesOfType(type));
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
  theTree.replace(dotDelimitedNode, toVector(values));
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
  return toPythonList(theTree.values(dotDelimitedNode));
}


boost::python::list PythonParseTree::children(const std::string & dotDelimitedNode) const
{
  return toPythonList(theTree.children(dotDelimitedNode));
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


BOOST_PYTHON_MODULE(libFWCoreParameterSet)
{

   class_<PythonParseTree>("PythonParseTree", init<std::string>())
       .def("modules",       &PythonParseTree::modules)
       .def("modulesOfType", &PythonParseTree::modulesOfType)
       .def("process",       &PythonParseTree::process)
       .def("replaceValue",  &PythonParseTree::replaceValue)
       .def("replaceValues", &PythonParseTree::replaceValues)
       .def("dump",          &PythonParseTree::dump)
       .def("typeOf",         &PythonParseTree::typeOf)
       .def("value",         &PythonParseTree::value)
       .def("values",        &PythonParseTree::values)
       .def("children",      &PythonParseTree::children)
       .def("dumpTree",      &PythonParseTree::dumpTree)
   ;

     
   register_exception_translator<edm::Exception>(PythonParseTree::exceptionTranslator);
}



#include "FWCore/ParameterSet/interface/PythonParseTreeSummary.h"
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
  while( is_ok ) {
    boost::python::extract<std::string>  x( l.pop( 0 ));

    if( x.check()) {
        result.push_back( x());
    } else {
        is_ok = false;
    }
  }
  return result;
}


PythonParseTreeSummary::PythonParseTreeSummary(const string & filename)
:  theTree(edm::pset::read_whole_file(filename))
{

  theTree.process();
}


boost::python::list 
PythonParseTreeSummary::modules() const
{
  return toPythonList(theTree.modules());
}


boost::python::list 
PythonParseTreeSummary::modulesOfType(const std::string & type) const
{
  return toPythonList(theTree.modulesOfType(type));
}


void PythonParseTreeSummary::process()
{
  theTree.process();
}


void PythonParseTreeSummary::replaceValue(const std::string & dotDelimitedNode,
                 const std::string & value)
{
  theTree.replace(dotDelimitedNode, value);
}


void PythonParseTreeSummary::replaceValues(const std::string & dotDelimitedNode,
                                           boost::python::list & values)
{
  theTree.replace(dotDelimitedNode, toVector(values));
}


void PythonParseTreeSummary::dump(const std::string & dotDelimitedNode) const
{
  theTree.print(dotDelimitedNode);
}


std::string PythonParseTreeSummary::value(const std::string & dotDelimitedNode) const
{
  return theTree.value(dotDelimitedNode);
}


boost::python::list PythonParseTreeSummary::values(const std::string & dotDelimitedNode) const
{
  return toPythonList(theTree.values(dotDelimitedNode));
}


boost::python::list PythonParseTreeSummary::children(const std::string & dotDelimitedNode) const
{
  return toPythonList(theTree.children(dotDelimitedNode));
}


std::string PythonParseTreeSummary::dumpTree() const
{
  std::ostringstream ost;
  theTree.top()->print(ost, edm::pset::Node::COMPRESSED);
  return ost.str();
}


BOOST_PYTHON_MODULE(libFWCoreParameterSet)
{

   class_<PythonParseTreeSummary>("PythonParseTreeSummary", init<std::string>())
       .def("modules",       &PythonParseTreeSummary::modules)
       .def("modulesOfType", &PythonParseTreeSummary::modulesOfType)
       .def("process",       &PythonParseTreeSummary::process)
       .def("replaceValue",  &PythonParseTreeSummary::replaceValue)
       .def("replaceValues", &PythonParseTreeSummary::replaceValues)
       .def("dump",          &PythonParseTreeSummary::dump)
       .def("value",         &PythonParseTreeSummary::value)
       .def("values",        &PythonParseTreeSummary::values)
       .def("children",      &PythonParseTreeSummary::children)
       .def("dumpTree",      &PythonParseTreeSummary::dumpTree)
   ;

     
}



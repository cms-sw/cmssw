#include "FWCore/ParameterSet/interface/PythonParseTreeSummary.h"
#include "FWCore/ParameterSet/interface/parse.h"

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


PythonParseTreeSummary::PythonParseTreeSummary(const string & filename)
{
  string configString;
  edm::pset::read_whole_file(filename, configString);

  boost::shared_ptr<edm::pset::NodePtrList> parsetree =
    edm::pset::parse(configString.c_str());

  theTweaker.process(parsetree);
}


boost::python::list 
PythonParseTreeSummary::modules() const
{
  return toPythonList(theTweaker.modules());
}


boost::python::list 
PythonParseTreeSummary::modulesOfType(const std::string & type) const
{
  return toPythonList(theTweaker.modulesOfType(type));
}


BOOST_PYTHON_MODULE(libFWCoreParameterSet)
{

   class_<PythonParseTreeSummary>("PythonParseTreeSummary", init<std::string>())
       .def("modules", &PythonParseTreeSummary::modules)
       .def("modulesOfType", &PythonParseTreeSummary::modulesOfType)
        ;
}



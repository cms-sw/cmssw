#ifndef PythonProcessDesc_h
#define PythonProcessDesc_h

#include "FWCore/ParameterSet/interface/PythonParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include <boost/python.hpp>

class PythonProcessDesc
{
public:
  PythonProcessDesc();
  /** This constructor will parse the given file or string
      and create two objects in python-land:
    * a PythonProcessDesc named 'processDesc'
    * a PythonParameterSet named 'processPSet'
    It decides whether it's a file or string by seeing if
    it ends in '.py'
  */
  PythonProcessDesc(const std::string & config);

  void addService(const PythonParameterSet & pset) {theServices.push_back(pset);}

  PythonParameterSet newPSet() const {return PythonParameterSet();}

  std::string dump() const;

  // makes a new (copy) of the ProcessDesc
  boost::shared_ptr<edm::ProcessDesc> processDesc() const;

private:
  void readFile(const std::string & fileName, boost::python::object & main_namespace);
  void readString(const std::string & pyConfig, boost::python::object & main_namespace);

  static bool initialized_;
  PythonParameterSet theProcessPSet;
  std::vector<PythonParameterSet> theServices;

};

#endif


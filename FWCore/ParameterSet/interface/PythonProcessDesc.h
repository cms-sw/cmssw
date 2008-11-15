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

  PythonProcessDesc(const std::string & config, int argc, char * argv[]);

  void addService(const PythonParameterSet & pset) {theServices.push_back(pset);}

  PythonParameterSet newPSet() const {return PythonParameterSet();}

  std::string dump() const;

  // makes a new (copy) of the ProcessDesc
  boost::shared_ptr<edm::ProcessDesc> processDesc() const;

private:
  void prepareToRead();
  void read(const std::string & config);
  void readFile(const std::string & fileName);
  void readString(const std::string & pyConfig);

  static bool initialized_;
  PythonParameterSet theProcessPSet;
  std::vector<PythonParameterSet> theServices;
  boost::python::object theMainModule;
  boost::python::object theMainNamespace;
};

#endif


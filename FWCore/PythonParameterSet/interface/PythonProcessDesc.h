#ifndef FWCore_PythonParameterSet_PythonProcessDesc_h
#define FWCore_PythonParameterSet_PythonProcessDesc_h

#include "FWCore/PythonParameterSet/interface/BoostPython.h"
#include "FWCore/PythonParameterSet/interface/PythonParameterSet.h"

#include <memory>

#include <string>
#include <vector>

namespace edm {
  class ParameterSet;
  class ProcessDesc;
}

class PythonProcessDesc {
public:
  PythonProcessDesc();
  /** This constructor will parse the given file or string
      and create two objects in python-land:
    * a PythonProcessDesc named 'processDesc'
    * a PythonParameterSet named 'processPSet'
    It decides whether it's a file or string by seeing if
    it ends in '.py'
  */
  PythonProcessDesc(std::string const& config);

  PythonProcessDesc(std::string const& config, int argc, char * argv[]);

  PythonParameterSet newPSet() const {return PythonParameterSet();}

  PythonParameterSet& pset() { return theProcessPSet;}
  
  std::string dump() const;

  // makes a new (copy) of the ParameterSet
  std::unique_ptr<edm::ParameterSet> parameterSet() const;

  // makes a new (copy) of a ProcessDesc
  // For backward compatibility only.  Remove when no longer needed.
  std::shared_ptr<edm::ProcessDesc> processDesc() const;

private:
  void prepareToRead();
  void read(std::string const& config);
  void readFile(std::string const& fileName);
  void readString(std::string const& pyConfig);

  PythonParameterSet theProcessPSet;
  boost::python::object theMainModule;
  boost::python::object theMainNamespace;
};

#endif

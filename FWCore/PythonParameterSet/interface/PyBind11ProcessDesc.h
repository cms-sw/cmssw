#ifndef FWCore_PyBind11ParameterSet_PyBind11ProcessDesc_h
#define FWCore_PyBind11ParameterSet_PyBind11ProcessDesc_h

#include "FWCore/PythonParameterSet/interface/Python11ParameterSet.h"

#include <memory>
#include <string>
#include <vector>

namespace edm {
  class ParameterSet;
  class ProcessDesc;
}  // namespace edm

class PyBind11InterpreterSentry {
public:
  PyBind11InterpreterSentry(bool ownsInterpreter);
  ~PyBind11InterpreterSentry();

  pybind11::object mainModule;

private:
  bool const ownsInterpreter_;
};

class PyBind11ProcessDesc {
public:
  PyBind11ProcessDesc();
  /** This constructor will parse the given file or string
      and create two objects in python-land:
    * a PythonProcessDesc named 'processDesc'
    * a PythonParameterSet named 'processPSet'
    It decides whether it's a file or string by seeing if
    it ends in '.py'
  */
  PyBind11ProcessDesc(std::string const& config);

  PyBind11ProcessDesc(std::string const& config, int argc, char* argv[]);

  ~PyBind11ProcessDesc();

  Python11ParameterSet newPSet() const { return Python11ParameterSet(); }

  Python11ParameterSet& pset() { return theProcessPSet; }

  std::string dump() const;

  // makes a new (copy) of the ParameterSet
  std::unique_ptr<edm::ParameterSet> parameterSet() const;

  // makes a new (copy) of a ProcessDesc
  // For backward compatibility only.  Remove when no longer needed.
  std::unique_ptr<edm::ProcessDesc> processDesc() const;

private:
  void prepareToRead();
  void read(std::string const& config);
  void readFile(std::string const& fileName);
  void readString(std::string const& pyConfig);

  Python11ParameterSet theProcessPSet;
  PyBind11InterpreterSentry theInterpreter;
};

#endif

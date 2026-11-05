#ifndef FWCore_PyBind11ParameterSet_PyBind11ProcessDesc_h
#define FWCore_PyBind11ParameterSet_PyBind11ProcessDesc_h

#include "FWCore/PythonParameterSet/interface/Python11ParameterSet.h"

#include <memory>
#include <string>
#include <utility>
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
  // process options to be set after reading the configuration, as pairs of
  // name and python expression: {"numberOfThreads", "4"} is equivalent to
  //   process.options.numberOfThreads = 4
  using ProcessOptions = std::vector<std::pair<std::string, std::string>>;

  PyBind11ProcessDesc();
  /** This constructor will parse the given file or string
      and create two objects in python-land:
    * a PythonProcessDesc named 'processDesc'
    * a PythonParameterSet named 'processPSet'
    It decides whether it's a file or string by seeing if
    it ends in '.py'
  */
  PyBind11ProcessDesc(std::string const& config, bool isFile, ProcessOptions const& options = {});

  PyBind11ProcessDesc(std::string const& config,
                      bool isFile,
                      const std::vector<std::string>& args,
                      ProcessOptions const& options = {});

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
  void read(std::string const& config, bool isFile, ProcessOptions const& options);
  void setProcessOptions(ProcessOptions const& options);

  Python11ParameterSet theProcessPSet;
  PyBind11InterpreterSentry theInterpreter;
};

#endif

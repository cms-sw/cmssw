#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/PythonParameterSet/interface/PythonParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/PythonParameterSet/src/initializeModule.h"

using namespace boost::python;

static
void
makePSetsFromFile(std::string const& fileName, boost::python::object& mainNamespace) {
  std::string initCommand("from FWCore.ParameterSet.Types import makeCppPSet\n"
                          "execfile('");
  initCommand += fileName + "')";

  handle<>(PyRun_String(initCommand.c_str(),
                        Py_file_input,
                        mainNamespace.ptr(),
                        mainNamespace.ptr()));
  std::string command("makeCppPSet(locals(), topPSet)");
  handle<>(PyRun_String(command.c_str(),
                        Py_eval_input,
                        mainNamespace.ptr(),
                        mainNamespace.ptr()));
}

static
void
makePSetsFromString(std::string const& module, boost::python::object& mainNamespace) {
  std::string command = module;
  command += "\nfrom FWCore.ParameterSet.Types import makeCppPSet\nmakeCppPSet(locals(), topPSet)";
  handle<>(PyRun_String(command.c_str(),
                        Py_file_input,
                        mainNamespace.ptr(),
                        mainNamespace.ptr()));
}

namespace edm {

  std::shared_ptr<ParameterSet>
  readConfig(std::string const& config) {
    PythonProcessDesc pythonProcessDesc(config);
    return pythonProcessDesc.parameterSet();
  }

  std::shared_ptr<ParameterSet>
  readConfig(std::string const& config, int argc, char* argv[]) {
    PythonProcessDesc pythonProcessDesc(config, argc, argv);
    return pythonProcessDesc.parameterSet();
  }

  void
  makeParameterSets(std::string const& configtext,
                  std::shared_ptr<ParameterSet>& main) {
    PythonProcessDesc pythonProcessDesc(configtext);
    main = pythonProcessDesc.parameterSet();
  }

  std::shared_ptr<ParameterSet>
  readPSetsFrom(std::string const& module) {
    python::initializeModule();

    boost::python::object mainModule = object(handle<>(borrowed(PyImport_AddModule(const_cast<char*>("__main__")))));

    boost::python::object mainNamespace = mainModule.attr("__dict__");
    PythonParameterSet theProcessPSet;
    mainNamespace["topPSet"] = ptr(&theProcessPSet);

    try {
      // if it ends with py, it's a file
      if(module.substr(module.size()-3) == ".py") {
        makePSetsFromFile(module,mainNamespace);
      } else {
        makePSetsFromString(module,mainNamespace);
      }
    }
    catch( error_already_set ) {
      pythonToCppException("Configuration");
      Py_Finalize();
    }
    auto returnValue = std::make_shared<ParameterSet>();
    theProcessPSet.pset().swap(*returnValue);
    return returnValue;
  }
} // namespace edm

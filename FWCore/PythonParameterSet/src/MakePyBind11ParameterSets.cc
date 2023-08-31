#include "FWCore/PythonParameterSet/interface/MakePyBind11ParameterSets.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/PythonParameterSet/interface/Python11ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PyBind11ProcessDesc.h"
#include "initializePyBind11Module.h"
#include <pybind11/embed.h>

static void makePSetsFromFile(std::string const& fileName) {
  std::string initCommand(
      "from FWCore.ParameterSet.Types import makeCppPSet\n"
      "exec(open('");
  initCommand += fileName + "').read())";
  pybind11::exec(initCommand);
  pybind11::exec("makeCppPSet(locals(), topPSet)");
}

static void makePSetsFromString(std::string const& module) {
  std::string command = module;
  command += "\nfrom FWCore.ParameterSet.Types import makeCppPSet\nmakeCppPSet(locals(), topPSet)";
  pybind11::exec(command);
}

namespace edm {
  namespace cmspybind11 {
    std::unique_ptr<ParameterSet> readConfig(std::string const& config) {
      PyBind11ProcessDesc pythonProcessDesc(config, true);
      return pythonProcessDesc.parameterSet();
    }

    std::unique_ptr<ParameterSet> readConfig(std::string const& config, int argc, char* argv[]) {
      PyBind11ProcessDesc pythonProcessDesc(config, true, argc, argv);
      return pythonProcessDesc.parameterSet();
    }

    void makeParameterSets(std::string const& configtext, std::unique_ptr<ParameterSet>& main) {
      PyBind11ProcessDesc pythonProcessDesc(configtext, false);
      main = pythonProcessDesc.parameterSet();
    }

    std::unique_ptr<ParameterSet> readPSetsFrom(std::string const& module) {
      pybind11::scoped_interpreter guard{};
      python::initializePyBind11Module();
      std::unique_ptr<ParameterSet> retVal;
      {
        Python11ParameterSet theProcessPSet;
        pybind11::object mainModule = pybind11::module::import("__main__");
        mainModule.attr("topPSet") = pybind11::cast(&theProcessPSet);

        try {
          // if it ends with py, it's a file
          if (module.substr(module.size() - 3) == ".py") {
            makePSetsFromFile(module);
          } else {
            makePSetsFromString(module);
          }
        } catch (pybind11::error_already_set const& e) {
          pythonToCppException("Configuration", e.what());
        }
        retVal = std::make_unique<edm::ParameterSet>(ParameterSet(theProcessPSet.pset()));
      }
      return retVal;
    }
  }  // namespace cmspybind11
}  // namespace edm

#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"

#include "FWCore/PythonParameterSet/interface/PythonParameterSet.h"
#include "FWCore/PythonParameterSet/src/initializeModule.h"


using namespace boost::python;

static 
void
makePSetsFromFile(const std::string& fileName, boost::python::object& mainNamespace)
{
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
makePSetsFromString(const std::string& module, boost::python::object& mainNamespace)
{
  std::string command = module;
  command += "\nfrom FWCore.ParameterSet.Types import makeCppPSet\nmakeCppPSet(locals(), topPSet)";
  handle<>(PyRun_String(command.c_str(),
                        Py_file_input,
                        mainNamespace.ptr(),
                        mainNamespace.ptr()));
}

namespace edm
{

  boost::shared_ptr<ProcessDesc>
  readConfig(const std::string & config)
  {
    PythonProcessDesc pythonProcessDesc(config);
    return pythonProcessDesc.processDesc();
  }

  boost::shared_ptr<edm::ProcessDesc>
  readConfig(const std::string & config, int argc, char * argv[])
  {
    PythonProcessDesc pythonProcessDesc(config, argc, argv);
    return pythonProcessDesc.processDesc();
  }


  void
  makeParameterSets(std::string const& configtext,
                  boost::shared_ptr<ParameterSet>& main,
                  boost::shared_ptr<std::vector<ParameterSet> >& serviceparams)
  {
    PythonProcessDesc pythonProcessDesc(configtext);
    boost::shared_ptr<edm::ProcessDesc> processDesc = pythonProcessDesc.processDesc();
    main = processDesc->getProcessPSet();
    serviceparams = processDesc->getServicesPSets();
  }

  boost::shared_ptr<ParameterSet> 
  readPSetsFrom(const std::string& module) {
    edm::python::initializeModule();
    
    boost::python::object mainModule = object(handle<>(borrowed(PyImport_AddModule(const_cast<char *>("__main__")))));
    
    boost::python::object mainNamespace = mainModule.attr("__dict__");
    PythonParameterSet theProcessPSet;
    mainNamespace["topPSet"] = ptr(&theProcessPSet);

    try {
      // if it ends with py, it's a file
      if(module.substr(module.size()-3) == ".py")
      {
        makePSetsFromFile(module,mainNamespace);
      }
      else
      {
        makePSetsFromString(module,mainNamespace);
      }
    }
    catch( error_already_set ) {
      edm::pythonToCppException("Configuration");
      Py_Finalize();
    }
    boost::shared_ptr<ParameterSet> returnValue(new ParameterSet);
    theProcessPSet.pset().swap(*returnValue);
    return returnValue;
    
  }

} // namespace edm

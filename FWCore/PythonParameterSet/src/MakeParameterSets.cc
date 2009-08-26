#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"

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

} // namespace edm

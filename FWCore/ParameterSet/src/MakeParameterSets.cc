#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/interface/ParseTree.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/parse.h"

namespace edm
{

  void setStrictParsing(bool strict)
  {
    edm::pset::ParseTree::setStrictParsing(strict);
  }


  void
  makeParameterSets(std::string const& configtext,
		    boost::shared_ptr<ParameterSet>& main,
		    boost::shared_ptr<std::vector<ParameterSet> >& serviceparams)
  {
    // Handle 'include' statements in a pre-processing step.
    //std::string finalConfigDoc;
    //edm::pset::ConfigurationPreprocessor preprocessor;
    //preprocessor.process(configtext, finalConfigDoc);
    edm::ProcessDesc processDesc(configtext);
    //edm::ProcessDesc processDesc( finalConfigDoc);

    main = processDesc.getProcessPSet();

    // Load every ParameterSet into the Registry
    pset::Registry* reg = pset::Registry::instance();

    pset::loadAllNestedParameterSets(reg, *main);
    serviceparams = processDesc.getServicesPSets();

  }

  boost::shared_ptr<ProcessDesc>
  readConfigFile(const std::string & fileName)
  {
    if (fileName.size() > 3 && fileName.substr(fileName.size()-3) == ".py") 
    {
      PythonProcessDesc pythonProcessDesc(fileName);
      return pythonProcessDesc.processDesc();
    }
    else
    {
      std::string configString(edm::pset::read_whole_file(fileName));
      return boost::shared_ptr<ProcessDesc>(new ProcessDesc(configString));
    }
  }

} // namespace edm

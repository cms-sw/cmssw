#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/PythonProcessDesc.h"
#include "FWCore/ParameterSet/interface/ParseTree.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/pythonFileToConfigure.h"

#include <iostream>
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
    edm::ProcessDesc processDesc(configtext);

    main = processDesc.getProcessPSet();
    serviceparams = processDesc.getServicesPSets();

  }

  boost::shared_ptr<ProcessDesc>
  readConfigFile(const std::string & fileName)
  {
    if (fileName.size() > 3 && fileName.substr(fileName.size()-3) == ".py") 
    {
      PythonProcessDesc pythonProcessDesc(fileName);
      return pythonProcessDesc.processDesc();
      //std::string configString(pythonFileToConfigure(fileName)); 
      //return boost::shared_ptr<ProcessDesc>(new ProcessDesc(configString));
    }
    else
    {
      std::string configString(read_whole_file(fileName));
      return boost::shared_ptr<ProcessDesc>(new ProcessDesc(configString));
    }
  }

} // namespace edm

#include <string>

#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ParseTree.h"
#include "FWCore/ParameterSet/interface/Registry.h"

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
    serviceparams = processDesc.getServicesPSets();

    // Load every ParameterSet into the Registry
    pset::Registry* reg = pset::Registry::instance();

    pset::loadAllNestedParameterSets(reg, *main);

    typedef std::vector<ParameterSet>::const_iterator iter;

    for (iter i=serviceparams->begin(), e=serviceparams->end(); i!=e; ++i)
      reg->insertMapped(*i);
  }
} // namespace edm

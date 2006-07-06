#include <fstream>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/src/ConfigurationPreprocessor.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/Registry.h"

using boost::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

namespace edm
{

  void
  makeParameterSets(string const& configtext,
		    shared_ptr<ParameterSet>& main,
		    shared_ptr<vector<ParameterSet> >& serviceparams)
  {
    // Handle 'include' statements in a pre-processing step.
    //string finalConfigDoc;
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

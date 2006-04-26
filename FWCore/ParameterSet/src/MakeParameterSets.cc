#include <fstream>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/src/ConfigurationPreprocessor.h"
#include "FWCore/ParameterSet/interface/ProcessPSetBuilder.h"
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
    string finalConfigDoc;
    edm::pset::ConfigurationPreprocessor preprocessor;
    preprocessor.process(configtext, finalConfigDoc);
    edm::ProcessPSetBuilder builder(finalConfigDoc);

    main = builder.getProcessPSet();
    serviceparams = builder.getServicesPSets();

    // Load every ParameterSet into the Registry
    pset::loadAllNestedParameterSets(*main);
    {
      // Should be able to use boost::lambda here. It did seem easy to
      // get it to work...
      pset::Registry* reg = pset::Registry::instance();
      std::vector<ParameterSet>::const_iterator i = serviceparams->begin();
      std::vector<ParameterSet>::const_iterator e = serviceparams->end();
      for (; i != e; ++i) reg->insertParameterSet(*i);
    }
  }
} // namespace edm

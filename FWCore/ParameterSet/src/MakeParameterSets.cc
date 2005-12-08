#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/ProcessPSetBuilder.h"


using boost::shared_ptr;
using std::vector;

namespace edm
{

  void
  makeParameterSets(std::string const& configtext,
		    shared_ptr<ParameterSet>& main,
		    shared_ptr<std::vector<ParameterSet> >& serviceparams)
  {
    edm::ProcessPSetBuilder builder(configtext);
    main = builder.getProcessPSet();
    serviceparams = builder.getServicesPSets();

    // NOTE: FIX WHEN POOL BUG IS FIXED.
    // For now, we have to always make use of the "LoadAllDictionaries" service.
    serviceparams->push_back(ParameterSet());
    serviceparams->back().addParameter<std::string>("@service_type", "LoadAllDictionaries");
  }
} // namespace edm

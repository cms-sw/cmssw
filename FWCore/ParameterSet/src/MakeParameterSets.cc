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
  }
} // namespace edm

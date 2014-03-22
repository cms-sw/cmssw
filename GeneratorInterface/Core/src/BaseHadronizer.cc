
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"

namespace gen {

const std::vector<std::string> BaseHadronizer::theSharedResources;

BaseHadronizer::BaseHadronizer( edm::ParameterSet const& ps )
{

   runInfo().setFilterEfficiency(
      ps.getUntrackedParameter<double>("filterEfficiency", -1.) );
   runInfo().setExternalXSecLO(
      GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("crossSection", -1.)) );
   runInfo().setExternalXSecNLO(
       GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("crossSectionNLO", -1.)) );

}

}

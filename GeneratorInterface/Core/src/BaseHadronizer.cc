
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"

namespace gen {

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

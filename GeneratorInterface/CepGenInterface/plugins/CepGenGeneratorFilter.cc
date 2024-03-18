// CepGen-CMSSW interfacing module
//   2022-2024, Laurent Forthomme

#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/CepGenInterface/interface/CepGenEventGenerator.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

namespace gen {
  typedef edm::GeneratorFilter<gen::CepGenEventGenerator, gen::ExternalDecayDriver> CepGenGeneratorFilter;
}  // namespace gen

using gen::CepGenGeneratorFilter;
DEFINE_FWK_MODULE(CepGenGeneratorFilter);

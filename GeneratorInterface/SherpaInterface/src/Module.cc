#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/SherpaInterface/interface/SherpaSource.h"
//#include "GeneratorInterface/SherpaInterface/interface/SherpaInitial.h"
#include "SHERPA-MC/Sherpa.H"

  using edm::SherpaSource;
// using edm::SherpaInitial;

  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_INPUT_SOURCE(SherpaSource);
//  DEFINE_ANOTHER_FWK_INPUT_SOURCE(SherpaInitial);



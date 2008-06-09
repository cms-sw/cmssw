#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenSource.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenEmptyEventFilter.h"

  using edm::AlpgenSource;

  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_INPUT_SOURCE(AlpgenSource);
  DEFINE_ANOTHER_FWK_MODULE(AlpgenEmptyEventFilter);

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenSource.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenEmptyEventFilter.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenProducer.h"

  using edm::AlpgenSource;
  using edm::AlpgenProducer;

  DEFINE_SEAL_MODULE();
  DEFINE_FWK_INPUT_SOURCE(AlpgenSource);
  DEFINE_FWK_MODULE(AlpgenEmptyEventFilter);
  DEFINE_FWK_MODULE(AlpgenProducer);

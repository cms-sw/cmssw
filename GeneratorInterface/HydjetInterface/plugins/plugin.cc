//
// $Id: plugin.cc,v 1.1 2007/06/21 13:55:57 mballint Exp $
//

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/HydjetInterface/interface/HydjetSource.h"
#include "GeneratorInterface/HydjetInterface/interface/HydjetProducer.h"

using edm::HydjetSource;
using edm::HydjetProducer;

DEFINE_FWK_INPUT_SOURCE(HydjetSource);
DEFINE_FWK_MODULE(HydjetProducer);

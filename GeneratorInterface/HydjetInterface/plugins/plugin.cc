//
// $Id: plugin.cc,v 1.3 2009/05/20 23:15:30 yilmaz Exp $
//

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "GeneratorInterface/HydjetInterface/interface/HydjetSource.h"
//#include "GeneratorInterface/HydjetInterface/interface/HydjetProducer.h"
#include "GeneratorInterface/HydjetInterface/interface/HydjetGeneratorFilter.h"

//using edm::HydjetSource;
//using edm::HydjetProducer;
using gen::HydjetGeneratorFilter;

//DEFINE_FWK_INPUT_SOURCE(HydjetSource);
//DEFINE_FWK_MODULE(HydjetProducer);
DEFINE_FWK_MODULE(HydjetGeneratorFilter);

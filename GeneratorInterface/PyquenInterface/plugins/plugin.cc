//
// $Id: plugin.cc,v 1.1 2007/06/21 13:55:59 mballint Exp $
//

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/PyquenInterface/interface/PyquenSource.h"
#include "GeneratorInterface/PyquenInterface/interface/PyquenProducer.h"


using edm::PyquenSource;
using edm::PyquenProducer;

DEFINE_FWK_INPUT_SOURCE(PyquenSource);
DEFINE_FWK_MODULE(PyquenProducer);

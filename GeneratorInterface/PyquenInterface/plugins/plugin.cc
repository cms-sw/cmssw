//
// $Id: plugin.cc,v 1.2 2008/04/09 19:02:37 marafino Exp $
//

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/PyquenInterface/interface/PyquenSource.h"
#include "GeneratorInterface/PyquenInterface/interface/PyquenProducer.h"

using edm::PyquenSource;
using edm::PyquenProducer;

DEFINE_FWK_INPUT_SOURCE(PyquenSource);
DEFINE_FWK_MODULE(PyquenProducer);

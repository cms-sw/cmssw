// File: FastPilupSubtractionJetProducer.cc
// Description:  see FastPilupSubtractionJetProducer.h
// Author:  Andreas Oehler, University Karlsruhe (TH)
// Creation Date:  Nov. 06 2006 Initial version.
//--------------------------------------------
#include <memory>

#include "FastPilupSubtractionJetProducer.h"

using namespace std;
using namespace reco;
using namespace JetReco;

namespace {
  const bool debug = false;

}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.
  FastPilupSubtractionJetProducer::FastPilupSubtractionJetProducer(const edm::ParameterSet& conf)
    : BasePilupSubtractionJetProducer (conf), alg_(conf)

  {}

  // run algorithm itself
  bool FastPilupSubtractionJetProducer::runAlgorithm (const InputCollection& fInput, 
		     OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);

    return true;
  }
}

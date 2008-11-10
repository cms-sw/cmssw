// File: AntiKtPilupSubtractionJetProducer.cc
// Description:  see AntiKtPilupSubtractionJetProducer.h
// Author:  Andreas Oehler, University Karlsruhe (TH)
// Creation Date:  Nov. 06 2006 Initial version.
//--------------------------------------------
#include <memory>

#include "AntiKtPilupSubtractionJetProducer.h"

using namespace std;
using namespace reco;
using namespace JetReco;

namespace {
  const bool debug = false;

}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.
  AntiKtPilupSubtractionJetProducer::AntiKtPilupSubtractionJetProducer(const edm::ParameterSet& conf)
    : BasePilupSubtractionJetProducer (conf), alg_(conf)

  {}

  // run algorithm itself
  bool AntiKtPilupSubtractionJetProducer::runAlgorithm (const InputCollection& fInput, 
		     OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);

    return true;
  }
}

// File: KtJetProducer.cc
// Description:  see KtJetProducer.h
// Author:  Andreas Oehler, University Karlsruhe (TH)
// Creation Date:  Nov. 06 2006 Initial version.
//--------------------------------------------
#include <memory>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "KtJetProducer.h"

using namespace std;
using namespace reco;
using namespace JetReco;

namespace {
  const bool debug = false;

}

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.
  KtJetProducer::KtJetProducer(const edm::ParameterSet& conf)
    : BaseJetProducer (conf), alg_(conf)

  {}

  // run algorithm itself
  bool KtJetProducer::runAlgorithm (const InputCollection& fInput, 
		     OutputCollection* fOutput) {
    alg_.run (fInput, fOutput);

    return true;
  }
  //  DEFINE_FWK_MODULE( KtJetProducer );
}


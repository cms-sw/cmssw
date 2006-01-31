// File: CorrMETProducer.cc
// Description:  see CorrMETProducer.h
// Author: R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 14, 2005 Initial version.
//
//--------------------------------------------
#include <memory>

#include "RecoMET/METProducers/interface/CorrMETProducer.h"
#include "DataFormats/METObjects/interface/METCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "FWCore/Framework/interface/Handle.h"

using namespace edm;
using namespace std;

namespace cms 
{

  CorrMETProducer::CorrMETProducer(ParameterSet const& conf) : alg_(false) 
  {
    produces<METCollection>(); 
  }

  CorrMETProducer::CorrMETProducer() : alg_(false) 
  {
    produces<METCollection>(); 
  }

  CorrMETProducer::~CorrMETProducer() {}

  void CorrMETProducer::produce(Event& event, const EventSetup& setup) {

    // Step A: Get Inputs.
    edm::Handle<METCollection> rawmet;
    event.getByLabel("testmet", rawmet); 
    // Step B: Create an empty output.
    std::auto_ptr<METCollection> result(new METCollection);
    // Step C: Invoke the algorithm. 
    alg_.run(rawmet.product(), *result);
    // Step D: Put output into the Event.
    event.put(result);

  }

}

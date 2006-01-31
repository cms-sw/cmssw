// File: TestMETProducer.cc
// Description:  see TestMETProducer.h
// Author: R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 14, 2005 Initial version.
//
//--------------------------------------------
#include <memory>

#include "RecoMET/METProducers/interface/TestMETProducer.h"
#include "DataFormats/METObjects/interface/METCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "FWCore/Framework/interface/Handle.h"

using namespace edm;
using namespace std;

namespace cms 
{

  TestMETProducer::TestMETProducer(ParameterSet const& conf) : alg_(false) 
  {
    produces<METCollection>(); 
  }

  TestMETProducer::TestMETProducer() : alg_(false) 
  {
    produces<METCollection>(); 
  }

  TestMETProducer::~TestMETProducer() {}

  void TestMETProducer::produce(Event& event, const EventSetup& setup) 
  {
    // Step A: Get Inputs.
    edm::Handle<CaloTowerCollection> towers;
    event.getByType(towers);
    // Step B: Create an empty output.
    std::auto_ptr<METCollection> result(new METCollection);
    // Step C: Invoke the algorithm. 
    alg_.run(towers.product(), *result);
    // Step D: Put output into the Event.
    event.put(result);
  }
}

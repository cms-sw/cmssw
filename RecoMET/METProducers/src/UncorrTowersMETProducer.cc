// File: UncorrCaloTowersMETProducer.cc
// Description:  see UncorrCaloTowersMETProducer.h
// Author: Michael Schmitt, The University of Florida
// Creation Date:  MHS May 14, 2005 Initial version.
//
//--------------------------------------------

#include "RecoMET/METProducers/interface/UncorrTowersMETProducer.h"
#include "DataFormats/METObjects/interface/TowerMETCollection.h"

using namespace edm;
using namespace std;

namespace cms {

  UncorrTowersMETProducer::UncorrTowersMETProducer(ParameterSet const& conf):
    alg_(false) {}
  UncorrTowersMETProducer::UncorrTowersMETProducer():
    alg_(false) {}

  UncorrTowersMETProducer::~UncorrTowersMETProducer() {}

  void UncorrTowersMETProducer::produce(Event& event, const EventSetup& setup) {

    // Step A: Get Inputs.
    edm::Handle<CaloTowerCollection> towers;
    event.getByLabel("CalTwr", towers); 

    // Step B: Create an empty output.
    std::auto_ptr<TowerMETCollection> result(new TowerMETCollection);

    // Step C: Invoke the algorithm. 
    alg_.run(towers.product(), NULL, NULL, *result);

    // Step D: Put output into the Event.
    event.put(result);

  }

}

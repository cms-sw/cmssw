// File: TestMETProducer.cc
// Description:  see TestMETProducer.h
// Author: M. Schmitt, R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 14, 2005 Initial version.
//
//--------------------------------------------
#include <memory>

#include "RecoMET/METProducers/interface/TowerMETProducer.h"
#include "DataFormats/METObjects/interface/TowerMETCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "FWCore/Framework/interface/Handle.h"

using namespace edm;
using namespace std;

namespace cms 
{

  TowerMETProducer::TowerMETProducer(const edm::ParameterSet& iConfig) : alg_() 
  {
    produces<TowerMETCollection>(); 
    //inputLabel = iConfig.getParameter<std::string>("inputLabel");
  }

  TowerMETProducer::TowerMETProducer() : alg_() 
  {
    produces<TowerMETCollection>(); 
  }

  TowerMETProducer::~TowerMETProducer() {}

  void TowerMETProducer::produce(Event& event, const EventSetup& setup) 
  {
    // Step A: Get Inputs.
    edm::Handle<CaloTowerCollection> towers;
    event.getByType(towers);
    // Step B: Create an empty output.
    std::auto_ptr<TowerMETCollection> result(new TowerMETCollection);
    // Step C: Invoke the algorithm. 
    alg_.run(towers.product(), *result);
    // Step D: Put output into the Event.
    event.put(result);
  }
}

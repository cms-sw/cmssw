// File: KtJetProducer.cc
// Description:  see KtJetProducer.h
// Author:  Fernando Varela Rodriguez, Boston University
// Creation Date:  Apr. 22 2005 Initial version.
//
//--------------------------------------------
#include <memory>

#include "RecoJets/JetProducers/interface/KtJetProducer.h"
#include "DataFormats/JetObjects/interface/CaloJetCollection.h"
#include "FWCore/CoreFramework/interface/Handle.h"

namespace cms
{

  // Constructor takes input parameters now: to be replaced with parameter set.
  KtJetProducer::KtJetProducer(const edm::ParameterSet& conf)
  : alg_(edm::getParameter<int>(conf, "ktAngle"),
	 edm::getParameter<int>(conf, "ktRecom"),
	 edm::getParameter<double>(conf, "ktECut"),
	 edm::getParameter<double>(conf, "ktRParam"))
  { }

  // Virtual destructor needed.
  KtJetProducer::~KtJetProducer() { }  

  // Functions that gets called by framework every event
  void KtJetProducer::produce(edm::Event& e, const edm::EventSetup&)
  {
    // Step A: Get Inputs 
    edm::Handle<CaloTowerCollection> towers;  //Fancy Event Pointer to CaloTowers
    e.getByLabel("CalTwr", towers);           //Set pointer to CaloTowers

    // Step B: Create empty output 
    std::auto_ptr<CaloJetCollection> result(new CaloJetCollection);  //Empty Jet Coll

    // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
    *result = (*(alg_.findJets(*(towers.product()))));  //Makes Full Jet Collection

    // Step D: Put outputs into event
    e.put(result);  //Puts Jet Collection into event
  }

}

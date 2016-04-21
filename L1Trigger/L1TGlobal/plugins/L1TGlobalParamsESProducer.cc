#include "L1TGlobalParamsESProducer.h"

// system include files
#include <memory>
#include <vector>

#include "boost/shared_ptr.hpp"

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/DataRecord/interface/L1TGlobalParametersRcd.h"

void L1TGlobalParamsESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<int>("NumberBxInEvent", 5);
  desc.add<unsigned int>("NumberPhysTriggers", 512);
  desc.add<unsigned int>("NumberL1Mu", 12);
  desc.add<unsigned int>("NumberL1EG", 12);
  desc.add<unsigned int>("NumberL1Jet", 12);
  desc.add<unsigned int>("NumberL1Tau", 8);
  desc.add<unsigned int>("NumberChips", 1);
  desc.add<unsigned int>("PinsOnChip", 512);
  std::vector<int> tmp = {1};
  desc.add<std::vector<int> >("OrderOfChip", tmp);
  descriptions.add("L1TGlobalParamsESProducer", desc);
}

L1TGlobalParamsESProducer::L1TGlobalParamsESProducer(const edm::ParameterSet& parSet) : 
  data_(new L1TGlobalParameters())						     
{
    // tell the framework what data is being produced
    setWhatProduced(
        this, &L1TGlobalParamsESProducer::produce);

    // set the number of bx in event
    data_.setGtTotalBxInEvent(parSet.getParameter<int>("NumberBxInEvent"));

    // set the number of physics trigger algorithms
    data_.setGtNumberPhysTriggers(parSet.getParameter<unsigned int>("NumberPhysTriggers"));

    // set the number of L1 muons received by GT
    data_.setGtNumberL1Mu(parSet.getParameter<unsigned int>("NumberL1Mu"));
    
    //  set the number of L1 e/gamma objects received by GT
    data_.setGtNumberL1EG(parSet.getParameter<unsigned int>("NumberL1EG"));
       
    // set the number of L1 central jets received by GT
    data_.setGtNumberL1Jet(parSet.getParameter<unsigned int>("NumberL1Jet"));
        
    // set the number of L1 tau jets received by GT
    data_.setGtNumberL1Tau(parSet.getParameter<unsigned int>("NumberL1Tau"));
       
    // hardware stuff
    
    // set the number of condition chips in GTL
    data_.setGtNumberChips(parSet.getParameter<unsigned int>("NumberChips"));
    
    // set the number of pins on the GTL condition chips
    data_.setGtPinsOnChip(parSet.getParameter<unsigned int>("PinsOnChip"));
    
    // set the correspondence "condition chip - GTL algorithm word"
    // in the hardware
    data_.setGtOrderOfChip(parSet.getParameter<std::vector<int> >("OrderOfChip"));

}

// destructor
L1TGlobalParamsESProducer::~L1TGlobalParamsESProducer() {

    // empty

}

// member functions

// method called to produce the data
boost::shared_ptr<L1TGlobalParameters> 
    L1TGlobalParamsESProducer::produce(
        const L1TGlobalParametersRcd& iRecord) {

    boost::shared_ptr<L1TGlobalParameters> pL1uGtStableParameters =
        boost::shared_ptr<L1TGlobalParameters>(data_.getWriteInstance());

    return pL1uGtStableParameters;

}

DEFINE_FWK_EVENTSETUP_MODULE(L1TGlobalParamsESProducer);

/**
 * \class L1TGlobalParamsESProducer
 * 
 * 
 * Description: ESProducer for L1 GT parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 *
 */

// this class header
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

// forward declarations

// constructor(s)
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


// StableParametersTrivialProducer


#include <memory>
#include <vector>
#include <boost/cstdint.hpp>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/L1TObjects/interface/L1TGlobalParameters.h"
#include "CondFormats/DataRecord/interface/L1TGlobalParametersRcd.h"

#include "L1Trigger/L1TGlobal/interface/GlobalParamsHelper.h"

// class declaration
class StableParametersTrivialProducer : public edm::ESProducer
{

public:

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    /// constructor
    StableParametersTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~StableParametersTrivialProducer();

    /// public methods

    /// L1 GT parameters
    std::shared_ptr<L1TGlobalParameters> produceGtStableParameters(
        const L1TGlobalParametersRcd&);

private:

    l1t::GlobalParamsHelper data_;

};

using namespace std;
using namespace edm;
using namespace l1t;

void StableParametersTrivialProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;

  // TotalBxInEvent = cms.int32(5),
  desc.add<int> ("TotalBxInEvent", 5)->setComment("stage2");
  
  //NumberPhysTriggers = cms.uint32(512)
  desc.add<unsigned int> ("NumberPhysTriggers", 512)->setComment("Number of physics trigger algorithms");
  
  //NumberL1Muon = cms.uint32(12)
  desc.add<unsigned int> ("NumberL1Muon", 12)->setComment("Number of L2 Muons");
  
  //NumberL1EGamma = cms.uint32(12),    
  desc.add<unsigned int> ("NumberL1EGamma", 12)->setComment("Number of L1 e/gamma objects");

  //NumberL1Jet = cms.uint32(12),
  desc.add<unsigned int> ("NumberL1Jet", 12)->setComment("Number of L1 jets");

  //NumberL1Tau = cms.uint32(8),
  desc.add<unsigned int> ("NumberL1Tau", 8)->setComment("Number of L1 taus");
    
  //NumberChips = cms.uint32(1),
  desc.add<unsigned int> ("NumberChips", 5)->setComment("Number of chips in Menu");

  //PinsOnChip = cms.uint32(512),
  desc.add<unsigned int> ("PinsOnChip", 512)->setComment("Number of pins on the GTL condition chips");

  //OrderOfChip = cms.vint32(1),
  vector<int> tmp = {1};
  desc.add<vector<int> > ("OrderOfChip", tmp)->setComment("Chip order");


  //
  // Deprecated Parameters:  These can be removed once the HLT inteface is updated, or HLT takes these conditions from Offline DB.
  //
  
  desc.add<unsigned int> ("NumberL1IsoEG", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("NumberL1JetCounts", 0)->setComment("Deprecated...");
  desc.add<int> ("UnitLength", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("NumberL1ForJet", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("IfCaloEtaNumberBits", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("IfMuEtaNumberBits", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("NumberL1TauJet", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("NumberL1Mu", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("NumberConditionChips", 0)->setComment("Deprecated...");
  desc.add<int> ("NumberPsbBoards", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("NumberL1CenJet", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("PinsOnConditionChip", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("NumberL1NoIsoEG", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("NumberTechnicalTriggers", 0)->setComment("Deprecated...");
  desc.add<unsigned int> ("NumberPhysTriggersExtended", 0)->setComment("Deprecated...");
  desc.add<int> ("WordLength", 0)->setComment("Deprecated...");
  vector<int> tmp2 = {1};
  desc.add<vector<int> > ("OrderConditionChip", tmp2)->setComment("Deprecated...");

  descriptions.add("L1TGlobalProducer", desc);
}


StableParametersTrivialProducer::StableParametersTrivialProducer(
								 const edm::ParameterSet& parSet) : 
  data_(new L1TGlobalParameters()) {

    // tell the framework what data is being produced
    setWhatProduced(
        this, &StableParametersTrivialProducer::produceGtStableParameters);


    // set the number of bx in event
    data_.setTotalBxInEvent(parSet.getParameter<int>("TotalBxInEvent"));

    // set the number of physics trigger algorithms
    data_.setNumberPhysTriggers(parSet.getParameter<unsigned int>("NumberPhysTriggers"));

    // set the number of L1 muons received by GT
    data_.setNumberL1Mu(parSet.getParameter<unsigned int>("NumberL1Muon"));
    
    //  set the number of L1 e/gamma objects received by GT
    data_.setNumberL1EG(parSet.getParameter<unsigned int>("NumberL1EGamma"));
       
    // set the number of L1 central jets received by GT
    data_.setNumberL1Jet(parSet.getParameter<unsigned int>("NumberL1Jet"));
        
    // set the number of L1 tau jets received by GT
    data_.setNumberL1Tau(parSet.getParameter<unsigned int>("NumberL1Tau"));
       
    // hardware stuff
    
    // set the number of condition chips in GTL
    data_.setNumberChips(parSet.getParameter<unsigned int>("NumberChips"));
    
    // set the number of pins on the GTL condition chips
    data_.setPinsOnChip(parSet.getParameter<unsigned int>("PinsOnChip"));
    
    // set the correspondence "condition chip - GTL algorithm word"
    // in the hardware
    data_.setOrderOfChip(parSet.getParameter<std::vector<int> >("OrderOfChip"));

}

// destructor
StableParametersTrivialProducer::~StableParametersTrivialProducer() {

    // empty

}

// member functions

// method called to produce the data
std::shared_ptr<L1TGlobalParameters> StableParametersTrivialProducer::produceGtStableParameters(const L1TGlobalParametersRcd& iRecord) {

  auto pL1uGtStableParameters = std::shared_ptr<L1TGlobalParameters>(data_.getWriteInstance());

  return pL1uGtStableParameters;
  
}

DEFINE_FWK_EVENTSETUP_MODULE(StableParametersTrivialProducer);

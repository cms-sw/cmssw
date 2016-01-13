#include <string>
#include <list>
#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "HLTrigger/HLTfilters/interface/HLTL1TSeed.h"
#include "HLTrigger/HLTfilters/interface/HLTL1TSeed.h"

using namespace std;


// constructors
HLTL1TSeed::HLTL1TSeed(const edm::ParameterSet& parSet) : 
  HLTStreamFilter(parSet),
  //useObjectMaps_(parSet.getParameter<bool>("L1UseL1TriggerObjectMaps")),
  //logicalExpression_(parSet.getParameter<string>("L1SeedsLogicalExpression")),
  muonCollectionsTag_(parSet.getParameter<edm::InputTag>("muonCollectionsTag")), // FIX WHEN UNPACKERS ADDED
  muonTag_(muonCollectionsTag_),
  muonToken_(consumes<l1t::MuonBxCollection>(muonTag_))
{


  // InputTag for L1 Global Trigger object maps
  //m_l1GtObjectMapTag(parSet.getParameter<edm::InputTag> (
  //        "L1GtObjectMapTag")),
  //m_l1GtObjectMapToken(consumes<L1GlobalTriggerObjectMapRecord>(m_l1GtObjectMapTag)),
  

  
  //if (m_l1SeedsLogicalExpression != "L1GlobalDecision") {
  // check also the logical expression - add/remove spaces if needed
  //m_l1AlgoLogicParser = L1GtLogicParser(m_l1SeedsLogicalExpression);
  // list of required algorithms for seeding
        // dummy values for tokenNumber and tokenResult
        //m_l1AlgoSeeds.reserve((m_l1AlgoLogicParser.operandTokenVector()).size());
        //m_l1AlgoSeeds = m_l1AlgoLogicParser.expressionSeedsOperandList();
        //size_t l1AlgoSeedsSize = m_l1AlgoSeeds.size();

        //
        //m_l1AlgoSeedsRpn.reserve(l1AlgoSeedsSize);
        //m_l1AlgoSeedsObjType.reserve(l1AlgoSeedsSize);
  //} else {
  //    m_l1GlobalDecision = true;
  //}


  cout << "DEBUG:  muonCollectionsTag:  " << muonCollectionsTag_ << "\n";
  cout << "DEBUG:  muonTag:  " << muonTag_ << "\n";

  //LogDebug("HLTL1TSeed") 
  //<< "\n";
    //<< "L1 Seeding using L1 trigger object maps:       "
    //<< useL1TriggerObjectMaps_ << "\n"
    //<< "  if false: seeding with all L1T objects\n"
    //<< "L1 Seeds Logical Expression:                   " << "\n      "
    //<< logicalExpression_ << "\n";
}

// destructor
HLTL1TSeed::~HLTL1TSeed() {
    // empty now
}

// member functions

void
HLTL1TSeed::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);

  // # default: true
  // #    seeding done via L1 trigger object maps, with objects that fired
  // #    only objects from the central BxInEvent (L1A) are used
  // # if false:
  // #    seeding is done ignoring if a L1 object fired or not,
  // #    adding all L1EXtra objects corresponding to the object types
  // #    used in all conditions from the algorithms in logical expression
  // #    for a given number of BxInEvent
  //desc.add<bool>("L1UseL1TriggerObjectMaps",true);

  // # logical expression for the required L1 algorithms;
  // # the algorithms are specified by name
  // # allowed operators: "AND", "OR", "NOT", "(", ")"
  // #
  // # by convention, "L1GlobalDecision" logical expression means global decision
  //desc.add<string>("L1SeedsLogicalExpression","");

  desc.add<edm::InputTag>("muonCollectionsTag",edm::InputTag("simGmtStage2Digis"));
  descriptions.add("hltL1TSeed", desc);
}

bool HLTL1TSeed::hltFilter(edm::Event& iEvent, const edm::EventSetup& evSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) {

  cout << "MY SIZE:  " << filterproduct.getCollectionTagsAsStrings().size() << "\n";

  // the filter object
  if (saveTags()) {
    
    cout << "INFO:  calling problem guy...\n";
    //cout << muonTag_.encode() << "\n";
    // I don't understand why this crashes:
    //filterproduct.addCollectionTag(muonTag_);
    cout << "INFO:  completed ok...\n"; 
  }
  
  //seedsL1TriggerObjectMaps(iEvent, filterproduct, l1GtTmAlgo.product(), gtReadoutRecordPtr, physicsDaqPartition))
  seedsAll(iEvent, filterproduct);
  return true;

}

// seeding is done ignoring if a L1 object fired or not
// if the event is selected at L1, fill all the L1 objects of types corresponding to the
// L1 conditions from the seeding logical expression for bunch crosses F, 0, 1
// directly from L1Extra and use them as seeds at HLT
// method and filter return true if at least an object is filled
bool HLTL1TSeed::seedsAll(edm::Event & iEvent, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

    //
    bool objectsInFilter = false;

    edm::Handle<l1t::MuonBxCollection> muons;
    iEvent.getByToken(muonToken_, muons);
    if (!muons.isValid()){ 
      edm::LogWarning("HLTL1TSeed")
	<< "\nWarning: L1MuonBxCollection with input tag "
	<< muonTag_
	<< "\nrequested in configuration, but not found in the event."
	<< "\nNo muons added to filterproduct."
	<< endl;	
    } else {
      cout << "DEBUG:  L1T adding muons to filterproduct...\n";

      l1t::MuonBxCollection::const_iterator iter;
      for (iter = muons->begin(0); iter != muons->end(0); ++iter){
	//objectsInFilter = true;
	l1t::MuonRef myref(muons, muons->key(iter));
	//filterproduct.addObject(trigger::TriggerL1Mu, myref);
      }
    }

    l1t::MuonBxCollection::const_iterator iter;



    /*
      int iObj = -1;

      iObj++;    
      int bxNr = objIter->bx();
      if ((bxNr >= minBxInEvent) && (bxNr <= maxBxInEvent))	    
      objectsInFilter = true;
      filterproduct.addObject(
      trigger::TriggerL1Mu,
      l1extra::L1MuonParticleRef(
      l1Muon, iObj));
    */
    return objectsInFilter;
}

// detailed print of filter content
void HLTL1TSeed::dumpTriggerFilterObjectWithRefs(trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  LogDebug("HLTL1TSeed") << "\nHLTL1TSeed::hltFilter "
			 << "\n  Dump TriggerFilterObjectWithRefs\n" << endl;
  
  //vector<l1extra::L1MuonParticleRef> seedsL1Mu;
  //filterproduct.getObjects(trigger::TriggerL1Mu, seedsL1Mu);
  //const size_t sizeSeedsL1Mu = seedsL1Mu.size();

  /*
  LogTrace("HLTL1TSeed") << "  L1Mu seeds:      " << sizeSeedsL1Mu << "\n"
			 << endl;

  for (size_t i = 0; i != sizeSeedsL1Mu; i++) {

    
    l1extra::L1MuonParticleRef obj = l1extra::L1MuonParticleRef(
								seedsL1Mu[i]);
    
        LogTrace("HLTL1TSeed") << "L1Mu     " << "\t" << "q*PT = "
                << obj->charge() * obj->pt() << "\t" << "eta =  " << obj->eta()
                << "\t" << "phi =  " << obj->phi() << "\t" << "BX = "
                << obj->bx();
  }
  */

  LogTrace("HLTL1TSeed") << " \n\n" << endl;

}

// register as framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTL1TSeed);

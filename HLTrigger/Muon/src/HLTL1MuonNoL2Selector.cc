//-------------------------------------------------
//
/**  \class HLTL1MuonNoL2Selector
 * 
 *   HLTL1MuonNoL2Selector:
 *   Simple selector to output a subset of L1 muon collection 
 *   with no L2 link.
 *   
 *   based on RecoMuon/L2MuonSeedGenerator
 *
 *
 *   \author  S. Folgueras
 */
//
//--------------------------------------------------

// Class Header
#include "HLTrigger/Muon/interface/HLTL1MuonNoL2Selector.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

using namespace std;
using namespace edm;
using namespace l1t;

// constructors
HLTL1MuonNoL2Selector::HLTL1MuonNoL2Selector(const edm::ParameterSet& iConfig) : 
  theL1Source_(iConfig.getParameter<InputTag>("InputObjects")),
  theL1MinPt_(iConfig.getParameter<double>("L1MinPt")),
  theL1MaxEta_(iConfig.getParameter<double>("L1MaxEta")),
  theL1MinQuality_(iConfig.getParameter<unsigned int>("L1MinQuality")),
  theL2CandTag_     (iConfig.getParameter< edm::InputTag > ("L2CandTag")),
  theL2CandToken_   (consumes<reco::RecoChargedCandidateCollection>(theL2CandTag_)),
  theL1CandTag_   (iConfig.getParameter<InputTag > ("L1CandTag")),
  theL1CandToken_ (consumes<trigger::TriggerFilterObjectWithRefs>(theL1CandTag_)),
  seedMapTag_( iConfig.getParameter<InputTag >("SeedMapTag") ),
  seedMapToken_(consumes<SeedMap>(seedMapTag_))
{
  muCollToken_ = consumes<MuonBxCollection>(theL1Source_);

  produces<MuonBxCollection>(); 
}

// destructor
HLTL1MuonNoL2Selector::~HLTL1MuonNoL2Selector(){
}

void
HLTL1MuonNoL2Selector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputObjects",edm::InputTag(""));
  desc.add<edm::InputTag>("L2CandTag",edm::InputTag("hltL2MuonCandidates"));
  desc.add<edm::InputTag>("L1CandTag",edm::InputTag(""));
  desc.add<edm::InputTag>("SeedMapTag",edm::InputTag("hltL2Muons"));
  desc.add<double>("L1MinPt",-1.);
  desc.add<double>("L1MaxEta",5.0);
  desc.add<unsigned int>("L1MinQuality",0);
  descriptions.add("hltL1MuonNoL2Selector",desc);
}

void HLTL1MuonNoL2Selector::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  using namespace std;
  using namespace reco;
  using namespace trigger;

  const std::string metname = "Muon|RecoMuon|HLTL1MuonNoL2Selector";

  unique_ptr<MuonBxCollection> output(new MuonBxCollection());
  
  // Get hold of L2 trks
  edm::Handle<RecoChargedCandidateCollection> L2cands;
  iEvent.getByToken(theL2CandToken_,L2cands);
  
  // get the L2 to L1 map object for this event
  //  HLTMuonL2ToL1TMap mapL2ToL1(theL1CandToken_, seedMapToken_, iEvent);

  // Muon particles 
  edm::Handle<MuonBxCollection> muColl;
  iEvent.getByToken(muCollToken_, muColl);
  LogTrace(metname) << "Number of muons " << muColl->size() << endl;
  
  edm::Handle<SeedMap> seedMapHandle;
  iEvent.getByToken(seedMapToken_, seedMapHandle);
  
  std::vector<l1t::MuonRef> firedL1Muons_;
  edm::Handle<trigger::TriggerFilterObjectWithRefs> L1Cands;
  iEvent.getByToken(theL1CandToken_, L1Cands);
  L1Cands->getObjects(trigger::TriggerL1Mu, firedL1Muons_);
      
  for (int ibx = muColl->getFirstBX(); ibx <= muColl->getLastBX(); ++ibx) {
    if (ibx != 0) continue;
    for (auto it = muColl->begin(ibx); it != muColl->end(ibx); it++){
      l1t::MuonRef l1muon(muColl, distance(muColl->begin(muColl->getFirstBX()),it) );
      
      // only select L1's that fired: 
      if(find(firedL1Muons_.begin(), firedL1Muons_.end(), l1muon) == firedL1Muons_.end()) continue;
      
      // Loop over L2's to find whether the L1 fired this L2. 
      bool isTriggeredByL1=false;
      for (RecoChargedCandidateCollection::const_iterator cand = L2cands->begin(); cand != L2cands->end(); cand++) {
	TrackRef l2muon = cand->get<TrackRef>();    
	const edm::RefVector<L2MuonTrajectorySeedCollection>& seeds = (*seedMapHandle)[l2muon->seedRef().castTo<edm::Ref<L2MuonTrajectorySeedCollection> >()];
	for(size_t i=0; i<seeds.size(); i++){
	  // Check if the L2 was seeded by a triggered L1, in such case skip the loop. 
	  if(find(firedL1Muons_.begin(), firedL1Muons_.end(), seeds[i]->l1tParticle()) != firedL1Muons_.end()){
	    isTriggeredByL1 = true;
	    break;
	  }
	}
	if (!isTriggeredByL1) {
	  output->push_back( ibx, *it);
	}
      }	
    }
  } // loop over L1
  
  iEvent.put(std::move(output));
}


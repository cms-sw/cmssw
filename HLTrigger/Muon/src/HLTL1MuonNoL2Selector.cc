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
  centralBxOnly_( iConfig.getParameter<bool>("CentralBxOnly") ),
  theL2CandTag_     (iConfig.getParameter< edm::InputTag > ("L2CandTag")),
  theL2CandToken_   (consumes<reco::RecoChargedCandidateCollection>(theL2CandTag_)),
  seedMapTag_( iConfig.getParameter<InputTag >("SeedMapTag") ),
  seedMapToken_(consumes<SeedMap>(seedMapTag_))
{
  muCollToken_ = consumes<MuonBxCollection>(theL1Source_);

  produces<MuonBxCollection>(); 
}

// destructor
HLTL1MuonNoL2Selector::~HLTL1MuonNoL2Selector()= default;

void
HLTL1MuonNoL2Selector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputObjects",edm::InputTag(""));
  desc.add<edm::InputTag>("L2CandTag",edm::InputTag("hltL2MuonCandidates"));
  desc.add<edm::InputTag>("SeedMapTag",edm::InputTag("hltL2Muons"));
  desc.add<double>("L1MinPt",-1.);
  desc.add<double>("L1MaxEta",5.0);
  desc.add<unsigned int>("L1MinQuality",0);
  // # OBSOLETE - these parameters are ignored, they are left only not to break old configurations
  // they will not be printed in the generated cfi.py file
  desc.addOptionalNode(edm::ParameterDescription<edm::InputTag>("L1CandTag", edm::InputTag(""), false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.add<bool>("CentralBxOnly", true);
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
  
  // Muon particles 
  edm::Handle<MuonBxCollection> muColl;
  iEvent.getByToken(muCollToken_, muColl);
  LogTrace(metname) << "Number of muons " << muColl->size() << endl;
  
  edm::Handle<SeedMap> seedMapHandle;
  iEvent.getByToken(seedMapToken_, seedMapHandle);
  
  for (int ibx = muColl->getFirstBX(); ibx <= muColl->getLastBX(); ++ibx) {
    if (centralBxOnly_ && (ibx != 0)) continue;
    for (auto it = muColl->begin(ibx); it != muColl->end(ibx); it++){
      l1t::MuonRef l1muon(muColl, distance(muColl->begin(muColl->getFirstBX()),it) );
      
      unsigned int quality = it->hwQual();
      float pt    =  it->pt();
      float eta   =  it->eta();
      
      if ( pt < theL1MinPt_ || std::abs(eta) > theL1MaxEta_  || quality <= theL1MinQuality_) continue;
      
      // Loop over L2's to find whether the L1 fired this L2. 
      bool isTriggeredByL1=false;
      for (auto const & cand : *L2cands) {
	TrackRef l2muon = cand.get<TrackRef>();    
	const edm::RefVector<L2MuonTrajectorySeedCollection>& seeds = (*seedMapHandle)[l2muon->seedRef().castTo<edm::Ref<L2MuonTrajectorySeedCollection> >()];
	for(auto const & seed : seeds){
	  // Check if the L2 was seeded by a triggered L1, in such case skip the loop. 
	  if(seed->l1tParticle()==l1muon) {
	    isTriggeredByL1 = true;
	    break;
	  }
	}
	if (isTriggeredByL1) break; // if I found a L2 I do not need to loop on the rest.
      }
      // Once we loop on all L2 decide:
      if (!isTriggeredByL1) {
	output->push_back( ibx, *it);
      }
    }
  } // loop over L1
  
  iEvent.put(std::move(output));
}

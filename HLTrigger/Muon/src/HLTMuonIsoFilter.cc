/** \class HLTMuonIsoFilter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/Muon/interface/HLTMuonIsoFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include <iostream>
//
// constructors and destructor
//
HLTMuonIsoFilter::HLTMuonIsoFilter(const edm::ParameterSet& iConfig) :
   candTag_ (iConfig.getParameter< edm::InputTag > ("CandTag") ),
   isoTag_  (iConfig.getParameter< edm::InputTag > ("IsoTag" ) ),
   min_N_   (iConfig.getParameter<int> ("MinN"))
{
   LogDebug("HLTMuonIsoFilter") << " candTag : " << candTag_.encode()
      << "  IsoTag : " << isoTag_.encode()
      << "  MinN : " << min_N_;

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTMuonIsoFilter::~HLTMuonIsoFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTMuonIsoFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace trigger;
   using namespace reco;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<TriggerFilterObjectWithRefs>
     filterproduct (new TriggerFilterObjectWithRefs(path(),module()));
   // Ref to Candidate object to be recorded in filter object
   RecoChargedCandidateRef ref;


   // get hold of trks
   Handle<TriggerFilterObjectWithRefs> mucands;
   iEvent.getByLabel (candTag_,mucands);

   Handle<MuIsoFlagMap> depMap;
   iEvent.getByLabel (isoTag_,depMap);

   // look at all mucands,  check cuts and add to filter object
   int n = 0;
   vector<RecoChargedCandidateRef> vcands;
   mucands->getObjects(TriggerMuon,vcands);
   for (unsigned int i=0; i<vcands.size(); i++) {
     RecoChargedCandidateRef candref =  RecoChargedCandidateRef(vcands[i]);
     TrackRef tk = candref->get<TrackRef>();
     MuIsoFlagMap::value_type muonIsIsolated = (*depMap)[tk];
     LogDebug("HLTMuonIsoFilter") << " Muon with q*pt= " << tk->charge()*tk->pt() << ", eta= " << tk->eta() << "; Is Muon isolated? " << muonIsIsolated;
     
     if (!muonIsIsolated) continue;
     
     n++;
     filterproduct->addObject(TriggerMuon,candref);
   }

   // filter decision
   const bool accept (n >= min_N_);

   // put filter object into the Event
   iEvent.put(filterproduct);

   LogDebug("HLTMuonIsoFilter") << " >>>>> Result of HLTMuonIsoFilter is " << accept << ", number of muons passing isolation cuts= " << n; 

   return accept;
}

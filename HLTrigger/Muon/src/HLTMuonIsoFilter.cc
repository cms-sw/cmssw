/** \class HLTMuonIsoFilter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/Muon/interface/HLTMuonIsoFilter.h"

#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Common/interface/AssociationMap.h"

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
   produces<reco::HLTFilterObjectWithRefs>();
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
   using namespace reco;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<HLTFilterObjectWithRefs>
     filterproduct (new HLTFilterObjectWithRefs(path(),module()));
   // Ref to Candidate object to be recorded in filter object
   RefToBase<Candidate> ref;


   // get hold of trks
   Handle<HLTFilterObjectWithRefs> mucands;
   iEvent.getByLabel (candTag_,mucands);

   Handle<MuIsoAssociationMap> depMap;
   iEvent.getByLabel (isoTag_,depMap);

   // look at all mucands,  check cuts and add to filter object
   int n = 0;
   for (unsigned int i=0; i<mucands->size(); i++) {
      RefToBase<Candidate> candref = mucands->getParticleRef(i);
      TrackRef tk = candref->get<TrackRef>();
      MuIsoAssociationMap::result_type muonIsIsolated = (*depMap)[tk];

      LogDebug("HLTMuonIsoFilter") 
         << " Is Muon isolated? " << muonIsIsolated;

      if (!muonIsIsolated) continue;

      n++;
      filterproduct->putParticle(candref);
   }

   // filter decision
   const bool accept (n >= min_N_);

   // put filter object into the Event
   iEvent.put(filterproduct);

   LogDebug("HLTMuonIsoFilter") 
     << " Muons passing HLT isolation cuts= " << n;

   return accept;
}

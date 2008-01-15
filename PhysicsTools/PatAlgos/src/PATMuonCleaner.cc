//
// $Id: PATMuonCleaner.cc,v 1.1 2008/01/15 13:30:13 lowette Exp $
//

#include "PhysicsTools/PatAlgos/interface/PATMuonCleaner.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TMath.h"

#include <vector>
#include <memory>

using pat::PATMuonCleaner;

template<class T>
class PtComparatorOfIndices {
  public:
      PtComparatorOfIndices(const T& coll) : coll_(coll) { }
      typedef size_t first_argument_type;
      typedef size_t second_argument_type;
      bool operator()( const size_t & t1, const size_t & t2 ) const {
          return coll_[t1].pt() > coll_[t2].pt();
      }
  private:
    const T & coll_;
};

PATMuonCleaner::PATMuonCleaner(const edm::ParameterSet & iConfig) :
  muonSrc_(iConfig.getParameter<edm::InputTag>( "muonSource" )) 
{
  // produces vector of muons
  produces<std::vector<reco::Muon> >();

  // producers also backmatch to the muons
  produces<reco::CandRefValueMap>();
}


PATMuonCleaner::~PATMuonCleaner() {
}


void PATMuonCleaner::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // Get the collection of muons from the event
  edm::Handle<edm::View<reco::Muon> > muonHandle;
  iEvent.getByLabel(muonSrc_, muonHandle);

  std::auto_ptr<reco::MuonCollection> selected(new reco::MuonCollection());
  std::auto_ptr<reco::CandRefValueMap> backRefs(new reco::CandRefValueMap());
  reco::CandRefValueMap::Filler backRefFiller(*backRefs);
  edm::RefProd<reco::MuonCollection>  ourRefProd = iEvent.getRefBeforePut<reco::MuonCollection>();
  std::vector< edm::RefToBase<reco::Candidate> > originalRefs;
  
  for (size_t m = 0, n = muonHandle->size(), accepted = 0; m < n; ++m) {
    const reco::Muon &srcMuon = (*muonHandle)[m];    

    // clone the muon so we can modify it
    reco::Muon ourMuon = srcMuon; 

    // perform the selection
    if (false) continue; // now there is no real selection for muons

    // write the muon
    accepted++;
    selected->push_back(ourMuon);

    // make the backRef
    edm::RefToBase<reco::Muon> backRef(muonHandle, m);
    originalRefs.push_back(edm::RefToBase<reco::Candidate>(backRef)); 
    // must be a RefToBase. push_back should convert it into edm::RefToBase<reco::Candidate>
  }

  // sort muons in pt. I can't just sort "selected" because I need the refs, so I'll sort indices
  // step 1: build list of indices
  size_t nselected = selected->size();
  std::vector<size_t> indices(nselected);
  for (size_t i = 0; i < nselected; ++i) indices[i] = i;
  // step 2: sort the list of indices
  PtComparatorOfIndices<reco::MuonCollection> pTComparator(*selected); 
  std::sort(indices.begin(), indices.end(), pTComparator);
  // step 3: use sorted indices
  std::auto_ptr<reco::MuonCollection> sorted(new reco::MuonCollection(nselected));
  std::vector< edm::RefToBase<reco::Candidate> > sortedRefs(nselected);
  for (size_t i = 0; i < nselected; ++i) {
        (*sorted)[i]     = (*selected)[indices[i]];
        sortedRefs[i] = originalRefs[i];
  }

  // fill in backrefs
  backRefFiller.insert(ourRefProd, sortedRefs.begin(), sortedRefs.end());
  backRefFiller.fill();

  // put objects in Event
  iEvent.put(sorted);
  iEvent.put(backRefs);
}



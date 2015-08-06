#ifndef UtilAlgos_PhysObjectMatcher_h
#define UtilAlgos_PhysObjectMatcher_h
/* \class PhysObjectMatcher.
 *
 * Extended version of reco::CandMatcher.
 * Tries to match elements from collection 1 to collection 2 with optional
 * resolution of ambiguities. Uses three helper classes for
 *  (1) the basic selection of the match (e.g. pdgId, charge, ..);
 *  (2) a distance measure (e.g. deltaR);
 *  (3) the ranking of several matches.
 *
 */
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/DeltaR.h"
#include "CommonTools/UtilAlgos/interface/MasterCollectionHelper.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Association.h"
#include "CommonTools/UtilAlgos/interface/MatchByDR.h"

// #include <iostream>

namespace reco {

  namespace helper {
    /// Default class for ranking matches: sorting by smaller distance.
    template<typename D, typename C1, typename C2> class LessByMatchDistance {
    public:
      LessByMatchDistance (const edm::ParameterSet& cfg,
			   const C1& c1, const C2& c2) :
	distance_(reco::modules::make<D>(cfg)), c1_(c1), c2_(c2) {}
      bool operator() (const std::pair<size_t,size_t>& p1,
		       const std::pair<size_t,size_t>& p2) const {
	return
	  distance_(c1_[p1.first],c2_[p1.second])<
	  distance_(c1_[p2.first],c2_[p2.second]);
      }
    private:
      D distance_;
      const C1& c1_;
      const C2& c2_;
    };
  }

  // Template arguments:
  // C1 .. candidate collection to be matched
  // C2 .. target of the match (typically MC)
  // S ... match (pre-)selector
  // D ... match (typically cut on some distance)
  //         default: deltaR cut
  // Q ... ranking of matches
  //         default: by smaller deltaR
  template<typename C1, typename C2, typename S,
	   typename D = reco::MatchByDR<typename C1::value_type,
					typename C2::value_type>,
	   typename Q =
	   helper::LessByMatchDistance< DeltaR<typename C1::value_type,
					       typename C2::value_type>,
					C1, C2 >
  >
  class PhysObjectMatcher : public edm::stream::EDProducer<> {
  public:
    PhysObjectMatcher(const edm::ParameterSet & cfg);
    ~PhysObjectMatcher();
  private:
    typedef typename C1::value_type T1;
    typedef typename C2::value_type T2;
    typedef edm::Association<C2> MatchMap;
    typedef std::pair<size_t, size_t> IndexPair;
    typedef std::vector<IndexPair> MatchContainer;
    void produce(edm::Event&, const edm::EventSetup&) override;
    edm::ParameterSet config_;
    edm::EDGetTokenT<C1> srcToken_;
    edm::EDGetTokenT<C2> matchedToken_;
    bool resolveAmbiguities_;            // resolve ambiguities after
                                         //   first pass?
    bool resolveByMatchQuality_;         // resolve by (global) quality
                                         //   of match (otherwise: by order
                                         //   of test candidates)
    bool select(const T1 & c1, const T2 & c2) const {
      return select_(c1, c2);
    }
    S select_;
    D distance_;
//     DeltaR<typename C1::value_type, typename C2::value_type> testDR_;
  };

  template<typename C1, typename C2, typename S, typename D, typename Q>
  PhysObjectMatcher<C1, C2, S, D, Q>::PhysObjectMatcher(const edm::ParameterSet & cfg) :
    config_(cfg),
    srcToken_(consumes<C1>(cfg.template getParameter<edm::InputTag>("src"))),
    matchedToken_(consumes<C2>(cfg.template getParameter<edm::InputTag>("matched"))),
    resolveAmbiguities_(cfg.template getParameter<bool>("resolveAmbiguities")),
    resolveByMatchQuality_(cfg.template getParameter<bool>("resolveByMatchQuality")),
    select_(reco::modules::make<S>(cfg)),
    distance_(reco::modules::make<D>(cfg)) {
    // definition of the product
    produces<MatchMap>();
    // set resolveByMatchQuality only if ambiguities are to be resolved
    resolveByMatchQuality_ = resolveByMatchQuality_ && resolveAmbiguities_;
  }

  template<typename C1, typename C2, typename S, typename D, typename Q>
  PhysObjectMatcher<C1, C2, S, D, Q>::~PhysObjectMatcher() { }

  template<typename C1, typename C2, typename S, typename D, typename Q>
  void PhysObjectMatcher<C1, C2, S, D, Q>::produce(edm::Event& evt, const edm::EventSetup&) {
    using namespace edm;
    using namespace std;
    typedef std::pair<size_t, size_t> IndexPair;
    typedef std::vector<IndexPair> MatchContainer;
    // get collections from event
    Handle<C2> matched;
    evt.getByToken(matchedToken_, matched);
    Handle<C1> cands;
    evt.getByToken(srcToken_, cands);
    // create product
    auto_ptr<MatchMap> matchMap(new MatchMap(matched));
    size_t size = cands->size();
    if( size != 0 ) {
      //
      // create helpers
      //
      Q comparator(config_,*cands,*matched);
      typename MatchMap::Filler filler(*matchMap);
      ::helper::MasterCollection<C1> master(cands);
      vector<int> indices(master.size(), -1);      // result: indices in target collection
      vector<bool> mLock(matched->size(),false);   // locks in target collection
      MatchContainer matchPairs;                   // container of matched pairs
      // loop over candidates
      for(size_t c = 0; c != size; ++ c) {
	const T1 & cand = (*cands)[c];
	// no global comparison of match quality -> reset the container for each candidate
	if ( !resolveByMatchQuality_ )  matchPairs.clear();
	// loop over target collection
	for(size_t m = 0; m != matched->size(); ++m) {
	  const T2 & match = (* matched)[m];
	  // check lock and preselection
	  if ( !mLock[m] && select(cand, match)) {
//  	    double dist = testDR_(cand,match);
//   	    cout << "dist between c = " << c << " and m = "
//   		 << m << " is " << dist << " at pts of "
//  		 << cand.pt() << " " << match.pt() << endl;
	    // matching requirement fulfilled -> store pair of indices
	    if ( distance_(cand,match) )  matchPairs.push_back(make_pair(c,m));
	  }
	}
	// if match(es) found and no global ambiguity resolution requested
	if ( matchPairs.size()>0 && !resolveByMatchQuality_ ) {
	  // look for and store best match
	  size_t idx = master.index(c);
	  assert(idx < indices.size());
	  size_t index = min_element(matchPairs.begin(), matchPairs.end(), comparator)->second;
	  indices[idx] = index;
	  // if ambiguity resolution by order of (reco) candidates:
	  //   lock element in target collection
	  if ( resolveAmbiguities_ )  mLock[index] = true;
// 	  {
// 	    MatchContainer::const_iterator i = min_element(matchPairs.begin(), matchPairs.end(), comparator);
//  	    cout << "smallest distance for c = " << c << " is "
//  		 << testDR_((*cands)[(*i).first],
// 			    (*matched)[(*i).second]) << endl;
// 	  }
	}
      }
      // ambiguity resolution by global match quality (if requested)
      if ( resolveByMatchQuality_ ) {
	// sort container of all matches by quality
	sort(matchPairs.begin(),matchPairs.end(),comparator);
	vector<bool> cLock(master.size(),false);
	// loop over sorted container
	for ( MatchContainer::const_iterator i=matchPairs.begin();
	      i!=matchPairs.end(); ++i ) {
	  size_t c = (*i).first;
	  size_t m = (*i).second;
// 	  cout << "rel dp = " << ((*cands)[c].pt()-(*matched)[m].pt())/(*matched)[m].pt() << endl;
	  // accept only pairs without any lock
	  if ( mLock[m] || cLock[c] )  continue;
	  // store index to target collection and lock the two items
	  size_t idx = master.index(c);
	  assert(idx < indices.size());
	  indices[idx] = m;
	  mLock[m] = true;
	  cLock[c] = true;
	}
      }
      filler.insert(master.get(), indices.begin(), indices.end());
      filler.fill();
    }
    evt.put(matchMap);
  }

}

#endif

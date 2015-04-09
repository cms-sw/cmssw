#ifndef UtilAlgos_NewMatcher_h
#define UtilAlgos_NewMatcher_h
/* \class Matcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/DeltaR.h"
#include "CommonTools/UtilAlgos/interface/MasterCollectionHelper.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/Common/interface/View.h"

namespace reco {
  namespace modulesNew {

    template<typename C1, typename C2,
             typename S, typename D = DeltaR<typename C1::value_type, typename C2::value_type> >
    class Matcher : public edm::EDProducer {
    public:
      Matcher(const edm::ParameterSet & cfg);
      ~Matcher();
    private:
      typedef typename C1::value_type T1;
      typedef typename C2::value_type T2;
      typedef edm::Association<C2> MatchMap;
      void produce(edm::Event&, const edm::EventSetup&) override;
      edm::EDGetTokenT<C1> srcToken_;
      edm::EDGetTokenT<C2> matchedToken_;
      double distMin_;
      double matchDistance(const T1 & c1, const T2 & c2) const {
	return distance_(c1, c2);
      }
      bool select(const T1 & c1, const T2 & c2) const {
	return select_(c1, c2);
      }
      S select_;
      D distance_;
    };

    namespace helper {
      typedef std::pair<size_t, double> MatchPair;

      struct SortBySecond {
	bool operator()(const MatchPair & p1, const MatchPair & p2) const {
	  return p1.second < p2.second;
	}
      };
    }

    template<typename C1, typename C2, typename S, typename D>
    Matcher<C1, C2, S, D>::Matcher(const edm::ParameterSet & cfg) :
      srcToken_(consumes<C1>(cfg.template getParameter<edm::InputTag>("src"))),
      matchedToken_(consumes<C2>(cfg.template getParameter<edm::InputTag>("matched"))),
      distMin_(cfg.template getParameter<double>("distMin")),
      select_(reco::modules::make<S>(cfg)),
      distance_(reco::modules::make<D>(cfg)) {
      produces<MatchMap>();
    }

    template<typename C1, typename C2, typename S, typename D>
    Matcher<C1, C2, S, D>::~Matcher() { }

    template<typename C1, typename C2, typename S, typename D>
    void Matcher<C1, C2, S, D>::produce(edm::Event& evt, const edm::EventSetup&) {
      using namespace edm;
      using namespace std;
      Handle<C2> matched;
      evt.getByToken(matchedToken_, matched);
      Handle<C1> cands;
      evt.getByToken(srcToken_, cands);
      auto_ptr<MatchMap> matchMap(new MatchMap(matched));
      size_t size = cands->size();
      if( size != 0 ) {
	typename MatchMap::Filler filler(*matchMap);
	::helper::MasterCollection<C1> master(cands, evt);
	std::vector<int> indices(master.size(), -1);
	for(size_t c = 0; c != size; ++ c) {
	  const T1 & cand = (*cands)[c];
	  vector<helper::MatchPair> v;
	  for(size_t m = 0; m != matched->size(); ++m) {
	    const T2 & match = (* matched)[m];
	    if (select(cand, match)) {
	      double dist = matchDistance(cand, match);
	      if (dist < distMin_) v.push_back(make_pair(m, dist));
	    }
	  }
	  if(v.size() > 0) {
	    size_t idx = master.index(c);
	    assert(idx < indices.size());
	    indices[idx] = min_element(v.begin(), v.end(), helper::SortBySecond())->first;
	  }
	}
	filler.insert(master.get(), indices.begin(), indices.end());
	filler.fill();
      }
      evt.put(matchMap);
    }

  }
}

#endif

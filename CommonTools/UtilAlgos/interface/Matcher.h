#ifndef UtilAlgos_Matcher_h
#define UtilAlgos_Matcher_h
/* \class Matcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/DeltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Common/interface/getRef.h"

namespace reco {
  namespace modules {
    template<typename C1, typename C2, typename M = edm::AssociationMap<edm::OneToOne<C1, C2> > >
    class MatcherBase : public edm::EDProducer {
    public:
      MatcherBase( const edm::ParameterSet & );
      ~MatcherBase();

    protected:
      typedef typename C1::value_type T1;
      typedef typename C2::value_type T2;
      typedef M MatchMap;

    private:
      void produce( edm::Event&, const edm::EventSetup&) override;
      edm::EDGetTokenT<C1> srcToken_;
      edm::EDGetTokenT<C2> matchedToken_;
      double distMin_;
      virtual double matchDistance( const T1 &, const T2 & ) const = 0;
      virtual bool select( const T1 &, const T2 & ) const = 0;
    };

    template<typename C1, typename C2,
             typename S, typename D = DeltaR<typename C1::value_type, typename C2::value_type>,
             typename M =  edm::AssociationMap<edm::OneToOne<C1, C2> > >
    class Matcher : public MatcherBase<C1, C2, M> {
    public:
      Matcher(  const edm::ParameterSet & cfg ) :
	MatcherBase<C1, C2, M>( cfg ),
        select_( reco::modules::make<S>( cfg ) ),
	distance_( reco::modules::make<D>( cfg ) ) { }
      ~Matcher() { }
    private:
      typedef typename MatcherBase<C1, C2, M>::T1 T1;
      typedef typename MatcherBase<C1, C2, M>::T2 T2;
      typedef typename MatcherBase<C1, C2, M>::MatchMap MatchMap;

      double matchDistance( const T1 & c1, const T2 & c2 ) const {
	return distance_( c1, c2 );
      }
      bool select( const T1 & c1, const T2 & c2 ) const {
	return select_( c1, c2 );
      }
      S select_;
      D distance_;
    };

    namespace helper {
      typedef std::pair<size_t, double> MatchPair;

      struct SortBySecond {
	bool operator()( const MatchPair & p1, const MatchPair & p2 ) const {
	  return p1.second < p2.second;
	}
      };
    }

    template<typename C1, typename C2, typename M>
    MatcherBase<C1, C2, M>::MatcherBase( const edm::ParameterSet & cfg ) :
      srcToken_( consumes<C1>( cfg.template getParameter<edm::InputTag>( "src" ) ) ),
      matchedToken_( consumes<C2>( cfg.template getParameter<edm::InputTag>( "matched" ) ) ),
      distMin_( cfg.template getParameter<double>( "distMin" ) ) {
      produces<MatchMap>();
    }

    template<typename C1, typename C2, typename M>
    MatcherBase<C1, C2, M>::~MatcherBase() { }

    template<typename C1, typename C2, typename M>
    void MatcherBase<C1, C2, M>::produce( edm::Event& evt, const edm::EventSetup& ) {
      using namespace edm;
      using namespace std;
      Handle<C2> matched;
      evt.getByToken( matchedToken_, matched );
      Handle<C1> cands;
      evt.getByToken( srcToken_, cands );
      typedef typename MatchMap::ref_type ref_type;
      typedef typename ref_type::key_type key_ref_type;
      typedef typename ref_type::value_type value_ref_type;
      auto_ptr<MatchMap> matchMap( new MatchMap( ref_type( key_ref_type( cands ),
							   value_ref_type( matched ) ) ) );
      for( size_t c = 0; c != cands->size(); ++ c ) {
	const T1 & cand = (*cands)[ c ];
	vector<helper::MatchPair> v;
	for( size_t m = 0; m != matched->size(); ++ m ) {
	  const T2 & match = ( * matched )[ m ];
	  if ( select( cand, match ) ) {
	    double dist = matchDistance( cand, match );
	    if ( dist < distMin_ ) v.push_back( make_pair( m, dist ) );
	  }
	}
	if ( v.size() > 0 ) {
	  size_t mMin = min_element( v.begin(), v.end(), helper::SortBySecond() )->first;
	  typedef typename MatchMap::key_type key_type;
	  typedef typename MatchMap::data_type data_type;
	  matchMap->insert( edm::getRef( cands, c ), edm::getRef( matched, mMin ) );
	}
      }
      evt.put( matchMap );
    }

  }
}

#endif

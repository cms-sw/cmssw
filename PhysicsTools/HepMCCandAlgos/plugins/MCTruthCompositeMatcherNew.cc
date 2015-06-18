/* \class MCTruthCompositeMatcher
 *
 * \author Luca Lista, INFN
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/CandUtils/interface/CandMatcherNew.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "DataFormats/Common/interface/Handle.h"

namespace reco {
  namespace modulesNew {

    class MCTruthCompositeMatcher : public edm::EDProducer {
    public:
      explicit MCTruthCompositeMatcher( const edm::ParameterSet & );
      ~MCTruthCompositeMatcher();
    private:
      edm::EDGetTokenT<CandidateView>  srcToken_;
      std::vector<edm::EDGetTokenT<reco::GenParticleMatch> > matchMapTokens_;
      std::vector<int> pdgId_;
      void produce( edm::Event & , const edm::EventSetup&) override;
    };

    MCTruthCompositeMatcher::MCTruthCompositeMatcher( const edm::ParameterSet & cfg ) :
      srcToken_(consumes<CandidateView>(cfg.getParameter<edm::InputTag>("src"))),
      matchMapTokens_( edm::vector_transform(cfg.template getParameter<std::vector<edm::InputTag> >( "matchMaps" ), [this](edm::InputTag const & tag){return consumes<reco::GenParticleMatch>(tag);} ) ),
      pdgId_(cfg.getParameter<std::vector<int> >("matchPDGId")) {
      produces<reco::GenParticleMatch>();
    }

    MCTruthCompositeMatcher::~MCTruthCompositeMatcher() {
    }

    void MCTruthCompositeMatcher::produce( edm::Event & evt , const edm::EventSetup & ) {
      using namespace edm;
      using namespace std;
      Handle<CandidateView> cands;
      evt.getByToken(srcToken_, cands);
      size_t nMaps = matchMapTokens_.size();
      std::vector<const GenParticleMatch *> maps;
      maps.reserve( nMaps );
      for( size_t i = 0; i != nMaps; ++ i ) {
	Handle<reco::GenParticleMatch> matchMap;
	evt.getByToken(matchMapTokens_[i], matchMap);
	maps.push_back(& * matchMap);
      }
      utilsNew::CandMatcher<GenParticleCollection> match(maps);
      auto_ptr<GenParticleMatch> matchMap(new GenParticleMatch(match.ref()));
      int size = cands->size();
      vector<int>::const_iterator begin = pdgId_.begin(), end = pdgId_.end();
      if(size != 0) {
	GenParticleMatch::Filler filler(*matchMap);
	vector<int> indices(size);
	for(int i = 0; i != size; ++ i) {
	  const Candidate & cand = (* cands)[i];
	  GenParticleRef mc = match[cand];
	  if(mc.isNull()) {
	    indices[i] = -1;
	  } else {
	    bool found = true;
	    if(begin!=end) found = find(begin, end, std::abs(mc->pdgId())) != end;
	    indices[i] = found ? int(mc.key()) : -1;
	  }
	}
	CandidateBaseRefProd ref(edm::makeRefToBaseProdFrom(cands->refAt(0), evt));
	filler.insert(ref, indices.begin(), indices.end());
	filler.fill();
      }
      evt.put(matchMap);
    }

  }
}

#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace reco {
  namespace modulesNew {

    typedef MCTruthCompositeMatcher MCTruthCompositeMatcherNew;

DEFINE_FWK_MODULE( MCTruthCompositeMatcherNew );

  }
}


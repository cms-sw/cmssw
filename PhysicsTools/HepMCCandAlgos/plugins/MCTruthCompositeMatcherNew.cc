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
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

namespace reco {
  namespace modulesNew {

    class MCTruthCompositeMatcher : public edm::EDProducer {
    public:
      explicit MCTruthCompositeMatcher( const edm::ParameterSet & );
      ~MCTruthCompositeMatcher();
    private:
      edm::InputTag src_;
      std::vector<edm::InputTag> matchMaps_;
      std::vector<int> pdgId_;
      void produce( edm::Event & , const edm::EventSetup&) override;
    };
    
    MCTruthCompositeMatcher::MCTruthCompositeMatcher( const edm::ParameterSet & cfg ) :
      src_(cfg.getParameter<edm::InputTag>("src")),
      matchMaps_(cfg.getParameter<std::vector<edm::InputTag> >("matchMaps")),
      pdgId_(cfg.getParameter<std::vector<int> >("matchPDGId")) {
      produces<reco::GenParticleMatch>();
    }
    
    MCTruthCompositeMatcher::~MCTruthCompositeMatcher() {
    }
    
    void MCTruthCompositeMatcher::produce( edm::Event & evt , const edm::EventSetup & ) {
      using namespace edm;
      using namespace std;
      Handle<CandidateView> cands;
      evt.getByLabel(src_, cands);
      size_t nMaps = matchMaps_.size();
      std::vector<const GenParticleMatch *> maps;
      maps.reserve( nMaps );
      for( size_t i = 0; i != nMaps; ++ i ) {
	Handle<reco::GenParticleMatch> matchMap;
	evt.getByLabel(matchMaps_[i], matchMap);
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
	CandidateBaseRefProd ref(cands->refAt(0));
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


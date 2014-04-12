#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/CaloRecHitCandidate.h"

namespace reco {
  namespace modules {

    template<typename HitCollection>
    class CaloRecHitCandidateProducer : public edm::EDProducer {
    public:
      /// constructor
      CaloRecHitCandidateProducer( const edm::ParameterSet & cfg ) :
	srcToken_( consumes<HitCollection>( cfg.template getParameter<edm::InputTag>( "src" ) ) ) { }
      /// destructor
      ~CaloRecHitCandidateProducer() { }

    private:
      /// process one event
      void produce( edm::Event &, const edm::EventSetup&) override;
      /// source collection tag
      edm::EDGetTokenT<HitCollection> srcToken_;
    };
  }
}

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Math/interface/Vector3D.h"

namespace reco {
  namespace modules {

    template<typename HitCollection>
    void CaloRecHitCandidateProducer<HitCollection>::produce( edm::Event & evt, const edm::EventSetup & ) {
      using namespace edm;
      using namespace reco;
      using namespace std;
      Handle<HitCollection> hits;
      evt.getByToken( srcToken_, hits );
      auto_ptr<CandidateCollection> cands( new CandidateCollection );
      size_t size = hits->size();
      cands->reserve( size );
      for( size_t idx = 0; idx != size; ++ idx ) {
	const CaloRecHit & hit = (*hits)[ idx ];
	/// don't know how to set eta and phi
	double eta = 0, phi = 0, energy = hit.energy();
	math::RhoEtaPhiVector p( 1, eta, phi );
	p *= ( energy / p.r() );
	CaloRecHitCandidate * c = new CaloRecHitCandidate( Candidate::LorentzVector( p.x(), p.y(), p.z(), energy ) );
	c->setCaloRecHit( RefToBase<CaloRecHit>( Ref<HitCollection>( hits, idx ) ) );
	cands->push_back( c );
      }
      evt.put( cands );
    }

  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

typedef reco::modules::CaloRecHitCandidateProducer<HFRecHitCollection> HFRecHitCandidateProducer;
typedef reco::modules::CaloRecHitCandidateProducer<ZDCRecHitCollection> ZDCRecHitCandidateProducer;
typedef reco::modules::CaloRecHitCandidateProducer<HBHERecHitCollection> HBHERecHitCandidateProducer;
typedef reco::modules::CaloRecHitCandidateProducer<HORecHitCollection> HORecHitCandidateProducer;

DEFINE_FWK_MODULE( HFRecHitCandidateProducer );
DEFINE_FWK_MODULE( ZDCRecHitCandidateProducer );
DEFINE_FWK_MODULE( HBHERecHitCandidateProducer );
DEFINE_FWK_MODULE( HORecHitCandidateProducer );

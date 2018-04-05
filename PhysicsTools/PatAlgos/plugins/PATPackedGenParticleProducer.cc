#include <string>


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "TrackingTools/IPTools/interface/IPTools.h" 
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Matrix/Vector.h"
#include <string>


namespace pat {
  class PATPackedGenParticleProducer : public edm::global::EDProducer<> {
  public:
    explicit PATPackedGenParticleProducer(const edm::ParameterSet&);
    ~PATPackedGenParticleProducer() override;
    
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
    
  private:
    const edm::EDGetTokenT<reco::GenParticleCollection>    Cands_;
    const edm::EDGetTokenT<reco::GenParticleCollection>    GenOrigs_;
    const edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> >    Asso_;
    const edm::EDGetTokenT<edm::Association<reco::GenParticleCollection> >    AssoOriginal_;
    const double maxRapidity_;
  };
}

pat::PATPackedGenParticleProducer::PATPackedGenParticleProducer(const edm::ParameterSet& iConfig) :
  Cands_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("inputCollection"))),
  GenOrigs_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("inputOriginal"))),
  Asso_(consumes<edm::Association<reco::GenParticleCollection> >(iConfig.getParameter<edm::InputTag>("map"))),
  AssoOriginal_(consumes<edm::Association<reco::GenParticleCollection> >(iConfig.getParameter<edm::InputTag>("inputCollection"))),
  maxRapidity_(iConfig.getParameter<double>("maxRapidity"))
{
  produces< std::vector<pat::PackedGenParticle> > ();
  produces< edm::Association< std::vector<pat::PackedGenParticle> > >();
}

pat::PATPackedGenParticleProducer::~PATPackedGenParticleProducer() {}

void pat::PATPackedGenParticleProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {


    edm::Handle<reco::GenParticleCollection> cands;
    iEvent.getByToken( Cands_, cands );
    std::vector<reco::Candidate>::const_iterator cand;

    //from prunedGenParticlesWithStatusOne to prunedGenParticles
    edm::Handle<edm::Association<reco::GenParticleCollection> > asso;
    iEvent.getByToken( Asso_, asso );

    edm::Handle<edm::Association<reco::GenParticleCollection> > assoOriginal;
    iEvent.getByToken( AssoOriginal_, assoOriginal);

    edm::Handle<reco::GenParticleCollection> genOrigs;
    iEvent.getByToken( GenOrigs_, genOrigs);
    std::vector<int> mapping(genOrigs->size(), -1);

    //invert the value map from Orig2New to New2Orig
    std::map< edm::Ref<reco::GenParticleCollection> ,  edm::Ref<reco::GenParticleCollection> > reverseMap;
    for(unsigned int ic=0, nc = genOrigs->size(); ic < nc; ++ic)
    {	
	edm::Ref<reco::GenParticleCollection> originalRef = edm::Ref<reco::GenParticleCollection>(genOrigs,ic);
	edm::Ref<reco::GenParticleCollection> newRef = (*assoOriginal)[originalRef];
	reverseMap.insert(std::pair<edm::Ref<reco::GenParticleCollection>,edm::Ref<reco::GenParticleCollection>>(newRef,originalRef));
    }

    auto outPtrP = std::make_unique<std::vector<pat::PackedGenParticle>>();

    unsigned int packed=0;
    for(unsigned int ic=0, nc = cands->size(); ic < nc; ++ic) {
        const reco::GenParticle &cand=(*cands)[ic];
	if(cand.status() ==1 && std::abs(cand.y()) < maxRapidity_)
        {
		// Obtain original gen particle collection reference from input reference and map
		edm::Ref<reco::GenParticleCollection> inputRef = edm::Ref<reco::GenParticleCollection>(cands,ic);
		edm::Ref<reco::GenParticleCollection> originalRef=reverseMap[inputRef];
		edm::Ref<reco::GenParticleCollection> finalPrunedRef=(*asso)[inputRef];
		mapping[originalRef.key()]=packed;
		packed++;
		if(finalPrunedRef.isNonnull()){ //this particle exists also in the final pruned
                        outPtrP->push_back( pat::PackedGenParticle(cand,finalPrunedRef));
		}else{
		  if(cand.numberOfMothers() > 0) {
			edm::Ref<reco::GenParticleCollection> newRef=(*asso)[cand.motherRef(0)];
	        	outPtrP->push_back( pat::PackedGenParticle(cand,newRef));
	  	  } else {
	        	outPtrP->push_back( pat::PackedGenParticle(cand,edm::Ref<reco::GenParticleCollection>()));
		  }
		}

	}	
    }


    edm::OrphanHandle<std::vector<pat::PackedGenParticle> > oh= iEvent.put(std::move(outPtrP));

    auto gp2pgp = std::make_unique<edm::Association<std::vector<pat::PackedGenParticle>>>(oh);
    edm::Association< std::vector<pat::PackedGenParticle> >::Filler gp2pgpFiller(*gp2pgp);
    gp2pgpFiller.insert(genOrigs, mapping.begin(), mapping.end());
    gp2pgpFiller.fill();
    iEvent.put(std::move(gp2pgp));
 

}


using pat::PATPackedGenParticleProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPackedGenParticleProducer);

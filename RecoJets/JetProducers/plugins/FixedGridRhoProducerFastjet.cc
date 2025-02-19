#include "RecoJets/JetProducers/plugins/FixedGridRhoProducerFastjet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

using namespace std;

FixedGridRhoProducerFastjet::FixedGridRhoProducerFastjet(const edm::ParameterSet& iConfig) :
  bge_( iConfig.getParameter<double>("maxRapidity"),
	iConfig.getParameter<double>("gridSpacing") )
{
  pfCandidatesTag_ = iConfig.getParameter<edm::InputTag>("pfCandidatesTag");
  produces<double>();
}

FixedGridRhoProducerFastjet::~FixedGridRhoProducerFastjet(){} 

void FixedGridRhoProducerFastjet::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

   edm::Handle<reco::PFCandidateCollection> pfColl;
   iEvent.getByLabel(pfCandidatesTag_,pfColl);
   std::vector<fastjet::PseudoJet> inputs;
   for ( reco::PFCandidateCollection::const_iterator ibegin = pfColl->begin(),
	   iend = pfColl->end(), i = ibegin; i != iend; ++i ){
     inputs.push_back( fastjet::PseudoJet(i->px(), i->py(), i->pz(), i->energy()) );
   }
   bge_.set_particles(inputs);
   std::auto_ptr<double> outputRho(new double(bge_.rho()));
   iEvent.put(outputRho);
}

DEFINE_FWK_MODULE(FixedGridRhoProducerFastjet);

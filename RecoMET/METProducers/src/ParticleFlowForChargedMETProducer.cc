#include "RecoMET/METProducers/interface/ParticleFlowForChargedMETProducer.h"

#include <DataFormats/VertexReco/interface/Vertex.h>
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>

using namespace edm;
using namespace std;
using namespace reco;

ParticleFlowForChargedMETProducer::ParticleFlowForChargedMETProducer(const edm::ParameterSet& iConfig)
{
  pfCollectionLabel    = iConfig.getParameter<edm::InputTag>("PFCollectionLabel");
  pvCollectionLabel    = iConfig.getParameter<edm::InputTag>("PVCollectionLabel");

  pfCandidatesToken = consumes<PFCandidateCollection>(pfCollectionLabel);
  pvCollectionToken = consumes<VertexCollection>(pvCollectionLabel);

  dzCut    = iConfig.getParameter<double>("dzCut");
  neutralEtThreshold    = iConfig.getParameter<double>("neutralEtThreshold");

  produces<PFCandidateCollection>();
}

void ParticleFlowForChargedMETProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  //Get the PV collection
  Handle<VertexCollection> pvCollection;
  iEvent.getByToken(pvCollectionToken, pvCollection);
  VertexCollection::const_iterator vertex = pvCollection->begin();

  //Get pfCandidates
  Handle<PFCandidateCollection> pfCandidates;
  iEvent.getByToken(pfCandidatesToken, pfCandidates);

  // the output collection
  auto chargedPFCandidates = std::make_unique<PFCandidateCollection>();
  if (!pvCollection->empty()) {
    for( unsigned i=0; i<pfCandidates->size(); i++ ) {
      const PFCandidate& pfCand = (*pfCandidates)[i];
      PFCandidatePtr pfCandPtr(pfCandidates, i);
      
      if (pfCandPtr->trackRef().isNonnull()) { 
	if (pfCandPtr->trackRef()->dz((*vertex).position()) < dzCut) {
	  chargedPFCandidates->push_back( pfCand );
	  chargedPFCandidates->back().setSourceCandidatePtr( pfCandPtr );
	}
	
      }
      else if (neutralEtThreshold>0 and 
	       pfCandPtr->pt()>neutralEtThreshold) {
	chargedPFCandidates->push_back( pfCand );
	chargedPFCandidates->back().setSourceCandidatePtr( pfCandPtr );
      }  
	

    }
  }


  iEvent.put(std::move(chargedPFCandidates));

  return;
}

ParticleFlowForChargedMETProducer::~ParticleFlowForChargedMETProducer(){}

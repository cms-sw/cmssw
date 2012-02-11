#include "RecoMET/METProducers/interface/PFCandidatesForTrackMETProducer.h"

#include <DataFormats/VertexReco/interface/Vertex.h>
#include <DataFormats/VertexReco/interface/VertexFwd.h>
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h>

using namespace edm;
using namespace std;
using namespace reco;

PFCandidatesForTrackMETProducer::PFCandidatesForTrackMETProducer(const edm::ParameterSet& iConfig)
{
  pfCollectionLabel    = iConfig.getParameter<edm::InputTag>("PFCollectionLabel");
  pvCollectionLabel    = iConfig.getParameter<edm::InputTag>("PVCollectionLabel");
  dzCut    = iConfig.getParameter<double>("dzCut");
  neutralEtThreshold    = iConfig.getParameter<double>("neutralEtThreshold");

  produces<PFCandidateCollection>();
}

void PFCandidatesForTrackMETProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  //Get the PV collection
  Handle<VertexCollection> pvCollection;
  iEvent.getByLabel(pvCollectionLabel, pvCollection);
  VertexCollection::const_iterator vertex = pvCollection->begin();

  //Get pfCandidates
  Handle<PFCandidateCollection> pfCandidates;
  iEvent.getByLabel(pfCollectionLabel, pfCandidates);

  // the output collection
  auto_ptr<PFCandidateCollection> chargedPFCandidates( new PFCandidateCollection ) ;
  if (pvCollection->size()>0) {
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


  iEvent.put(chargedPFCandidates); 

  return;
}

void PFCandidatesForTrackMETProducer::beginJob(){return;}
void PFCandidatesForTrackMETProducer::endJob(){return;}
void PFCandidatesForTrackMETProducer::beginRun(Run&, const EventSetup&){return;}
void PFCandidatesForTrackMETProducer::endRun(Run&, const EventSetup&){return;}
PFCandidatesForTrackMETProducer::~PFCandidatesForTrackMETProducer(){}

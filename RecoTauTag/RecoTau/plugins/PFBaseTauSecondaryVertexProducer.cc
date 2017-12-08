/* class PFBaseTauSecondaryVertexProducer
 * EDProducer of the 
 * authors: Ian M. Nugent
 * This work is based on the impact parameter work by Rosamaria Venditti and reconstructing the 3 prong taus.
 * The idea of the fully reconstructing the tau using a kinematic fit comes from
 * Lars Perchalla and Philip Sauerland Theses under Achim Stahl supervision. This
 * work was continued by Ian M. Nugent and Vladimir Cherepanov.
 * Thanks goes to Christian Veelken and Evan Klose Friis for their help and suggestions.
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

#include "DataFormats/TauReco/interface/PFBaseTau.h"
#include "DataFormats/TauReco/interface/PFBaseTauFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFBaseTauSecondaryVertexProducer : public edm::global::EDProducer<> {
 public:

  explicit PFBaseTauSecondaryVertexProducer(const edm::ParameterSet& iConfig);
  ~PFBaseTauSecondaryVertexProducer() override;
  void produce(edm::StreamID, edm::Event&,const edm::EventSetup&) const override;
 private:
  const edm::EDGetTokenT<std::vector<reco::PFBaseTau> > PFTauToken_;
};

PFBaseTauSecondaryVertexProducer::PFBaseTauSecondaryVertexProducer(const edm::ParameterSet& iConfig):
  PFTauToken_(consumes<std::vector<reco::PFBaseTau> >(iConfig.getParameter<edm::InputTag>("PFTauTag")))
{
  produces<edm::AssociationVector<PFBaseTauRefProd, std::vector<std::vector<reco::VertexRef> > > >();
  produces<VertexCollection>("PFTauSecondaryVertices");
}

PFBaseTauSecondaryVertexProducer::~PFBaseTauSecondaryVertexProducer(){

}

namespace {
  const reco::Track* getTrack(const reco::Candidate& cand) {
    const pat::PackedCandidate* pCand = dynamic_cast<const pat::PackedCandidate*>(&cand);
    if (pCand && pCand->hasTrackDetails())
    	return &pCand->pseudoTrack();
    return nullptr;
  }
}

void PFBaseTauSecondaryVertexProducer::produce(edm::StreamID, edm::Event& iEvent,const edm::EventSetup& iSetup) const {
  // Obtain 
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);

  edm::Handle<std::vector<reco::PFBaseTau> > Tau;
  iEvent.getByToken(PFTauToken_,Tau);

  // Set Association Map
  auto AVPFTauSV = std::make_unique<edm::AssociationVector<PFBaseTauRefProd, std::vector<std::vector<reco::VertexRef>>>>(PFBaseTauRefProd(Tau));
  auto VertexCollection_out = std::make_unique<VertexCollection>();
  reco::VertexRefProd VertexRefProd_out = iEvent.getRefBeforePut<reco::VertexCollection>("PFTauSecondaryVertices");

  // For each Tau Run Algorithim
  if(Tau.isValid()) {
    for(reco::PFBaseTauCollection::size_type iPFTau = 0; iPFTau < Tau->size(); iPFTau++) {
      reco::PFBaseTauRef RefPFTau(Tau, iPFTau);
      std::vector<reco::VertexRef> SV;
      if(RefPFTau->decayMode()>=5){
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Get tracks form PFTau daugthers
	std::vector<reco::TransientTrack> transTrk;
	const std::vector<reco::CandidatePtr> cands = RefPFTau->signalPFChargedHadrCands(); 
	for (std::vector<reco::CandidatePtr>::const_iterator iter = cands.begin(); iter!=cands.end(); ++iter) {
	  if((*iter).isNull()) continue;
	  const reco::Track* track = getTrack(**iter);
	  if(track == nullptr) continue;
	  transTrk.push_back(transTrackBuilder->build(*track));
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Fit the secondary vertex
	if(transTrk.size()>=2){
	  TransientVertex transVtx;
	  bool fitOk = true;
	  KalmanVertexFitter kvf(true);
	  try{
	    transVtx = kvf.vertex(transTrk); //KalmanVertexFitter  
	  }catch(...){
	    fitOk = false;
	  }
	  if(!transVtx.hasRefittedTracks()) fitOk = false;
	  if(transVtx.refittedTracks().size()!=transTrk.size()) fitOk=false;
	  if(fitOk){
	    SV.push_back(reco::VertexRef(VertexRefProd_out, VertexCollection_out->size()));
	    VertexCollection_out->push_back(transVtx);
	  }
	}
      }
      AVPFTauSV->setValue(iPFTau, SV);
    }
  }
  iEvent.put(std::move(VertexCollection_out),"PFTauSecondaryVertices");
  iEvent.put(std::move(AVPFTauSV));
}

DEFINE_FWK_MODULE(PFBaseTauSecondaryVertexProducer);

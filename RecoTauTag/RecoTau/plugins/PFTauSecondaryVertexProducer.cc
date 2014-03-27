/* class PFTauSecondaryVertexProducer
 * EDProducer of the 
 * authors: Ian M. Nugent
 * This work is based on the impact parameter work by Rosamaria Venditti and reconstructing the 3 prong taus.
 * The idea of the fully reconstructing the tau using a kinematic fit comes from
 * Lars Perchalla and Philip Sauerland Theses under Achim Stahl supervision. This
 * work was continued by Ian M. Nugent and Vladimir Cherepanov.
 * Thanks goes to Christian Veelken and Evan Klose Friis for their help and suggestions.
 */

#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"

#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

class PFTauSecondaryVertexProducer : public EDProducer {
 public:
  enum Alg{useInputPV=0, usePVwithMaxSumPt, useTauPV};

  explicit PFTauSecondaryVertexProducer(const edm::ParameterSet& iConfig);
  ~PFTauSecondaryVertexProducer();
  virtual void produce(edm::Event&,const edm::EventSetup&);
 private:
  edm::InputTag PFTauTag_;
  edm::EDGetTokenT<std::vector<reco::PFTau> > PFTauToken_;
};

PFTauSecondaryVertexProducer::PFTauSecondaryVertexProducer(const edm::ParameterSet& iConfig):
  PFTauToken_(consumes<std::vector<reco::PFTau> >(iConfig.getParameter<edm::InputTag>("PFTauTag")))
{
  produces<edm::AssociationVector<PFTauRefProd, std::vector<std::vector<reco::VertexRef> > > >();
  produces<VertexCollection>("PFTauSecondaryVertices");
}

PFTauSecondaryVertexProducer::~PFTauSecondaryVertexProducer(){

}

void PFTauSecondaryVertexProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  // Obtain 
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);

  edm::Handle<std::vector<reco::PFTau> > Tau;
  iEvent.getByToken(PFTauToken_,Tau);

  // Set Association Map
  auto_ptr<edm::AssociationVector<PFTauRefProd, std::vector<std::vector<reco::VertexRef> > > > AVPFTauSV(new edm::AssociationVector<PFTauRefProd, std::vector<std::vector<reco::VertexRef> > >(PFTauRefProd(Tau)));
  std::auto_ptr<VertexCollection>  VertexCollection_out= std::auto_ptr<VertexCollection>(new VertexCollection);
  reco::VertexRefProd VertexRefProd_out = iEvent.getRefBeforePut<reco::VertexCollection>("PFTauSecondaryVertices");

  // For each Tau Run Algorithim
  if(Tau.isValid()) {
    for(reco::PFTauCollection::size_type iPFTau = 0; iPFTau < Tau->size(); iPFTau++) {
      reco::PFTauRef RefPFTau(Tau, iPFTau);
      std::vector<reco::VertexRef> SV;
      if(RefPFTau->decayMode()==10){
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Get tracks form PFTau daugthers
	std::vector<reco::TransientTrack> transTrk;
	TransientVertex transVtx;
	const std::vector<edm::Ptr<reco::PFCandidate> > cands = RefPFTau->signalPFChargedHadrCands(); 
	for (std::vector<edm::Ptr<reco::PFCandidate> >::const_iterator iter = cands.begin(); iter!=cands.end(); ++iter) {
	  if(iter->get()->trackRef().isNonnull())transTrk.push_back(transTrackBuilder->build(iter->get()->trackRef()));
	  else if(iter->get()->gsfTrackRef().isNonnull())transTrk.push_back(transTrackBuilder->build(iter->get()->gsfTrackRef()));
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Fit the secondary vertex
	bool FitOk(true);
	KalmanVertexFitter kvf(true);
	try{
	  transVtx = kvf.vertex(transTrk); //KalmanVertexFitter  
	}catch(...){
	  FitOk=false;
	}
	if(!transVtx.hasRefittedTracks()) FitOk=false;
	if(transVtx.refittedTracks().size()!=transTrk.size()) FitOk=false;
	if(FitOk){
	  SV.push_back(reco::VertexRef(VertexRefProd_out, VertexCollection_out->size()));
	  VertexCollection_out->push_back(transVtx);
	}
      }
      AVPFTauSV->setValue(iPFTau, SV);
    }
  }
  iEvent.put(VertexCollection_out,"PFTauSecondaryVertices");
  iEvent.put(AVPFTauSV);
}

DEFINE_FWK_MODULE(PFTauSecondaryVertexProducer);

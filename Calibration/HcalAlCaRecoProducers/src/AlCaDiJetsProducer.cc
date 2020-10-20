// system include files
#include <memory>
#include <string>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include <iostream>

//
// class declaration
//

class AlCaDiJetsProducer : public edm::EDProducer {
public:
  explicit AlCaDiJetsProducer(const edm::ParameterSet&);
  ~AlCaDiJetsProducer() override;
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  bool select(reco::PFJetCollection);

  // ----------member data ---------------------------

  edm::InputTag labelPFJet_, labelHBHE_, labelHF_, labelHO_, labelPFCandidate_, labelVertex_;  //labelTrigger_,
  double minPtJet_;
  int nAll_, nSelect_;

  edm::EDGetTokenT<reco::PFJetCollection> tok_PFJet_;
  edm::EDGetTokenT<edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>> tok_HBHE_;
  edm::EDGetTokenT<edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>> tok_HF_;
  edm::EDGetTokenT<edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>> tok_HO_;
  //edm::EDGetTokenT<edm::TriggerResults>                                                   tok_TrigRes_;
  edm::EDGetTokenT<reco::PFCandidateCollection> tok_PFCand_;
  edm::EDGetTokenT<reco::VertexCollection> tok_Vertex_;
};

AlCaDiJetsProducer::AlCaDiJetsProducer(const edm::ParameterSet& iConfig) : nAll_(0), nSelect_(0) {
  // Take input
  labelPFJet_ = iConfig.getParameter<edm::InputTag>("PFjetInput");
  labelHBHE_ = iConfig.getParameter<edm::InputTag>("HBHEInput");
  labelHF_ = iConfig.getParameter<edm::InputTag>("HFInput");
  labelHO_ = iConfig.getParameter<edm::InputTag>("HOInput");
  //labelTrigger_    = iConfig.getParameter<edm::InputTag>("TriggerResults");
  labelPFCandidate_ = iConfig.getParameter<edm::InputTag>("particleFlowInput");
  labelVertex_ = iConfig.getParameter<edm::InputTag>("VertexInput");
  minPtJet_ = iConfig.getParameter<double>("MinPtJet");

  tok_PFJet_ = consumes<reco::PFJetCollection>(labelPFJet_);
  tok_HBHE_ = consumes<edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>>(labelHBHE_);
  tok_HF_ = consumes<edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>>(labelHF_);
  tok_HO_ = consumes<edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>>(labelHO_);
  //tok_TrigRes_= consumes<edm::TriggerResults>(labelTrigger_);
  tok_PFCand_ = consumes<reco::PFCandidateCollection>(labelPFCandidate_);
  tok_Vertex_ = consumes<reco::VertexCollection>(labelVertex_);

  // register your products
  produces<reco::PFJetCollection>(labelPFJet_.encode());
  produces<edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>>(labelHBHE_.encode());
  produces<edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>>(labelHF_.encode());
  produces<edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>>(labelHO_.encode());
  //produces<edm::TriggerResults>(labelTrigger_.encode());
  produces<reco::PFCandidateCollection>(labelPFCandidate_.encode());
  produces<reco::VertexCollection>(labelVertex_.encode());
}

AlCaDiJetsProducer::~AlCaDiJetsProducer() {}

void AlCaDiJetsProducer::beginJob() {}

void AlCaDiJetsProducer::endJob() {
  edm::LogVerbatim("AlcaDiJets") << "Accepts " << nSelect_ << " events from a total of " << nAll_ << " events";
}

bool AlCaDiJetsProducer::select(reco::PFJetCollection jt) {
  if (jt.size() < 2)
    return false;
  if (((jt.at(0)).pt()) < minPtJet_)
    return false;
  return true;
}
// ------------ method called to produce the data  ------------
void AlCaDiJetsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  nAll_++;

  // Access the collections from iEvent
  edm::Handle<reco::PFJetCollection> pfjet;
  iEvent.getByToken(tok_PFJet_, pfjet);
  if (!pfjet.isValid()) {
    edm::LogWarning("AlCaDiJets") << "AlCaDiJetsProducer: Error! can't get product " << labelPFJet_;
    return;
  }
  const reco::PFJetCollection pfjets = *(pfjet.product());

  edm::Handle<reco::PFCandidateCollection> pfc;
  iEvent.getByToken(tok_PFCand_, pfc);
  if (!pfc.isValid()) {
    edm::LogWarning("AlCaDiJets") << "AlCaDiJetsProducer: Error! can't get product " << labelPFCandidate_;
    return;
  }
  const reco::PFCandidateCollection pfcand = *(pfc.product());

  edm::Handle<reco::VertexCollection> vt;
  iEvent.getByToken(tok_Vertex_, vt);
  if (!vt.isValid()) {
    edm::LogWarning("AlCaDiJets") << "AlCaDiJetsProducer: Error! can't get product " << labelVertex_;
    return;
  }
  const reco::VertexCollection vtx = *(vt.product());

  edm::Handle<edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>> hbhe;
  iEvent.getByToken(tok_HBHE_, hbhe);
  if (!hbhe.isValid()) {
    edm::LogWarning("AlCaDiJets") << "AlCaDiJetsProducer: Error! can't get product " << labelHBHE_;
    return;
  }
  const edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>> Hithbhe = *(hbhe.product());

  edm::Handle<edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>> ho;
  iEvent.getByToken(tok_HO_, ho);
  if (!ho.isValid()) {
    edm::LogWarning("AlCaDiJets") << "AlCaDiJetsProducer: Error! can't get product " << labelHO_;
    return;
  }
  const edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>> Hitho = *(ho.product());

  edm::Handle<edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>> hf;
  iEvent.getByToken(tok_HF_, hf);
  if (!hf.isValid()) {
    edm::LogWarning("AlCaDiJets") << "AlCaDiJetsProducer: Error! can't get product " << labelHF_;
    return;
  }
  const edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>> Hithf = *(hf.product());

  // See if this event is useful
  bool accept = select(pfjets);
  if (accept) {
    nSelect_++;

    //Copy from standard place
    auto miniPFjetCollection = std::make_unique<reco::PFJetCollection>();
    for (reco::PFJetCollection::const_iterator pfjetItr = pfjets.begin(); pfjetItr != pfjets.end(); pfjetItr++) {
      miniPFjetCollection->push_back(*pfjetItr);
    }

    auto miniPFCandCollection = std::make_unique<reco::PFCandidateCollection>();
    for (reco::PFCandidateCollection::const_iterator pfcItr = pfcand.begin(); pfcItr != pfcand.end(); pfcItr++) {
      miniPFCandCollection->push_back(*pfcItr);
    }

    auto miniVtxCollection = std::make_unique<reco::VertexCollection>();
    for (reco::VertexCollection::const_iterator vtxItr = vtx.begin(); vtxItr != vtx.end(); vtxItr++) {
      miniVtxCollection->push_back(*vtxItr);
    }

    auto miniHBHECollection =
        std::make_unique<edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>>();
    for (edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>::const_iterator hbheItr =
             Hithbhe.begin();
         hbheItr != Hithbhe.end();
         hbheItr++) {
      miniHBHECollection->push_back(*hbheItr);
    }

    auto miniHOCollection = std::make_unique<edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>>();
    for (edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>::const_iterator hoItr = Hitho.begin();
         hoItr != Hitho.end();
         hoItr++) {
      miniHOCollection->push_back(*hoItr);
    }

    auto miniHFCollection = std::make_unique<edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>>();
    for (edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>::const_iterator hfItr = Hithf.begin();
         hfItr != Hithf.end();
         hfItr++) {
      miniHFCollection->push_back(*hfItr);
    }

    //Put them in the event
    iEvent.put(std::move(miniPFjetCollection), labelPFJet_.encode());
    iEvent.put(std::move(miniHBHECollection), labelHBHE_.encode());
    iEvent.put(std::move(miniHFCollection), labelHF_.encode());
    iEvent.put(std::move(miniHOCollection), labelHO_.encode());
    //iEvent.put(std::move(miniTriggerCollection),     labelTrigger_.encode());
    iEvent.put(std::move(miniPFCandCollection), labelPFCandidate_.encode());
    iEvent.put(std::move(miniVtxCollection), labelVertex_.encode());
  }
  return;
}

DEFINE_FWK_MODULE(AlCaDiJetsProducer);

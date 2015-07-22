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
#include<iostream>

//
// class declaration
//

class AlCaGammaJetProducer : public edm::EDProducer {

public:
  explicit AlCaGammaJetProducer(const edm::ParameterSet&);
  ~AlCaGammaJetProducer();
  virtual void beginJob() ;
  virtual void produce(edm::Event &, const edm::EventSetup&);
  virtual void endJob();

private:
  bool select(const reco::PhotonCollection&, const reco::PFJetCollection&);

  // ----------member data ---------------------------
  
  edm::InputTag   labelPhoton_, labelPFJet_, labelHBHE_, labelHF_, labelHO_, labelTrigger_, labelPFCandidate_, labelVertex_, labelPFMET_, labelGsfEle_, labelRho_, labelConv_, labelBeamSpot_, labelLoosePhot_, labelTightPhot_;
  double          minPtJet_, minPtPhoton_;
  int             nAll_, nSelect_;
  
  edm::EDGetTokenT<reco::PhotonCollection>                                                tok_Photon_; 
  edm::EDGetTokenT<reco::PFJetCollection>                                                 tok_PFJet_;
  edm::EDGetTokenT<edm::SortedCollection<HBHERecHit,edm::StrictWeakOrdering<HBHERecHit>>> tok_HBHE_;
  edm::EDGetTokenT<edm::SortedCollection<HFRecHit,edm::StrictWeakOrdering<HFRecHit>>>     tok_HF_;
  edm::EDGetTokenT<edm::SortedCollection<HORecHit,edm::StrictWeakOrdering<HORecHit>>>     tok_HO_;
  edm::EDGetTokenT<edm::TriggerResults>                                                   tok_TrigRes_;
  edm::EDGetTokenT<reco::PFCandidateCollection>                                           tok_PFCand_;
  edm::EDGetTokenT<reco::VertexCollection>                                                tok_Vertex_;
  edm::EDGetTokenT<reco::PFMETCollection>                                                 tok_PFMET_;
  edm::EDGetTokenT<reco::GsfElectronCollection>                                           tok_GsfElec_;
  edm::EDGetTokenT<double>                                                                tok_Rho_;
  edm::EDGetTokenT<reco::ConversionCollection>                                            tok_Conv_;
  edm::EDGetTokenT<reco::BeamSpot>                                                        tok_BS_;
  edm::EDGetTokenT<edm::ValueMap<Bool_t> >                                                tok_loosePhoton_;
  edm::EDGetTokenT<edm::ValueMap<Bool_t> >                                                tok_tightPhoton_;
};

AlCaGammaJetProducer::AlCaGammaJetProducer(const edm::ParameterSet& iConfig) : nAll_(0), nSelect_(0) {
   // Take input 
  labelPhoton_     = iConfig.getParameter<edm::InputTag>("PhoInput");
  labelPFJet_      = iConfig.getParameter<edm::InputTag>("PFjetInput");
  labelHBHE_       = iConfig.getParameter<edm::InputTag>("HBHEInput");
  labelHF_         = iConfig.getParameter<edm::InputTag>("HFInput");
  labelHO_         = iConfig.getParameter<edm::InputTag>("HOInput");
  labelTrigger_    = iConfig.getParameter<edm::InputTag>("TriggerResults");
  labelPFCandidate_= iConfig.getParameter<edm::InputTag>("particleFlowInput");
  labelVertex_     = iConfig.getParameter<edm::InputTag>("VertexInput");
  labelPFMET_      = iConfig.getParameter<edm::InputTag>("METInput");
  labelGsfEle_     = iConfig.getParameter<edm::InputTag>("gsfeleInput");
  labelRho_        = iConfig.getParameter<edm::InputTag>("rhoInput");
  labelConv_       = iConfig.getParameter<edm::InputTag>("ConversionsInput");
  labelBeamSpot_   = iConfig.getParameter<edm::InputTag>("BeamSpotInput");
  labelLoosePhot_  = iConfig.getParameter<edm::InputTag>("PhoLoose");
  labelTightPhot_  = iConfig.getParameter<edm::InputTag>("PhoTight");
  minPtJet_        = iConfig.getParameter<double>("MinPtJet");
  minPtPhoton_     = iConfig.getParameter<double>("MinPtPhoton");

  tok_Photon_ = consumes<reco::PhotonCollection>(labelPhoton_);
  tok_PFJet_  = consumes<reco::PFJetCollection>(labelPFJet_);
  tok_HBHE_   = consumes<edm::SortedCollection<HBHERecHit,edm::StrictWeakOrdering<HBHERecHit>>>(labelHBHE_);
  tok_HF_     = consumes<edm::SortedCollection<HFRecHit,edm::StrictWeakOrdering<HFRecHit>>>(labelHF_);
  tok_HO_     = consumes<edm::SortedCollection<HORecHit,edm::StrictWeakOrdering<HORecHit>>>(labelHO_);
  tok_TrigRes_= consumes<edm::TriggerResults>(labelTrigger_);
  tok_PFCand_ = consumes<reco::PFCandidateCollection>(labelPFCandidate_);
  tok_Vertex_ = consumes<reco::VertexCollection>(labelVertex_);
  tok_PFMET_  = consumes<reco::PFMETCollection>(labelPFMET_);
  tok_loosePhoton_ = consumes<edm::ValueMap<Bool_t> >(labelLoosePhot_);
  tok_tightPhoton_ = consumes<edm::ValueMap<Bool_t> >(labelTightPhot_);
  tok_GsfElec_ = consumes<reco::GsfElectronCollection>(labelGsfEle_);
  tok_Rho_ = consumes<double>(labelRho_);
  tok_Conv_        = consumes<reco::ConversionCollection>(labelConv_);
  tok_BS_          = consumes<reco::BeamSpot>(labelBeamSpot_);

  // register your products
  produces<reco::PhotonCollection>(labelPhoton_.encode());
  produces<reco::PFJetCollection>(labelPFJet_.encode());
  produces<edm::SortedCollection<HBHERecHit,edm::StrictWeakOrdering<HBHERecHit>>>(labelHBHE_.encode());
  produces<edm::SortedCollection<HFRecHit,edm::StrictWeakOrdering<HFRecHit>>>(labelHF_.encode());
  produces<edm::SortedCollection<HORecHit,edm::StrictWeakOrdering<HORecHit>>>(labelHO_.encode());
  produces<edm::TriggerResults>(labelTrigger_.encode());
  produces<std::vector<Bool_t>>(labelLoosePhot_.encode());
  produces<std::vector<Bool_t>>(labelTightPhot_.encode());
  produces<double>(labelRho_.encode());
  produces<reco::PFCandidateCollection>(labelPFCandidate_.encode());
  produces<reco::VertexCollection>(labelVertex_.encode());
  produces<reco::PFMETCollection>(labelPFMET_.encode());
  produces<reco::GsfElectronCollection>(labelGsfEle_.encode());
  produces<reco::ConversionCollection>(labelConv_.encode());
  produces<reco::BeamSpot>(labelBeamSpot_.encode());
 
}

AlCaGammaJetProducer::~AlCaGammaJetProducer() { }

void AlCaGammaJetProducer::beginJob() { }

void AlCaGammaJetProducer::endJob() {
  edm::LogInfo("AlcaGammaJet") << "Accepts " << nSelect_ << " events from a total of " << nAll_ << " events";
}

bool AlCaGammaJetProducer::select (const reco::PhotonCollection &ph, const reco::PFJetCollection &jt) {

  // Check the requirement for minimum pT
  if (ph.size() == 0) return false;
  bool ok(false);
  for (reco::PFJetCollection::const_iterator itr=jt.begin();
       itr!=jt.end(); ++itr) {
    if (itr->pt() >= minPtJet_) {
      ok = true;
      break;
    }
  }
  if (!ok) return ok;
  for (reco::PhotonCollection::const_iterator itr=ph.begin();
       itr!=ph.end(); ++itr) {
    if (itr->pt() >= minPtPhoton_) return ok;
  }
  return false;
}
// ------------ method called to produce the data  ------------
void AlCaGammaJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  nAll_++;

  // Access the collections from iEvent
  edm::Handle<reco::PhotonCollection> phoHandle;
  iEvent.getByToken(tok_Photon_,phoHandle);
  if (!phoHandle.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get the product " << labelPhoton_;
    return ;
  }
  const reco::PhotonCollection photon = *(phoHandle.product());

  edm::Handle<reco::PFJetCollection> pfjet;
  iEvent.getByToken(tok_PFJet_,pfjet);
  if (!pfjet.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelPFJet_;
    return ;
  }
  const reco::PFJetCollection pfjets = *(pfjet.product());

  edm::Handle<reco::PFCandidateCollection> pfc;
  iEvent.getByToken(tok_PFCand_,pfc);
  if (!pfc.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelPFCandidate_;
    return ;
  }
  const reco::PFCandidateCollection pfcand = *(pfc.product());

  edm::Handle<reco::VertexCollection> vt;
  iEvent.getByToken(tok_Vertex_,vt);
  if (!vt.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelVertex_;
    return ;
  }
  const reco::VertexCollection vtx = *(vt.product());

  edm::Handle<reco::PFMETCollection> pfmt;
  iEvent.getByToken(tok_PFMET_,pfmt);
  if (!pfmt.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelPFMET_;
    return ;
  }
  const reco::PFMETCollection pfmet = *(pfmt.product());

  edm::Handle<edm::SortedCollection<HBHERecHit,edm::StrictWeakOrdering<HBHERecHit> > > hbhe;
  iEvent.getByToken(tok_HBHE_,hbhe);
  if (!hbhe.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelHBHE_;
    return ;
  }
  const edm::SortedCollection<HBHERecHit,edm::StrictWeakOrdering<HBHERecHit> > Hithbhe = *(hbhe.product());

  edm::Handle<edm::SortedCollection<HORecHit,edm::StrictWeakOrdering<HORecHit> > > ho;
  iEvent.getByToken(tok_HO_,ho);
  if(!ho.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelHO_;
    return ;
  }
  const edm::SortedCollection<HORecHit,edm::StrictWeakOrdering<HORecHit> > Hitho = *(ho.product());
    
  edm::Handle<edm::SortedCollection<HFRecHit,edm::StrictWeakOrdering<HFRecHit> > > hf;
  iEvent.getByToken(tok_HF_,hf);
  if(!hf.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelHF_;
    return ;
  }
  const edm::SortedCollection<HFRecHit,edm::StrictWeakOrdering<HFRecHit> > Hithf = *(hf.product());

  edm::Handle<edm::TriggerResults> trig;
  iEvent.getByToken(tok_TrigRes_,trig);
  if (!trig.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelTrigger_;
    return ;
  }
  const edm::TriggerResults trigres = *(trig.product());

  edm::Handle<double> rh;
  iEvent.getByToken(tok_Rho_,rh);
  if (!rh.isValid()) {
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelRho_;
    return ;
  }
  const double rho_val = *(rh.product());

  edm::Handle<reco::GsfElectronCollection> gsf;
  iEvent.getByToken(tok_GsfElec_,gsf);
  if (!gsf.isValid()){
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelGsfEle_;
    return ;
  }
  const reco::GsfElectronCollection gsfele = *(gsf.product());

  edm::Handle<reco::ConversionCollection> con;
  iEvent.getByToken(tok_Conv_,con);
  if (!con.isValid()){
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelConv_;
    return ;
  }
  const reco::ConversionCollection conv = *(con.product());

  edm::Handle<reco::BeamSpot> bs;
  iEvent.getByToken(tok_BS_,bs);
  if (!bs.isValid()){
    edm::LogWarning("AlCaGammaJet") << "AlCaGammaJetProducer: Error! can't get product " << labelBeamSpot_;
    return ;
  }
  const reco::BeamSpot beam = *(bs.product());

  // declare variables
  // copy from standard place, if the event is useful
  std::auto_ptr<reco::PFJetCollection>  miniPFjetCollection(new reco::PFJetCollection);
  std::auto_ptr<reco::PhotonCollection> miniPhotonCollection(new reco::PhotonCollection);
  std::auto_ptr<reco::PFCandidateCollection> miniPFCandCollection(new reco::PFCandidateCollection);
  std::auto_ptr<reco::VertexCollection> miniVtxCollection(new reco::VertexCollection);
  std::auto_ptr<reco::PFMETCollection> miniPFMETCollection(new reco::PFMETCollection);
  std::auto_ptr<edm::SortedCollection<HBHERecHit,edm::StrictWeakOrdering<HBHERecHit>>>  miniHBHECollection(new edm::SortedCollection<HBHERecHit,edm::StrictWeakOrdering<HBHERecHit>>);
  std::auto_ptr<edm::SortedCollection<HORecHit,edm::StrictWeakOrdering<HORecHit>>>  miniHOCollection(new edm::SortedCollection<HORecHit,edm::StrictWeakOrdering<HORecHit>>);
  std::auto_ptr<edm::SortedCollection<HFRecHit,edm::StrictWeakOrdering<HFRecHit>>>  miniHFCollection(new edm::SortedCollection<HFRecHit,edm::StrictWeakOrdering<HFRecHit>>);
  std::auto_ptr<reco::GsfElectronCollection> miniGSFeleCollection(new reco::GsfElectronCollection);
  std::auto_ptr<reco::ConversionCollection> miniConversionCollection(new reco::ConversionCollection);

  std::auto_ptr<reco::BeamSpot> miniBeamSpotCollection(new reco::BeamSpot(beam.position(),beam.sigmaZ(),
									    beam.dxdz(),beam.dydz(),beam.BeamWidthX(),
									    beam.covariance(),beam.type()));
    
  std::auto_ptr<edm::TriggerResults> miniTriggerCollection(new edm::TriggerResults);

  std::auto_ptr<double> miniRhoCollection(new double);
  std::auto_ptr<std::vector<Bool_t> > miniLoosePhoton(new std::vector<Bool_t>());
  std::auto_ptr<std::vector<Bool_t> > miniTightPhoton(new std::vector<Bool_t>());


  // See if this event is useful
  bool accept = select(photon, pfjets);
  if (accept) {
    nSelect_++;

    //Copy from standard place
    for(reco::PFJetCollection::const_iterator pfjetItr=pfjets.begin();
        pfjetItr!=pfjets.end(); pfjetItr++) {
      miniPFjetCollection->push_back(*pfjetItr);
    }

    for(reco::PhotonCollection::const_iterator phoItr=photon.begin();
        phoItr!=photon.end(); phoItr++) {
      miniPhotonCollection->push_back(*phoItr);
    }

    for(reco::PFCandidateCollection::const_iterator pfcItr=pfcand.begin();
        pfcItr!=pfcand.end(); pfcItr++) {
      miniPFCandCollection->push_back(*pfcItr);
    }

    for(reco::VertexCollection::const_iterator vtxItr=vtx.begin();
        vtxItr!=vtx.end(); vtxItr++) {
      miniVtxCollection->push_back(*vtxItr);
    }

    for(reco::PFMETCollection::const_iterator pfmetItr=pfmet.begin();
        pfmetItr!=pfmet.end(); pfmetItr++) {
      miniPFMETCollection->push_back(*pfmetItr);
    }

    for(edm::SortedCollection<HBHERecHit,edm::StrictWeakOrdering<HBHERecHit> >::const_iterator hbheItr=Hithbhe.begin(); 
	hbheItr!=Hithbhe.end(); hbheItr++) {
      miniHBHECollection->push_back(*hbheItr);
    }

    for(edm::SortedCollection<HORecHit,edm::StrictWeakOrdering<HORecHit> >::const_iterator hoItr=Hitho.begin();
        hoItr!=Hitho.end(); hoItr++) {
      miniHOCollection->push_back(*hoItr);
    }

    for(edm::SortedCollection<HFRecHit,edm::StrictWeakOrdering<HFRecHit> >::const_iterator hfItr=Hithf.begin();
        hfItr!=Hithf.end(); hfItr++) {
      miniHFCollection->push_back(*hfItr);
    }

    for(reco::GsfElectronCollection::const_iterator gsfItr=gsfele.begin();
        gsfItr!=gsfele.end(); gsfItr++) {
      miniGSFeleCollection->push_back(*gsfItr);
    }

    for(reco::ConversionCollection::const_iterator convItr=conv.begin();
        convItr!=conv.end(); convItr++) {
      miniConversionCollection->push_back(*convItr);
    }

    *miniTriggerCollection = trigres;
    *miniRhoCollection = rho_val;

    edm::Handle<edm::ValueMap<Bool_t> > loosePhotonQual;
    iEvent.getByToken(tok_loosePhoton_, loosePhotonQual);
    edm::Handle<edm::ValueMap<Bool_t> > tightPhotonQual;
    iEvent.getByToken(tok_tightPhoton_, tightPhotonQual);
    if (loosePhotonQual.isValid() && tightPhotonQual.isValid()) {
      miniLoosePhoton->reserve(miniPhotonCollection->size());
      miniTightPhoton->reserve(miniPhotonCollection->size());
      for (int iPho=0; iPho<int(miniPhotonCollection->size()); ++iPho) {
	edm::Ref<reco::PhotonCollection> photonRef(phoHandle,iPho);
	if (!photonRef) {
	  std::cout << "failed ref" << std::endl;
	  miniLoosePhoton->push_back(-1);
	  miniTightPhoton->push_back(-1);
	}
	else {
	  miniLoosePhoton->push_back((*loosePhotonQual)[photonRef]);
	  miniTightPhoton->push_back((*tightPhotonQual)[photonRef]);
	}
      }
    }
  }

  //Put them in the event
  iEvent.put( miniPhotonCollection,      labelPhoton_.encode());
  iEvent.put( miniPFjetCollection,       labelPFJet_.encode());
  iEvent.put( miniHBHECollection,        labelHBHE_.encode());
  iEvent.put( miniHFCollection,          labelHF_.encode());
  iEvent.put( miniHOCollection,          labelHO_.encode());
  iEvent.put( miniTriggerCollection,     labelTrigger_.encode());
  iEvent.put( miniPFCandCollection,      labelPFCandidate_.encode());
  iEvent.put( miniVtxCollection,         labelVertex_.encode());
  iEvent.put( miniPFMETCollection,       labelPFMET_.encode());
  iEvent.put( miniGSFeleCollection,      labelGsfEle_.encode());
  iEvent.put( miniRhoCollection,         labelRho_.encode());
  iEvent.put( miniConversionCollection,  labelConv_.encode());
  iEvent.put( miniBeamSpotCollection,    labelBeamSpot_.encode());
  iEvent.put( miniLoosePhoton,           labelLoosePhot_.encode());
  iEvent.put( miniTightPhoton,           labelTightPhot_.encode());

  return;

}

DEFINE_FWK_MODULE(AlCaGammaJetProducer); 

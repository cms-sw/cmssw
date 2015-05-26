// -*- C++ -*-
//#define DebugLog

// system include files
#include <memory>
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <boost/regex.hpp> 

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

//
// class declaration
//

class AlCaHBHEMuonProducer : public edm::EDProducer {
public:
  explicit AlCaHBHEMuonProducer(const edm::ParameterSet&);
  ~AlCaHBHEMuonProducer();
  
  virtual void produce(edm::Event &, const edm::EventSetup&);
 
private:

  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  bool         select(const reco::MuonCollection &);
  
  // ----------member data ---------------------------
  int                        nRun, nAll, nGood;
  edm::InputTag              labelBS_, labelVtx_ ;
  edm::InputTag              labelEB_, labelEE_, labelHBHE_, labelMuon_;
  double                     pMuonMin_;

  edm::EDGetTokenT<reco::BeamSpot>                        tok_BS_;
  edm::EDGetTokenT<reco::VertexCollection>                tok_Vtx_;
  edm::EDGetTokenT<EcalRecHitCollection>                  tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>                  tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>                  tok_HBHE_;
  edm::EDGetTokenT<reco::MuonCollection>                  tok_Muon_;
};


AlCaHBHEMuonProducer::AlCaHBHEMuonProducer(const edm::ParameterSet& iConfig) :
  nRun(0), nAll(0), nGood(0) {
  //Get the run parameters
  labelBS_             = iConfig.getParameter<edm::InputTag>("BeamSpotLabel");
  labelVtx_            = iConfig.getParameter<edm::InputTag>("VertexLabel");
  labelEB_             = iConfig.getParameter<edm::InputTag>("EBRecHitLabel");
  labelEE_             = iConfig.getParameter<edm::InputTag>("EERecHitLabel");
  labelHBHE_           = iConfig.getParameter<edm::InputTag>("HBHERecHitLabel");
  labelMuon_           = iConfig.getParameter<edm::InputTag>("MuonLabel");
  pMuonMin_            = iConfig.getParameter<double>("MinimumMuonP");

  // define tokens for access
  tok_Vtx_      = consumes<reco::VertexCollection>(labelVtx_);
  tok_BS_       = consumes<reco::BeamSpot>(labelBS_);
  tok_EB_       = consumes<EcalRecHitCollection>(labelEB_);
  tok_EE_       = consumes<EcalRecHitCollection>(labelEE_);
  tok_HBHE_     = consumes<HBHERecHitCollection>(labelHBHE_);
  tok_Muon_     = consumes<reco::MuonCollection>(labelMuon_);

  edm::LogInfo("HcalIsoTrack") << "Parameters read from config file \n" 
			       << "\t minP of muon " << pMuonMin_
			       << "\t input labels " << labelBS_ << " " 
			       << labelVtx_ <<" " << labelEB_ << " " << labelEE_
			       <<" " << labelHBHE_ << " " << labelMuon_;

  //saves the following collections
  produces<reco::BeamSpot>(labelBS_.label());
  produces<reco::VertexCollection>(labelVtx_.label());
  produces<EcalRecHitCollection>(labelEB_.instance());
  produces<EcalRecHitCollection>(labelEE_.instance());
  produces<HBHERecHitCollection>(labelHBHE_.label());
  produces<reco::MuonCollection>(labelMuon_.label());
}

AlCaHBHEMuonProducer::~AlCaHBHEMuonProducer() { }

void AlCaHBHEMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  nAll++;
#ifdef DebugLog
  edm::LogInfo("HcalHBHEMuon") << "AlCaHBHEMuonProducer::Run " 
			       << iEvent.id().run() << " Event " 
			       << iEvent.id().event() << " Luminosity " 
			       << iEvent.luminosityBlock() << " Bunch " 
			       << iEvent.bunchCrossing();
#endif
  
  //Step1: Get all the relevant containers
  edm::Handle<reco::BeamSpot> bmspot;
  iEvent.getByToken(tok_BS_, bmspot);
  if (!bmspot.isValid()){
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelBS_;
    return;
  }
  const reco::BeamSpot beam = *(bmspot.product());


  edm::Handle<reco::VertexCollection> vt;
  iEvent.getByToken(tok_Vtx_, vt);  
  if (!vt.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelVtx_;
    return ;
  }
  const reco::VertexCollection vtx = *(vt.product());

  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelEB_;
    return ;
  }
  const EcalRecHitCollection ebcoll = *(barrelRecHitsHandle.product());

  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelEE_;
    return ;
  }
  const EcalRecHitCollection eecoll = *(endcapRecHitsHandle.product());

  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_HBHE_, hbhe);
  if (!hbhe.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelHBHE_;
    return ;
  }
  const HBHERecHitCollection hbhecoll = *(hbhe.product());

  edm::Handle<reco::MuonCollection> muonhandle;
  iEvent.getByToken(tok_Muon_, muonhandle);
  if (!muonhandle.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelMuon_;
    return ;
  }
  const reco::MuonCollection muons = *(muonhandle.product());

#ifdef DebugLog
  edm::LogInfo("HcalHBHEMuon") << "AlCaHBHEMuonProducer::Has obtained all the collections";
#endif

  //For accepted events
  bool accept = select(muons);
  std::auto_ptr<reco::BeamSpot>         outputBeamSpot(new reco::BeamSpot(beam.position(),beam.sigmaZ(),
									  beam.dxdz(),beam.dydz(),beam.BeamWidthX(),
									  beam.covariance(),beam.type()));
  std::auto_ptr<reco::VertexCollection> outputVColl(new reco::VertexCollection);
  std::auto_ptr<EBRecHitCollection>     outputEBColl(new EBRecHitCollection);
  std::auto_ptr<EERecHitCollection>     outputEEColl(new EERecHitCollection);
  std::auto_ptr<HBHERecHitCollection>   outputHBHEColl(new HBHERecHitCollection);
  std::auto_ptr<reco::MuonCollection>   outputMColl(new reco::MuonCollection);
  if (accept) {
    nGood++;
 
    for (reco::VertexCollection::const_iterator vtr=vtx.begin(); vtr!=vtx.end(); ++vtr)
      outputVColl->push_back(*vtr);

    for (edm::SortedCollection<EcalRecHit>::const_iterator ehit=ebcoll.begin(); ehit!=ebcoll.end(); ++ehit)
      outputEBColl->push_back(*ehit);

    for (edm::SortedCollection<EcalRecHit>::const_iterator ehit=eecoll.begin(); ehit!=eecoll.end(); ++ehit)
      outputEEColl->push_back(*ehit);

    for (std::vector<HBHERecHit>::const_iterator hhit=hbhecoll.begin(); hhit!=hbhecoll.end(); ++hhit)
      outputHBHEColl->push_back(*hhit);

    for (reco::MuonCollection::const_iterator muon=muons.begin(); muon!=muons.end(); ++muon)
      outputMColl->push_back(*muon);
  }

  iEvent.put(outputBeamSpot,       labelBS_.label());
  iEvent.put(outputVColl,          labelVtx_.label());
  iEvent.put(outputEBColl,         labelEB_.instance());
  iEvent.put(outputEEColl,         labelEE_.instance());
  iEvent.put(outputHBHEColl,       labelHBHE_.label());
  iEvent.put(outputMColl,          labelMuon_.label());
}

void AlCaHBHEMuonProducer::beginJob() { }

void AlCaHBHEMuonProducer::endJob() {
  edm::LogInfo("HcalHBHEMuon") << "Finds " << nGood << " good tracks in " 
			       << nAll << " events from " << nRun << " runs";
}

void AlCaHBHEMuonProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogInfo("HcalHBHEMuon") << "Run[" << nRun << "] " << iRun.run(); 
}

void AlCaHBHEMuonProducer::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun++;
  edm::LogInfo("HcalHBHEMuon") << "endRun[" << nRun << "] " << iRun.run();
}

bool AlCaHBHEMuonProducer::select(const reco::MuonCollection & muons) {

  bool ok(false);
  for (unsigned int k=0; k<muons.size(); ++k) {
    if (muons[k].p() > pMuonMin_) {
      ok = true; break;
    }
  }
  return ok;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AlCaHBHEMuonProducer);

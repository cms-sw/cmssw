// -*- C++ -*-
//#define DebugLog

// system include files
#include <atomic>
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
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h" 
#include "DataFormats/HcalDetId/interface/HcalDetId.h" 
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalIsolatedTrack/interface/HcalIsolatedTrackCandidate.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"


//
// class declaration
//

namespace AlCaIsoTracks {
  struct Counters {
    Counters() : nAll_(0), nGood_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_;
  };
}

class AlCaIsoTracksProducer : public edm::stream::EDProducer<edm::GlobalCache<AlCaIsoTracks::Counters> > {
public:
  explicit AlCaIsoTracksProducer(edm::ParameterSet const&, const AlCaIsoTracks::Counters* count);
  ~AlCaIsoTracksProducer();
  
  static std::unique_ptr<AlCaIsoTracks::Counters> initializeGlobalCache(edm::ParameterSet const& ) {
    return std::unique_ptr<AlCaIsoTracks::Counters>(new AlCaIsoTracks::Counters());
  }

  virtual void produce(edm::Event &, edm::EventSetup const&) override;
  virtual void endStream() override;
  static  void globalEndJob(const AlCaIsoTracks::Counters* counters);
 
private:

  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  reco::HcalIsolatedTrackCandidateCollection* select(edm::Handle<edm::TriggerResults>& triggerResults, const std::vector<std::string> & triggerNames_, edm::Handle<reco::TrackCollection>& trkCollection, math::XYZPoint& leadPV,edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle, edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle, edm::Handle<HBHERecHitCollection>& hbhe, double ptL1, double etaL1, double phiL1);
  void setPtEtaPhi(std::vector< edm::Ref<l1extra::L1JetParticleCollection> >& objref, double &ptL1, double &etaL1, double &phiL1);

  // ----------member data ---------------------------
  HLTConfigProvider               hltConfig_;
  std::vector<std::string>        trigNames_, HLTNames_;
  unsigned int                    nRun_, nAll_, nGood_;
  spr::trackSelectionParameters   selectionParameter_;
  std::string                     theTrackQuality_, processName_;
  double                          maxRestrictionPt_, slopeRestrictionPt_;
  double                          a_mipR_, a_coneR_, a_charIsoR_;
  double                          pTrackMin_, eEcalMax_, eIsolation_;
  edm::InputTag                   labelTriggerEvent_, labelTriggerResults_;
  edm::InputTag                   labelGenTrack_, labelRecVtx_,  labelHltGT_;
  edm::InputTag                   labelEB_, labelEE_, labelHBHE_, labelBS_;
  std::string                     labelIsoTk_;
  const MagneticField            *bField;
  const CaloGeometry             *geo;

  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>  tok_hltGT_;
  edm::EDGetTokenT<trigger::TriggerEvent>                 tok_trigEvt_;
  edm::EDGetTokenT<edm::TriggerResults>                   tok_trigRes_;
  edm::EDGetTokenT<reco::TrackCollection>                 tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection>                tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>                        tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection>                  tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>                  tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>                  tok_hbhe_;
};


AlCaIsoTracksProducer::AlCaIsoTracksProducer(edm::ParameterSet const& iConfig, const AlCaIsoTracks::Counters* counters) :
  nRun_(0), nAll_(0), nGood_(0) {
  //Get the run parameters
  const double isolationRadius(28.9);
  trigNames_                          = iConfig.getParameter<std::vector<std::string> >("Triggers");
  theTrackQuality_                    = iConfig.getParameter<std::string>("TrackQuality");
  processName_                        = iConfig.getParameter<std::string>("ProcessName");
  maxRestrictionPt_                   = iConfig.getParameter<double>("MinTrackPt");
  slopeRestrictionPt_                 = iConfig.getParameter<double>("SlopeTrackPt");
  selectionParameter_.minPt           = (slopeRestrictionPt_ > 0) ? (maxRestrictionPt_ - 2.5/slopeRestrictionPt_) : maxRestrictionPt_;
  selectionParameter_.minQuality      = reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameter_.maxDxyPV        = iConfig.getParameter<double>("MaxDxyPV");
  selectionParameter_.maxDzPV         = iConfig.getParameter<double>("MaxDzPV");
  selectionParameter_.maxChi2         = iConfig.getParameter<double>("MaxChi2");
  selectionParameter_.maxDpOverP      = iConfig.getParameter<double>("MaxDpOverP");
  selectionParameter_.minOuterHit     = iConfig.getParameter<int>("MinOuterHit");
  selectionParameter_.minLayerCrossed = iConfig.getParameter<int>("MinLayerCrossed");
  selectionParameter_.maxInMiss       = iConfig.getParameter<int>("MaxInMiss");
  selectionParameter_.maxOutMiss      = iConfig.getParameter<int>("MaxOutMiss");
  a_coneR_                            = iConfig.getParameter<double>("ConeRadius");
  a_charIsoR_                         = a_coneR_ + isolationRadius;
  a_mipR_                             = iConfig.getParameter<double>("ConeRadiusMIP");
  pTrackMin_                          = iConfig.getParameter<double>("MinimumTrackP");
  eEcalMax_                           = iConfig.getParameter<double>("MaximumEcalEnergy");
  eIsolation_                         = iConfig.getParameter<double>("IsolationEnergy");
  labelGenTrack_                      = iConfig.getParameter<edm::InputTag>("TrackLabel");
  labelRecVtx_                        = iConfig.getParameter<edm::InputTag>("VertexLabel");
  labelBS_                            = iConfig.getParameter<edm::InputTag>("BeamSpotLabel");
  labelEB_                            = iConfig.getParameter<edm::InputTag>("EBRecHitLabel");
  labelEE_                            = iConfig.getParameter<edm::InputTag>("EERecHitLabel");
  labelHBHE_                          = iConfig.getParameter<edm::InputTag>("HBHERecHitLabel");
  labelHltGT_                         = iConfig.getParameter<edm::InputTag>("L1GTSeedLabel");
  labelTriggerEvent_                  = iConfig.getParameter<edm::InputTag>("TriggerEventLabel");
  labelTriggerResults_                = iConfig.getParameter<edm::InputTag>("TriggerResultLabel");
  labelIsoTk_                         = iConfig.getParameter<std::string>("IsoTrackLabel");

  // define tokens for access
  tok_hltGT_    = consumes<trigger::TriggerFilterObjectWithRefs>(labelHltGT_);
  tok_trigEvt_  = consumes<trigger::TriggerEvent>(labelTriggerEvent_);
  tok_trigRes_  = consumes<edm::TriggerResults>(labelTriggerResults_);
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);
  tok_recVtx_   = consumes<reco::VertexCollection>(labelRecVtx_);
  tok_bs_       = consumes<reco::BeamSpot>(labelBS_);
  tok_EB_       = consumes<EcalRecHitCollection>(labelEB_);
  tok_EE_       = consumes<EcalRecHitCollection>(labelEE_);
  tok_hbhe_     = consumes<HBHERecHitCollection>(labelHBHE_);

  edm::LogInfo("HcalIsoTrack") <<"Parameters read from config file \n" 
			       <<"\t minPt "           << maxRestrictionPt_
			       <<":"                   << slopeRestrictionPt_
			       <<"\t theTrackQuality " << theTrackQuality_
			       <<"\t minQuality "      << selectionParameter_.minQuality
			       <<"\t maxDxyPV "        << selectionParameter_.maxDxyPV          
			       <<"\t maxDzPV "         << selectionParameter_.maxDzPV          
			       <<"\t maxChi2 "         << selectionParameter_.maxChi2          
			       <<"\t maxDpOverP "      << selectionParameter_.maxDpOverP
			       <<"\t minOuterHit "     << selectionParameter_.minOuterHit
			       <<"\t minLayerCrossed " << selectionParameter_.minLayerCrossed
			       <<"\t maxInMiss "       << selectionParameter_.maxInMiss
			       <<"\t maxOutMiss "      << selectionParameter_.maxOutMiss
			       <<"\t a_coneR "         << a_coneR_
			       <<"\t a_charIsoR "      << a_charIsoR_
			       <<"\t a_mipR "          << a_mipR_
			       <<"\t pTrackMin "       << pTrackMin_
			       <<"\t eEcalMax "        << eEcalMax_
			       <<"\t eIsolation "      << eIsolation_
			       << "\tProcess "         << processName_;
  for (unsigned int k=0; k<trigNames_.size(); ++k)
    edm::LogInfo("HcalIsoTrack") << "Trigger[" << k << "] " << trigNames_[k];

  //create also IsolatedPixelTrackCandidateCollection which contains isolation info and reference to primary track
  produces<reco::HcalIsolatedTrackCandidateCollection>(labelIsoTk_);
  produces<reco::VertexCollection>(labelRecVtx_.label());
  produces<EcalRecHitCollection>(labelEB_.instance());
  produces<EcalRecHitCollection>(labelEE_.instance());
  produces<HBHERecHitCollection>(labelHBHE_.label());

  edm::LogInfo("HcalIsoTrack") << " Expected to produce the collections:\n"
			       << "reco::HcalIsolatedTrackCandidateCollection "
			       << " with label HcalIsolatedTrackCollection\n"
			       << "reco::VertexCollection with label " << labelRecVtx_.label() << "\n"
			       << "EcalRecHitCollection with label EcalRecHitsEB\n"
			       << "EcalRecHitCollection with label EcalRecHitsEE\n"
			       << "HBHERecHitCollection with label " << labelHBHE_.label();
}


AlCaIsoTracksProducer::~AlCaIsoTracksProducer() { }


void AlCaIsoTracksProducer::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {

  nAll_++;
#ifdef DebugLog
  edm::LogInfo("HcalIsoTrack") << "Run " << iEvent.id().run() << " Event " 
			       << iEvent.id().event() << " Luminosity " 
			       << iEvent.luminosityBlock() << " Bunch " 
			       << iEvent.bunchCrossing();
#endif
  bool valid(true);
  //Step1: Get all the relevant containers
  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt_, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelTriggerEvent_;
    valid = false;
  }
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(tok_trigRes_, triggerResults);
  if (!triggerResults.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelTriggerResults_;
    valid = false;
  }

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);
  if (!trkCollection.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelGenTrack_;
    valid = false;
  }
  reco::TrackCollection::const_iterator trkItr;

  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);  
  if (!recVtxs.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelGenTrack_;
    valid = false;
  }

  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);
  math::XYZPoint leadPV(0,0,0);
  if (valid) {
    if (recVtxs->size()>0 && !((*recVtxs)[0].isFake())) {
      leadPV = math::XYZPoint((*recVtxs)[0].x(),(*recVtxs)[0].y(),
			      (*recVtxs)[0].z());
    } else if (beamSpotH.isValid()) {
      leadPV = beamSpotH->position();
    }
  }
#ifdef DebugLog
  edm::LogInfo("HcalIsoTrack") << "Primary Vertex " << leadPV;
#endif

  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEB_;
    valid = false;
  }
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEE_;
    valid = false;
  }
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);
  if (!hbhe.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelHBHE_;
    valid = false;
  }

  //Get L1 trigger object
  double ptL1(0), etaL1(0), phiL1(0);
  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1trigobj;
  iEvent.getByToken(tok_hltGT_, l1trigobj);

  if (l1trigobj.isValid()) {
    std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1tauobjref;
    l1trigobj->getObjects(trigger::TriggerL1TauJet, l1tauobjref);
    setPtEtaPhi(l1tauobjref,ptL1,etaL1,phiL1);

    std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1jetobjref;
    l1trigobj->getObjects(trigger::TriggerL1CenJet, l1jetobjref);
    setPtEtaPhi(l1jetobjref,ptL1,etaL1,phiL1);

    std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1forjetobjref;
    l1trigobj->getObjects(trigger::TriggerL1ForJet, l1forjetobjref);
    setPtEtaPhi(l1forjetobjref,ptL1,etaL1,phiL1);
  } else {
    valid = false;
  }

  std::auto_ptr<reco::HcalIsolatedTrackCandidateCollection> outputHcalIsoTrackColl(new reco::HcalIsolatedTrackCandidateCollection);
  std::auto_ptr<reco::VertexCollection> outputVColl(new reco::VertexCollection);
  std::auto_ptr<EBRecHitCollection>     outputEBColl(new EBRecHitCollection);
  std::auto_ptr<EERecHitCollection>     outputEEColl(new EERecHitCollection);
  std::auto_ptr<HBHERecHitCollection>   outputHBHEColl(new HBHERecHitCollection);

  //For valid HLT record
  if (!valid) {
    edm::LogWarning("HcalIsoTrack") << "Error! Can't get some of the products";
  } else {
    trigger::TriggerEvent triggerEvent = *(triggerEventHandle.product());
    if (triggerResults.isValid()) {
      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
      const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
      reco::HcalIsolatedTrackCandidateCollection* isotk = select(triggerResults, triggerNames_, trkCollection, leadPV, barrelRecHitsHandle, endcapRecHitsHandle, hbhe, ptL1, etaL1, phiL1);
#ifdef DebugLog
      edm::LogInfo("HcalIsoTrack") << "AlCaIsoTracksProducer::select returns "
				   << isotk->size() << " isolated tracks";
#endif
    
      if (isotk->size() > 0) {
	for (reco::HcalIsolatedTrackCandidateCollection::const_iterator itr=isotk->begin(); itr!=isotk->end(); ++itr)
	  outputHcalIsoTrackColl->push_back(*itr);
	
	for (reco::VertexCollection::const_iterator vtx=recVtxs->begin(); vtx!=recVtxs->end(); ++vtx)
	  outputVColl->push_back(*vtx);
      
	for (edm::SortedCollection<EcalRecHit>::const_iterator ehit=barrelRecHitsHandle->begin(); ehit!=barrelRecHitsHandle->end(); ++ehit)
	  outputEBColl->push_back(*ehit);
    
	for (edm::SortedCollection<EcalRecHit>::const_iterator ehit=endcapRecHitsHandle->begin(); ehit!=endcapRecHitsHandle->end(); ++ehit)
	  outputEEColl->push_back(*ehit);
    
	for (std::vector<HBHERecHit>::const_iterator hhit=hbhe->begin(); hhit!=hbhe->end(); ++hhit)
	  outputHBHEColl->push_back(*hhit);
	++nGood_;
      }
    }
  }
  iEvent.put(outputHcalIsoTrackColl, labelIsoTk_);
  iEvent.put(outputVColl,            labelRecVtx_.label());
  iEvent.put(outputEBColl,           labelEB_.instance());
  iEvent.put(outputEEColl,           labelEE_.instance());
  iEvent.put(outputHBHEColl,         labelHBHE_.label());
}

void AlCaIsoTracksProducer::endStream() {
  globalCache()->nAll_  += nAll_;
  globalCache()->nGood_ += nGood_;
}

void AlCaIsoTracksProducer::globalEndJob(const AlCaIsoTracks::Counters* count) {
  edm::LogInfo("HcalIsoTrack") << "Finds " << count->nGood_ <<" good tracks in "
			       << count->nAll_ << " events";
}

void AlCaIsoTracksProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(false);
  edm::LogInfo("HcalIsoTrack") << "Run[" << nRun_ << "] " << iRun.run() 
			       << " hltconfig.init " << hltConfig_.init(iRun,iSetup,processName_,changed);

  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  bField = bFieldH.product();
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  geo    = pG.product();
}

void AlCaIsoTracksProducer::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  edm::LogInfo("HcalIsoTrack") << "endRun [" << nRun_ << "] " << iRun.run();
  ++nRun_;
}

reco::HcalIsolatedTrackCandidateCollection* 
AlCaIsoTracksProducer::select(edm::Handle<edm::TriggerResults>& triggerResults, 
			      const std::vector<std::string> & triggerNames_,
			      edm::Handle<reco::TrackCollection>& trkCollection,
			      math::XYZPoint& leadPV,edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle, 
			      edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle, 
			      edm::Handle<HBHERecHitCollection>& hbhe,
			      double ptL1, double etaL1, double phiL1) {
  
  reco::HcalIsolatedTrackCandidateCollection* trackCollection=new reco::HcalIsolatedTrackCandidateCollection;
  bool ok(false);

  // Find a good HLT trigger
  for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
    int hlt    = triggerResults->accept(iHLT);
    for (unsigned int i=0; i<trigNames_.size(); ++i) {
      if (triggerNames_[iHLT].find(trigNames_[i].c_str())!=std::string::npos) {
	if (hlt > 0) {
	  ok = true;
	}
	edm::LogInfo("HcalIsoTrack") << "The trigger we are looking for "
				     << triggerNames_[iHLT] << " Flag " 
				     << hlt << ":" << ok;
      }
    }
  }

  //Propagate tracks to calorimeter surface)
  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_,
		     trkCaloDirections, false);

  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
  unsigned int nTracks(0), nselTracks(0);
  for (trkDetItr = trkCaloDirections.begin(),nTracks=0; 
       trkDetItr != trkCaloDirections.end(); trkDetItr++,nTracks++) {
    const reco::Track* pTrack = &(*(trkDetItr->trkItr));
    math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), 
			       pTrack->pz(), pTrack->p());
#ifdef DebugLog
    edm::LogInfo("HcalIsoTrack") << "This track : " << nTracks 
				 << " (pt|eta|phi|p) :" << pTrack->pt() 
				 << "|" << pTrack->eta() << "|" 
				 << pTrack->phi() << "|" << pTrack->p();
#endif	    
    //Selection of good track
    selectionParameter_.minPt = maxRestrictionPt_ - std::abs(pTrack->eta())/slopeRestrictionPt_;
    bool qltyFlag  = spr::goodTrack(pTrack,leadPV,selectionParameter_,false);
#ifdef DebugLog
    edm::LogInfo("HcalIsoTrack") << "qltyFlag|okECAL|okHCAL : " << qltyFlag
				 << "|" << trkDetItr->okECAL << "|"  
				 << trkDetItr->okHCAL;
#endif
    if (qltyFlag && trkDetItr->okECAL && trkDetItr->okHCAL) {
      double t_p        = pTrack->p();
      nselTracks++;
      int    nRH_eMipDR(0), nNearTRKs(0);
      double eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, 
				      endcapRecHitsHandle,
				      trkDetItr->pointHCAL,
				      trkDetItr->pointECAL, a_mipR_,
				      trkDetItr->directionECAL, 
				      nRH_eMipDR);
      double hmaxNearP = spr::chargeIsolationCone(nTracks,
						  trkCaloDirections,
						  a_charIsoR_,
						  nNearTRKs, false);
#ifdef DebugLog
      edm::LogInfo("HcalIsoTrack") << "This track : " << nTracks 
				   << " (pt|eta|phi|p) :"  << pTrack->pt() 
				   << "|" << pTrack->eta() << "|" 
				   << pTrack->phi() << "|" << t_p 
				   << " e_MIP " << eMipDR 
				   << " Chg Isolation " << hmaxNearP;
#endif
      if (t_p>pTrackMin_ && eMipDR<eEcalMax_ && hmaxNearP<eIsolation_) {
	reco::HcalIsolatedTrackCandidate newCandidate(v4);
	newCandidate.SetMaxP(hmaxNearP);
	newCandidate.SetEnergyEcal(eMipDR);
	newCandidate.setL1(ptL1,etaL1,phiL1);
	newCandidate.SetEtaPhiEcal((trkDetItr->pointECAL).eta(),
				   (trkDetItr->pointECAL).phi());
	HcalDetId detId = HcalDetId(trkDetItr->detIdHCAL);
	newCandidate.SetEtaPhiHcal((trkDetItr->pointHCAL).eta(),
				   (trkDetItr->pointHCAL).phi(),
				   detId.ieta(), detId.iphi());
	int indx(0);
	reco::TrackCollection::const_iterator trkItr1;
	for (trkItr1=trkCollection->begin(); trkItr1 != trkCollection->end(); ++trkItr1,++indx) {
	  const reco::Track* pTrack1 = &(*trkItr1);
	  if (pTrack1 == pTrack) {
	    reco::TrackRef tRef = reco::TrackRef(trkCollection,indx);
	    newCandidate.setTrack(tRef);
	    break;
	  }
	}
	trackCollection->push_back(newCandidate);
      }
    }
  }
  return trackCollection;
}

void AlCaIsoTracksProducer::setPtEtaPhi(std::vector< edm::Ref<l1extra::L1JetParticleCollection> >& objref, double &ptL1, double &etaL1, double &phiL1) {

  for (unsigned int p=0; p<objref.size(); p++) {
    if (objref[p]->pt()>ptL1) {
      ptL1  = objref[p]->pt(); 
      phiL1 = objref[p]->phi();
      etaL1 = objref[p]->eta();
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AlCaIsoTracksProducer);

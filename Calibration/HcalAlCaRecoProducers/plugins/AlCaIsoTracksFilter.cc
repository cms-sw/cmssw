// system include files
#include <atomic>
#include <memory>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/Common/interface/Handle.h"
//Tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
// RecHits
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
//Triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

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
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
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

class AlCaIsoTracksFilter : public edm::stream::EDFilter<edm::GlobalCache<AlCaIsoTracks::Counters> > {
public:
  explicit AlCaIsoTracksFilter(edm::ParameterSet const&, const AlCaIsoTracks::Counters* count);
  ~AlCaIsoTracksFilter();
    
  static std::unique_ptr<AlCaIsoTracks::Counters> initializeGlobalCache(edm::ParameterSet const& iConfig) {
    return std::unique_ptr<AlCaIsoTracks::Counters>(new AlCaIsoTracks::Counters());
  }

  virtual bool filter(edm::Event&, edm::EventSetup const&) override;
  virtual void endStream() override;
  static  void globalEndJob(const AlCaIsoTracks::Counters* counters);
  static  void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  
  // ----------member data ---------------------------
  HLTConfigProvider             hltConfig_;
  std::vector<std::string>      trigNames_, HLTNames_;
  spr::trackSelectionParameters selectionParameter_;
  std::string                   theTrackQuality_, processName_;
  double                        maxRestrictionPt_, slopeRestrictionPt_;
  double                        a_mipR_, a_coneR_, a_charIsoR_;
  double                        pTrackMin_, eEcalMax_, eIsolation_;
  unsigned int                  nRun_, nAll_, nGood_;
  edm::InputTag                 triggerEvent_, theTriggerResultsLabel;
  edm::InputTag                 labelGenTrack_, labelRecVtx_;
  edm::InputTag                 labelEB_, labelEE_, labelHBHE_;
  edm::EDGetTokenT<trigger::TriggerEvent>  tok_trigEvt_;
  edm::EDGetTokenT<edm::TriggerResults>    tok_trigRes_;
  edm::EDGetTokenT<reco::TrackCollection>  tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>         tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>   tok_hbhe_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
AlCaIsoTracksFilter::AlCaIsoTracksFilter(const edm::ParameterSet& iConfig, const AlCaIsoTracks::Counters* count) :
  nRun_(0), nAll_(0), nGood_(0) {
  //now do what ever initialization is needed
  const double isolationRadius(28.9);
  trigNames_                          = iConfig.getParameter<std::vector<std::string> >("Triggers");
  theTrackQuality_                    = iConfig.getParameter<std::string>("TrackQuality");
  processName_                        = iConfig.getParameter<std::string>("ProcessName");
  reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality_);
  maxRestrictionPt_                   = iConfig.getParameter<double>("MinTrackPt");
  slopeRestrictionPt_                 = iConfig.getParameter<double>("SlopeTrackPt");
  selectionParameter_.minPt           = maxRestrictionPt_;
  selectionParameter_.minQuality      = trackQuality_;
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
  triggerEvent_                       = iConfig.getParameter<edm::InputTag>("TriggerEventLabel");
  theTriggerResultsLabel              = iConfig.getParameter<edm::InputTag>("TriggerResultLabel");
  labelGenTrack_                      = iConfig.getParameter<edm::InputTag>("TrackLabel");
  labelRecVtx_                        = iConfig.getParameter<edm::InputTag>("VertexLabel");
  labelEB_                            = iConfig.getParameter<edm::InputTag>("EBRecHitLabel");
  labelEE_                            = iConfig.getParameter<edm::InputTag>("EERecHitLabel");
  labelHBHE_                          = iConfig.getParameter<edm::InputTag>("HBHERecHitLabel");

  // define tokens for access
  tok_trigEvt_  = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes_  = consumes<edm::TriggerResults>(theTriggerResultsLabel);
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);
  tok_recVtx_   = consumes<reco::VertexCollection>(labelRecVtx_);
  tok_bs_       = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("BeamSpotLabel"));
 
  tok_EB_       = consumes<EcalRecHitCollection>(labelEB_);
  tok_EE_       = consumes<EcalRecHitCollection>(labelEE_);
  tok_hbhe_     = consumes<HBHERecHitCollection>(labelHBHE_);

  edm::LogInfo("HcalIsoTrack") <<"Parameters read from config file \n" 
			       <<"\t minPt "          << maxRestrictionPt_
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
			       <<"\t a_mipR "          << a_mipR_;
  for (unsigned int k=0; k<trigNames_.size(); ++k)
    edm::LogInfo("HcalIsoTrack") << "Trigger[" << k << "] " << trigNames_[k];
} // AlCaIsoTracksFilter::AlCaIsoTracksFilter  constructor


AlCaIsoTracksFilter::~AlCaIsoTracksFilter() {}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool AlCaIsoTracksFilter::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  bool accept(false);
  ++nAll_;
  LogDebug("HcalIsoTrack") << "Run " << iEvent.id().run() << " Event " 
			   << iEvent.id().event() << " Luminosity " 
			   << iEvent.luminosityBlock() << " Bunch " 
			   << iEvent.bunchCrossing();

  //Step1: Find if the event passes one of the chosen triggers
  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt_, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Error! Can't get the product "
				    << triggerEvent_.label() ;
  } else {
    triggerEvent = *(triggerEventHandle.product());

    /////////////////////////////TriggerResults
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByToken(tok_trigRes_, triggerResults);
    if (triggerResults.isValid()) {
      bool ok(false);
      std::vector<std::string> modules;
      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
      const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
      for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
	int hlt    = triggerResults->accept(iHLT);
	for (unsigned int i=0; i<trigNames_.size(); ++i) {
          if (triggerNames_[iHLT].find(trigNames_[i].c_str())!=std::string::npos) {
	    if (hlt > 0) {
	      ok = true;
	    }
	    LogDebug("HcalIsoTrack") <<"This is the trigger we are looking for "
				     << triggerNames_[iHLT] << " Flag " << hlt 
				     << ":" << ok;
          }
        }
      }
      if (ok) {
	//Step2: Get geometry/B-field information
	//Get magnetic field
	edm::ESHandle<MagneticField> bFieldH;
	iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
	const MagneticField *bField = bFieldH.product();
	// get handles to calogeometry and calotopology
	edm::ESHandle<CaloGeometry> pG;
	iSetup.get<CaloGeometryRecord>().get(pG);
	const CaloGeometry* geo = pG.product();
  
	//Also relevant information to extrapolate tracks to Hcal surface
	bool okC(true);
	//Get track collection
	edm::Handle<reco::TrackCollection> trkCollection;
	iEvent.getByToken(tok_genTrack_, trkCollection);
	if (!trkCollection.isValid()) {
	  edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelGenTrack_;
	  okC = false;
	}
	reco::TrackCollection::const_iterator trkItr;

	//Define the best vertex and the beamspot
	edm::Handle<reco::VertexCollection> recVtxs;
	iEvent.getByToken(tok_recVtx_, recVtxs);  
	edm::Handle<reco::BeamSpot> beamSpotH;
	iEvent.getByToken(tok_bs_, beamSpotH);
	math::XYZPoint leadPV(0,0,0);
	if (recVtxs->size()>0 && !((*recVtxs)[0].isFake())) {
	  leadPV = math::XYZPoint((*recVtxs)[0].x(),(*recVtxs)[0].y(),
				  (*recVtxs)[0].z());
	} else if (beamSpotH.isValid()) {
	  leadPV = beamSpotH->position();
	}
	LogDebug("HcalIsoTrack") << "Primary Vertex " << leadPV;
  
	// RecHits
	edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
	iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
	if (!barrelRecHitsHandle.isValid()) {
	  edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEB_;
	  okC = false;
	}
	edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
	iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
	if (!endcapRecHitsHandle.isValid()) {
	  edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelEE_;
	  okC = false;
	}
	edm::Handle<HBHERecHitCollection> hbhe;
	iEvent.getByToken(tok_hbhe_, hbhe);
	if (!hbhe.isValid()) {
	  edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelHBHE_;
	  okC = false;
	}
	  
	if (okC) {
	  //Step3 propagate the tracks to calorimeter surface and find
	  // candidates for isolated tracks
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
	    LogDebug("HcalIsoTrack") << "This track : " << nTracks 
				     << " (pt|eta|phi|p) :" << pTrack->pt() 
				     << "|" << pTrack->eta() << "|" 
				     << pTrack->phi() << "|" << pTrack->p();
	    
	    //Selection of good track
	    selectionParameter_.minPt = maxRestrictionPt_ - std::abs(pTrack->eta())/slopeRestrictionPt_;
	    bool qltyFlag  = spr::goodTrack(pTrack,leadPV,selectionParameter_,false);
	    LogDebug("HcalIsoTrack") << "qltyFlag|okECAL|okHCAL : " << qltyFlag
				     << "|" << trkDetItr->okECAL << "|" 
				     << trkDetItr->okHCAL;
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
	      LogDebug("HcalIsoTrack") << "This track : " << nTracks 
				       << " (pt|eta|phi|p) :"  << pTrack->pt() 
				       << "|" << pTrack->eta() << "|" 
				       << pTrack->phi() << "|" << t_p
				       << "e_MIP " << eMipDR 
				       << " Chg Isolation " << hmaxNearP;
	      if (t_p>pTrackMin_ && eMipDR<eEcalMax_ && hmaxNearP<eIsolation_) 
		accept = true;
	    }
	  }
	}
      }
    }
  }
  // Step 4:  Return the acceptance flag
  if (accept) ++nGood_;
  return accept;

}  // AlCaIsoTracksFilter::filter

// ------------ method called once each job just after ending the event loop  ------------
void AlCaIsoTracksFilter::endStream() {
  globalCache()->nAll_  += nAll_;
  globalCache()->nGood_ += nGood_;
}

void AlCaIsoTracksFilter::globalEndJob(const AlCaIsoTracks::Counters* count) {
  edm::LogInfo("HcalIsoTrack") << "Selects " << count->nGood_ << " in " 
			       << count->nAll_ << " events";
}


// ------------ method called when starting to processes a run  ------------
void AlCaIsoTracksFilter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(false);
  edm::LogInfo("HcalIsoTrack") << "Run[" << nRun_ << "] " << iRun.run() 
			       << " hltconfig.init " << hltConfig_.init(iRun,iSetup,processName_,changed);
}

// ------------ method called when ending the processing of a run  ------------
void AlCaIsoTracksFilter::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  ++nRun_;
  edm::LogInfo("HcalIsoTrack") << "endRun[" << nRun_ << "] " << iRun.run();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void AlCaIsoTracksFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlCaIsoTracksFilter);

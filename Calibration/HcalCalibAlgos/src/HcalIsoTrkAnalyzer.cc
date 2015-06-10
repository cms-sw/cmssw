// system include files
#include <memory>

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TInterpreter.h"

//Tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//Triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
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

//#define DebugLog

class HcalIsoTrkAnalyzer : public edm::EDAnalyzer {

public:
  explicit HcalIsoTrkAnalyzer(const edm::ParameterSet&);
  ~HcalIsoTrkAnalyzer();
 
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
 
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);

  bool                       changed;
  edm::Service<TFileService> fs;
  HLTConfigProvider          hltConfig_;
  std::vector<std::string>   trigNames, HLTNames;
  std::vector<int>           trigKount, trigPass;
  spr::trackSelectionParameters selectionParameters;
  double                     a_mipR, a_coneR, a_charIsoR;
  double                     pTrackMin_, eEcalMax_, eIsolation_;
  int                        nRun, nAll, nGood;
  edm::InputTag              triggerEvent_, theTriggerResultsLabel;
  std::string                labelGenTrack_, labelRecVtx_, labelEB_, labelEE_;
  std::string                theTrackQuality, processName, labelHBHE_;
  std::string                l1Filter, l2Filter, l3Filter;
  edm::EDGetTokenT<trigger::TriggerEvent>  tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults>    tok_trigRes;

  edm::EDGetTokenT<reco::TrackCollection>  tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>         tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>   tok_hbhe_;

  TTree                     *tree;
  int                        t_Run, t_Event, t_ieta; 
  double                     t_EventWeight, t_l1pt, t_l1eta, t_l1phi;
  double                     t_l3pt, t_l3eta, t_l3phi, t_p, t_mindR1;
  double                     t_mindR2, t_eMipDR, t_eHcal, t_hmaxNearP;
  bool                       t_selectTk,t_qltyFlag,t_qltyMissFlag,t_qltyPVFlag;
  std::vector<unsigned int> *t_DetIds;
  std::vector<double>       *t_HitEnergies, pbin;
  std::vector<bool>         *t_trgbits; 
  std::vector<std::string>   trgnames;
};

HcalIsoTrkAnalyzer::HcalIsoTrkAnalyzer(const edm::ParameterSet& iConfig) : 
  changed(false), nRun(0), nAll(0), nGood(0) {
  //now do whatever initialization is needed
  trigNames                           = iConfig.getParameter<std::vector<std::string> >("Triggers");
  theTrackQuality                     = iConfig.getParameter<std::string>("TrackQuality");
  processName                         = iConfig.getParameter<std::string>("ProcessName");
  l1Filter                            = iConfig.getParameter<std::string>("L1Filter");
  l2Filter                            = iConfig.getParameter<std::string>("L2Filter");
  l3Filter                            = iConfig.getParameter<std::string>("L3Filter");
  reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);
  selectionParameters.minPt           = iConfig.getParameter<double>("MinTrackPt");
  selectionParameters.minQuality      = trackQuality_;
  selectionParameters.maxDxyPV        = iConfig.getParameter<double>("MaxDxyPV");
  selectionParameters.maxDzPV         = iConfig.getParameter<double>("MaxDzPV");
  selectionParameters.maxChi2         = iConfig.getParameter<double>("MaxChi2");
  selectionParameters.maxDpOverP      = iConfig.getParameter<double>("MaxDpOverP");
  selectionParameters.minOuterHit     = iConfig.getParameter<int>("MinOuterHit");
  selectionParameters.minLayerCrossed = iConfig.getParameter<int>("MinLayerCrossed");
  selectionParameters.maxInMiss       = iConfig.getParameter<int>("MaxInMiss");
  selectionParameters.maxOutMiss      = iConfig.getParameter<int>("MaxOutMiss");
  a_coneR                             = iConfig.getParameter<double>("ConeRadius");
  a_charIsoR                          = a_coneR + 28.9;
  a_mipR                              = iConfig.getParameter<double>("ConeRadiusMIP");
  pTrackMin_                          = iConfig.getParameter<double>("MinimumTrackP");
  eEcalMax_                           = iConfig.getParameter<double>("MaximumEcalEnergy");
  eIsolation_                         = iConfig.getParameter<double>("IsolationEnergy");
  triggerEvent_                       = iConfig.getParameter<edm::InputTag>("TriggerEventLabel");
  theTriggerResultsLabel              = iConfig.getParameter<edm::InputTag>("TriggerResultLabel");
  labelGenTrack_                      = iConfig.getParameter<std::string>("TrackLabel");
  labelRecVtx_                        = iConfig.getParameter<std::string>("VertexLabel");
  labelEB_                            = iConfig.getParameter<std::string>("EBRecHitLabel");
  labelEE_                            = iConfig.getParameter<std::string>("EERecHitLabel");
  labelHBHE_                          = iConfig.getParameter<std::string>("HBHERecHitLabel");
  edm::InputTag labelBS               = iConfig.getParameter<edm::InputTag>("BeamSpotLabel");
  std::string modnam                  = iConfig.getUntrackedParameter<std::string>("ModuleName","");
  std::string prdnam                  = iConfig.getUntrackedParameter<std::string>("ProducerName","");

  // define tokens for access
  tok_trigEvt   = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes   = consumes<edm::TriggerResults>(theTriggerResultsLabel);
  tok_bs_       = consumes<reco::BeamSpot>(labelBS);
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);
  if (modnam == "") {
    tok_recVtx_   = consumes<reco::VertexCollection>(labelRecVtx_);
    tok_EB_       = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit",labelEB_));
    tok_EE_       = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit",labelEE_));
    tok_hbhe_     = consumes<HBHERecHitCollection>(labelHBHE_);
    edm::LogInfo("HcalIsoTrack") << "Labels used " << triggerEvent_ << " "
				 << theTriggerResultsLabel << " "
				 << labelBS << " " << labelRecVtx_ << " " 
				 << labelGenTrack_ << " " 
				 << edm::InputTag("ecalRecHit",labelEB_) << " " 
				 << edm::InputTag("ecalRecHit",labelEE_) << " "
				 << labelHBHE_;
  } else {
    tok_recVtx_   = consumes<reco::VertexCollection>(edm::InputTag(modnam,labelRecVtx_,prdnam));
    tok_EB_       = consumes<EcalRecHitCollection>(edm::InputTag(modnam,labelEB_,prdnam));
    tok_EE_       = consumes<EcalRecHitCollection>(edm::InputTag(modnam,labelEE_,prdnam));
    tok_hbhe_     = consumes<HBHERecHitCollection>(edm::InputTag(modnam,labelHBHE_,prdnam));
    edm::LogInfo("HcalIsoTrack") << "Labels used "   << triggerEvent_ 
				 << "\n            " << theTriggerResultsLabel 
				 << "\n            " << labelBS 
				 << "\n            " << edm::InputTag(modnam,labelRecVtx_,prdnam)
				 << "\n            " << labelGenTrack_
				 << "\n            " << edm::InputTag(modnam,labelEB_,prdnam)
				 << "\n            " << edm::InputTag(modnam,labelEE_,prdnam)
				 << "\n            " << edm::InputTag(modnam,labelHBHE_,prdnam);
  }

  std::vector<int> dummy(trigNames.size(),0);
  trigKount = trigPass = dummy;
  edm::LogInfo("HcalIsoTrack") <<"Parameters read from config file \n" 
			       <<"\t minPt "           << selectionParameters.minPt   
			       <<"\t theTrackQuality " << theTrackQuality
			       <<"\t minQuality "      << selectionParameters.minQuality
			       <<"\t maxDxyPV "        << selectionParameters.maxDxyPV          
			       <<"\t maxDzPV "         << selectionParameters.maxDzPV          
			       <<"\t maxChi2 "         << selectionParameters.maxChi2          
			       <<"\t maxDpOverP "      << selectionParameters.maxDpOverP
			       <<"\t minOuterHit "     << selectionParameters.minOuterHit
			       <<"\t minLayerCrossed " << selectionParameters.minLayerCrossed
			       <<"\t maxInMiss "       << selectionParameters.maxInMiss
			       <<"\t maxOutMiss "      << selectionParameters.maxOutMiss
			       <<"\t a_coneR "         << a_coneR          
			       <<"\t a_charIsoR "      << a_charIsoR          
			       <<"\t a_mipR "          << a_mipR;
  edm::LogInfo("HcalIsoTrack") << "Process " << processName << " L1Filter:" 
			       << l1Filter << " L2Filter:" << l2Filter 
			       << " L3Filter:" << l3Filter;
  for (unsigned int k=0; k<trigNames.size(); ++k)
    edm::LogInfo("HcalIsoTrack") << "Trigger[" << k << "] " << trigNames[k];
}

HcalIsoTrkAnalyzer::~HcalIsoTrkAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void HcalIsoTrkAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  t_Run   = iEvent.id().run();
  t_Event = iEvent.id().event();
  nAll++;
#ifdef DebugLog
  edm::LogInfo("HcalIsoTrack") << "Run " << t_Run << " Event " << t_Event 
			       << " Luminosity " << iEvent.luminosityBlock() 
			       << " Bunch " << iEvent.bunchCrossing();
#endif
  //Get magnetic field and ECAL channel status
  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField *bField = bFieldH.product();

  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  
  bool okC(true);
  //Get track collection
  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);
  reco::TrackCollection::const_iterator trkItr;
  if (!trkCollection.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Cannot access the collection " << labelGenTrack_;
    okC = false;
  }
 
  //event weight for FLAT sample
  t_EventWeight = 1.0;

  //Define the best vertex and the beamspot
  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);  
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);
  math::XYZPoint leadPV(0,0,0);
  if (recVtxs.isValid() && recVtxs->size()>0 && !((*recVtxs)[0].isFake())) {
    leadPV = math::XYZPoint( (*recVtxs)[0].x(),(*recVtxs)[0].y(), (*recVtxs)[0].z() );
  } else if (beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
#ifdef DebugLog
  edm::LogInfo("HcalIsoTrack") << "Primary Vertex " << leadPV;
  if (beamSpotH.isValid()) edm::LogInfo("HcalIsoTrack") << "Beam Spot " 
							<< beamSpotH->position();
#endif  
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
	  
  //Propagate tracks to calorimeter surface)
  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality,
		     trkCaloDirections, false);
  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;

  //Trigger
  trgnames.clear();
  t_trgbits->clear();
  for (unsigned int i=0; i<trigNames.size(); ++i) t_trgbits->push_back(false);

  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Error! Can't get the product "
				    << triggerEvent_.label() ;
  } else if (okC) {
    triggerEvent = *(triggerEventHandle.product());
    
    const trigger::TriggerObjectCollection& TOC(triggerEvent.getObjects());
    /////////////////////////////TriggerResults
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByToken(tok_trigRes, triggerResults);
    if (triggerResults.isValid()) {
      std::vector<std::string> modules;
      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
      const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
      for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
	bool ok(false);
	int hlt    = triggerResults->accept(iHLT);
	for (unsigned int i=0; i<trigNames.size(); ++i) {
          if (triggerNames_[iHLT].find(trigNames[i].c_str())!=std::string::npos) {
	    t_trgbits->at(i) = (hlt>0);
	    trigKount[i]++;
	    if (hlt > 0) {
	      ok = true;
	      trigPass[i]++;
	    }
#ifdef DebugLog
	    edm::LogInfo("HcalIsoTrack") << "This trigger "
					 << triggerNames_[iHLT] << " Flag " 
					 << hlt << ":" << ok;
#endif
          }
        }

	if (ok) {
	  unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[iHLT]);
	  const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(triggerindx));
	  std::vector<math::XYZTLorentzVector> vec[3];
	  //loop over all trigger filters in event (i.e. filters passed)
	  for (unsigned int ifilter=0; ifilter<triggerEvent.sizeFilters();
	       ++ifilter) {  
	    std::vector<int> Keys;
	    std::string label = triggerEvent.filterTag(ifilter).label();
	    //loop over keys to objects passing this filter
	    for (unsigned int imodule=0; imodule<moduleLabels.size(); 
		 imodule++) {
	      if (label.find(moduleLabels[imodule]) != std::string::npos) {
#ifdef DebugLog
		edm::LogInfo("HcalIsoTrack") << "FilterName " << label;
#endif
		for (unsigned int ifiltrKey=0; ifiltrKey<triggerEvent.filterKeys(ifilter).size(); ++ifiltrKey) {
		  Keys.push_back(triggerEvent.filterKeys(ifilter)[ifiltrKey]);
		  const trigger::TriggerObject& TO(TOC[Keys[ifiltrKey]]);
		  math::XYZTLorentzVector v4(TO.px(), TO.py(), TO.pz(), TO.energy());
		  if (label.find(l2Filter) != std::string::npos) {
		    vec[1].push_back(v4);
		  } else if (label.find(l3Filter) != std::string::npos) {
		    vec[2].push_back(v4);
		  } else if (label.find(l1Filter) != std::string::npos ||
			     l1Filter == "") {
		    vec[0].push_back(v4);
		  }
#ifdef DebugLog
		  edm::LogInfo("HcalIsoTrack") << "key " << ifiltrKey<<" : pt "
					       << TO.pt() << " eta "<< TO.eta()
					       << " phi " << TO.phi()<<" mass "
					       << TO.mass() << " Id "<<TO.id();
#endif
		}
#ifdef DebugLog
		edm::LogInfo("HcalIsoTrack") << "sizes " << vec[0].size() << ":"
					     << vec[1].size() <<":" <<vec[2].size();
#endif
	      }
	    }
	  }
	  double dr;
	  //// deta, dphi and dR for leading L1 object with L2 objects
	  math::XYZTLorentzVector mindRvec1;
	  double mindR1(999);
	  for (int lvl=1; lvl<3; lvl++) {
	    for (unsigned int i=0; i<vec[lvl].size(); i++) {
	      dr   = dR(vec[0][0],vec[lvl][i]);
#ifdef DebugLog
	      edm::LogInfo("HcalIsoTrack") << "lvl " << lvl << " i " << i 
					   << " dR " << dr;
#endif
	      if (dr<mindR1) {
		mindR1    = dr;
		mindRvec1 = vec[lvl][i];
	      }
	    }
	  }
	  
	  t_l1pt  = vec[0][0].pt();
	  t_l1eta = vec[0][0].eta();
	  t_l1phi = vec[0][0].phi();
          if (vec[2].size()>0) {
	    t_l3pt  = vec[2][0].pt();
	    t_l3eta = vec[2][0].eta();
	    t_l3phi = vec[2][0].phi();
	  }
	  //Loop over tracks
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
					 << pTrack->phi() << "|" <<pTrack->p();
#endif
	    math::XYZTLorentzVector mindRvec2;
	    t_mindR2 = 999;

	    for (unsigned int k=0; k<vec[2].size(); ++k) {
	      dr   = dR(vec[2][k],v4); //changed 1 to 2
	      if (dr<t_mindR2) {
		t_mindR2  = dr;
		mindRvec2 = vec[2][k];
	      }
	    }
	    t_mindR1 = dR(vec[0][0],v4);
#ifdef DebugLog
	    edm::LogInfo("HcalIsoTrack") << "Closest L3 object at mindr :" 
					 << t_mindR2 << " is " << mindRvec2 
					 << " and from L1 " << t_mindR1;
#endif	    
	    //Selection of good track
	    t_selectTk = spr::goodTrack(pTrack,leadPV,selectionParameters,false);
	    spr::trackSelectionParameters oneCutParameters = selectionParameters;
 	    oneCutParameters.maxDxyPV  = 10;
	    oneCutParameters.maxDzPV   = 100;
	    oneCutParameters.maxInMiss = 2;
	    oneCutParameters.maxOutMiss= 2;
	    bool qltyFlag  = spr::goodTrack(pTrack,leadPV,oneCutParameters,false);
	    oneCutParameters           = selectionParameters;
	    oneCutParameters.maxDxyPV  = 10;
	    oneCutParameters.maxDzPV   = 100;
	    t_qltyMissFlag = spr::goodTrack(pTrack,leadPV,oneCutParameters,false);
	    oneCutParameters           = selectionParameters;
	    oneCutParameters.maxInMiss = 2;
	    oneCutParameters.maxOutMiss= 2;
	    t_qltyPVFlag   = spr::goodTrack(pTrack,leadPV,oneCutParameters,false);
	    t_ieta = 0;
	    if (trkDetItr->okHCAL) {
	      HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
	      t_ieta = detId.ieta();
	    }
#ifdef DebugLog
	    edm::LogInfo("HcalIsoTrack") << "qltyFlag|okECAL|okHCAL : " 
					 << qltyFlag << "|" << trkDetItr->okECAL
					 << "|" << trkDetItr->okHCAL;
#endif
	    t_qltyFlag = (qltyFlag && trkDetItr->okECAL && trkDetItr->okHCAL);
	    t_p        = pTrack->p();
	    if (t_qltyFlag) {
              nselTracks++;
	      int nRH_eMipDR(0), nNearTRKs(0), nRecHits(-999);
	      t_eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, 
					 endcapRecHitsHandle,
					 trkDetItr->pointHCAL,
					 trkDetItr->pointECAL,
					 a_mipR, trkDetItr->directionECAL, 
					 nRH_eMipDR);
	      t_DetIds->clear(); t_HitEnergies->clear();
	      std::vector<DetId> ids;
	      t_eHcal = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, 
					trkDetItr->pointECAL, a_coneR, 
					trkDetItr->directionHCAL,nRecHits, 
					ids, *t_HitEnergies);
	      for (unsigned int k=0; k<ids.size(); ++k) {
		t_DetIds->push_back(ids[k].rawId());
	      }
	      t_hmaxNearP = spr::chargeIsolationCone(nTracks,trkCaloDirections,
						     a_charIsoR, nNearTRKs, 
						     false);
#ifdef DebugLog
	      edm::LogInfo("HcalIsoTrack") << "This track : " << nTracks 
					   << " (pt|eta|phi|p) :"  << pTrack->pt() 
					   << "|" << pTrack->eta() << "|" 
					   << pTrack->phi() << "|" << t_p;
	      edm::LogInfo("HcalIsoTrack") << "e_MIP " << t_eMipDR 
					   << " Chg Isolation " << t_hmaxNearP 
					   << " eHcal" << t_eHcal << " ieta " 
					   << t_ieta << " Quality " 
					   << t_qltyMissFlag << ":" 
					   << t_qltyPVFlag << ":" << t_selectTk;
	      for (unsigned int lll=0;lll<t_DetIds->size();lll++) {
		edm::LogInfo("HcalIsoTrack") << "det id is = " <<t_DetIds->at(lll)
					     << "   hit enery is  = "  
					     << t_HitEnergies->at(lll) ;
	      }
#endif
	      if (t_p>pTrackMin_ && t_eMipDR<eEcalMax_ && 
		  t_hmaxNearP<eIsolation_) {
		tree->Fill();
		nGood++;
#ifdef DebugLog
		for (unsigned int k=0; k<t_trgbits->size(); k++) 
		  edm::LogInfo("HcalIsoTrack") << "trigger bit is  = " 
					       << t_trgbits->at(k);
#endif
	      }
	    }
	  }
	}
      }
      // check if trigger names in (new) config                       
      if (changed) {
	changed = false;
#ifdef DebugLog
	edm::LogInfo("HcalIsoTrack") << "New trigger menu found !!!";
#endif
	const unsigned int n(hltConfig_.size());
	for (unsigned itrig=0; itrig<triggerNames_.size(); itrig++) {
	  unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[itrig]);
	  if (triggerindx >= n)
	    edm::LogInfo("HcalIsoTrack") << triggerNames_[itrig] << " " 
					 << triggerindx << " does not exist in "
					 << "the current menu";
#ifdef DebugLog
	  else
	    edm::LogInfo("HcalIsoTrack") << triggerNames_[itrig] << " " 
					 << triggerindx << " exists";
#endif
	}
      }
    }
  }
}

void HcalIsoTrkAnalyzer::beginJob() {

  tree = fs->make<TTree>("CalibTree", "CalibTree");
     
  tree->Branch("t_Run",         &t_Run,         "t_Run/I");
  tree->Branch("t_Event",       &t_Event,       "t_Event/I");
  tree->Branch("t_ieta",        &t_ieta,        "t_ieta/I");
  tree->Branch("t_EventWeight", &t_EventWeight, "t_EventWeight/D");
  tree->Branch("t_l1pt",        &t_l1pt,        "t_l1pt/D");
  tree->Branch("t_l1eta",       &t_l1eta,       "t_l1eta/D");
  tree->Branch("t_l1phi",       &t_l1phi,       "t_l1phi/D"); 
  tree->Branch("t_l3pt",        &t_l3pt,        "t_l3pt/D");
  tree->Branch("t_l3eta",       &t_l3eta,       "t_l3eta/D"); 
  tree->Branch("t_l3phi",       &t_l3phi,       "t_l3phi/D");
  tree->Branch("t_p",           &t_p,           "t_p/D");
  tree->Branch("t_mindR1",      &t_mindR1,      "t_mindR1/D");
  tree->Branch("t_mindR2",      &t_mindR2,      "t_mindR2/D");
  tree->Branch("t_eMipDR",      &t_eMipDR,      "t_eMipDR/D");
  tree->Branch("t_eHcal",       &t_eHcal,       "t_eHcal/D");
  tree->Branch("t_hmaxNearP",   &t_hmaxNearP,   "t_hmaxNearP/D");
  tree->Branch("t_selectTk",    &t_selectTk,    "t_selectTk/O");
  tree->Branch("t_qltyFlag",    &t_qltyFlag,    "t_qltyFlag/O");
  tree->Branch("t_qltyMissFlag",&t_qltyMissFlag,"t_qltyMissFlag/O");
  tree->Branch("t_qltyPVFlag",  &t_qltyPVFlag,  "t_qltyPVFlag/O)");

  t_DetIds      = new std::vector<unsigned int>();
  t_HitEnergies = new std::vector<double>();
  t_trgbits     = new std::vector<bool>();
  tree->Branch("t_DetIds",      "std::vector<unsigned int>", &t_DetIds);
  tree->Branch("t_HitEnergies", "std::vector<double>",       &t_HitEnergies);
  tree->Branch("t_trgbits","std::vector<bool>", &t_trgbits); 
}

// ------------ method called once each job just after ending the event loop  ------------
void HcalIsoTrkAnalyzer::endJob() {
  edm::LogInfo("HcalIsoTrack") << "Finds " << nGood << " good tracks in " 
			       << nAll << " events from " << nRun << " runs";
  for (unsigned int k=0; k<trigNames.size(); ++k)
    edm::LogInfo("HcalIsoTrack") << "Trigger[" << k << "]: " << trigNames[k] 
				 << " Events " << trigKount[k] << " Passed " 
				 << trigPass[k];
}

// ------------ method called when starting to processes a run  ------------
void HcalIsoTrkAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogInfo("HcalIsoTrack") << "Run[" << nRun << "] " << iRun.run() 
			       << " hltconfig.init " << hltConfig_.init(iRun,iSetup,processName,changed);
}

// ------------ method called when ending the processing of a run  ------------
void HcalIsoTrkAnalyzer::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun++;
  edm::LogInfo("HcalIsoTrack") << "endRun[" << nRun << "] " << iRun.run();
}

// ------------ method called when starting to processes a luminosity block  ------------
void HcalIsoTrkAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method called when ending the processing of a luminosity block  ------------
void HcalIsoTrkAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalIsoTrkAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

double HcalIsoTrkAnalyzer::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return reco::deltaR(vec1.eta(),vec1.phi(),vec2.eta(),vec2.phi());
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalIsoTrkAnalyzer);


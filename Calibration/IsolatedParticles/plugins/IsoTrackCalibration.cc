//#define DebugLog
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
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/JetReco/interface/PFJet.h"
//#include "DataFormats/PatCandidates/interface/Jet.h"

class IsoTrackCalibration : public edm::EDAnalyzer {

public:
  explicit IsoTrackCalibration(const edm::ParameterSet&);
  ~IsoTrackCalibration();
 
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
 
  double dEta(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dPhi(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);

  bool                       changed;
  edm::Service<TFileService> fs;
  HLTConfigProvider          hltConfig_;
  std::vector<std::string>   trigNames, HLTNames;
  int                        verbosity;
  spr::trackSelectionParameters selectionParameters;
  std::string                theTrackQuality;
  double                     a_mipR, a_coneR, a_charIsoR;
  bool                       qcdMC;
  int                        nRun;

  edm::InputTag              triggerEvent_, theTriggerResultsLabel;
  edm::EDGetTokenT<trigger::TriggerEvent>  tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults>    tok_trigRes;

  edm::EDGetTokenT<reco::TrackCollection>  tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>         tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>   tok_hbhe_;
  edm::EDGetTokenT<GenEventInfoProduct>    tok_ew_; 

  TTree* tree;
  int                        t_Run, t_Event, t_ieta; 
  double                     t_EventWeight, t_l1pt, t_l1eta, t_l1phi;
  double                     t_l3pt, t_l3eta, t_l3phi, t_p, t_mindR1;
  double                     t_mindR2, t_eMipDR, t_eHcal, t_hmaxNearP;
  bool                       t_selectTk, t_qltyMissFlag, t_qltyPVFlag;
  std::vector<unsigned int> *t_DetIds;
  std::vector<double>       *t_HitEnergies;
};

IsoTrackCalibration::IsoTrackCalibration(const edm::ParameterSet& iConfig) : 
  changed(false), nRun(0) {
   //now do whatever initialization is needed
  verbosity                           = iConfig.getUntrackedParameter<int>("Verbosity",0);
  trigNames                           = iConfig.getUntrackedParameter<std::vector<std::string> >("Triggers");
  theTrackQuality                     = iConfig.getUntrackedParameter<std::string>("TrackQuality","highPurity");
  reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);
  selectionParameters.minPt           = iConfig.getUntrackedParameter<double>("MinTrackPt", 10.0);
  selectionParameters.minQuality      = trackQuality_;
  selectionParameters.maxDxyPV        = iConfig.getUntrackedParameter<double>("MaxDxyPV", 0.2);
  selectionParameters.maxDzPV         = iConfig.getUntrackedParameter<double>("MaxDzPV",  5.0);
  selectionParameters.maxChi2         = iConfig.getUntrackedParameter<double>("MaxChi2",  5.0);
  selectionParameters.maxDpOverP      = iConfig.getUntrackedParameter<double>("MaxDpOverP",  0.1);
  selectionParameters.minOuterHit     = iConfig.getUntrackedParameter<int>("MinOuterHit", 4);
  selectionParameters.minLayerCrossed = iConfig.getUntrackedParameter<int>("MinLayerCrossed", 8);
  selectionParameters.maxInMiss       = iConfig.getUntrackedParameter<int>("MaxInMiss", 0);
  selectionParameters.maxOutMiss      = iConfig.getUntrackedParameter<int>("MaxOutMiss", 0);
  a_coneR                             = iConfig.getUntrackedParameter<double>("ConeRadius",34.98);
  a_charIsoR                          = a_coneR + 28.9;
  a_mipR                              = iConfig.getUntrackedParameter<double>("ConeRadiusMIP",14.0);
  qcdMC                               = iConfig.getUntrackedParameter<bool>("IsItQCDMC", false);
  bool isItAOD                        = iConfig.getUntrackedParameter<bool>("IsItAOD", true);
  triggerEvent_                       = edm::InputTag("hltTriggerSummaryAOD","","HLT");
  theTriggerResultsLabel              = edm::InputTag("TriggerResults","","HLT");

  // define tokens for access
  tok_trigEvt   = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes   = consumes<edm::TriggerResults>(theTriggerResultsLabel);
  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_   = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_       = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  tok_ew_       = consumes<GenEventInfoProduct>(edm::InputTag("generator")); 
 
  if (isItAOD) {
    tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));
    tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("reducedHcalRecHits", "hbhereco"));
  } else {
    tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
    tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEE"));
    tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  }

  if (verbosity>=0) {
    edm::LogInfo("IsoTrack") <<"Parameters read from config file \n" 
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
			     <<"\t a_mipR "          << a_mipR 
			     <<"\t qcdMC "           << qcdMC;
    edm::LogInfo("IsoTrack") << trigNames.size() << " triggers to be studied:";
    for (unsigned int k=0; k<trigNames.size(); ++k)
      edm::LogInfo("IsoTrack") << "Trigger[" << k << "] : " << trigNames[k];
  }
}

IsoTrackCalibration::~IsoTrackCalibration() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}

void IsoTrackCalibration::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup) {

  t_Run   = iEvent.id().run();
  t_Event = iEvent.id().event();
  if (verbosity%10 > 0) 
    edm::LogInfo("IsoTrack") << "Run " << t_Run << " Event " << t_Event 
			     << " Luminosity " << iEvent.luminosityBlock() 
			     << " Bunch " << iEvent.bunchCrossing() 
			     << " starts";

  //Get magnetic field and ECAL channel status
  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField *bField = bFieldH.product();
  
  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);

  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  
  //Get track collection
  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);
  reco::TrackCollection::const_iterator trkItr;
 
  //event weight for FLAT sample
  t_EventWeight = 1.0;
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(tok_ew_, genEventInfo);
  if (genEventInfo.isValid()) t_EventWeight = genEventInfo->weight();  

  //Define the best vertex and the beamspot
  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);  
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);
  math::XYZPoint leadPV(0,0,0);
  if (recVtxs->size()>0 && !((*recVtxs)[0].isFake())) {
    leadPV = math::XYZPoint( (*recVtxs)[0].x(),(*recVtxs)[0].y(), (*recVtxs)[0].z() );
  } else if (beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
  if ((verbosity/100)%10>0) {
    edm::LogInfo("IsoTrack") << "Primary Vertex " << leadPV;
    if (beamSpotH.isValid()) edm::LogInfo("IsoTrack") << " Beam Spot " 
						      << beamSpotH->position();
  }
  
  // RecHits
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);

  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("IsoTrack") << "Error! Can't get the product "
				<< triggerEvent_.label();
  } else {
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
	    if (hlt > 0) ok = true;
	    if (verbosity%10 > 0)
	      edm::LogInfo("IsoTrack") << "This is the trigger we are looking for " 
				       << triggerNames_[iHLT] << " Flag " 
				       << hlt << ":" << ok;
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
		if (verbosity%10 > 0) 
		  edm::LogInfo("IsoTrack") << "FilterName " << label;
		for (unsigned int ifiltrKey=0; ifiltrKey<triggerEvent.filterKeys(ifilter).size(); ++ifiltrKey) {
		  Keys.push_back(triggerEvent.filterKeys(ifilter)[ifiltrKey]);
		  const trigger::TriggerObject& TO(TOC[Keys[ifiltrKey]]);
		  math::XYZTLorentzVector v4(TO.px(), TO.py(), TO.pz(), TO.energy());
		  if (qcdMC) {
		    if (label.find("hltL1s") != std::string::npos) {
		      vec[0].push_back(v4);
		    } else if (label.find("PFJet") != std::string::npos) {
		      vec[2].push_back(v4);
		    } else {
		      vec[1].push_back(v4);
		    }
		  } else {
		    if (label.find("L2Filter") != std::string::npos) {
		      vec[1].push_back(v4);
		    } else if (label.find("Filter") != std::string::npos) {
		      vec[2].push_back(v4);
		    } else {
		      vec[0].push_back(v4);
		    }
		  }
		  if (verbosity%10 > 2)
		    edm::LogInfo("IsoTrack") << "key " << ifiltrKey << " : pt " 
					     << TO.pt() << " eta " << TO.eta() 
					     << " phi " << TO.phi() << " mass "
					     << TO.mass() << " Id " << TO.id();

		}
		if (verbosity%10 > 0) 
		  edm::LogInfo("IsoTrack") << "sizes " << vec[0].size() 
					   << ":" << vec[1].size() << ":" 
					   << vec[2].size();

	      }
	    }
	  }

	  double deta, dphi, dr;
	  //// deta, dphi and dR for leading L1 object with L2 objects
	  math::XYZTLorentzVector mindRvec1;
	  double mindR1(999);
	  for (int lvl=1; lvl<3; lvl++) {
	    for (unsigned int i=0; i<vec[lvl].size(); i++) {
	      deta = dEta(vec[0][0],vec[lvl][i]);
	      dphi = dPhi(vec[0][0],vec[lvl][i]);
	      dr   = dR(vec[0][0],vec[lvl][i]);
	      if (verbosity%10 > 2) 
		edm::LogInfo("IsoTrack") << "lvl " <<lvl << " i " << i 
					 << " deta " << deta << " dphi " 
					 << dphi << " dR " << dr;
	      if (dr<mindR1) {
		mindR1    = dr;
		mindRvec1 = vec[lvl][i];
	      }
	    }
	  }

	  t_l1pt  = vec[0][0].pt();
	  t_l1eta = vec[0][0].eta();
	  t_l1phi = vec[0][0].phi();
	  t_l3pt  = vec[2][0].pt();
	  t_l3eta = vec[2][0].eta();
	  t_l3phi = vec[2][0].phi();

	  //Propagate tracks to calorimeter surface)
	  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
	  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality,
			     trkCaloDirections, ((verbosity/100)%10>2));
	  //Loop over tracks
	  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
	  unsigned int nTracks(0), nselTracks(0);
	  for (trkDetItr = trkCaloDirections.begin(),nTracks=0; 
	       trkDetItr != trkCaloDirections.end(); trkDetItr++,nTracks++) {
	    const reco::Track* pTrack = &(*(trkDetItr->trkItr));
            math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), 
				       pTrack->pz(), pTrack->p());
	    if (verbosity%10 > 0) 
	      edm::LogInfo("IsoTrack") << "This track : " << nTracks 
				       << " (pt/eta/phi/p) :" << pTrack->pt() 
				       << "/" << pTrack->eta() << "/" 
				       << pTrack->phi() << "/" << pTrack->p();
	    math::XYZTLorentzVector mindRvec2;
	    t_mindR2 = 999;

	    for (unsigned int k=0; k<vec[2].size(); ++k) {
	      dr   = dR(vec[2][k],v4); //changed 1 to 2
	      if (dr<t_mindR2) {
		t_mindR2  = dr;
		mindRvec2 = vec[2][k];
	      }
	    }
	    t_mindR1 = dR(mindRvec1,v4);
	    if (verbosity%10 > 2)
	      edm::LogInfo("IsoTrack") << "Closest L3 object at mindr :" 
				       << t_mindR2 << " is " << mindRvec2 
				       << " and from L1 " << t_mindR1;
	    
	    //Selection of good track
	    t_selectTk = spr::goodTrack(pTrack,leadPV,selectionParameters,((verbosity/100)%10>2));
	    spr::trackSelectionParameters oneCutParameters = selectionParameters;
 	    oneCutParameters.maxDxyPV  = 10;
	    oneCutParameters.maxDzPV   = 100;
	    oneCutParameters.maxInMiss = 2;
	    oneCutParameters.maxOutMiss= 2;
	    bool qltyFlag  = spr::goodTrack(pTrack,leadPV,oneCutParameters,((verbosity/100)%10>2));
	    oneCutParameters           = selectionParameters;
	    oneCutParameters.maxDxyPV  = 10;
	    oneCutParameters.maxDzPV   = 100;
	    t_qltyMissFlag = spr::goodTrack(pTrack,leadPV,oneCutParameters,((verbosity/100)%10>2));
	    oneCutParameters           = selectionParameters;
	    oneCutParameters.maxInMiss = 2;
	    oneCutParameters.maxOutMiss= 2;
	    t_qltyPVFlag   = spr::goodTrack(pTrack,leadPV,oneCutParameters,((verbosity/100)%10>2));
	    t_ieta = 0;
	    if (trkDetItr->okHCAL) {
	      HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
	      t_ieta = detId.ieta();
	    }
	    if (verbosity%10 > 0) 
	      edm::LogInfo("IsoTrack") << "qltyFlag|okECAL|okHCAL : " 
				       << qltyFlag << "|" << trkDetItr->okECAL 
				       << "/" << trkDetItr->okHCAL;
	    if (qltyFlag && trkDetItr->okECAL && trkDetItr->okHCAL) {
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
						     ((verbosity/100)%10>2));
	      t_p = pTrack->p();
              if (verbosity%10 > 0) {
		edm::LogInfo("IsoTrack") << "This track : " << nTracks 
					 << " (pt/eta/phi/p) :" << pTrack->pt()
					 << "/" << pTrack->eta() << "/" 
					 << pTrack->phi() << "/" << t_p << "\n"
					 << "e_MIP " << t_eMipDR 
					 << " Chg Isolation " << t_hmaxNearP 
					 << " eHcal" << t_eHcal << " ieta " 
					 << t_ieta << " Quality " 
					 << t_qltyMissFlag << ":" 
					 << t_qltyPVFlag << ":" << t_selectTk;
	      }

	      tree->Fill();
	    }
	  }
//	  break;
	}
      }
      // check if trigger names in (new) config                       
      if (changed) {
	changed = false;
	if ((verbosity/10)%10 > 1) {
	  edm::LogInfo("IsoTrack") << "New trigger menu found !!!";
	  const unsigned int n(hltConfig_.size());
	  for (unsigned itrig=0; itrig<triggerNames_.size(); itrig++) {
	    unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[itrig]);
	    if (triggerindx >= n)
	      edm::LogInfo("IsoTrack") << triggerNames_[itrig] << " " 
				       << triggerindx << " does not exist";
	    else
	      edm::LogInfo("IsoTrack") << triggerNames_[itrig] << " " 
				       << triggerindx << " exists";
	  }
	}
      }
    }
  }
}

void IsoTrackCalibration::beginJob() {

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
  tree->Branch("t_qltyMissFlag",&t_qltyMissFlag,"t_qltyMissFlag/O");
  tree->Branch("t_qltyPVFlag",  &t_qltyPVFlag,  "t_qltyPVFlag/O)");

  t_DetIds      = new std::vector<unsigned int>();
  t_HitEnergies = new std::vector<double>();

  tree->Branch("t_DetIds",      "std::vector<unsigned int>", &t_DetIds);
  tree->Branch("t_HitEnergies", "std::vector<double>",       &t_HitEnergies);
}

// ------------ method called once each job just after ending the event loop  ------------
void IsoTrackCalibration::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void IsoTrackCalibration::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogInfo("IsoTrack") << "Run[" << nRun <<"] " << iRun.run() 
			   << " hltconfig.init " 
			   << hltConfig_.init(iRun,iSetup,"HLT",changed);
}

// ------------ method called when ending the processing of a run  ------------
void IsoTrackCalibration::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun++;
  edm::LogInfo("IsoTrack") << "endRun[" << nRun << "] " << iRun.run();
}

// ------------ method called when starting to processes a luminosity block  ------------
void IsoTrackCalibration::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method called when ending the processing of a luminosity block  ------------
void IsoTrackCalibration::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void IsoTrackCalibration::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

double IsoTrackCalibration::dEta(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (vec1.eta()-vec2.eta());
}

double IsoTrackCalibration::dPhi(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return reco::deltaPhi(vec1.phi(),vec2.phi());
}

double IsoTrackCalibration::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return reco::deltaR(vec1.eta(),vec1.phi(),vec2.eta(),vec2.phi());
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsoTrackCalibration);


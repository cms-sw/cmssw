// system include files
#include <atomic>
#include <memory>
#include <string>
#include <vector>

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

//Generator information
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

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

class HcalIsoTrkAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit HcalIsoTrkAnalyzer(edm::ParameterSet const&);
  ~HcalIsoTrkAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void beginJob() override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
 
  int fillTree(std::vector< math::XYZTLorentzVector>& vecL1,
	       std::vector< math::XYZTLorentzVector>& vecL3,
	       math::XYZPoint& leadPV,
	       std::vector<spr::propagatedTrackDirection>& trkCaloDirections,
	       const CaloGeometry* geo,
	       edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle,
	       edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle,
	       edm::Handle<HBHERecHitCollection>& hbhe,
	       edm::Handle<reco::GenParticleCollection>& genParticles);
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double trackP(const reco::Track* ,
		const edm::Handle<reco::GenParticleCollection>&);

  unsigned int                  nRun_;
  edm::Service<TFileService>    fs;
  HLTConfigProvider             hltConfig_;
  std::vector<std::string>      trigNames_;
  spr::trackSelectionParameters selectionParameter_;
  int                           dataType_, mode_;
  double                        maxRestrictionP_, slopeRestrictionP_;
  double                        a_mipR_, a_coneR_, a_charIsoR_, hcalScale_;
  double                        a_coneR1_, a_coneR2_;
  double                        pTrackMin_, eEcalMax_, eIsolate1_, eIsolate2_;
  edm::InputTag                 triggerEvent_, theTriggerResultsLabel_;
  std::string                   labelGenTrack_, labelRecVtx_, labelEB_,labelEE_;
  std::string                   theTrackQuality_, processName_, labelHBHE_;
  std::string                   l1Filter_, l2Filter_, l3Filter_;
  bool                          ignoreTrigger_, useRaw_;
  edm::EDGetTokenT<trigger::TriggerEvent>       tok_trigEvt_;
  edm::EDGetTokenT<edm::TriggerResults>         tok_trigRes_;
  edm::EDGetTokenT<reco::GenParticleCollection> tok_parts_;
  edm::EDGetTokenT<reco::TrackCollection>       tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection>      tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>              tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection>        tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>        tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>        tok_hbhe_;
  edm::EDGetTokenT<GenEventInfoProduct>         tok_ew_; 

  TTree                     *tree, *tree2;
  int                        t_Run, t_Event, t_ieta, t_goodPV, t_DataType; 
  int                        t_nVtx, t_nTrk;
  double                     t_EventWeight, t_p, t_pt, t_phi;
  double                     t_l1pt, t_l1eta, t_l1phi;
  double                     t_l3pt, t_l3eta, t_l3phi;
  double                     t_mindR1, t_mindR2;
  double                     t_eMipDR, t_hmaxNearP, t_gentrackP;
  double                     t_eHcal, t_eHcal10, t_eHcal30;
  bool                       t_selectTk,t_qltyFlag,t_qltyMissFlag,t_qltyPVFlag;
  std::vector<unsigned int> *t_DetIds, *t_DetIds1, *t_DetIds3;
  std::vector<double>       *t_HitEnergies, *t_HitEnergies1, *t_HitEnergies3;
  std::vector<bool>         *t_trgbits; 
  int                        t_Tracks, t_TracksProp, t_TracksSaved;
  std::vector<int>          *t_ietaAll, *t_ietaGood;
};

HcalIsoTrkAnalyzer::HcalIsoTrkAnalyzer(const edm::ParameterSet& iConfig) : 
  nRun_(0) {

  usesResource("TFileService");

  //now do whatever initialization is needed
  const double isolationRadius(28.9), innerR(10.0), outerR(30.0);
  trigNames_                          = iConfig.getParameter<std::vector<std::string> >("Triggers");
  theTrackQuality_                    = iConfig.getParameter<std::string>("TrackQuality");
  processName_                        = iConfig.getParameter<std::string>("ProcessName");
  l1Filter_                           = iConfig.getParameter<std::string>("L1Filter");
  l2Filter_                           = iConfig.getParameter<std::string>("L2Filter");
  l3Filter_                           = iConfig.getParameter<std::string>("L3Filter");
  reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameter_.minPt           = iConfig.getParameter<double>("MinTrackPt");;
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
  a_coneR1_                           = a_coneR_ + innerR;
  a_coneR2_                           = a_coneR_ + outerR;
  a_mipR_                             = iConfig.getParameter<double>("ConeRadiusMIP");
  pTrackMin_                          = iConfig.getParameter<double>("MinimumTrackP");
  eEcalMax_                           = iConfig.getParameter<double>("MaximumEcalEnergy");
  maxRestrictionP_                    = iConfig.getParameter<double>("MaxTrackP");
  slopeRestrictionP_                  = iConfig.getParameter<double>("SlopeTrackP");
  eIsolate1_                          = iConfig.getParameter<double>("IsolationEnergyStr");
  eIsolate2_                          = iConfig.getParameter<double>("IsolationEnergySft");
  triggerEvent_                       = iConfig.getParameter<edm::InputTag>("TriggerEventLabel");
  theTriggerResultsLabel_             = iConfig.getParameter<edm::InputTag>("TriggerResultLabel");
  labelGenTrack_                      = iConfig.getParameter<std::string>("TrackLabel");
  labelRecVtx_                        = iConfig.getParameter<std::string>("VertexLabel");
  labelEB_                            = iConfig.getParameter<std::string>("EBRecHitLabel");
  labelEE_                            = iConfig.getParameter<std::string>("EERecHitLabel");
  labelHBHE_                          = iConfig.getParameter<std::string>("HBHERecHitLabel");
  std::string labelBS                 = iConfig.getParameter<std::string>("BeamSpotLabel");
  std::string modnam                  = iConfig.getUntrackedParameter<std::string>("ModuleName","");
  std::string prdnam                  = iConfig.getUntrackedParameter<std::string>("ProducerName","");
  ignoreTrigger_                      = iConfig.getUntrackedParameter<bool>("IgnoreTriggers", false);
  useRaw_                             = iConfig.getUntrackedParameter<bool>("UseRaw", false);
  hcalScale_                          = iConfig.getUntrackedParameter<double>("HcalScale", 1.0);
  dataType_                           = iConfig.getUntrackedParameter<int>("DataType", 0);
  mode_                               = iConfig.getUntrackedParameter<int>("OutMode", 0);

  // define tokens for access
  tok_trigEvt_  = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes_  = consumes<edm::TriggerResults>(theTriggerResultsLabel_);
  tok_bs_       = consumes<reco::BeamSpot>(labelBS);
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);
  tok_ew_       = consumes<GenEventInfoProduct>(edm::InputTag("generator")); 
  tok_parts_    = consumes<reco::GenParticleCollection>(edm::InputTag("genParticles"));
  if (modnam == "") {
    tok_recVtx_   = consumes<reco::VertexCollection>(labelRecVtx_);
    tok_EB_       = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit",labelEB_));
    tok_EE_       = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit",labelEE_));
    tok_hbhe_     = consumes<HBHERecHitCollection>(labelHBHE_);
    edm::LogInfo("HcalIsoTrack") << "Labels used " << triggerEvent_ << " "
				 << theTriggerResultsLabel_ << " "
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
				 << "\n            " << theTriggerResultsLabel_ 
				 << "\n            " << labelBS 
				 << "\n            " << edm::InputTag(modnam,labelRecVtx_,prdnam)
				 << "\n            " << labelGenTrack_
				 << "\n            " << edm::InputTag(modnam,labelEB_,prdnam)
				 << "\n            " << edm::InputTag(modnam,labelEE_,prdnam)
				 << "\n            " << edm::InputTag(modnam,labelHBHE_,prdnam);
  }

  edm::LogInfo("HcalIsoTrack") <<"Parameters read from config file \n" 
			       <<"\t minPt "           << selectionParameter_.minPt
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
			       << ":" << a_coneR1_ << ":" << a_coneR2_
			       <<"\t a_charIsoR "      << a_charIsoR_
			       <<"\t a_mipR "          << a_mipR_
			       <<"\n pTrackMin_ "      << pTrackMin_
			       <<"\t eEcalMax_ "       << eEcalMax_
			       <<"\t maxRestrictionP_ "<< maxRestrictionP_
			       <<"\t slopeRestrictionP_ " << slopeRestrictionP_
			       <<"\t eIsolateStrong_ " << eIsolate1_
			       <<"\t eIsolateSoft_ "   << eIsolate2_
			       <<"\t hcalScale_ "      << hcalScale_
			       <<"\t useRaw_ "         << useRaw_
			       <<"\t ignoreTrigger_ "  << ignoreTrigger_
			       <<"\t dataType_      "  << dataType_;
  edm::LogInfo("HcalIsoTrack") << "Process " << processName_ << " L1Filter:" 
			       << l1Filter_ << " L2Filter:" << l2Filter_
			       << " L3Filter:" << l3Filter_;
  for (unsigned int k=0; k<trigNames_.size(); ++k) {
    edm::LogInfo("HcalIsoTrack") << "Trigger[" << k << "] " << trigNames_[k];
  }
}

HcalIsoTrkAnalyzer::~HcalIsoTrkAnalyzer() { }

void HcalIsoTrkAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {

  t_Run      = iEvent.id().run();
  t_Event    = iEvent.id().event();
  t_DataType = dataType_;
#ifdef DebugLog
  edm::LogInfo("HcalIsoTrack") << "Run " << t_Run << " Event " << t_Event 
			       << " type " << t_DataType << " Luminosity " 
			       << iEvent.luminosityBlock() << " Bunch " 
			       << iEvent.bunchCrossing();
#endif
  //Get magnetic field and ECAL channel status
  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField *bField = bFieldH.product();

  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();

  //=== genParticle information
  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(tok_parts_, genParticles);
  
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
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(tok_ew_, genEventInfo);
  if (genEventInfo.isValid()) t_EventWeight = genEventInfo->weight();  

  //Define the best vertex and the beamspot
  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);  
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);
  math::XYZPoint leadPV(0,0,0);
  t_goodPV = t_nVtx = 0;
  if (recVtxs.isValid() && recVtxs->size()>0) {
    t_nVtx = recVtxs->size();
    for (unsigned int k=0; k<recVtxs->size(); ++k) {
      if (!((*recVtxs)[k].isFake()) && ((*recVtxs)[k].ndof() > 4)) {
	if (t_goodPV == 0) leadPV = math::XYZPoint((*recVtxs)[k].x(),(*recVtxs)[k].y(),(*recVtxs)[k].z());
	t_goodPV++;
      }
    }
  } 
  if (t_goodPV == 0 && beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
#ifdef DebugLog
  edm::LogInfo("HcalIsoTrack") << "Primary Vertex " << leadPV << " out of "
			       << t_goodPV << " vertex";
  if (beamSpotH.isValid()) {
    edm::LogInfo("HcalIsoTrack") << " Beam Spot " << beamSpotH->position();
  }
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
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_,
		     trkCaloDirections, false);
  std::vector<math::XYZTLorentzVector> vecL1, vecL3;
  t_Tracks     = trkCollection->size();
  t_TracksProp = trkCaloDirections.size();
  t_ietaAll->clear(); t_ietaGood->clear();
#ifdef DebugLog
  edm::LogInfo("HcalIsoTrack") << "# of propagated tracks " << t_TracksProp
			       << " out of " << t_Tracks << " with Trigger "
			       << ignoreTrigger_;
#endif
  //Trigger
  t_trgbits->clear();
  t_TracksSaved = 0;
  for (unsigned int i=0; i<trigNames_.size(); ++i) t_trgbits->push_back(false);
  if (ignoreTrigger_) {
    t_l1pt  = t_l1eta = t_l1phi = 0;
    t_l3pt  = t_l3eta = t_l3phi = 0;
    t_TracksSaved = fillTree(vecL1, vecL3, leadPV, trkCaloDirections, geo, 
			     barrelRecHitsHandle, endcapRecHitsHandle, hbhe,
			     genParticles);
  } else {
    trigger::TriggerEvent triggerEvent;
    edm::Handle<trigger::TriggerEvent> triggerEventHandle;
    iEvent.getByToken(tok_trigEvt_, triggerEventHandle);
    if (!triggerEventHandle.isValid()) {
    edm::LogWarning("HcalIsoTrack") << "Error! Can't get the product "
				    << triggerEvent_.label() ;
    } else if (okC) {
      triggerEvent = *(triggerEventHandle.product());
      const trigger::TriggerObjectCollection& TOC(triggerEvent.getObjects());
      /////////////////////////////TriggerResults
      edm::Handle<edm::TriggerResults> triggerResults;
      iEvent.getByToken(tok_trigRes_, triggerResults);
      bool done(false);
      if (triggerResults.isValid()) {
	std::vector<std::string> modules;
	const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
	const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
	int ntksave(0);
	for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
	  bool ok(false);
	  int hlt    = triggerResults->accept(iHLT);
	  if (trigNames_.size() > 0) {
	    for (unsigned int i=0; i<trigNames_.size(); ++i) {
	      if (triggerNames_[iHLT].find(trigNames_[i].c_str())!=std::string::npos) {
		t_trgbits->at(i) = (hlt>0);
		if (hlt > 0) ok = true;
#ifdef DebugLog
		edm::LogInfo("HcalIsoTrack") << "This trigger "
					     << triggerNames_[iHLT] << " Flag " 
					     << hlt << ":" << ok;
#endif
	      }
	    }
	  } else {
	    ok = (hlt>0);
	  }

	  if (ok) {
	    unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[iHLT]);
	    const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(triggerindx));
	    std::vector<math::XYZTLorentzVector> vecL2;
	    vecL1.clear(); vecL3.clear();
	    //loop over all trigger filters in event (i.e. filters passed)
	    for (unsigned int ifilter=0; ifilter<triggerEvent.sizeFilters();
		 ++ifilter) {  
	      std::vector<int> Keys;
	      std::string label = triggerEvent.filterTag(ifilter).label();
	      //loop over keys to objects passing this filter
	      for (unsigned int imodule=0; imodule<moduleLabels.size(); imodule++) {
		if (label.find(moduleLabels[imodule]) != std::string::npos) {
#ifdef DebugLog
		  edm::LogInfo("HcalIsoTrack") << "FilterName " << label;
#endif
		  for (unsigned int ifiltrKey=0; ifiltrKey<triggerEvent.filterKeys(ifilter).size(); ++ifiltrKey) {
		    Keys.push_back(triggerEvent.filterKeys(ifilter)[ifiltrKey]);
		    const trigger::TriggerObject& TO(TOC[Keys[ifiltrKey]]);
		    math::XYZTLorentzVector v4(TO.px(), TO.py(), TO.pz(), TO.energy());
		    if (label.find(l2Filter_) != std::string::npos) {
		      vecL2.push_back(v4);
		    } else if (label.find(l3Filter_) != std::string::npos) {
		      vecL3.push_back(v4);
		    } else if (label.find(l1Filter_) != std::string::npos || l1Filter_ == "") {
		      vecL1.push_back(v4);
		    }
#ifdef DebugLog
		    edm::LogInfo("HcalIsoTrack") << "key " <<ifiltrKey<<" : pt "
						 << TO.pt() << " eta "<<TO.eta()
						 << " phi " <<TO.phi()<<" mass "
						 << TO.mass() <<" Id "<<TO.id();
#endif
		  }
#ifdef DebugLog
		  edm::LogInfo("HcalIsoTrack") << "sizes " << vecL1.size() <<":"
					       << vecL2.size() <<":" <<vecL3.size();
#endif
		}
	      }
	    }
	    //// deta, dphi and dR for leading L1 object with L2 objects
	    math::XYZTLorentzVector mindRvec1;
	    double mindR1(999);
	    for (unsigned int i=0; i<vecL2.size(); i++) {
	      double dr   = dR(vecL1[0],vecL2[i]);
#ifdef DebugLog
	      edm::LogInfo("HcalIsoTrack") << "lvl2[" << i << "] dR " << dr;
#endif
	      if (dr<mindR1) {
		mindR1    = dr;
		mindRvec1 = vecL2[i];
	      }
	    }
#ifdef DebugLog
	    edm::LogInfo("HcalIsoTrack") << "L2 object closest to L1 "
					 << mindRvec1 << " at Dr " << mindR1;
#endif
	  
	    if (vecL1.size()>0) {
	      t_l1pt  = vecL1[0].pt();
	      t_l1eta = vecL1[0].eta();
	      t_l1phi = vecL1[0].phi();
	    } else {
	      t_l1pt  = t_l1eta = t_l1phi = 0;
	    }
	    if (vecL3.size()>0) {
	      t_l3pt  = vecL3[0].pt();
	      t_l3eta = vecL3[0].eta();
	      t_l3phi = vecL3[0].phi();
	    } else {
	      t_l3pt  = t_l3eta = t_l3phi = 0;
	    }
	    // Now fill in the tree for each selected track
	    if (!done) {
	      ntksave = fillTree(vecL1, vecL3, leadPV, trkCaloDirections, geo, 
				 barrelRecHitsHandle,endcapRecHitsHandle, hbhe,
				 genParticles);
	      done = true;
	    } 
	    t_TracksSaved += ntksave;
	  }
	}
      }
    }
  }
  tree2->Fill();
}

void HcalIsoTrkAnalyzer::beginJob() {

  tree = fs->make<TTree>("CalibTree", "CalibTree");
     
  tree->Branch("t_Run",         &t_Run,         "t_Run/I");
  tree->Branch("t_Event",       &t_Event,       "t_Event/I");
  tree->Branch("t_DataType",    &t_DataType,    "t_DataType/I");
  tree->Branch("t_ieta",        &t_ieta,        "t_ieta/I");
  tree->Branch("t_EventWeight", &t_EventWeight, "t_EventWeight/D");
  tree->Branch("t_nVtx",        &t_nVtx,        "t_nVtx/I");
  tree->Branch("t_nTrk",        &t_nTrk,        "t_nTrk/I");
  tree->Branch("t_goodPV",      &t_goodPV,      "t_goodPV/I");
  if (((mode_/10)%10) == 1) {
    tree->Branch("t_l1pt",        &t_l1pt,        "t_l1pt/D");
    tree->Branch("t_l1eta",       &t_l1eta,       "t_l1eta/D");
    tree->Branch("t_l1phi",       &t_l1phi,       "t_l1phi/D"); 
    tree->Branch("t_l3pt",        &t_l3pt,        "t_l3pt/D");
    tree->Branch("t_l3eta",       &t_l3eta,       "t_l3eta/D"); 
    tree->Branch("t_l3phi",       &t_l3phi,       "t_l3phi/D");
  }
  tree->Branch("t_p",           &t_p,           "t_p/D");
  tree->Branch("t_pt",          &t_pt,          "t_pt/D");
  tree->Branch("t_phi",         &t_phi,         "t_phi/D");
  tree->Branch("t_mindR1",      &t_mindR1,      "t_mindR1/D");
  tree->Branch("t_mindR2",      &t_mindR2,      "t_mindR2/D");
  tree->Branch("t_eMipDR",      &t_eMipDR,      "t_eMipDR/D");
  tree->Branch("t_eHcal",       &t_eHcal,       "t_eHcal/D");
  tree->Branch("t_eHcal10",     &t_eHcal10,     "t_eHcal10/D");
  tree->Branch("t_eHcal30",     &t_eHcal30,     "t_eHcal30/D");
  tree->Branch("t_hmaxNearP",   &t_hmaxNearP,   "t_hmaxNearP/D");
  tree->Branch("t_selectTk",    &t_selectTk,    "t_selectTk/O");
  tree->Branch("t_qltyFlag",    &t_qltyFlag,    "t_qltyFlag/O");
  tree->Branch("t_qltyMissFlag",&t_qltyMissFlag,"t_qltyMissFlag/O");
  tree->Branch("t_qltyPVFlag",  &t_qltyPVFlag,  "t_qltyPVFlag/O)");
  tree->Branch("t_gentrackP",   &t_gentrackP,   "t_gentrackP/D");

  t_DetIds       = new std::vector<unsigned int>();
  t_DetIds1      = new std::vector<unsigned int>();
  t_DetIds3      = new std::vector<unsigned int>();
  t_HitEnergies  = new std::vector<double>();
  t_HitEnergies1 = new std::vector<double>();
  t_HitEnergies3 = new std::vector<double>();
  t_trgbits      = new std::vector<bool>();
  tree->Branch("t_DetIds",      "std::vector<unsigned int>", &t_DetIds);
  tree->Branch("t_HitEnergies", "std::vector<double>",       &t_HitEnergies);
  if (((mode_/10)%10) == 1) {
    tree->Branch("t_trgbits",     "std::vector<bool>",         &t_trgbits); 
  }
  if ((mode_%10) == 1) {
    tree->Branch("t_DetIds1",      "std::vector<unsigned int>", &t_DetIds1);
    tree->Branch("t_DetIds3",      "std::vector<unsigned int>", &t_DetIds3);
    tree->Branch("t_HitEnergies1", "std::vector<double>",       &t_HitEnergies1);
    tree->Branch("t_HitEnergies3", "std::vector<double>",       &t_HitEnergies3);
  }
  tree2 = fs->make<TTree>("EventInfo", "Event Information");
     
  tree2->Branch("t_Tracks",      &t_Tracks,      "t_Tracks/I");
  tree2->Branch("t_TracksProp",  &t_TracksProp,  "t_TracksProp/I");
  tree2->Branch("t_TracksSaved", &t_TracksSaved, "t_TracksSaved/I");
  t_ietaAll     = new std::vector<int>();
  t_ietaGood    = new std::vector<int>();
  tree2->Branch("t_ietaAll",    "std::vector<int>",          &t_ietaAll);
  tree2->Branch("t_ietaGood",   "std::vector<int>",          &t_ietaGood);
}


// ------------ method called when starting to processes a run  ------------
void HcalIsoTrkAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed_(true);
  bool flag =  hltConfig_.init(iRun,iSetup,processName_,changed_);
  edm::LogInfo("HcalIsoTrack") << "Run[" << nRun_ << "] " << iRun.run() 
			       << " process " << processName_ << " init flag "
			       << flag << " change flag " << changed_;

  // check if trigger names in (new) config                       
  if (changed_) {
    changed_ = false;
#ifdef DebugLog
    edm::LogInfo("HcalIsoTrack") << "New trigger menu found !!!";
#endif
    const unsigned int n(hltConfig_.size());
    for (unsigned itrig=0; itrig<trigNames_.size(); itrig++) {
      unsigned int triggerindx = hltConfig_.triggerIndex(trigNames_[itrig]);
      if (triggerindx >= n) {
	edm::LogWarning("HcalIsoTrack") << trigNames_[itrig] << " " 
					<< triggerindx << " does not exist in "
					<< "the current menu";
#ifdef DebugLog
      } else {
	edm::LogInfo("HcalIsoTrack") << trigNames_[itrig] << " " 
				     << triggerindx << " exists";
#endif
      }
    }
  }
}

// ------------ method called when ending the processing of a run  ------------
void HcalIsoTrkAnalyzer::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun_++;
  edm::LogInfo("HcalIsoTrack") << "endRun[" << nRun_ << "] " << iRun.run();
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

int HcalIsoTrkAnalyzer::fillTree(std::vector< math::XYZTLorentzVector>& vecL1,
				 std::vector< math::XYZTLorentzVector>& vecL3,
				 math::XYZPoint& leadPV,
				 std::vector<spr::propagatedTrackDirection>& trkCaloDirections,
				 const CaloGeometry* geo,
				 edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle,
				 edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle,
				 edm::Handle<HBHERecHitCollection>& hbhe,
				 edm::Handle<reco::GenParticleCollection>& genParticles) {

  int nsave = 0;
  //Loop over tracks
  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
  unsigned int nTracks(0), nselTracks(0);
  t_nTrk = trkCaloDirections.size();
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
    t_mindR2 = 999;
    for (unsigned int k=0; k<vecL3.size(); ++k) {
      double dr   = dR(vecL3[k],v4); 
      if (dr<t_mindR2) {
	t_mindR2  = dr;
      }
    }
    t_mindR1 = (vecL1.size() > 0) ? dR(vecL1[0],v4) : 999;
#ifdef DebugLog
    edm::LogInfo("HcalIsoTrack") << "Closest L3 object at dr :" 
				 << t_mindR2 << " and from L1 " << t_mindR1;
#endif	    
    t_ieta         = 0;
    if (trkDetItr->okHCAL) {
      HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
      t_ieta = detId.ieta();
      if (t_p > 40.0 && t_p <= 60.0) t_ietaAll->push_back(t_ieta);
    }
    //Selection of good track
    t_selectTk = spr::goodTrack(pTrack,leadPV,selectionParameter_,false);
    spr::trackSelectionParameters oneCutParameters = selectionParameter_;
    oneCutParameters.maxDxyPV  = 10;
    oneCutParameters.maxDzPV   = 100;
    oneCutParameters.maxInMiss = 2;
    oneCutParameters.maxOutMiss= 2;
    bool qltyFlag  = spr::goodTrack(pTrack,leadPV,oneCutParameters,false);
    oneCutParameters           = selectionParameter_;
    oneCutParameters.maxDxyPV  = 10;
    oneCutParameters.maxDzPV   = 100;
    t_qltyMissFlag = spr::goodTrack(pTrack,leadPV,oneCutParameters,false);
    oneCutParameters           = selectionParameter_;
    oneCutParameters.maxInMiss = 2;
    oneCutParameters.maxOutMiss= 2;
    t_qltyPVFlag   = spr::goodTrack(pTrack,leadPV,oneCutParameters,false);
    double eIsolation = maxRestrictionP_*exp(slopeRestrictionP_*std::abs((double)t_ieta));
    if (eIsolation < eIsolate1_) eIsolation = eIsolate1_;
    if (eIsolation < eIsolate2_) eIsolation = eIsolate2_;
#ifdef DebugLog
    edm::LogInfo("HcalIsoTrack") << "qltyFlag|okECAL|okHCAL : " 
				 << qltyFlag << "|" << trkDetItr->okECAL
				 << "|" << trkDetItr->okHCAL
				 << " eIsolation " << eIsolation;
#endif
    t_qltyFlag = (qltyFlag && trkDetItr->okECAL && trkDetItr->okHCAL);
    if (t_qltyFlag) {
      nselTracks++;
      int nRH_eMipDR(0), nNearTRKs(0);
      t_eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, 
				 endcapRecHitsHandle, trkDetItr->pointHCAL,
				 trkDetItr->pointECAL, a_mipR_, 
				 trkDetItr->directionECAL, nRH_eMipDR);
      t_hmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections,
					     a_charIsoR_, nNearTRKs, false);
      t_gentrackP = trackP(pTrack, genParticles);
      if (t_eMipDR < eEcalMax_ && t_hmaxNearP < eIsolation) {
	t_DetIds->clear();  t_HitEnergies->clear();
	t_DetIds1->clear(); t_HitEnergies1->clear();
	t_DetIds3->clear(); t_HitEnergies3->clear();
	int nRecHits(-999), nRecHits1(-999), nRecHits3(-999);
	std::vector<DetId>  ids, ids1, ids3;
	t_eHcal       = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, 
					trkDetItr->pointECAL, a_coneR_, 
					trkDetItr->directionHCAL,nRecHits, 
					ids, *t_HitEnergies, useRaw_);
	t_eHcal      *= hcalScale_;
	double ehcal(0);
	for (unsigned int k=0; k<ids.size(); ++k) {
	  t_DetIds->push_back(ids[k].rawId());
	  ehcal += ((*t_HitEnergies)[k]);
	}
	ehcal        *= hcalScale_;
	if (std::abs(ehcal-t_eHcal) > 0.001) 
	  edm::LogWarning("HcalIsoTrack") << "Check inconsistent energies: "
					  << t_eHcal << ":" << ehcal
					  << " from " << t_DetIds->size()
					  << " cells\n";
	//----- hcal energy in the extended cone 1 (a_coneR+10) --------------
	t_eHcal10 = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL,
				    trkDetItr->pointECAL, a_coneR1_,
				    trkDetItr->directionHCAL,nRecHits1,
				    ids1, *t_HitEnergies1);
	t_eHcal10    *= hcalScale_;
	t_DetIds1->reserve(ids1.size());
	for (unsigned int k = 0; k < ids1.size(); ++k) {
	  t_DetIds1->push_back(ids1[k].rawId());
	}
	//----- hcal energy in the extended cone 3 (a_coneR+30) --------------
	t_eHcal30 = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL,
				    trkDetItr->pointECAL, a_coneR2_,
				    trkDetItr->directionHCAL,nRecHits3,
				    ids3, *t_HitEnergies3);
	t_eHcal30    *= hcalScale_;
	t_DetIds3->reserve(ids3.size());
	for (unsigned int k = 0; k < ids3.size(); ++k) {
	  t_DetIds3->push_back(ids3[k].rawId());
	}
	t_p           = pTrack->p();
	t_pt          = pTrack->pt();
	t_phi         = pTrack->phi();
	
#ifdef DebugLog
	edm::LogInfo("HcalIsoTrack") << "This track : " << nTracks 
				     << " (pt|eta|phi|p) :"  << t_pt
				     << "|" << pTrack->eta() << "|" 
				     << t_phi << "|" << t_p
				     << " Generator Level p " << t_gentrackP;
	edm::LogInfo("HcalIsoTrack") << "e_MIP " << t_eMipDR 
				     << " Chg Isolation " << t_hmaxNearP 
				     << " eHcal" << t_eHcal << " ieta " 
				     << t_ieta << " Quality " << t_qltyMissFlag
				     << ":" << t_qltyPVFlag << ":"<< t_selectTk;
	for (unsigned int lll=0;lll<t_DetIds->size();lll++) {
	  edm::LogInfo("HcalIsoTrack") << "det id is = " << t_DetIds->at(lll)
				       << "   hit enery is  = "  
				       << t_HitEnergies->at(lll);
	}
	for (unsigned int lll=0;lll<t_DetIds1->size();lll++) {
	  edm::LogInfo("HcalIsoTrack") << "det id is = " << t_DetIds1->at(lll)
				       << "   hit enery is  = "  
				       << t_HitEnergies1->at(lll);
	}
	for (unsigned int lll=0;lll<t_DetIds3->size();lll++) {
	  edm::LogInfo("HcalIsoTrack") << "det id is = " << t_DetIds3->at(lll)
				       << "   hit enery is  = "  
				       << t_HitEnergies3->at(lll);
	}
#endif
	if (t_p>pTrackMin_) {
	  tree->Fill();
	  nsave++;
	  if (t_p > 40.0 && t_p <= 60.0 && t_selectTk) 
	    t_ietaGood->push_back(t_ieta);
#ifdef DebugLog
	  for (unsigned int k=0; k<t_trgbits->size(); k++) {
	    edm::LogInfo("HcalIsoTrack") << "trigger bit is  = " 
					 << t_trgbits->at(k);
	  }
#endif
	}
      }
    }
  }
  return nsave;
}

double HcalIsoTrkAnalyzer::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return reco::deltaR(vec1.eta(),vec1.phi(),vec2.eta(),vec2.phi());
}

double HcalIsoTrkAnalyzer::trackP(const reco::Track* pTrack,
				  const edm::Handle<reco::GenParticleCollection>& genParticles) {

  double pmom = -1.0;
  if (genParticles.isValid()) {
    reco::GenParticleCollection::const_iterator p;
    double mindR(999.9);
    for (p=genParticles->begin(); p!=genParticles->end(); ++p) {
      double dR = reco::deltaR(pTrack->eta(), pTrack->phi(),
			       p->momentum().Eta(), p->momentum().Phi());
      if (dR < mindR) {
	mindR = dR; pmom = p->momentum().R();
      }
    }
  }
  return pmom;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalIsoTrkAnalyzer);

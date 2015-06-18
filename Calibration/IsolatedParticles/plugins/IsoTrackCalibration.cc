//#define DebugLog
// system include files
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
// Jets
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

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
 
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);

  bool                       changed;
  edm::Service<TFileService> fs;
  HLTConfigProvider          hltConfig_;
  std::vector<std::string>   trigNames, HLTNames;
  std::vector<int>           trigKount, trigPass;
  int                        verbosity;
  spr::trackSelectionParameters selectionParameters;
  std::string                theTrackQuality, processName;
  std::string                l1Filter, l2Filter, l3Filter;
  double                     a_mipR, a_coneR, a_charIsoR;
  double                     pTrackMin_, eEcalMax_, eIsolation_;
  bool                       qcdMC;
  int                        nRun, nAll, nGood;
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
  edm::EDGetTokenT<reco::GenJetCollection> tok_jets_;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo> > tok_pu_;

  TTree                     *tree;
  int                        t_Run, t_Event, t_ieta; 
  double                     t_EventWeight, t_l1pt, t_l1eta, t_l1phi;
  double                     t_l3pt, t_l3eta, t_l3phi, t_p, t_mindR1;
  double                     t_mindR2, t_eMipDR, t_eHcal, t_hmaxNearP;
  double                     t_npvtruth, t_npvObserved;
  bool                       t_selectTk,t_qltyFlag,t_qltyMissFlag,t_qltyPVFlag;
  std::vector<bool>         *t_trgbits; 
  std::vector<unsigned int> *t_DetIds;
  std::vector<double>       *t_HitEnergies, pbin, vbin;
  TProfile                  *h_RecHit_iEta, *h_RecHit_num;
  TH1I                      *h_iEta, *h_tketa0[5], *h_tketa1[5], *h_tketa2[5];
  TH1I                      *h_tketa3[5], *h_tketa4[5], *h_tketa5[5];
  TH1F                      *h_Rechit_E, *h_jetp;
  TH1F 			    *h_jetpt[4];
  TH1I                      *h_tketav1[5][6], *h_tketav2[5][6];
};

IsoTrackCalibration::IsoTrackCalibration(const edm::ParameterSet& iConfig) : 
  changed(false), nRun(0), nAll(0), nGood(0), 
  t_trgbits(0), t_DetIds(0), t_HitEnergies(0) {
  //now do whatever initialization is needed
  verbosity                           = iConfig.getUntrackedParameter<int>("Verbosity",0);
  trigNames                           = iConfig.getUntrackedParameter<std::vector<std::string> >("Triggers");
  theTrackQuality                     = iConfig.getUntrackedParameter<std::string>("TrackQuality","highPurity");
  processName                         = iConfig.getUntrackedParameter<std::string>("ProcessName","HLT");
  l1Filter                            = iConfig.getUntrackedParameter<std::string>("L1Filter", "hltL1sL1SingleJet");
  l2Filter                            = iConfig.getUntrackedParameter<std::string>("L2Filter", "L2Filter");
  l3Filter                            = iConfig.getUntrackedParameter<std::string>("L3Filter", "Filter");
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
  pTrackMin_                          = iConfig.getUntrackedParameter<double>("MinimumTrackP",20.0);
  eEcalMax_                           = iConfig.getUntrackedParameter<double>("MaximumEcalEnergy",2.0);
  eIsolation_                         = iConfig.getUntrackedParameter<double>("IsolationEnergy",2.0);
  qcdMC                               = iConfig.getUntrackedParameter<bool>("IsItQCDMC", true);
  bool isItAOD                        = iConfig.getUntrackedParameter<bool>("IsItAOD", false);
  triggerEvent_                       = edm::InputTag("hltTriggerSummaryAOD","",processName);
  theTriggerResultsLabel              = edm::InputTag("TriggerResults","",processName);

  // define tokens for access
  tok_trigEvt   = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes   = consumes<edm::TriggerResults>(theTriggerResultsLabel);
  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_   = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_       = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  tok_ew_       = consumes<GenEventInfoProduct>(edm::InputTag("generator")); 
  tok_pu_       = consumes<std::vector<PileupSummaryInfo>>(iConfig.getParameter<edm::InputTag>("PUinfo"));
 
  if (isItAOD) {
    tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));
    tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("reducedHcalRecHits", "hbhereco"));
  } else {
    tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
    tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEE"));
    tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  }
  tok_jets_     = consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("JetSource"));

  std::vector<int>dummy(trigNames.size(),0);
  trigKount = trigPass = dummy;
#ifdef DebugLog
  if (verbosity>=0) {
    std::cout <<"Parameters read from config file \n" 
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
	      <<"\t qcdMC "           << qcdMC
	      << std::endl;
    std::cout << "Process " << processName << " L1Filter:" << l1Filter
	      << " L2Filter:" << l2Filter << " L3Filter:" << l3Filter
	      << std::endl;
    std::cout << trigNames.size() << " triggers to be studied";
    for (unsigned int k=0; k<trigNames.size(); ++k)
      std::cout << ": " << trigNames[k];
    std::cout << std::endl;
  }
#endif
}

IsoTrackCalibration::~IsoTrackCalibration() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  if (t_trgbits)     delete t_trgbits;
  if (t_DetIds)      delete t_DetIds;
  if (t_HitEnergies) delete t_HitEnergies;

}

void IsoTrackCalibration::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup) {

  t_Run   = iEvent.id().run();
  t_Event = iEvent.id().event();
  nAll++;
#ifdef DebugLog
  if (verbosity%10 > 0) 
    std::cout << "Run " << t_Run << " Event " << t_Event << " Luminosity " 
	      << iEvent.luminosityBlock() << " Bunch " << iEvent.bunchCrossing()
	      << " starts ==========" << std::endl;
#endif
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
 
  //event weight for FLAT sample and PU information
  t_EventWeight = 1.0;
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(tok_ew_, genEventInfo);
  if (genEventInfo.isValid()) t_EventWeight = genEventInfo->weight();  
  t_npvtruth = t_npvObserved = 999;
  edm::Handle<std::vector< PileupSummaryInfo > >  PupInfo;
  iEvent.getByToken(tok_pu_, PupInfo);
  if (PupInfo.isValid()) {
    std::vector<PileupSummaryInfo>::const_iterator PVI;
    for (PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
      int BX = PVI->getBunchCrossing();
      if (BX==0) {
	t_npvtruth    = PVI->getTrueNumInteractions();
	t_npvObserved = PVI->getPU_NumInteractions();
	break;
      }
    }
  }
  int ivtx = (int)(vbin.size()-2);
  for (unsigned int iv=1; iv<vbin.size(); ++iv) {
#ifdef DebugLog
    std::cout << "Bin " << iv << " " << vbin[iv-1] << ":" << vbin[iv] << std::endl;
#endif
    if (t_npvtruth <= vbin[iv]) {
      ivtx = iv-1; break;
    }
  }
#ifdef DebugLog
  if (verbosity == 0) 
    std::cout << "PU Vertex " << t_npvtruth << "/" << t_npvObserved << " IV "
	      << ivtx << ":" << vbin.size() << std::endl;
#endif
  //=== genJet information
  edm::Handle<reco::GenJetCollection> genJets;
  iEvent.getByToken(tok_jets_, genJets);
  if (genJets.isValid()) {
    for (unsigned iGenJet = 0; iGenJet < genJets->size(); ++iGenJet) {
      const reco::GenJet& genJet = (*genJets) [iGenJet];
      double genJetPt  = genJet.pt();
      double genJetEta = genJet.eta();
      h_jetpt[0]->Fill(genJetPt);
      h_jetpt[1]->Fill(genJetPt,t_EventWeight);
      if (genJetEta>-2.5 && genJetEta<2.5) {
	h_jetpt[2]->Fill(genJetPt);
	h_jetpt[3]->Fill(genJetPt,t_EventWeight);
      }
    }
  }

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
#ifdef DebugLog
  if ((verbosity/100)%10>0) {
    std::cout << "Primary Vertex " << leadPV;
    if (beamSpotH.isValid()) std::cout << " Beam Spot " 
				       << beamSpotH->position();
    std::cout << std::endl;
  }
#endif  
  // RecHits
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);
  HBHERecHitCollection::const_iterator rhitItr;

  for (rhitItr=hbhe->begin();rhitItr!=hbhe->end();rhitItr++) {
    double rec_energy = rhitItr->energy();
    int    rec_ieta   = rhitItr->id().ieta();
    int    rec_depth  = rhitItr->id().depth();
    int    rec_zside  = rhitItr->id().zside();
    double num1_1     = rec_zside*(rec_ieta+0.2*(rec_depth-1));
#ifdef DebugLog
    if (verbosity%10>0)
      std::cout << "detid/rechit/ieta/zside/depth/num  = " << rhitItr->id() 
		<< "/" << rec_energy << "/" << rec_ieta << "/" << rec_zside 
		<< "/" << rec_depth << "/" << num1_1 << std::endl;
#endif
    h_iEta->Fill(rec_ieta);
    h_Rechit_E->Fill(rec_energy);
    h_RecHit_iEta->Fill(rec_ieta,rec_energy);
    h_RecHit_num->Fill(num1_1,rec_energy);
  }
	  
  //Propagate tracks to calorimeter surface)
  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality,
		     trkCaloDirections, ((verbosity/100)%10>2));
  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
  for (trkDetItr = trkCaloDirections.begin(); 
       trkDetItr != trkCaloDirections.end(); trkDetItr++) {
    if (trkDetItr->okHCAL) {
      HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
      int tk_ieta = detId.ieta();
      const reco::Track* pTrack = &(*(trkDetItr->trkItr));
      double tk_p = pTrack->p();
      h_tketa0[0]->Fill(tk_ieta);
      h_tketav1[ivtx][0]->Fill(tk_ieta);
#ifdef DebugLog
      std::cout << "Fill for " << tk_ieta << " in " << ivtx << ":" 
		<<  h_tketav1[ivtx][0]->GetName() << std::endl;
#endif
      for (unsigned int k=1; k<pbin.size(); ++k) {
	if (tk_p >= pbin[k-1] && tk_p < pbin[k]) {
	  h_tketa0[k]->Fill(tk_ieta);
	  h_tketav1[ivtx][k]->Fill(tk_ieta);
#ifdef DebugLog
	  std::cout << "Fill for " << tk_ieta << ":" << tk_p << " in " << ivtx
		    << ":" <<  h_tketav1[ivtx][k]->GetName() << std::endl;
#endif
	  break;
	}
      }
    }
  }

  //Trigger
  t_trgbits->clear();
  for (unsigned int i=0; i<trigNames.size(); ++i) t_trgbits->push_back(false);

  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
#ifdef DebugLog
    std::cout << "Error! Can't get the product "<< triggerEvent_.label() 
	      << std::endl;
#endif
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
	    t_trgbits->at(i) = (hlt>0);
	    trigKount[i]++;
	    if (hlt > 0) {
	      ok = true;
	      trigPass[i]++;
	    }
#ifdef DebugLog
	    if (verbosity%10 > 0)
	      std::cout << "This is the trigger we are looking for "
			<< triggerNames_[iHLT] << " Flag " << hlt << ":"
			<< ok << std::endl;
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
		if (verbosity%10 > 0) std::cout << "FilterName " << label << std::endl;
#endif
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
		    if (label.find(l2Filter) != std::string::npos) {
		      vec[1].push_back(v4);
		    } else if (label.find(l3Filter) != std::string::npos) {
		      vec[2].push_back(v4);
		    } else if (label.find(l1Filter) != std::string::npos ||
			       l1Filter == "") {
		      vec[0].push_back(v4);
		    }
		  }
#ifdef DebugLog
		  if (verbosity%10 > 2)
		    std::cout << "key " << ifiltrKey << " : pt " << TO.pt() 
			      << " eta " << TO.eta() << " phi " << TO.phi() 
			      << " mass " << TO.mass() << " Id " << TO.id() 
			      << std::endl;
#endif
		}
#ifdef DebugLog
		if (verbosity%10 > 0) std::cout << "sizes " << vec[0].size() 
						<< ":" << vec[1].size() << ":" 
						<< vec[2].size() << std::endl;
#endif
	      }
	    }
	  }
	  double dr;
	  //// dR for leading L1 object with L2 objects
	  math::XYZTLorentzVector mindRvec1;
	  double mindR1(999);
	  for (int lvl=1; lvl<3; lvl++) {
	    for (unsigned int i=0; i<vec[lvl].size(); i++) {
	      dr   = dR(vec[0][0],vec[lvl][i]);
#ifdef DebugLog
	      if (verbosity%10 > 2)  std::cout << "lvl " <<lvl << " i " << i 
					       << " dR " << dr << std::endl;
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
	    if (verbosity%10> 0) 
	      std::cout << "This track : " << nTracks << " (pt/eta/phi/p) :" 
			<< pTrack->pt() << "/" << pTrack->eta() << "/" 
			<< pTrack->phi() << "/" << pTrack->p() << std::endl;
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
	    if (verbosity%10> 2)
	      std::cout << "Closest L3 object at mindr :" << t_mindR2 << " is "
			<< mindRvec2 << " and from L1 " << t_mindR1 <<std::endl;
#endif
	    
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
#ifdef DebugLog
	    if (verbosity%10 > 0) 
	      std::cout << "qltyFlag|okECAL|okHCAL : " << qltyFlag << "|" 
			<< trkDetItr->okECAL << "/" << trkDetItr->okHCAL 
			<< std::endl;
#endif
	    t_qltyFlag = (qltyFlag && trkDetItr->okECAL && trkDetItr->okHCAL);
	    t_p        = pTrack->p();
	    h_tketa1[0]->Fill(t_ieta);
	    for (unsigned int k=1; k<pbin.size(); ++k) {
	      if (t_p >= pbin[k-1] && t_p < pbin[k]) {
		h_tketa1[k]->Fill(t_ieta);
		break;
	      }
	    }
	    if (t_qltyFlag) {
              nselTracks++;
	      h_tketa2[0]->Fill(t_ieta);
	      for (unsigned int k=1; k<pbin.size(); ++k) {
		if (t_p >= pbin[k-1] && t_p < pbin[k]) {
		  h_tketa2[k]->Fill(t_ieta);
		  break;
		}
	      }
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
	      if (t_hmaxNearP < 2) {
		h_tketa3[0]->Fill(t_ieta);
		for (unsigned int k=1; k<pbin.size(); ++k) {
		  if (t_p >= pbin[k-1] && t_p < pbin[k]) {
		    h_tketa3[k]->Fill(t_ieta);
		    break;
		  }
		}
		if (t_eMipDR < 1) {
		  h_tketa4[0]->Fill(t_ieta);
		  for (unsigned int k=1; k<pbin.size(); ++k) {
		    if (t_p >= pbin[k-1] && t_p < pbin[k]) {
		      h_tketa4[k]->Fill(t_ieta);
		      break;
		    }
		  }
		  if (t_mindR1 > 1) {
		    h_tketa5[0]->Fill(t_ieta);
		    h_tketav2[ivtx][0]->Fill(t_ieta);
		    for (unsigned int k=1; k<pbin.size(); ++k) {
		      if (t_p >= pbin[k-1] && t_p < pbin[k]) {
			h_tketa5[k]->Fill(t_ieta);
			h_tketav2[ivtx][k]->Fill(t_ieta);
			break;
		      }
		    }
		  }
		}
	      }
#ifdef DebugLog
	      if (verbosity%10 > 0) {
		std::cout << "This track : " << nTracks << " (pt/eta/phi/p) :" 
			  << pTrack->pt() << "/" << pTrack->eta() << "/" 
			  << pTrack->phi() << "/" << t_p << std::endl;
		std::cout << "e_MIP " << t_eMipDR << " Chg Isolation "
			  << t_hmaxNearP << " eHcal" << t_eHcal << " ieta " 
			  << t_ieta << " Quality " << t_qltyMissFlag
			  << ":" << t_qltyPVFlag << ":" << t_selectTk 
			  << std::endl;
		for (unsigned int lll=0;lll<t_DetIds->size();lll++) {
		  std::cout << "det id is = " << t_DetIds->at(lll)  << "  " 
			    << " hit enery is  = "  << t_HitEnergies->at(lll) 
			    << std::endl;         
		}
	      }
#endif
	      if (t_p>pTrackMin_ && t_eMipDR<eEcalMax_ && 
		  t_hmaxNearP<eIsolation_) {
#ifdef DebugLog
 	        if (verbosity%10> 2) {
		  for (unsigned int k=0; k<t_trgbits->size(); k++) 
		    std::cout <<"trigger bit is  = " << t_trgbits->at(k) 
			      << std::endl; 
		}
#endif
		tree->Fill();
		nGood++;
	      }
	    }
	  }
	}
      }
      t_trgbits->clear(); t_DetIds->clear(); t_HitEnergies->clear();
      // check if trigger names in (new) config                       
      if (changed) {
	changed = false;
#ifdef DebugLog
	if (verbosity%10> 1) {
	  std::cout<<"New trigger menu found !!!" << std::endl;
	  const unsigned int n(hltConfig_.size());
	  for (unsigned itrig=0; itrig<triggerNames_.size(); itrig++) {
	    unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[itrig]);
	    std::cout << triggerNames_[itrig] << " " << triggerindx << " ";
	    if (triggerindx >= n)
	      std::cout << "does not exist in the current menu" << std::endl;
	    else
	      std::cout << "exists" << std::endl;
	  }
	}
#endif
      }
    }
  }
}

void IsoTrackCalibration::beginJob() {
  h_RecHit_iEta = fs->make<TProfile>("rechit_ieta","Rec hit vs. ieta",60,-30,30,0,1000);
  h_RecHit_num = fs->make<TProfile>("rechit_num","Rec hit vs. num",100,0,20,0,1000);
  h_iEta       = fs->make<TH1I>("iEta","iEta",60,-30,30);
  h_Rechit_E   = fs->make<TH1F>("Rechit_E","Rechit_E",100,0,1000);

  double prange[5] = {20,30,40,60,100};
  for (int k=0; k<5; ++k) pbin.push_back(prange[k]);
  std::string type[6] = {"All", "Trigger OK", "Tree Selected", 
			 "Charge Isolation", "MIP Cut", "L1 Cut"};
  double vrange[6] = {0, 10, 20, 30, 40, 1000};
  for (int k=0; k<6; ++k) vbin.push_back(vrange[k]);
  char name[20], namp[20], title[100];
  for (unsigned int k=0; k<pbin.size(); ++k) {
    if (k == 0) sprintf (namp, "all momentum");
    else        sprintf (namp, "p = %4.0f:%4.0f GeV", pbin[k-1], pbin[k]);
    sprintf (name, "TrackEta0%d", k);
    sprintf (title, "Track #eta for tracks with %s (%s)",namp,type[0].c_str());
    h_tketa0[k] = fs->make<TH1I>(name, title, 60, -30, 30);
    sprintf (name, "TrackEta1%d", k);
    sprintf (title, "Track #eta for tracks with %s (%s)",namp,type[1].c_str());
    h_tketa1[k] = fs->make<TH1I>(name, title, 60, -30, 30);
    sprintf (name, "TrackEta2%d", k);
    sprintf (title, "Track #eta for tracks with %s (%s)",namp,type[2].c_str());
    h_tketa2[k] = fs->make<TH1I>(name, title, 60, -30, 30);
    sprintf (name, "TrackEta3%d", k);
    sprintf (title, "Track #eta for tracks with %s (%s)",namp,type[3].c_str());
    h_tketa3[k] = fs->make<TH1I>(name, title, 60, -30, 30);
    sprintf (name, "TrackEta4%d", k);
    sprintf (title, "Track #eta for tracks with %s (%s)",namp,type[4].c_str());
    h_tketa4[k] = fs->make<TH1I>(name, title, 60, -30, 30);
    sprintf (name, "TrackEta5%d", k);
    sprintf (title, "Track #eta for tracks with %s (%s)",namp,type[5].c_str());
    h_tketa5[k] = fs->make<TH1I>(name, title, 60, -30, 30);
    for (unsigned int l=0; l<vbin.size()-1; ++l) {
      int v1 = (int)(vbin[l]);
      int v2 = (int)(vbin[l+1]);
      sprintf (name, "TrackEta1%dVtx%d", k, l);
      sprintf (title, "Track #eta for tracks with %s (%s and PU %d:%d)",namp,type[0].c_str(),v1,v2);
      h_tketav1[l][k] = fs->make<TH1I>(name, title, 60, -30, 30);
      sprintf (name, "TrackEta2%dVtx%d", k, l);
      sprintf (title, "Track #eta for tracks with %s (%s and PU %d:%d)",namp,type[5].c_str(),v1,v2);
      h_tketav2[l][k] = fs->make<TH1I>(name, title, 60, -30, 30);
    }
  }
  h_jetpt[0] = fs->make<TH1F>("Jetpt0","Jet p_T (All)", 500,0.,2500.);
  h_jetpt[1] = fs->make<TH1F>("Jetpt1","Jet p_T (All Weighted)", 500,0.,2500.);
  h_jetpt[2] = fs->make<TH1F>("Jetpt2","Jet p_T (|#eta| < 2.5)", 500,0.,2500.);
  h_jetpt[3] = fs->make<TH1F>("Jetpt3","Jet p_T (|#eta| < 2.5 Weighted)", 500,0.,2500.);
  tree = fs->make<TTree>("CalibTree", "CalibTree");

     
  tree->Branch("t_Run",         &t_Run,         "t_Run/I");
  tree->Branch("t_Event",       &t_Event,       "t_Event/I");
  tree->Branch("t_ieta",        &t_ieta,        "t_ieta/I");
  tree->Branch("t_EventWeight", &t_EventWeight, "t_EventWeight/D");
  tree->Branch("t_npvtruth",    &t_npvtruth,    "t_npvtruth/D");
  tree->Branch("t_npvObserved", &t_npvObserved, "t_npvObserved/D");
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
  tree->Branch("t_trgbits",     "std::vector<bool>",         &t_trgbits); 
}

// ------------ method called once each job just after ending the event loop  ------------
void IsoTrackCalibration::endJob() {
  edm::LogWarning("IsoTrack") << "Finds " << nGood << " good tracks in " 
			      << nAll << " events from " << nRun << " runs";
  for (unsigned int k=0; k<trigNames.size(); ++k)
    edm::LogWarning("IsoTrack") << "Trigger[" << k << "]: " << trigNames[k] 
				<< " Events " << trigKount[k] << " Passed " 
				<< trigPass[k];
}

// ------------ method called when starting to processes a run  ------------
void IsoTrackCalibration::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogWarning("IsoTrack") << "Run[" << nRun <<"] " << iRun.run()
			      << " hltconfig.init " << hltConfig_.init(iRun,iSetup,processName,changed);
}

// ------------ method called when ending the processing of a run  ------------
void IsoTrackCalibration::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun++;
  edm::LogWarning("IsoTrack") << "endRun[" << nRun << "] " << iRun.run();
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

double IsoTrackCalibration::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return reco::deltaR(vec1.eta(),vec1.phi(),vec2.eta(),vec2.phi());
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsoTrackCalibration);


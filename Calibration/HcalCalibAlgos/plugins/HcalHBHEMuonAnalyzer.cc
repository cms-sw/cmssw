#include <memory>
#include <iostream>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include "TPRegexp.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

//////////////trigger info////////////////////////////////////

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h" 
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h" 
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h" 
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h" 

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//#define EDM_ML_DEBUG

class HcalHBHEMuonAnalyzer : public edm::EDAnalyzer {

public:
  explicit HcalHBHEMuonAnalyzer(const edm::ParameterSet&);
  ~HcalHBHEMuonAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
   
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  virtual void endJob() ;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  void   clearVectors();
  int    matchId(const HcalDetId&, const HcalDetId&);
  double activeLength(const DetId&);
  bool   isGoodVertex(const reco::Vertex& vtx);
  // ----------member data ---------------------------
  HLTConfigProvider          hltConfig_;
  edm::Service<TFileService> fs;
  edm::InputTag              HLTriggerResults_;
  std::string                labelEBRecHit_, labelEERecHit_;
  std::string                labelVtx_, labelHBHERecHit_, labelMuon_;
  int                        verbosity_, maxDepth_, kount_;
  bool                       useRaw_;
  const int                  MaxDepth=7;

  edm::EDGetTokenT<edm::TriggerResults>                   tok_trigRes_;
  edm::EDGetTokenT<reco::VertexCollection>                tok_Vtx_;
  edm::EDGetTokenT<EcalRecHitCollection>                  tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>                  tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>                  tok_HBHE_;
  edm::EDGetTokenT<reco::MuonCollection>                  tok_Muon_;

  //////////////////////////////////////////////////////
  std::vector<double>       ptGlob_, etaGlob_, phiGlob_, chiGlobal_;
  std::vector<double>       globalMuonHits_,matchedStat_,globalTrckPt_;
  std::vector<double>       globalTrckEta_,globalTrckPhi_,trackerLayer_;
  std::vector<double>       innerTrackpt_,innerTracketa_,innerTrackphi_;
  std::vector<double>       matchedId_,numPixelLayers_;
  std::vector<double>       chiTracker_,dxyTracker_,dzTracker_;
  std::vector<double>       outerTrackPt_,outerTrackEta_,outerTrackPhi_;
  std::vector<double>       outerTrackChi_,outerTrackHits_,outerTrackRHits_;
  std::vector<double>       tight_LongPara_,tight_PixelHits_,tight_TransImpara_;
  std::vector<bool>         innerTrack_, outerTrack_, globalTrack_;
  std::vector<double>       isolationR04_,isolationR03_;
  std::vector<double>       energyMuon_,hcalEnergy_,ecalEnergy_,hoEnergy_;
  std::vector<double>       ecal3x3Energy_,hcal1x1Energy_, pMuon_, hcalHot_;
  std::vector<unsigned int> ecalDetId_,hcalDetId_,ehcalDetId_;
  std::vector<double>       hcalDepth1Energy_, hcalDepth1ActiveLength_;
  std::vector<double>       hcalDepth2Energy_, hcalDepth2ActiveLength_;
  std::vector<double>       hcalDepth3Energy_, hcalDepth3ActiveLength_;
  std::vector<double>       hcalDepth4Energy_, hcalDepth4ActiveLength_;
  std::vector<double>       hcalDepth5Energy_, hcalDepth5ActiveLength_;
  std::vector<double>       hcalDepth6Energy_, hcalDepth6ActiveLength_;
  std::vector<double>       hcalDepth7Energy_, hcalDepth7ActiveLength_;
  std::vector<double>       hcalDepth1EnergyHot_, hcalDepth1ActiveLengthHot_;
  std::vector<double>       hcalDepth2EnergyHot_, hcalDepth2ActiveLengthHot_;
  std::vector<double>       hcalDepth3EnergyHot_, hcalDepth3ActiveLengthHot_;
  std::vector<double>       hcalDepth4EnergyHot_, hcalDepth4ActiveLengthHot_;
  std::vector<double>       hcalDepth5EnergyHot_, hcalDepth5ActiveLengthHot_;
  std::vector<double>       hcalDepth6EnergyHot_, hcalDepth6ActiveLengthHot_;
  std::vector<double>       hcalDepth7EnergyHot_, hcalDepth7ActiveLengthHot_;
  std::vector<double>       hcalActiveLength_,    hcalActiveLengthHot_;
  std::vector<HcalDDDRecConstants::HcalActiveLength> actHB, actHE;
  std::vector<std::string>  all_triggers;
  ////////////////////////////////////////////////////////////

  TTree                    *tree_;
  std::vector<bool>         muon_is_good_, muon_global_, muon_tracker_;
  std::vector<int>          hltresults;
  unsigned int              runNumber_, eventNumber_ , lumiNumber_, bxNumber_;
 };

HcalHBHEMuonAnalyzer::HcalHBHEMuonAnalyzer(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  kount_            = 0;
  HLTriggerResults_ = iConfig.getParameter<edm::InputTag>("HLTriggerResults");
  labelVtx_         = iConfig.getParameter<std::string>("LabelVertex");
  labelEBRecHit_    = iConfig.getParameter<std::string>("LabelEBRecHit");
  labelEERecHit_    = iConfig.getParameter<std::string>("LabelEERecHit");
  labelHBHERecHit_  = iConfig.getParameter<std::string>("LabelHBHERecHit");
  labelMuon_        = iConfig.getParameter<std::string>("LabelMuon");
  verbosity_        = iConfig.getUntrackedParameter<int>("Verbosity",0);
  maxDepth_         = iConfig.getUntrackedParameter<int>("MaxDepth",4);
  if (maxDepth_ > MaxDepth) maxDepth_ = MaxDepth;
  else if (maxDepth_ < 1)   maxDepth_ = 4;
  std::string modnam = iConfig.getUntrackedParameter<std::string>("ModuleName","");
  std::string procnm = iConfig.getUntrackedParameter<std::string>("ProcessName","");
  useRaw_            = iConfig.getUntrackedParameter<bool>("UseRaw",false);

  tok_trigRes_  = consumes<edm::TriggerResults>(HLTriggerResults_);
  if (modnam == "") {
    tok_Vtx_      = consumes<reco::VertexCollection>(labelVtx_);
    tok_EB_       = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit",labelEBRecHit_));
    tok_EE_       = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit",labelEERecHit_));
    tok_HBHE_     = consumes<HBHERecHitCollection>(labelHBHERecHit_);
    tok_Muon_     = consumes<reco::MuonCollection>(labelMuon_);
    edm::LogInfo("HBHEMuon")  << "Labels used " << HLTriggerResults_ << " "
			      << labelVtx_ << " " << labelEBRecHit_ << " "
			      << labelEERecHit_ << " " << labelHBHERecHit_
			      << " " << labelMuon_;
  } else {
    tok_Vtx_      = consumes<reco::VertexCollection>(edm::InputTag(modnam,labelVtx_,procnm));
    tok_EB_       = consumes<EcalRecHitCollection>(edm::InputTag(modnam,labelEBRecHit_,procnm));
    tok_EE_       = consumes<EcalRecHitCollection>(edm::InputTag(modnam,labelEERecHit_,procnm));
    tok_HBHE_     = consumes<HBHERecHitCollection>(edm::InputTag(modnam,labelHBHERecHit_,procnm));
    tok_Muon_     = consumes<reco::MuonCollection>(edm::InputTag(modnam,labelMuon_,procnm));
    edm::LogInfo("HBHEMuon")   << "Labels used "   << HLTriggerResults_
			       << "\n            " << edm::InputTag(modnam,labelVtx_,procnm)
			       << "\n            " << edm::InputTag(modnam,labelEBRecHit_,procnm)
			       << "\n            " << edm::InputTag(modnam,labelEERecHit_,procnm)
			       << "\n            " << edm::InputTag(modnam,labelHBHERecHit_,procnm)
			       << "\n            " << edm::InputTag(modnam,labelMuon_,procnm);
  }
}

HcalHBHEMuonAnalyzer::~HcalHBHEMuonAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void HcalHBHEMuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ++kount_;
  clearVectors();
  // depthHE is the first depth index for HE for |ieta| = 16
  // It used to be 3 for all runs preceding 2017 and 4 beyond that
  int depthHE = (maxDepth_ <= 6) ? 3 : 4;
  runNumber_   = iEvent.id().run();
  eventNumber_ = iEvent.id().event();
  lumiNumber_  = iEvent.id().luminosityBlock();
  bxNumber_    = iEvent.bunchCrossing();
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HBHEMuon") << "Run " << runNumber_ << " Event " << eventNumber_
			   << " Lumi " << lumiNumber_ << " BX " << bxNumber_
			   << std::endl;
#endif  
  edm::Handle<edm::TriggerResults> _Triggers;
  iEvent.getByToken(tok_trigRes_, _Triggers); 
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HBHEMuon") << "Size of all triggers "  
			   << all_triggers.size() << std::endl;
#endif
  int Ntriggers = all_triggers.size();
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HBHEMuon") << "Size of HLT MENU: " << _Triggers->size()
			   << std::endl;
#endif
  if (_Triggers.isValid()) {
    const edm::TriggerNames &triggerNames_ = iEvent.triggerNames(*_Triggers);
    std::vector<int> index;
    for (int i=0; i<Ntriggers; i++) {
      index.push_back(triggerNames_.triggerIndex(all_triggers[i]));
      int triggerSize = int( _Triggers->size());
#ifdef EDM_ML_DEBUG
      edm::LogInfo("HBHEMuon") << "outside loop " << index[i]
			       << "\ntriggerSize " << triggerSize
			       << std::endl;
#endif
      if (index[i] < triggerSize) {
	hltresults.push_back(_Triggers->accept(index[i]));
#ifdef EDM_ML_DEBUG
	edm::LogInfo("HBHEMuon") << "Trigger_info " << triggerSize
				 << " triggerSize " << index[i]
				 << " trigger_index " << hltresults.at(i)
				 << " hltresult" << std::endl;
#endif
      } else {
	edm::LogInfo("HBHEMuon") << "Requested HLT path \"" 
				 << "\" does not exist\n";
      }
    }
  }

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  
  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField* bField = bFieldH.product();
  
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus);
  const EcalChannelStatus* theEcalChStatus = ecalChStatus.product();
  
  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  const CaloTopology *caloTopology = theCaloTopology.product();
  
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();

  // Relevant blocks from iEvent
  edm::Handle<reco::VertexCollection> vtx;
  iEvent.getByToken(tok_Vtx_, vtx);
  
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_HBHE_, hbhe);
  
  edm::Handle<reco::MuonCollection> _Muon;
  iEvent.getByToken(tok_Muon_, _Muon);
  
  // get handles to calogeometry and calotopology
  if (!(vtx.isValid()))                  return;
  reco::VertexCollection::const_iterator firstGoodVertex = vtx->end();
  for (reco::VertexCollection::const_iterator it = vtx->begin(); it != firstGoodVertex; it++) {
    if (isGoodVertex(*it)) {
      firstGoodVertex = it;
      break;
    }
  }
  // require a good vertex
  if (firstGoodVertex == vtx->end())     return;
  
  bool accept(false);
  if (_Muon.isValid() && barrelRecHitsHandle.isValid() && 
      endcapRecHitsHandle.isValid() && hbhe.isValid()) { 
    for (reco::MuonCollection::const_iterator RecMuon = _Muon->begin(); RecMuon!= _Muon->end(); ++RecMuon)  {
      
      if (RecMuon->p() > 10.0) accept = true;

      muon_is_good_.push_back(RecMuon->isPFMuon());
      muon_global_.push_back(RecMuon->isGlobalMuon());
      muon_tracker_.push_back(RecMuon->isTrackerMuon());
      ptGlob_.push_back((RecMuon)->pt());
      etaGlob_.push_back(RecMuon->eta());
      phiGlob_.push_back(RecMuon->phi());
      energyMuon_.push_back(RecMuon->energy());	
      pMuon_.push_back(RecMuon->p());
#ifdef EDM_ML_DEBUG
      edm::LogInfo("HBHEMuon") << "Energy:" << RecMuon->energy() << " P:"
			       << RecMuon->p() << std::endl;
#endif
      // acessing tracker hits info
      if (RecMuon->track().isNonnull()) {
	trackerLayer_.push_back(RecMuon->track()->hitPattern().trackerLayersWithMeasurement());
      } else {
	trackerLayer_.push_back(-1);
      }
      if (RecMuon->innerTrack().isNonnull()) {
	innerTrack_.push_back(true);
	numPixelLayers_.push_back(RecMuon->innerTrack()->hitPattern().pixelLayersWithMeasurement());
	chiTracker_.push_back(RecMuon->innerTrack()->normalizedChi2());
	dxyTracker_.push_back(fabs(RecMuon->innerTrack()->dxy((*firstGoodVertex).position())));
	dzTracker_.push_back(fabs(RecMuon->innerTrack()->dz((*firstGoodVertex).position())));
	innerTrackpt_.push_back(RecMuon->innerTrack()->pt());
	innerTracketa_.push_back(RecMuon->innerTrack()->eta());
	innerTrackphi_.push_back(RecMuon->innerTrack()->phi());
	tight_PixelHits_.push_back(RecMuon->innerTrack()->hitPattern().numberOfValidPixelHits());
      } else {
	innerTrack_.push_back(false);
	numPixelLayers_.push_back(0);
	chiTracker_.push_back(0);
        dxyTracker_.push_back(0);
	dzTracker_.push_back(0);
	innerTrackpt_.push_back(0);
	innerTracketa_.push_back(0);
	innerTrackphi_.push_back(0);
	tight_PixelHits_.push_back(0);
      }
      // outer track info
      if (RecMuon->outerTrack().isNonnull()) {
	outerTrack_.push_back(true);
	outerTrackPt_.push_back(RecMuon->outerTrack()->pt());
	outerTrackEta_.push_back(RecMuon->outerTrack()->eta());
	outerTrackPhi_.push_back(RecMuon->outerTrack()->phi());
	outerTrackChi_.push_back(RecMuon->outerTrack()->normalizedChi2());
	outerTrackHits_.push_back(RecMuon->outerTrack()->numberOfValidHits());
	outerTrackRHits_.push_back(RecMuon->outerTrack()->recHitsSize());
      } else {
	outerTrack_.push_back(false);
	outerTrackPt_.push_back(0);
	outerTrackEta_.push_back(0);
	outerTrackPhi_.push_back(0);
	outerTrackChi_.push_back(0);
	outerTrackHits_.push_back(0);
	outerTrackRHits_.push_back(0);
      }
      // Tight Muon cuts
      if (RecMuon->globalTrack().isNonnull())  {	 	
	globalTrack_.push_back(true);
	chiGlobal_.push_back(RecMuon->globalTrack()->normalizedChi2());
	globalMuonHits_.push_back(RecMuon->globalTrack()->hitPattern().numberOfValidMuonHits());
	matchedStat_.push_back(RecMuon->numberOfMatchedStations());  
	globalTrckPt_.push_back(RecMuon->globalTrack()->pt());
	globalTrckEta_.push_back(RecMuon->globalTrack()->eta());
	globalTrckPhi_.push_back(RecMuon->globalTrack()->phi()); 
	tight_TransImpara_.push_back(fabs(RecMuon->muonBestTrack()->dxy((*firstGoodVertex).position())));
	tight_LongPara_.push_back(fabs(RecMuon->muonBestTrack()->dz((*firstGoodVertex).position())));
      } else {
	globalTrack_.push_back(false);
	chiGlobal_.push_back(0);
	globalMuonHits_.push_back(0);
	matchedStat_.push_back(0);
	globalTrckPt_.push_back(0);
	globalTrckEta_.push_back(0);
	globalTrckPhi_.push_back(0);
	tight_TransImpara_.push_back(0);
	tight_LongPara_.push_back(0);
      }
      
      isolationR04_.push_back(((RecMuon->pfIsolationR04().sumChargedHadronPt + std::max(0.,RecMuon->pfIsolationR04().sumNeutralHadronEt + RecMuon->pfIsolationR04().sumPhotonEt - (0.5 *RecMuon->pfIsolationR04().sumPUPt))) / RecMuon->pt()) );
      
      isolationR03_.push_back(((RecMuon->pfIsolationR03().sumChargedHadronPt + std::max(0.,RecMuon->pfIsolationR03().sumNeutralHadronEt + RecMuon->pfIsolationR03().sumPhotonEt - (0.5 * RecMuon->pfIsolationR03().sumPUPt))) / RecMuon->pt()));

      ecalEnergy_.push_back(RecMuon->calEnergy().emS9);		 
      hcalEnergy_.push_back(RecMuon->calEnergy().hadS9);
      hoEnergy_.push_back(RecMuon->calEnergy().hoS9);
      
      double eEcal(0), eHcal(0), activeLengthTot(0), activeLengthHotTot(0);
      double eHcalDepth[MaxDepth], eHcalDepthHot[MaxDepth];
      double activeL[MaxDepth], activeHotL[MaxDepth];
      unsigned int isHot(0);
      bool         tmpmatch(false);
      for (int i=0; i<MaxDepth; ++i) 
	eHcalDepth[i] = eHcalDepthHot[i] = activeL[i] = activeHotL[i] = -10000;
      
      if (RecMuon->innerTrack().isNonnull()) {
	const reco::Track* pTrack = (RecMuon->innerTrack()).get();
	spr::propagatedTrackID trackID = spr::propagateCALO(pTrack, geo, bField, (((verbosity_/100)%10>0)));
	
	ecalDetId_.push_back((trackID.detIdECAL)()); 
	hcalDetId_.push_back((trackID.detIdHCAL)());  
	ehcalDetId_.push_back((trackID.detIdEHCAL)());
  
	HcalDetId check;
	std::pair<bool,HcalDetId> info = spr::propagateHCALBack(pTrack,  geo, bField, (((verbosity_/100)%10>0)));
	if (info.first) { 
	  check = info.second;
	}	

	bool okE = trackID.okECAL;
	if (okE) {
	  const DetId isoCell(trackID.detIdECAL);
	  std::pair<double,bool> e3x3 = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle,*theEcalChStatus,geo,caloTopology,sevlv.product(),1,1,-100.0,-100.0,-500.0,500.0,false);
	  eEcal = e3x3.first;
	  okE   = e3x3.second;
	}
#ifdef EDM_ML_DEBUG
	edm::LogInfo("HBHEMuon") << "Propagate Track to ECAL: " << okE << ":"
				 << trackID.okECAL << " E "
				 << eEcal << std::endl;
#endif

	if (trackID.okHCAL) {
	  const DetId closestCell(trackID.detIdHCAL);
	  HcalDetId hcidt(closestCell.rawId());  
	  if ((hcidt.ieta() == check.ieta()) && (hcidt.iphi() == check.iphi()))
	    tmpmatch= true;

	  HcalSubdetector subdet = HcalDetId(closestCell).subdet();
	  int             ieta   = HcalDetId(closestCell).ieta();
	  int             iphi   = HcalDetId(closestCell).iphi();
	  bool            hborhe = (std::abs(ieta) == 16);

	  eHcal = spr::eHCALmatrix(theHBHETopology, closestCell, hbhe,0,0, false, true, -100.0, -100.0, -100.0, -100.0, -500.,500.,useRaw_);
	  std::vector<std::pair<double,int> > ehdepth;
	  spr::energyHCALCell((HcalDetId) closestCell, hbhe, ehdepth, maxDepth_, -100.0, -100.0, -100.0, -100.0, -500.0, 500.0, useRaw_, (((verbosity_/1000)%10)>0));
	  for (unsigned int i=0; i<ehdepth.size(); ++i) {
	    eHcalDepth[ehdepth[i].second-1] = ehdepth[i].first;
	    HcalSubdetector subdet0 = (hborhe) ? ((ehdepth[i].second >= depthHE) ? HcalEndcap : HcalBarrel) : subdet;
	    HcalDetId hcid0(subdet0,ieta,iphi,ehdepth[i].second);
	    double actL = activeLength(DetId(hcid0));
	    activeL[ehdepth[i].second-1] = actL;
	    activeLengthTot += actL;
#ifdef EDM_ML_DEBUG
	    if ((verbosity_%10) > 0)
	      edm::LogInfo("HBHEMuon") << hcid0 << " E " << ehdepth[i].first
				       << " L " << actL << std::endl;
#endif
	  }
	  
	  HcalDetId           hotCell;
	  spr::eHCALmatrix(geo, theHBHETopology, closestCell, hbhe, 1,1, hotCell, false, useRaw_, false);
	  isHot = matchId(closestCell,hotCell);
	  if (hotCell != HcalDetId()) {
	    subdet = HcalDetId(hotCell).subdet();
	    ieta   = HcalDetId(hotCell).ieta();
	    iphi   = HcalDetId(hotCell).iphi();
	    hborhe = (std::abs(ieta) == 16);
	    std::vector<std::pair<double,int> > ehdepth;
	    spr::energyHCALCell(hotCell, hbhe, ehdepth, maxDepth_, -100.0, -100.0, -100.0, -100.0, -500.0, 500.0, useRaw_, false);//(((verbosity_/1000)%10)>0    ));
	    for (unsigned int i=0; i<ehdepth.size(); ++i) {
	      eHcalDepthHot[ehdepth[i].second-1] = ehdepth[i].first;
	      HcalSubdetector subdet0 = (hborhe) ? ((ehdepth[i].second >= depthHE) ? HcalEndcap : HcalBarrel) : subdet;
	      HcalDetId hcid0(subdet0,ieta,iphi,ehdepth[i].second);
	      double actL = activeLength(DetId(hcid0));
	      activeHotL[ehdepth[i].second-1] = actL;
	      activeLengthHotTot += actL;
#ifdef EDM_ML_DEBUG
	      if ((verbosity_%10) > 0)
		edm::LogInfo("HBHEMuon") << hcid0 << " E " << ehdepth[i].first
					 << " L " << actL << std::endl;
#endif
	    }
	  }
	}
#ifdef EDM_ML_DEBUG
	edm::LogInfo("HBHEMuon") << "Propagate Track to HCAL: " 
				 << trackID.okHCAL << " Match " << tmpmatch
				 << " Hot " << isHot << " Energy "
				 << eHcal << std::endl;
#endif
	
      } else {
	ecalDetId_.push_back(0);
	hcalDetId_.push_back(0);
	ehcalDetId_.push_back(0);
      }
      
      matchedId_.push_back(tmpmatch); 
      ecal3x3Energy_.push_back(eEcal);
      hcal1x1Energy_.push_back(eHcal);
      hcalDepth1Energy_.push_back(eHcalDepth[0]);
      hcalDepth2Energy_.push_back(eHcalDepth[1]);
      hcalDepth3Energy_.push_back(eHcalDepth[2]);
      hcalDepth4Energy_.push_back(eHcalDepth[3]);
      hcalDepth5Energy_.push_back(eHcalDepth[4]);
      hcalDepth6Energy_.push_back(eHcalDepth[5]);
      hcalDepth7Energy_.push_back(eHcalDepth[6]);
      hcalDepth1ActiveLength_.push_back(activeL[0]);
      hcalDepth2ActiveLength_.push_back(activeL[1]);
      hcalDepth3ActiveLength_.push_back(activeL[2]);
      hcalDepth4ActiveLength_.push_back(activeL[3]);
      hcalDepth5ActiveLength_.push_back(activeL[4]);
      hcalDepth6ActiveLength_.push_back(activeL[5]);
      hcalDepth7ActiveLength_.push_back(activeL[6]);
      hcalActiveLength_.push_back(activeLengthTot);
      hcalHot_.push_back(isHot);
      hcalDepth1EnergyHot_.push_back(eHcalDepthHot[0]);
      hcalDepth2EnergyHot_.push_back(eHcalDepthHot[1]);
      hcalDepth3EnergyHot_.push_back(eHcalDepthHot[2]);
      hcalDepth4EnergyHot_.push_back(eHcalDepthHot[3]);
      hcalDepth5EnergyHot_.push_back(eHcalDepthHot[4]);
      hcalDepth6EnergyHot_.push_back(eHcalDepthHot[5]);
      hcalDepth7EnergyHot_.push_back(eHcalDepthHot[6]);
      hcalDepth1ActiveLengthHot_.push_back(activeHotL[0]);
      hcalDepth2ActiveLengthHot_.push_back(activeHotL[1]);
      hcalDepth3ActiveLengthHot_.push_back(activeHotL[2]);
      hcalDepth4ActiveLengthHot_.push_back(activeHotL[3]);
      hcalDepth5ActiveLengthHot_.push_back(activeHotL[4]);
      hcalDepth6ActiveLengthHot_.push_back(activeHotL[5]);
      hcalDepth7ActiveLengthHot_.push_back(activeHotL[6]);
      hcalActiveLengthHot_.push_back(activeLengthHotTot);
    }
  }
  if (accept) tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void HcalHBHEMuonAnalyzer::beginJob() {
  
  tree_ = fs->make<TTree>("TREE", "TREE");
  tree_->Branch("Event_No",          &eventNumber_);
  tree_->Branch("Run_No",            &runNumber_);
  tree_->Branch("LumiNumber",        &lumiNumber_);
  tree_->Branch("BXNumber",          &bxNumber_);
  tree_->Branch("pt_of_muon",        &ptGlob_);
  tree_->Branch("eta_of_muon",       &etaGlob_);
  tree_->Branch("phi_of_muon",       &phiGlob_);
  tree_->Branch("energy_of_muon",    &energyMuon_);
  tree_->Branch("p_of_muon",         &pMuon_);
  tree_->Branch("PF_Muon",           &muon_is_good_);
  tree_->Branch("Global_Muon",       &muon_global_);
  tree_->Branch("Tracker_muon",      &muon_tracker_);
  
  tree_->Branch("hcal_3into3",      &hcalEnergy_);
  tree_->Branch("hcal_1x1",         &hcal1x1Energy_);
  tree_->Branch("hcal_detID",       &hcalDetId_);
  tree_->Branch("hcal_cellHot",     &hcalHot_);
  tree_->Branch("activeLength",     &hcalActiveLength_);
  tree_->Branch("hcal_edepth1",     &hcalDepth1Energy_);
  tree_->Branch("hcal_edepth2",     &hcalDepth2Energy_);
  tree_->Branch("hcal_edepth3",     &hcalDepth3Energy_);
  tree_->Branch("hcal_edepth4",     &hcalDepth4Energy_);
  tree_->Branch("hcal_activeL1",    &hcalDepth1ActiveLength_);
  tree_->Branch("hcal_activeL2",    &hcalDepth2ActiveLength_);
  tree_->Branch("hcal_activeL3",    &hcalDepth3ActiveLength_);
  tree_->Branch("hcal_activeL4",    &hcalDepth4ActiveLength_);
  tree_->Branch("activeLengthHot",  &hcalActiveLengthHot_);
  tree_->Branch("hcal_edepthHot1",  &hcalDepth1EnergyHot_);
  tree_->Branch("hcal_edepthHot2",  &hcalDepth2EnergyHot_);
  tree_->Branch("hcal_edepthHot3",  &hcalDepth3EnergyHot_);
  tree_->Branch("hcal_edepthHot4",  &hcalDepth4EnergyHot_);
  tree_->Branch("hcal_activeHotL1", &hcalDepth1ActiveLength_);
  tree_->Branch("hcal_activeHotL2", &hcalDepth2ActiveLength_);
  tree_->Branch("hcal_activeHotL3", &hcalDepth3ActiveLength_);
  tree_->Branch("hcal_activeHotL4", &hcalDepth4ActiveLength_);
  
  if (maxDepth_ > 4) {
    tree_->Branch("hcal_edepth5",     &hcalDepth5Energy_);
    tree_->Branch("hcal_activeL5",    &hcalDepth5ActiveLength_);
    tree_->Branch("hcal_edepthHot5",  &hcalDepth5EnergyHot_);
    tree_->Branch("hcal_activeHotL5", &hcalDepth5ActiveLength_);
    if (maxDepth_ > 5) {
      tree_->Branch("hcal_edepth6",     &hcalDepth6Energy_);
      tree_->Branch("hcal_activeL6",    &hcalDepth6ActiveLength_);
      tree_->Branch("hcal_edepthHot6",  &hcalDepth6EnergyHot_);
      tree_->Branch("hcal_activeHotL6", &hcalDepth6ActiveLength_);
      if (maxDepth_ > 6) {
	tree_->Branch("hcal_edepth7",     &hcalDepth7Energy_);
	tree_->Branch("hcal_activeL7",    &hcalDepth7ActiveLength_);
	tree_->Branch("hcal_edepthHot7",  &hcalDepth7EnergyHot_);
	tree_->Branch("hcal_activeHotL7", &hcalDepth7ActiveLength_);
      }
    }
  }
  

  tree_->Branch("TrackerLayer",                     &trackerLayer_);
  tree_->Branch("matchedId",                        &matchedId_);
  tree_->Branch("innerTrack",                       &innerTrack_);
  tree_->Branch("innerTrackpt",                     &innerTrackpt_);
  tree_->Branch("innerTracketa",                    &innerTracketa_);
  tree_->Branch("innerTrackphi",                    &innerTrackphi_);
  tree_->Branch("MatchedStat",                      &matchedStat_);
  tree_->Branch("GlobalTrckPt",                     &globalTrckPt_);
  tree_->Branch("GlobalTrckEta",                    &globalTrckEta_);
  tree_->Branch("GlobalTrckPhi",                    &globalTrckPhi_);
  tree_->Branch("NumPixelLayers",                   &numPixelLayers_);
  tree_->Branch("chiTracker",                       &chiTracker_);
  tree_->Branch("DxyTracker",                       &dxyTracker_);
  tree_->Branch("DzTracker",                        &dzTracker_);
  tree_->Branch("OuterTrack",                       &outerTrack_);
  tree_->Branch("OuterTrackPt",                     &outerTrackPt_);
  tree_->Branch("OuterTrackEta",                    &outerTrackEta_);
  tree_->Branch("OuterTrackPhi",                    &outerTrackPhi_);
  tree_->Branch("OuterTrackHits",                   &outerTrackHits_);
  tree_->Branch("OuterTrackRHits",                  &outerTrackRHits_);
  tree_->Branch("OuterTrackChi",                    &outerTrackChi_);
  tree_->Branch("GlobalTrack",                      &globalTrack_);
  tree_->Branch("GlobTrack_Chi",                    &chiGlobal_);
  tree_->Branch("Global_Muon_Hits",                 &globalMuonHits_);
  tree_->Branch("MatchedStations",                  &matchedStat_);
  tree_->Branch("Global_Track_Pt",                  &globalTrckPt_);
  tree_->Branch("Global_Track_Eta",                 &globalTrckEta_);
  tree_->Branch("Global_Track_Phi",                 &globalTrckPhi_);
  ///////////////////////////////
  tree_->Branch("Tight_LongitudinalImpactparameter",&tight_LongPara_);
  tree_->Branch("Tight_TransImpactparameter",       &tight_TransImpara_);
  tree_->Branch("InnerTrackPixelHits",              &tight_PixelHits_);
  tree_->Branch("IsolationR04",                     &isolationR04_);
  tree_->Branch("IsolationR03",                     &isolationR03_);

  tree_->Branch("ecal_3into3",                      &ecalEnergy_);
  tree_->Branch("ecal_3x3",                         &ecal3x3Energy_);
  tree_->Branch("ecal_detID",                       &ecalDetId_);
  tree_->Branch("ehcal_detID",                      &ehcalDetId_);
  tree_->Branch("tracker_3into3",                   &hoEnergy_);
  
  ///////////////////////////////
  tree_->Branch("hltresults",                       &hltresults);
  tree_->Branch("all_triggers",                     &all_triggers);

}

// ------------ method called once each job just after ending the event loop  ------------
void HcalHBHEMuonAnalyzer::endJob() {}

// ------------ method called when starting to processes a run  ------------
void HcalHBHEMuonAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {

  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);
  const HcalDDDRecConstants & hdc = (*pHRNDC);
  actHB.clear();
  actHE.clear();
  actHB = hdc.getThickActive(0);
  actHE = hdc.getThickActive(1);
   
  bool changed = true;
  all_triggers.clear();
  if (hltConfig_.init(iRun, iSetup,"HLT" , changed)) {
    // if init returns TRUE, initialisation has succeeded!
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HBHEMuon") << "HLT config with process name " 
			     << "HLT" << " successfully extracted"
			     << std::endl;
#endif
//  std::string string_search[5]={"HLT_IsoMu_","HLT_L1SingleMu_","HLT_L2Mu","HLT_Mu","HLT_RelIso1p0Mu"};
    std::string string_search[6]={"HLT_IsoMu17","HLT_IsoMu20","HLT_IsoMu24","HLT_IsoMu27","HLT_Mu45","HLT_Mu50"};
  
    unsigned int ntriggers = hltConfig_.size();
    for (unsigned int t=0;t<ntriggers;++t) {
      std::string hltname(hltConfig_.triggerName(t));
      for (unsigned int ik=0; ik<6; ++ik) {
	if (hltname.find(string_search[ik])!=std::string::npos ){
	  all_triggers.push_back(hltname);
	  break;
	}
      }
    }//loop over ntriggers
    edm::LogInfo("HBHEMuon") << "All triggers size in begin run " 
			     << all_triggers.size() << std::endl;
  } else {
    edm::LogError("HBHEMuon") << "Error! HLT config extraction with process name " 
			      << "HLT" << " failed";
  }
  
}


// ------------ method called when ending the processing of a run  ------------
void HcalHBHEMuonAnalyzer::endRun(edm::Run const&, edm::EventSetup const&) { }

// ------------ method called when starting to processes a luminosity block  ------------
void HcalHBHEMuonAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) { }

// ------------ method called when ending the processing of a luminosity block  ------------
void HcalHBHEMuonAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) { }

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalHBHEMuonAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HcalHBHEMuonAnalyzer::clearVectors() {
  ///clearing vectots
  eventNumber_ = -99999;
  runNumber_   = -99999;
  lumiNumber_  = -99999;
  bxNumber_    = -99999;
  muon_is_good_.clear();
  muon_global_.clear();
  muon_tracker_.clear();
  ptGlob_.clear();
  etaGlob_.clear(); 
  phiGlob_.clear(); 
  energyMuon_.clear();
  pMuon_.clear();
  trackerLayer_.clear();
  matchedId_.clear();
  innerTrack_.clear();
  numPixelLayers_.clear();
  chiTracker_.clear();
  dxyTracker_.clear();
  dzTracker_.clear();
  innerTrackpt_.clear();
  innerTracketa_.clear();
  innerTrackphi_.clear();
  tight_PixelHits_.clear();
  outerTrack_.clear();
  outerTrackPt_.clear();
  outerTrackEta_.clear();
  outerTrackPhi_.clear();
  outerTrackHits_.clear();
  outerTrackRHits_.clear();
  outerTrackChi_.clear();
  globalTrack_.clear();
  chiGlobal_.clear();
  globalMuonHits_.clear();
  matchedStat_.clear();
  globalTrckPt_.clear();
  globalTrckEta_.clear();
  globalTrckPhi_.clear();
  tight_TransImpara_.clear();
  tight_LongPara_.clear();

  isolationR04_.clear();
  isolationR03_.clear();
  ecalEnergy_.clear();
  hcalEnergy_.clear();
  hoEnergy_.clear();
  ecalDetId_.clear();
  hcalDetId_.clear();
  ehcalDetId_.clear();
  ecal3x3Energy_.clear();
  hcal1x1Energy_.clear();
  hcalHot_.clear();
  hcalDepth1Energy_.clear();
  hcalDepth2Energy_.clear();
  hcalDepth3Energy_.clear();
  hcalDepth4Energy_.clear();
  hcalDepth5Energy_.clear();
  hcalDepth6Energy_.clear();
  hcalDepth7Energy_.clear();
  hcalDepth1ActiveLength_.clear();
  hcalDepth2ActiveLength_.clear();
  hcalDepth3ActiveLength_.clear();
  hcalDepth4ActiveLength_.clear();
  hcalDepth5ActiveLength_.clear();
  hcalDepth6ActiveLength_.clear();
  hcalDepth7ActiveLength_.clear();
  hcalActiveLengthHot_.clear();
  hcalDepth1EnergyHot_.clear();
  hcalDepth2EnergyHot_.clear();
  hcalDepth3EnergyHot_.clear();
  hcalDepth4EnergyHot_.clear();
  hcalDepth5EnergyHot_.clear();
  hcalDepth6EnergyHot_.clear();
  hcalDepth7EnergyHot_.clear();
  hcalDepth1ActiveLengthHot_.clear();
  hcalDepth2ActiveLengthHot_.clear();
  hcalDepth3ActiveLengthHot_.clear();
  hcalDepth4ActiveLengthHot_.clear();
  hcalDepth5ActiveLengthHot_.clear();
  hcalDepth6ActiveLengthHot_.clear();
  hcalDepth7ActiveLengthHot_.clear();
  hltresults.clear();
}

int HcalHBHEMuonAnalyzer::matchId(const HcalDetId& id1, const HcalDetId& id2) {

  HcalDetId kd1(id1.subdet(),id1.ieta(),id1.iphi(),1);
  HcalDetId kd2(id2.subdet(),id2.ieta(),id2.iphi(),1);
  int match = ((kd1 == kd2) ? 1 : 0);
  return match;
}

double HcalHBHEMuonAnalyzer::activeLength(const DetId& id_) {
  HcalDetId id(id_);
  int ieta = id.ietaAbs();
  int depth= id.depth();
  double lx(0);
  if (id.subdet() == HcalBarrel) {
    for (unsigned int i=0; i<actHB.size(); ++i) {
      if (ieta == actHB[i].ieta && depth == actHB[i].depth) {
	lx = actHB[i].thick;
	break;
      }
    }
  } else {
    for (unsigned int i=0; i<actHE.size(); ++i) {
      if (ieta == actHE[i].ieta && depth == actHE[i].depth) {
	lx = actHE[i].thick;
	break;
      }
    }
  }
  return lx;
}

bool HcalHBHEMuonAnalyzer::isGoodVertex(const reco::Vertex& vtx) {
  if (vtx.isFake())                   return false;
  if (vtx.ndof() < 4)                 return false;
  if (vtx.position().Rho() > 2.)      return false;
  if (fabs(vtx.position().Z()) > 24.) return false;
  return true;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalHBHEMuonAnalyzer);


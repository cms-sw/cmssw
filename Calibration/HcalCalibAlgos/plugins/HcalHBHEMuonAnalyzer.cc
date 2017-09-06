#include <memory>
#include <iostream>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include "TPRegexp.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
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

#define EDM_ML_DEBUG

class HcalHBHEMuonAnalyzer :  public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit HcalHBHEMuonAnalyzer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  void   clearVectors();
  int    matchId(const HcalDetId&, const HcalDetId&);
  double activeLength(const DetId&);
  bool   isGoodVertex(const reco::Vertex& vtx);

  // ----------member data ---------------------------
  HLTConfigProvider          hltConfig_;
  edm::Service<TFileService> fs;
  edm::InputTag              HLTriggerResults_;
  edm::InputTag              labelEBRecHit_, labelEERecHit_, labelHBHERecHit_;
  std::string                labelVtx_, labelBS_, labelMuon_;
  std::vector<std::string>   triggers_;
  bool                       useRaw_, unCorrect_, collapseDepth_, saveCorrect_;
  int                        verbosity_, maxDepth_, kount_;
  static const int           depthMax_=7;
  const HcalDDDRecConstants *hdc;

  edm::EDGetTokenT<edm::TriggerResults>                   tok_trigRes_;
  edm::EDGetTokenT<reco::BeamSpot>                        tok_bs_;
  edm::EDGetTokenT<reco::VertexCollection>                tok_Vtx_;
  edm::EDGetTokenT<EcalRecHitCollection>                  tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>                  tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>                  tok_HBHE_;
  edm::EDGetTokenT<reco::MuonCollection>                  tok_Muon_;
  
  //////////////////////////////////////////////////////
  std::vector<double>       muon_trkKink,muon_chi2LocalPosition, muon_segComp, tight_validFraction_;		
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
  std::vector<double>       hcalDepthEnergy_[depthMax_];
  std::vector<double>       hcalDepthActiveLength_[depthMax_];
  std::vector<double>       hcalDepthEnergyHot_[depthMax_];
  std::vector<double>       hcalDepthActiveLengthHot_[depthMax_];
  std::vector<double>       hcalDepthEnergyCorr_[depthMax_];
  std::vector<double>       hcalDepthEnergyHotCorr_[depthMax_];
  std::vector<double>       hcalActiveLength_,    hcalActiveLengthHot_;
  std::vector<HcalDDDRecConstants::HcalActiveLength> actHB, actHE;
  std::vector<std::string>  all_triggers;
  ////////////////////////////////////////////////////////////
  
  TTree                    *tree_;
  std::vector<bool>         muon_is_good_, muon_global_, muon_tracker_;
  std::vector<int>          hltresults;
  unsigned int              runNumber_, eventNumber_ , lumiNumber_, bxNumber_;
};

HcalHBHEMuonAnalyzer::HcalHBHEMuonAnalyzer(const edm::ParameterSet& iConfig) : hdc(nullptr) {
  
  usesResource(TFileService::kSharedResource);
  //now do what ever initialization is needed
  kount_            = 0;
  HLTriggerResults_ = iConfig.getParameter<edm::InputTag>("HLTriggerResults");
  labelBS_          = iConfig.getParameter<std::string>("LabelBeamSpot");
  labelVtx_         = iConfig.getParameter<std::string>("LabelVertex");
  labelEBRecHit_    = iConfig.getParameter<edm::InputTag>("LabelEBRecHit");
  labelEERecHit_    = iConfig.getParameter<edm::InputTag>("LabelEERecHit");
  labelHBHERecHit_  = iConfig.getParameter<edm::InputTag>("LabelHBHERecHit");
  labelMuon_        = iConfig.getParameter<std::string>("LabelMuon");
  triggers_         = iConfig.getParameter<std::vector<std::string>>("Triggers");
  useRaw_           = iConfig.getParameter<bool>("UseRaw");
  unCorrect_        = iConfig.getParameter<bool>("UnCorrect");
  collapseDepth_    = iConfig.getParameter<bool>("CollapseDepth");
  saveCorrect_      = iConfig.getParameter<bool>("SaveCorrect");
  verbosity_        = iConfig.getUntrackedParameter<int>("Verbosity",0);
  maxDepth_         = iConfig.getUntrackedParameter<int>("MaxDepth",4);
  if      (maxDepth_ > depthMax_) maxDepth_ = depthMax_;
  else if (maxDepth_ < 1)         maxDepth_ = 4;
  std::string modnam = iConfig.getUntrackedParameter<std::string>("ModuleName","");
  std::string procnm = iConfig.getUntrackedParameter<std::string>("ProcessName","");

  tok_trigRes_  = consumes<edm::TriggerResults>(HLTriggerResults_);
  tok_bs_       = consumes<reco::BeamSpot>(labelBS_);
  tok_EB_       = consumes<EcalRecHitCollection>(labelEBRecHit_);
  tok_EE_       = consumes<EcalRecHitCollection>(labelEERecHit_);
  tok_HBHE_     = consumes<HBHERecHitCollection>(labelHBHERecHit_);
  if (modnam == "") {
    tok_Vtx_      = consumes<reco::VertexCollection>(labelVtx_);
    tok_Muon_     = consumes<reco::MuonCollection>(labelMuon_);
    edm::LogVerbatim("HBHEMuon")  << "Labels used " << HLTriggerResults_ << " "
				  << labelVtx_ << " " << labelEBRecHit_ << " "
				  << labelEERecHit_ << " " << labelHBHERecHit_
				  << " " << labelMuon_;
  } else {
    tok_Vtx_      = consumes<reco::VertexCollection>(edm::InputTag(modnam,labelVtx_,procnm));
    tok_Muon_     = consumes<reco::MuonCollection>(edm::InputTag(modnam,labelMuon_,procnm));
    edm::LogVerbatim("HBHEMuon")   << "Labels used "   << HLTriggerResults_
				   << "\n            " << edm::InputTag(modnam,labelVtx_,procnm)
				   << "\n            " << labelEBRecHit_
				   << "\n            " << labelEERecHit_
				   << "\n            " << labelHBHERecHit_
				   << "\n            " << edm::InputTag(modnam,labelMuon_,procnm);
  }
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
  edm::LogVerbatim("HBHEMuon") << "Run " << runNumber_ << " Event "
			       << eventNumber_ << " Lumi " << lumiNumber_ 
			       << " BX " << bxNumber_ << std::endl;
#endif  
  edm::Handle<edm::TriggerResults> _Triggers;
  iEvent.getByToken(tok_trigRes_, _Triggers); 
#ifdef EDM_ML_DEBUG
  if ((verbosity_/10000)%10>0) 
    edm::LogVerbatim("HBHEMuon") << "Size of all triggers "  
				 << all_triggers.size() << std::endl;
#endif
  int Ntriggers = all_triggers.size();
#ifdef EDM_ML_DEBUG
  if ((verbosity_/10000)%10>0) 
    edm::LogVerbatim("HBHEMuon") << "Size of HLT MENU: " << _Triggers->size()
				 << std::endl;
#endif
  if (_Triggers.isValid()) {
    const edm::TriggerNames &triggerNames_ = iEvent.triggerNames(*_Triggers);
    std::vector<int> index;
    for (int i=0; i<Ntriggers; i++) {
      index.push_back(triggerNames_.triggerIndex(all_triggers[i]));
      int triggerSize = int( _Triggers->size());
#ifdef EDM_ML_DEBUG
      if ((verbosity_/10000)%10>0) 
	edm::LogVerbatim("HBHEMuon") << "outside loop " << index[i]
				     << "\ntriggerSize " << triggerSize
				     << std::endl;
#endif
      if (index[i] < triggerSize) {
	hltresults.push_back(_Triggers->accept(index[i]));
#ifdef EDM_ML_DEBUG
	if ((verbosity_/10000)%10>0) 
	  edm::LogVerbatim("HBHEMuon") << "Trigger_info " << triggerSize
				       << " triggerSize " << index[i]
				       << " trigger_index " << hltresults.at(i)
				       << " hltresult" << std::endl;
#endif
      } else {
	if ((verbosity_/10000)%10>0) 
	  edm::LogVerbatim("HBHEMuon") << "Requested HLT path \"" 
				       << "\" does not exist\n";
      }
    }
  }

  // get handles to calogeometry and calotopology
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

  edm::ESHandle<HcalRespCorrs> resp;
  iSetup.get<HcalRespCorrsRcd>().get(resp);
  HcalRespCorrs* respCorrs = new HcalRespCorrs(*resp.product());
  respCorrs->setTopo(theHBHETopology);

  // Relevant blocks from iEvent
  edm::Handle<reco::VertexCollection> vtx;
  iEvent.getByToken(tok_Vtx_, vtx);
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);

  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);

  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_HBHE_, hbhe);

  edm::Handle<reco::MuonCollection> _Muon;
  iEvent.getByToken(tok_Muon_, _Muon);

  // require a good vertex
  math::XYZPoint pvx;
  bool goodVtx(false);
  if (vtx.isValid()) {
    reco::VertexCollection::const_iterator firstGoodVertex = vtx->end();
    for (reco::VertexCollection::const_iterator it = vtx->begin(); 
	 it != firstGoodVertex; it++) {
      if (isGoodVertex(*it)) {
	firstGoodVertex = it;
	break;
      }
    }
    if (firstGoodVertex != vtx->end()) {
      pvx     = firstGoodVertex->position();
      goodVtx = true;
    }
  }
  if (!goodVtx) {
    if (beamSpotH.isValid()) {
      pvx     = beamSpotH->position();
      goodVtx = true;
    }
  }
  if (!goodVtx) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HBHEMuon") << "No Good Vertex found == Reject\n";
#endif
    return;
  }
  
  bool accept(false);
  if (_Muon.isValid() && barrelRecHitsHandle.isValid() && 
      endcapRecHitsHandle.isValid() && hbhe.isValid()) { 
    for (reco::MuonCollection::const_iterator RecMuon = _Muon->begin(); RecMuon!= _Muon->end(); ++RecMuon)  {
      
      if ((RecMuon->p()>10.0) && (RecMuon->track().isNonnull())) accept = true;

      muon_is_good_.push_back(RecMuon->isPFMuon());
      muon_global_.push_back(RecMuon->isGlobalMuon());
      muon_tracker_.push_back(RecMuon->isTrackerMuon());
      ptGlob_.push_back((RecMuon)->pt());
      etaGlob_.push_back(RecMuon->eta());
      phiGlob_.push_back(RecMuon->phi());
      energyMuon_.push_back(RecMuon->energy());	
      pMuon_.push_back(RecMuon->p());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HBHEMuon") << "Energy:" << RecMuon->energy() << " P:"
				   << RecMuon->p() << std::endl;
#endif
      muon_trkKink.push_back(RecMuon->combinedQuality().trkKink);
      muon_chi2LocalPosition.push_back(RecMuon->combinedQuality().chi2LocalPosition);
      muon_segComp.push_back(muon::segmentCompatibility(*RecMuon));
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
	dxyTracker_.push_back(fabs(RecMuon->innerTrack()->dxy(pvx)));
	dzTracker_.push_back(fabs(RecMuon->innerTrack()->dz(pvx)));
	innerTrackpt_.push_back(RecMuon->innerTrack()->pt());
	innerTracketa_.push_back(RecMuon->innerTrack()->eta());
	innerTrackphi_.push_back(RecMuon->innerTrack()->phi());
	tight_PixelHits_.push_back(RecMuon->innerTrack()->hitPattern().numberOfValidPixelHits());
	tight_validFraction_.push_back(RecMuon->innerTrack()->validFraction());
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
	tight_validFraction_.push_back(-99);
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
	tight_TransImpara_.push_back(fabs(RecMuon->muonBestTrack()->dxy(pvx)));
	tight_LongPara_.push_back(fabs(RecMuon->muonBestTrack()->dz(pvx)));
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
      double eHcalDepth[depthMax_], eHcalDepthHot[depthMax_];
      double eHcalDepthC[depthMax_], eHcalDepthHotC[depthMax_];
      double activeL[depthMax_], activeHotL[depthMax_];
      HcalDetId eHcalDetId[depthMax_];
      unsigned int isHot(0);
      bool         tmpmatch(false);
      for (int i=0; i<depthMax_; ++i) {
	eHcalDepth[i]  = eHcalDepthHot[i]  = 0;
	eHcalDepthC[i] = eHcalDepthHotC[i] = 0;
	activeL[i]     = activeHotL[i]     = 0;
      }
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
	edm::LogVerbatim("HBHEMuon") << "Propagate Track to ECAL: " << okE 
				     << ":" << trackID.okECAL << " E "
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
	  for (int i=0; i<depthMax_; ++i) eHcalDetId[i] = HcalDetId();
	  for (unsigned int i=0; i<ehdepth.size(); ++i) {
	    HcalSubdetector subdet0 = (hborhe) ? ((ehdepth[i].second >= depthHE) ? HcalEndcap : HcalBarrel) : subdet;
	    HcalDetId hcid0(subdet0,ieta,iphi,ehdepth[i].second);
	    double actL = activeLength(DetId(hcid0));
	    double ene  = ehdepth[i].first;
	    double enec(ene);
	    if (unCorrect_) {
	      double corr = (respCorrs->getValues(DetId(hcid0)))->getValue();
	      if (corr != 0) ene /= corr;
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("HBHEMuon") << hcid0 << " corr " << corr;
#endif
	    }
	    int depth = ehdepth[i].second - 1;
	    if (collapseDepth_) {
	      HcalDetId id = hdc->mergedDepthDetId(hcid0);
	      depth        = id.depth() - 1;
	    }
	    eHcalDepth[depth] += ene;
	    eHcalDepthC[depth]+= enec;
	    activeL[depth]    += actL;
	    activeLengthTot   += actL;
#ifdef EDM_ML_DEBUG
	    if ((verbosity_%10) > 0)
	      edm::LogVerbatim("HBHEMuon") << hcid0 << " E " << ene << " L " 
					   << actL << std::endl;
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
	    for (int i=0; i<depthMax_; ++i) eHcalDetId[i] = HcalDetId();
	    for (unsigned int i=0; i<ehdepth.size(); ++i) {
	      HcalSubdetector subdet0 = (hborhe) ? ((ehdepth[i].second >= depthHE) ? HcalEndcap : HcalBarrel) : subdet;
	      HcalDetId hcid0(subdet0,ieta,iphi,ehdepth[i].second);
	      double actL = activeLength(DetId(hcid0));
	      double ene  = ehdepth[i].first;
	      double enec(ene);
	      if (unCorrect_) {
		double corr = (respCorrs->getValues(DetId(hcid0)))->getValue();
		if (corr != 0) ene /= corr;
#ifdef EDM_ML_DEBUG
		edm::LogVerbatim("HBHEMuon") << hcid0 << " corr " << corr;
#endif
	      }
	      int depth = ehdepth[i].second - 1;
	      if (collapseDepth_) {
		HcalDetId id = hdc->mergedDepthDetId(hcid0);
		depth        = id.depth() - 1;
	      }
	      eHcalDepthHot[depth] += ene;
	      eHcalDepthHotC[depth]+= enec;
	      activeHotL[depth]    += actL;
	      activeLengthHotTot   += actL;
#ifdef EDM_ML_DEBUG
	      if ((verbosity_%10) > 0)
		edm::LogVerbatim("HBHEMuon") << hcid0 << " E " << ene 
					     << " L " << actL << std::endl;
#endif
	    }
	  }
	}
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("HBHEMuon") << "Propagate Track to HCAL: " 
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
      for (int i=0; i<depthMax_; ++i)  {
	hcalDepthEnergy_[i].push_back(eHcalDepth[i]);
	hcalDepthActiveLength_[i].push_back(activeL[i]);
	hcalDepthEnergyHot_[i].push_back(eHcalDepthHot[i]);
	hcalDepthActiveLengthHot_[i].push_back(activeHotL[i]);
	hcalDepthEnergyCorr_[i].push_back(eHcalDepthC[i]);
	hcalDepthEnergyHotCorr_[i].push_back(eHcalDepthHotC[i]);
      }
      hcalActiveLength_.push_back(activeLengthTot);
      hcalHot_.push_back(isHot);
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
  
  tree_->Branch("hcal_3into3",       &hcalEnergy_);
  tree_->Branch("hcal_1x1",          &hcal1x1Energy_);
  tree_->Branch("hcal_detID",        &hcalDetId_);
  tree_->Branch("hcal_cellHot",      &hcalHot_);
  tree_->Branch("activeLength",      &hcalActiveLength_);
  tree_->Branch("activeLengthHot",   &hcalActiveLengthHot_);
  char name[100];
  for (int k=0; k<maxDepth_; ++k) {
    sprintf (name, "hcal_edepth%d", (k+1));
    tree_->Branch(name, &hcalDepthEnergy_[k]);
    sprintf (name, "hcal_activeL%d", (k+1));
    tree_->Branch(name,  &hcalDepthActiveLength_[k]);
    sprintf (name, "hcal_edepthHot%d", (k+1));
    tree_->Branch(name,  &hcalDepthEnergyHot_[k]);
    sprintf (name, "hcal_activeHotL%d", (k+1));
    tree_->Branch(name, &hcalDepthActiveLength_[k]);
    if (saveCorrect_) {
      sprintf (name, "hcal_edepthCorrect%d", (k+1));
      tree_->Branch(name, &hcalDepthEnergyCorr_[k]);
      sprintf (name, "hcal_edepthHotCorrect%d", (k+1));
      tree_->Branch(name,  &hcalDepthEnergyHotCorr_[k]);
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
  
  tree_->Branch("muon_trkKink", &muon_trkKink);
  tree_->Branch("muon_chi2LocalPosition", &muon_chi2LocalPosition);
  tree_->Branch("muon_segComp", &muon_segComp);
  tree_->Branch("tight_validFraction", &tight_validFraction_);
}

// ------------ method called when starting to processes a run  ------------
void HcalHBHEMuonAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {

  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);
  hdc = pHRNDC.product();
  actHB.clear();
  actHE.clear();
  actHB = hdc->getThickActive(0);
  actHE = hdc->getThickActive(1);
  
  bool changed = true;
  all_triggers.clear();
  if (hltConfig_.init(iRun, iSetup,"HLT" , changed)) {
    // if init returns TRUE, initialisation has succeeded!
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HBHEMuon") << "HLT config with process name " 
				 << "HLT" << " successfully extracted"
				 << std::endl;
#endif
    unsigned int ntriggers = hltConfig_.size();
    for (unsigned int t=0;t<ntriggers;++t) {
      std::string hltname(hltConfig_.triggerName(t));
      for (unsigned int ik=0; ik<6; ++ik) {
	if (hltname.find(triggers_[ik])!=std::string::npos ){
	  all_triggers.push_back(hltname);
	  break;
	}
      }
    }//loop over ntriggers
    edm::LogVerbatim("HBHEMuon") << "All triggers size in begin run " 
				 << all_triggers.size() << std::endl;
  } else {
    edm::LogError("HBHEMuon") << "Error! HLT config extraction with process name " 
			      << "HLT" << " failed";
  }

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalHBHEMuonAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HLTriggerResults",edm::InputTag("TriggerResults","","HLT"));
  desc.add<std::string>("LabelBeamSpot","offlineBeamSpot");
  desc.add<std::string>("LabelVertex","offlinePrimaryVertices");
  desc.add<edm::InputTag>("LabelEBRecHit",edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  desc.add<edm::InputTag>("LabelEERecHit",edm::InputTag("ecalRecHit","EcalRecHitsEE"));
  desc.add<edm::InputTag>("LabelHBHERecHit",edm::InputTag("hbhereco"));
  desc.add<std::string>("LabelMuon","muons");
// std::vector<std::string> trig = {"HLT_IsoMu_","HLT_L1SingleMu_","HLT_L2Mu","HLT_Mu","HLT_RelIso1p0Mu"};
  std::vector<std::string> trig = {"HLT_IsoMu17","HLT_IsoMu20",
				   "HLT_IsoMu24","HLT_IsoMu27",
				   "HLT_Mu45","HLT_Mu50"};
  desc.add<std::vector<std::string>>("Triggers",trig);
  desc.add<bool>("UseRaw",false);
  desc.add<bool>("UnCorrect",false);
  desc.add<bool>("CollapseDepth",false);
  desc.add<bool>("SaveCorrect",false);
  desc.addUntracked<std::string>("ModuleName","");
  desc.addUntracked<std::string>("ProcessName","");
  desc.addUntracked<int>("Verbosity",0);
  desc.addUntracked<int>("MaxDepth",4);
  descriptions.add("hcalHBHEMuon",desc);
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
  hcalActiveLengthHot_.clear();
  for (int i=0; i<maxDepth_; ++i) {
    hcalDepthEnergy_[i].clear();
    hcalDepthActiveLength_[i].clear();
    hcalDepthEnergyHot_[i].clear();
    hcalDepthActiveLengthHot_[i].clear();
    hcalDepthEnergyCorr_[i].clear();
    hcalDepthEnergyHotCorr_[i].clear();
  }
  hltresults.clear();
  muon_trkKink.clear();
  muon_chi2LocalPosition.clear();
  muon_segComp.clear();
  tight_validFraction_.clear();
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
  int zside= id.zside();
  int iphi = id.iphi();
  double lx(0);
  if (id.subdet() == HcalBarrel) {
    for (unsigned int i=0; i<actHB.size(); ++i) {
      if ((ieta == actHB[i].ieta) && (depth == actHB[i].depth) && 
	  (zside == actHB[i].zside) && 
	  (std::find(actHB[i].iphis.begin(),actHB[i].iphis.end(),iphi) !=
	   actHB[i].iphis.end())) {
	lx = actHB[i].thick;
	break;
      }
    }
  } else {
    for (unsigned int i=0; i<actHE.size(); ++i) {
      if ((ieta == actHE[i].ieta) && (depth == actHE[i].depth) && 
	  (zside == actHE[i].zside) && 
	  (std::find(actHE[i].iphis.begin(),actHE[i].iphis.end(),iphi) !=
	   actHE[i].iphis.end())) {
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

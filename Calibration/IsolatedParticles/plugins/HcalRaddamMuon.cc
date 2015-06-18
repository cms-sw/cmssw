#include <memory>
#include <iostream>
#include <vector>

#include <TTree.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
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
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

class HcalDDDRecConstants {
public:
  HcalDDDRecConstants();
  ~HcalDDDRecConstants();

  struct HcalActiveLength {
    int    ieta, depth;
    double eta, thick;
    HcalActiveLength(int ie=0, int d=0, double et=0, 
                     double t=0) : ieta(ie), depth(d), eta(et), thick(t) {}
  };
 
  std::vector<HcalActiveLength> getThickActive(const int type) const;

private:
  std::vector<HcalActiveLength> actHB, actHE;
};

HcalDDDRecConstants::HcalDDDRecConstants() {
  int ietaHB[18]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
		      11, 12, 13, 14, 15, 15, 16, 16};
  int depthHB[18]  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		      1, 1, 1, 1, 1, 2, 1, 2};
  double etaHB[18] = {0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785,
		      0.5655, 0.6525, 0.7395, 0.8265, 0.9135, 1.0005,
		      1.0875, 1.1745, 1.2615, 1.2615, 1.3485, 1.3485};
  double actLHB[18]= {7.35696, 7.41268, 7.52454, 7.69339, 7.92051, 8.20761,
		      8.55688, 8.97096, 9.45298, 10.0066, 10.6360, 11.3460,
		      12.1419, 13.0297, 10.1832, 3.83301, 2.61066, 5.32410};
  actHB.clear();
  for (int i=0; i<18; ++i) {
    HcalDDDRecConstants::HcalActiveLength act(ietaHB[i],depthHB[i],etaHB[i],actLHB[i]);
    actHB.push_back(act);
  }
  
  int ietaHE[28]   = {16, 17, 18, 18, 19, 19, 20, 20, 21, 21,
		      22, 22, 23, 23, 24, 24, 25, 25, 26, 26,
		      27, 27, 27, 28, 28, 28, 29, 29};
  int depthHE[28]  = {3, 1, 1, 2, 1, 2, 1, 2, 1, 2,
		      1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
		      1, 2, 3, 1, 2, 3, 1, 2};
  double etaHE[28] = {1.3485, 1.4355, 1.5225, 1.5225, 1.6095, 1.6095, 1.6965,
		      1.6965, 1.7850, 1.7850, 1.8800, 1.8800, 1.9865, 1.9865,
		      2.1075, 2.1075, 2.2470, 2.2470, 2.4110, 2.4110, 2.5750,
		      2.5750, 2.5750, 2.7590, 2.7590, 2.8250, 2.9340, 2.9340};
  double actLHE[28]= {4.23487, 8.05342, 2.21090, 5.69774, 2.57831, 5.21078,
		      2.54554, 5.14455, 2.51790, 5.08871, 2.49347, 5.03933,
		      2.47129, 4.99449, 2.45137, 4.95424, 2.43380, 4.91873,
		      2.41863, 4.88808, 1.65913, 0.74863, 4.86612, 1.65322,
		      0.74596, 4.84396, 1.64930, 0.744198};
  actHE.clear();
  for (int i=0; i<28; ++i) {
    HcalDDDRecConstants::HcalActiveLength act(ietaHE[i],depthHE[i],etaHE[i],actLHE[i]);
    actHE.push_back(act);
  }
}

HcalDDDRecConstants::~HcalDDDRecConstants() {
  std::cout << "HcalDDDRecConstants::destructed!!!" << std::endl;
}

std::vector<HcalDDDRecConstants::HcalActiveLength>
HcalDDDRecConstants::getThickActive(const int type) const {

  if (type == 0) return actHB;
  else           return actHE;
}

class HcalRaddamMuon : public edm::EDAnalyzer {

public:
  explicit HcalRaddamMuon(const edm::ParameterSet&);
  ~HcalRaddamMuon();

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

  // ----------member data ---------------------------
  std::vector<double> PtGlob ,track_cosmic_positionIX ,track_cosmic_positionIY , track_cosmic_positionIZ, track_cosmic_positionOX, track_cosmic_positionOY, track_cosmic_positionOZ;

  std::vector<double> track_cosmic_momentumOX,track_cosmic_momentumOY,track_cosmic_momentumOZ,track_cosmic_momentumIX,track_cosmic_momentumIY,track_cosmic_momentumIZ, track_cosmic_detIDinner, track_cosmic_detIDouter;
  std::vector<double> EtaGlob;
  std::vector<double> PhiGlob,chiGlobal,GlobalMuonHits,MatchedStat,GlobalTrckPt,GlobalTrckEta,GlobalTrckPhi;
  std::vector<double> TrackerLayer,innerTrackpt,innerTracketa,innerTrackphi;
  std::vector<double> NumPixelLayers,chiTracker,DxyTracker,DzTracker;
  std::vector<double> OuterTrackPt,OuterTrackEta,OuterTrackPhi,OuterTrackChi,OuterTrackHits,OuterTrackRHits;
  std::vector<double> trackerlayer_hits,No_pixelLayers,NormChi2,ImpactParameter;
  std::vector<double> Tight_GlobalMuonTrkFit,Tight_MuonHits,Tight_MatchedStations,Tight_LongPara,Tight_PixelHits,Tight_TrkerLayers,Tight_TransImpara,High_TrackLayers;
  std::vector<bool>   innerTrack, OuterTrack, GlobalTrack;
  std::vector<double> IsolationR04,IsolationR03;
  std::vector<double> Energy,MuonHcalEnergy,MuonEcalEnergy,MuonHOEnergy,MuonEcal3x3Energy,MuonHcal1x1Energy, Pmuon;
  std::vector<unsigned int> MuonEcalDetId,MuonHcalDetId,MuonEHcalDetId, MuonHcalHot, MuonHcalHotCalo;
  std::vector<double> MuonHcalDepth1Energy,MuonHcalDepth2Energy,MuonHcalDepth3Energy,MuonHcalDepth4Energy,MuonHcalDepth5Energy,MuonHcalDepth6Energy,MuonHcalDepth7Energy;
  std::vector<double> MuonHcalDepth1HotEnergy,MuonHcalDepth2HotEnergy,MuonHcalDepth3HotEnergy,MuonHcalDepth4HotEnergy,MuonHcalDepth5HotEnergy,MuonHcalDepth6HotEnergy,MuonHcalDepth7HotEnergy;

  //
  std::vector<double> MuonHcalDepth1EnergyCalo,MuonHcalDepth2EnergyCalo,MuonHcalDepth3EnergyCalo,MuonHcalDepth4EnergyCalo,MuonHcalDepth5EnergyCalo,MuonHcalDepth6EnergyCalo,MuonHcalDepth7EnergyCalo;
  std::vector<double> MuonHcalDepth1HotEnergyCalo,MuonHcalDepth2HotEnergyCalo,MuonHcalDepth3HotEnergyCalo,MuonHcalDepth4HotEnergyCalo,MuonHcalDepth5HotEnergyCalo,MuonHcalDepth6HotEnergyCalo,MuonHcalDepth7HotEnergyCalo;

  //
  std::vector<double> MuonHcalActiveLength;
  std::vector<HcalDDDRecConstants::HcalActiveLength> actHB , actHE;
  int maxDepth_,type; 
  edm::Service<TFileService> fs;
  //////////////////////////////////////////////////////
  HLTConfigProvider hltConfig_;
  int               verbosity_;
  bool              isAOD_, isSLHC_;
  std::string       hltlabel_;
  std::vector<std::string> all_triggers,all_triggers1,all_triggers2,all_triggers3,all_triggers4,all_triggers5;
  ////////////////////////////////////////////////////////////

  std::vector<bool>  isHB, isHE;
  TTree *TREE;
  std::vector<bool>  all_ifTriggerpassed;
  std::vector<bool>  muon_is_good, muon_global, muon_tracker, Trk_match_MuStat;
  std::vector<int>   hltresults;
  std::vector<float> energy_hb,time_hb;
  std::vector<std::string> hltpaths,TrigName_;
  std::vector<int>    v_RH_h3x3_ieta;
  std::vector<int>    v_RH_h3x3_iphi;
  std::vector<double> v_RH_h3x3_ene, PxGlob, PyGlob,PzGlob,Pthetha;
  std::vector<double>  PCharge,PChi2,PD0, PD0Error,dxyWithBS,dzWithBS,PdxyTrack, PdzTrack,PNormalizedChi2, PNDoF, PValidHits, PLostHits, NPvx, NPvy, NPvz, NQOverP, NQOverPError, NTrkMomentum, NRefPointX, NRefPointY, NRefPointZ;
  std::vector<bool> NTrkQuality;
  double h3x3, h3x3Calo; 
  unsigned int RunNumber, EventNumber , LumiNumber, BXNumber;
  double _RecoMuon1TrackIsoSumPtMaxCutValue_03, _RecoMuon1TrackIsoSumPtMaxCutValue_04; 
  int           ntriggers;
  edm::InputTag HLTriggerResults_;
  std::string theTrackQuality;
  edm::InputTag muonsrc_;
  std::vector <double> track_cosmic_xposition , track_cosmic_yposition, track_cosmic_zposition, track_cosmic_xmomentum,track_cosmic_ymomentum, track_cosmic_zmomentum, track_cosmic_rad, track_cosmic_detid;

  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hcal_;
  edm::EDGetTokenT<edm::TriggerResults>    tok_trigRes_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>         tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>   tok_hbhe_;
  edm::EDGetTokenT<reco::MuonCollection>   tok_muon_;
};

HcalRaddamMuon::HcalRaddamMuon(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  HLTriggerResults_ = iConfig.getUntrackedParameter<edm::InputTag>("HLTriggerResults_");
  muonsrc_          = iConfig.getUntrackedParameter<edm::InputTag>("MuonSource");
  verbosity_        = iConfig.getUntrackedParameter<int>("Verbosity",0);
  isAOD_            = iConfig.getUntrackedParameter<bool>("IsAOD",false);
  isSLHC_           = iConfig.getUntrackedParameter<bool>("IsSLHC",true);
  maxDepth_         = iConfig.getUntrackedParameter<int>("MaxDepth",4);

  if (maxDepth_ > 7)      maxDepth_ = 7;
  else if (maxDepth_ < 1) maxDepth_ = 4;

  tok_hcal_    = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","HcalHits"));
  tok_trigRes_ = consumes<edm::TriggerResults>(HLTriggerResults_);
  tok_recVtx_  = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_      = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  if (isAOD_) {
    tok_EB_    = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    tok_EE_    = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));
    if (isSLHC_) {
      tok_hbhe_= consumes<HBHERecHitCollection>(edm::InputTag("reducedHcalRecHits","hbheUpgradeReco"));
    } else {
      tok_hbhe_= consumes<HBHERecHitCollection>(edm::InputTag("reducedHcalRecHits", "hbhereco"));
    }
  } else {
    tok_EB_    = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
    tok_EE_    = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEE"));
    if (isSLHC_) {
      tok_hbhe_= consumes<HBHERecHitCollection>(edm::InputTag("hbheUpgradeReco"));
    } else {
      tok_hbhe_= consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
    }
  }
  tok_muon_    = consumes<reco::MuonCollection>(muonsrc_);
}

HcalRaddamMuon::~HcalRaddamMuon() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}



//
// member functions
//

// ------------ method called for each event  ------------
void HcalRaddamMuon::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  clearVectors();
  RunNumber   = iEvent.id().run();
  EventNumber = iEvent.id().event();
  LumiNumber  = iEvent.id().luminosityBlock();
  BXNumber = iEvent.bunchCrossing();
 
  edm::Handle<edm::PCaloHitContainer> calosimhits;
  iEvent.getByToken(tok_hcal_,calosimhits);
 
  edm::Handle<edm::TriggerResults> _Triggers;
  iEvent.getByToken(tok_trigRes_,_Triggers); 
  
  if ((verbosity_%10)>1) std::cout << "size of all triggers " 
				   << all_triggers.size() << std::endl;
  int Ntriggers = all_triggers.size();
  
  if ((verbosity_%10)>1) std::cout << "size of HLT MENU: "
				   << _Triggers->size() << std::endl;
  
  if (_Triggers.isValid()) {
    const edm::TriggerNames &triggerNames_ = iEvent.triggerNames(*_Triggers);
    
    std::vector<int> index;
    for (int i=0;i < Ntriggers;i++) {
      index.push_back(triggerNames_.triggerIndex(all_triggers[i]));
      int triggerSize =int( _Triggers->size());
      if ((verbosity_%10)>2) std::cout << "outside loop " << index[i]
				       << "\ntriggerSize " << triggerSize
				       << std::endl;
      if (index[i] < triggerSize) {
        hltresults.push_back(_Triggers->accept(index[i])) ;
	if ((verbosity_%10)>2) std::cout << "trigger_info " << triggerSize
					 << " triggerSize " << index[i]
					 << " trigger_index " << hltresults.at(i)
					 << " hltresult " << std::endl;
      } else {
	edm::LogInfo("TriggerBlock") << "Requested HLT path \"" <<  "\" does not exist";
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
  iSetup.get<IdealGeometryRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();

  edm::Handle<reco::BeamSpot> bmspot;
  iEvent.getByToken(tok_bs_,bmspot);

  edm::Handle<reco::VertexCollection> vtx;
  iEvent.getByToken(tok_recVtx_,vtx);
  
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EB_,barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_,endcapRecHitsHandle);
  
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_,hbhe);
    
  edm::Handle<reco::MuonCollection> _Muon;
  iEvent.getByToken(tok_muon_,_Muon);
  const reco::Vertex& vertex = (*(vtx)->begin());  
 
  math::XYZPoint bspot;
  bspot= (bmspot.isValid()) ? bmspot->position() : math::XYZPoint(0,0,0);
  
  if (_Muon.isValid()) { 
    for (reco::MuonCollection::const_iterator RecMuon = _Muon->begin(); RecMuon!= _Muon->end(); ++RecMuon)  {
      muon_is_good.push_back(RecMuon->isPFMuon());
      muon_global.push_back(RecMuon->isGlobalMuon());
      muon_tracker.push_back(RecMuon->isTrackerMuon());
      PtGlob.push_back((RecMuon)->pt());
      EtaGlob.push_back(RecMuon->eta());
      PhiGlob.push_back(RecMuon->phi());
      Energy.push_back(RecMuon->energy());	
      Pmuon.push_back(RecMuon->p());
      //      if (RecMuon->isPFMuon()) goodEvent = true;
      // acessing tracker hits info
      if (RecMuon->track().isNonnull()) {
	TrackerLayer.push_back(RecMuon->track()->hitPattern().trackerLayersWithMeasurement());
      } else {
	TrackerLayer.push_back(-1);
      }
      if (RecMuon->innerTrack().isNonnull()) {
	innerTrack.push_back(true);
	NumPixelLayers.push_back(RecMuon->innerTrack()->hitPattern().pixelLayersWithMeasurement());
	chiTracker.push_back(RecMuon->innerTrack()->normalizedChi2());
	DxyTracker.push_back(fabs(RecMuon->innerTrack()->dxy((vertex).position())));
	DzTracker.push_back(fabs(RecMuon->innerTrack()->dz((vertex).position())));
	innerTrackpt.push_back(RecMuon->innerTrack()->pt());
	innerTracketa.push_back(RecMuon->innerTrack()->eta());
	innerTrackphi.push_back(RecMuon->innerTrack()->phi());
	Tight_PixelHits.push_back(RecMuon->innerTrack()->hitPattern().numberOfValidPixelHits());
      } else {
	innerTrack.push_back(false);
	NumPixelLayers.push_back(0);
	chiTracker.push_back(0);
	DxyTracker.push_back(0);
	DzTracker.push_back(0);
	innerTrackpt.push_back(0);
	innerTracketa.push_back(0);
	innerTrackphi.push_back(0);
	Tight_PixelHits.push_back(0);
      }
      // outer track info
      
      if (RecMuon->outerTrack().isNonnull()) {
	OuterTrack.push_back(true);
	OuterTrackPt.push_back(RecMuon->outerTrack()->pt());
	OuterTrackEta.push_back(RecMuon->outerTrack()->eta());
	OuterTrackPhi.push_back(RecMuon->outerTrack()->phi());
	OuterTrackChi.push_back(RecMuon->outerTrack()->normalizedChi2());
	OuterTrackHits.push_back(RecMuon->outerTrack()->numberOfValidHits());
	OuterTrackRHits.push_back(RecMuon->outerTrack()->recHitsSize());
      } else {
	OuterTrack.push_back(false);
	OuterTrackPt.push_back(0);
	OuterTrackEta.push_back(0);
	OuterTrackPhi.push_back(0);
	OuterTrackChi.push_back(0);
	OuterTrackHits.push_back(0);
	OuterTrackRHits.push_back(0);
      }
      // Tight Muon cuts
      if (RecMuon->globalTrack().isNonnull())  {	 	
	GlobalTrack.push_back(true);
	chiGlobal.push_back(RecMuon->globalTrack()->normalizedChi2());
	GlobalMuonHits.push_back(RecMuon->globalTrack()->hitPattern().numberOfValidMuonHits());
	MatchedStat.push_back(RecMuon->numberOfMatchedStations());  
	GlobalTrckPt.push_back(RecMuon->globalTrack()->pt());
	GlobalTrckEta.push_back(RecMuon->globalTrack()->eta());
	GlobalTrckPhi.push_back(RecMuon->globalTrack()->phi()); 
	Tight_TransImpara.push_back(fabs(RecMuon->muonBestTrack()->dxy(vertex.position())));
	Tight_LongPara.push_back(fabs(RecMuon->muonBestTrack()->dz(vertex.position())));
      } else {
	GlobalTrack.push_back(false);
	chiGlobal.push_back(0);
	GlobalMuonHits.push_back(0);
	MatchedStat.push_back(0);
	GlobalTrckPt.push_back(0);
	GlobalTrckEta.push_back(0);
	GlobalTrckPhi.push_back(0);
	Tight_TransImpara.push_back(0);
	Tight_LongPara.push_back(0);
      }
    
      IsolationR04.push_back(((RecMuon->pfIsolationR04().sumChargedHadronPt + std::max(0.,RecMuon->pfIsolationR04().sumNeutralHadronEt + RecMuon->pfIsolationR04().sumPhotonEt - (0.5 *RecMuon->pfIsolationR04().sumPUPt))) / RecMuon->pt()) );
      
      IsolationR03.push_back(((RecMuon->pfIsolationR03().sumChargedHadronPt + std::max(0.,RecMuon->pfIsolationR03().sumNeutralHadronEt + RecMuon->pfIsolationR03().sumPhotonEt - (0.5 * RecMuon->pfIsolationR03().sumPUPt))) / RecMuon->pt()));                                                                              

      MuonEcalEnergy.push_back(RecMuon->calEnergy().emS9);		 
      MuonHcalEnergy.push_back(RecMuon->calEnergy().hadS9);
      MuonHOEnergy.push_back(RecMuon->calEnergy().hoS9);
      
      double eEcal(0),eHcal(0),activeL(0),eHcalDepth[7],eHcalDepthHot[7],eHcalDepthCalo[7],eHcalDepthHotCalo[7];
      unsigned int isHot = 0;
      unsigned int isHotCalo = 0;

      for (int i=0; i<7; ++i) eHcalDepth[i]=eHcalDepthHot[i]=eHcalDepthCalo[i]=eHcalDepthHotCalo[i]=-10000 ;
      
      if (RecMuon->innerTrack().isNonnull()) {
	const reco::Track* pTrack = (RecMuon->innerTrack()).get();
	spr::propagatedTrackID trackID = spr::propagateCALO(pTrack, geo, bField, (((verbosity_/100)%10>0)));
	
	MuonEcalDetId.push_back((trackID.detIdECAL)()); 
	MuonHcalDetId.push_back((trackID.detIdHCAL)());  
	MuonEHcalDetId.push_back((trackID.detIdEHCAL)());  
	
	if(trackID.okECAL){
	  const DetId isoCell(trackID.detIdECAL);
	  std::pair<double,bool> e3x3 = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle,*theEcalChStatus,geo,caloTopology,sevlv.product(),1,1,-100.0,-100.0,-500.0,500.0,false);
	  
	  eEcal = e3x3.first;
	  //std::cout<<"eEcal"<<eEcal<<std::endl;
	}
	
	if (trackID.okHCAL) {
	  const DetId closestCell(trackID.detIdHCAL);
	  eHcal = spr::eHCALmatrix(theHBHETopology, closestCell, hbhe,0,0, false, true, -100.0, -100.0, -100.0, -100.0, -500.,500.);
	  
	  //std::cout<<"eHcal"<<eHcal<<std::endl;
	  std::vector<std::pair<double,int> > ehdepth;
	  spr::energyHCALCell((HcalDetId) closestCell, hbhe, ehdepth, maxDepth_, -100.0, -100.0, -100.0, -100.0, -500.0, 500.0, (((verbosity_/1000)%10)>0));
	  for (unsigned int i=0; i<ehdepth.size(); ++i) {
	    eHcalDepth[ehdepth[i].second-1] = ehdepth[i].first;
	    //std::cout<<eHcalDepth[ehdepth[i].second-1]<<std::endl;
	  }

	  eHcal = spr::eHCALmatrix(theHBHETopology, closestCell, calosimhits,0,0, false, true, -100.0, -100.0, -100.0, -100.0, -500.,500.);
	  
	  //std::cout<<"eHcal"<<eHcal<<std::endl;
	  const DetId closestCellCalo(trackID.detIdHCAL);
	  std::vector<std::pair<double,int> > ehdepthCalo;
	  spr::energyHCALCell((HcalDetId) closestCellCalo, calosimhits, ehdepthCalo, maxDepth_, -100.0, -100.0, -100.0, -100.0, -500.0, 500.0, (((verbosity_/1000)%10)>0));
	  for (unsigned int i=0; i<ehdepthCalo.size(); ++i) {
	    eHcalDepthCalo[ehdepthCalo[i].second-1] = ehdepthCalo[i].first;
	    //std::cout<<eHcalDepth[ehdepth[i].second-1]<<std::endl;
	  }
	  
	  HcalDetId hcid0(closestCell.rawId());
	  activeL = activeLength(trackID.detIdHCAL);
	  
	  std::cout<<activeL<<std::endl;
	  HcalDetId hotCell, hotCellCalo;
	  h3x3 = spr::eHCALmatrix(geo,theHBHETopology, closestCell, hbhe, 1,1, hotCell, false, false);
	  h3x3Calo = spr::eHCALmatrix(geo,theHBHETopology, closestCellCalo, calosimhits, 1,1, hotCellCalo, false, false);

	  isHot = matchId(closestCell,hotCell);
	  isHotCalo = matchId(closestCellCalo,hotCellCalo);

	  // std::cout<<"hcal 3X3  < "<<h3x3<<">" << " ClosestCell <" << (HcalDetId)(closestCell) << "> hotCell id < " << hotCell << "> isHot" << isHot << std::endl;
	  if (hotCell != HcalDetId()) {
	    std::vector<std::pair<double,int> > ehdepth;
	    //   spr::energyHCALCell(hotCell, hbhe, ehdepth, maxDepth_, -100.0, -100.0, -100.0, -100.0, -500.0, 500.0, false);//(((verbosity_/1000)%10)>0    ));
	    spr::energyHCALCell(hotCell, hbhe, ehdepth, maxDepth_, -100.0, -100.0, -100.0, -100.0, -500.0, 500.0, false);
	    for (unsigned int i=0; i<ehdepth.size(); ++i) {
	      eHcalDepthHot[ehdepth[i].second-1] = ehdepth[i].first;
	      //  std::cout<<eHcalDepthHot[ehdepth[i].second-1]<<std::endl;
	    }
	  }

	  if (hotCellCalo != HcalDetId()) {
	    std::vector<std::pair<double,int> > ehdepthCalo;

	    spr::energyHCALCell(hotCellCalo, calosimhits, ehdepthCalo, maxDepth_, -100.0, -100.0, -100.0, -100.0, -500.0, 500.0, false);
	    for (unsigned int i=0; i<ehdepthCalo.size(); ++i) {
	      eHcalDepthHotCalo[ehdepthCalo[i].second-1] = ehdepthCalo[i].first;
	      //  std::cout<<eHcalDepthHot[ehdepth[i].second-1]<<std::endl;                                                                         
	    }
	  }
	}
      } else {
	MuonEcalDetId.push_back(0);
	MuonHcalDetId.push_back(0);
	MuonEHcalDetId.push_back(0);
      }

      MuonEcal3x3Energy.push_back(eEcal);
      MuonHcal1x1Energy.push_back(eHcal);
      MuonHcalDepth1Energy.push_back(eHcalDepth[0]);
      MuonHcalDepth2Energy.push_back(eHcalDepth[1]);
      MuonHcalDepth3Energy.push_back(eHcalDepth[2]);
      MuonHcalDepth4Energy.push_back(eHcalDepth[3]);
      MuonHcalDepth5Energy.push_back(eHcalDepth[4]);
      MuonHcalDepth6Energy.push_back(eHcalDepth[5]);
      MuonHcalDepth7Energy.push_back(eHcalDepth[6]);
      MuonHcalDepth1HotEnergy.push_back(eHcalDepthHot[0]);
      MuonHcalDepth2HotEnergy.push_back(eHcalDepthHot[1]);
      MuonHcalDepth3HotEnergy.push_back(eHcalDepthHot[2]);
      MuonHcalDepth4HotEnergy.push_back(eHcalDepthHot[3]);
      MuonHcalDepth5HotEnergy.push_back(eHcalDepthHot[4]);
      MuonHcalDepth6HotEnergy.push_back(eHcalDepthHot[5]);
      MuonHcalDepth7HotEnergy.push_back(eHcalDepthHot[6]);
      MuonHcalHot.push_back(isHot);
      
      //
      MuonHcalDepth1EnergyCalo.push_back(eHcalDepthCalo[0]);
      MuonHcalDepth2EnergyCalo.push_back(eHcalDepthCalo[1]);
      MuonHcalDepth3EnergyCalo.push_back(eHcalDepthCalo[2]);
      MuonHcalDepth4EnergyCalo.push_back(eHcalDepthCalo[3]);
      MuonHcalDepth5EnergyCalo.push_back(eHcalDepthCalo[4]);
      MuonHcalDepth6EnergyCalo.push_back(eHcalDepthCalo[5]);
      MuonHcalDepth7EnergyCalo.push_back(eHcalDepthCalo[6]);
      MuonHcalDepth1HotEnergyCalo.push_back(eHcalDepthHotCalo[0]);
      MuonHcalDepth2HotEnergyCalo.push_back(eHcalDepthHotCalo[1]);
      MuonHcalDepth3HotEnergyCalo.push_back(eHcalDepthHotCalo[2]);
      MuonHcalDepth4HotEnergyCalo.push_back(eHcalDepthHotCalo[3]);
      MuonHcalDepth5HotEnergyCalo.push_back(eHcalDepthHotCalo[4]);
      MuonHcalDepth6HotEnergyCalo.push_back(eHcalDepthHotCalo[5]);
      MuonHcalDepth7HotEnergyCalo.push_back(eHcalDepthHotCalo[6]);
      MuonHcalHotCalo.push_back(isHotCalo);
      
      //
      MuonHcalActiveLength.push_back(activeL);
    }
  }
  TREE->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void HcalRaddamMuon::beginJob() {
  
  TREE = fs->make<TTree>("TREE", "TREE");
  TREE->Branch("Event_No",&EventNumber);
  TREE->Branch("Run_No",&RunNumber);
  TREE->Branch("LumiNumber",&LumiNumber);
  TREE->Branch("BXNumber",&BXNumber);
  TREE->Branch("pt_of_muon",&PtGlob);
  TREE->Branch("eta_of_muon",&EtaGlob);
  TREE->Branch("phi_of_muon",&PhiGlob);
  TREE->Branch("energy_of_muon",&Energy);
  TREE->Branch("p_of_muon",&Pmuon);
  TREE->Branch("PF_Muon",&muon_is_good);
  TREE->Branch("Global_Muon",&muon_global);
  TREE->Branch("Tracker_muon",&muon_tracker);
  
  
  TREE->Branch("hcal_3into3",&MuonHcalEnergy);
  TREE->Branch("hcal_1x1",&MuonHcal1x1Energy);
  TREE->Branch("hcal_detID",&MuonHcalDetId);
  TREE->Branch("hcal_edepth1",&MuonHcalDepth1Energy);
  TREE->Branch("hcal_edepth2",&MuonHcalDepth2Energy);
  TREE->Branch("hcal_edepth3",&MuonHcalDepth3Energy);
  TREE->Branch("hcal_edepth4",&MuonHcalDepth4Energy);
  TREE->Branch("hcal_edepthHot1",&MuonHcalDepth1HotEnergy);
  TREE->Branch("hcal_edepthHot2",&MuonHcalDepth2HotEnergy);
  TREE->Branch("hcal_edepthHot3",&MuonHcalDepth3HotEnergy);
  TREE->Branch("hcal_edepthHot4",&MuonHcalDepth4HotEnergy);

  TREE->Branch("hcal_edepth1PSim",&MuonHcalDepth1EnergyCalo);
  TREE->Branch("hcal_edepth2PSim",&MuonHcalDepth2EnergyCalo);
  TREE->Branch("hcal_edepth3PSim",&MuonHcalDepth3EnergyCalo);
  TREE->Branch("hcal_edepth4PSim",&MuonHcalDepth4EnergyCalo);
  TREE->Branch("hcal_edepthHot1PSim",&MuonHcalDepth1HotEnergyCalo);
  TREE->Branch("hcal_edepthHot2PSim",&MuonHcalDepth2HotEnergyCalo);
  TREE->Branch("hcal_edepthHot3PSim",&MuonHcalDepth3HotEnergyCalo);
  TREE->Branch("hcal_edepthHot4PSim",&MuonHcalDepth4HotEnergyCalo);
 
  if (maxDepth_ > 4) {
    TREE->Branch("hcal_edepth5PSim",&MuonHcalDepth5EnergyCalo);
    TREE->Branch("hcal_edepthHot5PSim",&MuonHcalDepth5HotEnergyCalo);
    if (maxDepth_ > 5) {
      TREE->Branch("hcal_edepth6PSim",&MuonHcalDepth6EnergyCalo);
      TREE->Branch("hcal_edepthHot6PSim",&MuonHcalDepth6HotEnergyCalo);
      if (maxDepth_ > 6) {
	TREE->Branch("hcal_edepth7PSim",&MuonHcalDepth7EnergyCalo);
	TREE->Branch("hcal_edepthHot7PSim",&MuonHcalDepth7HotEnergyCalo);
      }
    }
  }
  
  TREE->Branch("TrackerLayer",&TrackerLayer);
  TREE->Branch("innerTrack",&innerTrack);
  TREE->Branch("innerTrackpt",&innerTrackpt);
  TREE->Branch("innerTracketa",&innerTracketa);
  TREE->Branch("innerTrackphi",&innerTrackphi);
  TREE->Branch("MatchedStat",&MatchedStat);
  TREE->Branch("GlobalTrckPt",&GlobalTrckPt);
  TREE->Branch("GlobalTrckEta",&GlobalTrckEta);
  TREE->Branch("GlobalTrckPhi",&GlobalTrckPhi);
  TREE->Branch("NumPixelLayers",&NumPixelLayers);
  TREE->Branch("chiTracker",&chiTracker);
  TREE->Branch("DxyTracker",&DxyTracker);
  TREE->Branch("DzTracker",&DzTracker);
  TREE->Branch("OuterTrack",&OuterTrack);
  TREE->Branch("OuterTrackPt",&OuterTrackPt);
  TREE->Branch("OuterTrackEta",&OuterTrackEta);
  TREE->Branch("OuterTrackPhi",&OuterTrackPhi);
  TREE->Branch("OuterTrackHits",&OuterTrackHits);
  TREE->Branch("OuterTrackRHits",&OuterTrackRHits);
  TREE->Branch("OuterTrackChi",&OuterTrackChi);
  TREE->Branch("GlobalTrack",&GlobalTrack);
  TREE->Branch("GlobTrack_Chi",&chiGlobal);
  TREE->Branch("Global_Muon_Hits",&GlobalMuonHits);
  TREE->Branch("MatchedStations",&MatchedStat);
  TREE->Branch("Global_Track_Pt",&GlobalTrckPt);
  TREE->Branch("Global_Track_Eta",&GlobalTrckEta);
  TREE->Branch("Global_Track_Phi",&GlobalTrckPhi);
  ///////////////////////////////
  TREE->Branch("Tight_LongitudinalImpactparameter",&Tight_LongPara);
  TREE->Branch("Tight_TransImpactparameter",&Tight_TransImpara);
  TREE->Branch("InnerTrackPixelHits",&Tight_PixelHits);
  TREE->Branch("IsolationR04",&IsolationR04);
  TREE->Branch("IsolationR03",&IsolationR03);

  TREE->Branch("hcal_cellHot",&MuonHcalHot);
  TREE->Branch("hcal_cellHotPSim",&MuonHcalHotCalo);

  TREE->Branch("ecal_3into3",&MuonEcalEnergy);
  TREE->Branch("ecal_3x3",&MuonEcal3x3Energy);
  TREE->Branch("ecal_detID",&MuonEcalDetId);
  TREE->Branch("ehcal_detID",&MuonEHcalDetId);
  TREE->Branch("tracker_3into3",&MuonHOEnergy);
  TREE->Branch("activeLength",&MuonHcalActiveLength);
  
  
  ///////////////////////////////
  TREE->Branch("hltresults",&hltresults);
  TREE->Branch("all_triggers",&all_triggers);
  TREE->Branch("rechit_energy",&energy_hb);
  TREE->Branch("rechit_time",&time_hb);
}

// ------------ method called once each job just after ending the event loop  ------------
void HcalRaddamMuon::endJob() {}

// ------------ method called when starting to processes a run  ------------
void HcalRaddamMuon::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  /*   edm::ESHandle<HcalDDDRecConstants> pHRNDC;
       iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);
       const HcalDDDRecConstants & hdc = (*pHRNDC);
  */
  
  HcalDDDRecConstants hdc;
  actHB.clear();
  actHE.clear();
  actHB = hdc.getThickActive(0);
  actHE = hdc.getThickActive(1);
   
  bool changed = true;
  all_triggers.clear();
  if (hltConfig_.init(iRun, iSetup,"HLT" , changed)) {
    // if init returns TRUE, initialisation has succeeded!
    edm::LogInfo("TriggerBlock") << "HLT config with process name " 
				 << "HLT" << " successfully extracted";
    std::string string_search[5]={"HLT_IsoMu_","HLT_L1SingleMu_","HLT_L2Mu","HLT_Mu","HLT_RelIso1p0Mu"};
    unsigned int ntriggers = hltConfig_.size();
    for(unsigned int t=0;t<ntriggers;++t){
      std::string hltname(hltConfig_.triggerName(t));
      for (unsigned int ik=0; ik<5; ++ik) {
	if (hltname.find(string_search[ik])!=std::string::npos ){
	  all_triggers.push_back(hltname);
	  break;
	}
      }
    }//loop over ntriggers
    //  std::cout<<"all triggers size in begin run"<<all_triggers.size()<<std::endl;
  } else {
    edm::LogError("TriggerBlock") << "Error! HLT config extraction with process name " 
				  << "HLT"<< " failed";
  }
  
}//firstmethod


// ------------ method called when ending the processing of a run  ------------
void HcalRaddamMuon::endRun(edm::Run const&, edm::EventSetup const&) { }

// ------------ method called when starting to processes a luminosity block  ------------
void HcalRaddamMuon::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) { }

// ------------ method called when ending the processing of a luminosity block  ------------
void HcalRaddamMuon::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) { }

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalRaddamMuon::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HcalRaddamMuon::clearVectors() {
  ///clearing vectots
  EventNumber = -99999;
  RunNumber = -99999;
  LumiNumber = -99999;
  BXNumber = -99999;
  energy_hb.clear();
  time_hb.clear();
  muon_is_good.clear();
  muon_global.clear();
  muon_tracker.clear();
  PtGlob.clear();
  EtaGlob.clear(); 
  PhiGlob.clear(); 
  Energy.clear();
  Pmuon.clear();
  TrackerLayer.clear();
  innerTrack.clear();
  NumPixelLayers.clear();
  chiTracker.clear();
  DxyTracker.clear();
  DzTracker.clear();
  innerTrackpt.clear();
  innerTracketa.clear();
  innerTrackphi.clear();
  Tight_PixelHits.clear();
  OuterTrack.clear();
  OuterTrackPt.clear();
  OuterTrackEta.clear();
  OuterTrackPhi.clear();
  OuterTrackHits.clear();
  OuterTrackRHits.clear();
  OuterTrackChi.clear();
  GlobalTrack.clear();
  chiGlobal.clear();
  GlobalMuonHits.clear();
  MatchedStat.clear();
  GlobalTrckPt.clear();
  GlobalTrckEta.clear();
  GlobalTrckPhi.clear();
  Tight_TransImpara.clear();
  Tight_LongPara.clear();

  IsolationR04.clear();
  IsolationR03.clear();
  MuonEcalEnergy.clear();
  MuonHcalEnergy.clear();
  MuonHOEnergy.clear();
  MuonEcalDetId.clear();
  MuonHcalDetId.clear();
  MuonEHcalDetId.clear();
  MuonEcal3x3Energy.clear();
  MuonHcal1x1Energy.clear();
  MuonHcalDepth1Energy.clear();
  MuonHcalDepth2Energy.clear();
  MuonHcalDepth3Energy.clear();
  MuonHcalDepth4Energy.clear();
  MuonHcalDepth5Energy.clear();
  MuonHcalDepth6Energy.clear();
  MuonHcalDepth7Energy.clear();

  MuonHcalDepth1HotEnergy.clear();
  MuonHcalDepth2HotEnergy.clear();
  MuonHcalDepth3HotEnergy.clear();
  MuonHcalDepth4HotEnergy.clear();
  MuonHcalDepth5HotEnergy.clear();
  MuonHcalDepth6HotEnergy.clear();
  MuonHcalDepth7HotEnergy.clear();
  MuonHcalHot.clear();
  MuonHcalActiveLength.clear();
  hltresults.clear();
}

int HcalRaddamMuon::matchId(const HcalDetId& id1, const HcalDetId& id2) {

  HcalDetId kd1(id1.subdet(),id1.ieta(),id1.iphi(),1);
  HcalDetId kd2(id2.subdet(),id2.ieta(),id2.iphi(),1);
  int match = ((kd1 == kd2) ? 1 : 0);
  return match;
}

double HcalRaddamMuon::activeLength(const DetId& id_) {
  HcalDetId id(id_);
  int ieta = id.ietaAbs();
  int depth= id.depth();
  double lx(0);
  if (id.subdet() == HcalBarrel) {
    //    std::cout<<"actHB.size()"<<actHB.size()<<std::endl;
    for (unsigned int i=0; i<actHB.size(); ++i) {
      if (ieta == actHB[i].ieta && depth == actHB[i].depth) {
	lx = actHB[i].thick;
	break;
      }
    }
  } else {
    //    std::cout<<"actHE.size()"<<actHE.size()<<std::endl; 
    for (unsigned int i=0; i<actHE.size(); ++i) {
      if (ieta == actHE[i].ieta && depth == actHE[i].depth) {
	lx = actHE[i].thick;
//  std::cout<<"actHE[i].thick"<<actHE[i].thick<<std::endl;
	break;
      }
    }
  }
  return lx;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalRaddamMuon);

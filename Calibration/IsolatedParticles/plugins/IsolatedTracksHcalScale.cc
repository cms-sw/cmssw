
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "Calibration/IsolatedParticles/plugins/IsolatedTracksHcalScale.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrixExtra.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

IsolatedTracksHcalScale::IsolatedTracksHcalScale(const edm::ParameterSet& iConfig) {

  //now do what ever initialization is needed
  doMC                                = iConfig.getUntrackedParameter<bool>("DoMC", false); 
  myverbose                           = iConfig.getUntrackedParameter<int>("Verbosity", 5          );
  theTrackQuality                     = iConfig.getUntrackedParameter<std::string>("TrackQuality","highPurity");
  reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);
  selectionParameters.minPt           = iConfig.getUntrackedParameter<double>("MinTrackPt", 10.0);
  selectionParameters.minQuality      = trackQuality_;
  selectionParameters.maxDxyPV        = iConfig.getUntrackedParameter<double>("MaxDxyPV", 0.2);
  selectionParameters.maxDzPV         = iConfig.getUntrackedParameter<double>("MaxDzPV",  5.0);
  selectionParameters.maxChi2         = iConfig.getUntrackedParameter<double>("MaxChi2",  5.0);
  selectionParameters.minOuterHit     = iConfig.getUntrackedParameter<int>("MinOuterHit", 4);
  selectionParameters.minLayerCrossed = iConfig.getUntrackedParameter<int>("MinLayerCrossed", 8);
  selectionParameters.maxInMiss       = iConfig.getUntrackedParameter<int>("MaxInMiss", 0);
  selectionParameters.maxOutMiss      = iConfig.getUntrackedParameter<int>("MaxOutMiss", 0);
  a_coneR                             = iConfig.getUntrackedParameter<double>("ConeRadius",34.98);
  a_charIsoR                          = a_coneR + 28.9;
  a_neutIsoR                          = a_charIsoR*0.726;
  a_mipR                              = iConfig.getUntrackedParameter<double>("ConeRadiusMIP",14.0);
  
  if(myverbose>=0) {
    std::cout <<"Parameters read from config file \n" 
	      <<" doMC "              << doMC
	      <<"\t myverbose "       << myverbose        
	      <<"\t minPt "           << selectionParameters.minPt   
              <<"\t theTrackQuality " << theTrackQuality
	      <<"\t minQuality "      << selectionParameters.minQuality          
	      <<"\t maxDxyPV "        << selectionParameters.maxDxyPV          
	      <<"\t maxDzPV "         << selectionParameters.maxDzPV          
	      <<"\t maxChi2 "         << selectionParameters.maxChi2          
	      <<"\t minOuterHit "     << selectionParameters.minOuterHit          
	      <<"\t minLayerCrossed " << selectionParameters.minLayerCrossed          
	      <<"\t maxInMiss "       << selectionParameters.maxInMiss          
	      <<"\t maxOutMiss "      << selectionParameters.maxOutMiss          
	      <<"\t a_coneR "         << a_coneR          
	      <<"\t a_charIsoR "      << a_charIsoR          
	      <<"\t a_neutIsoR "      << a_neutIsoR          
	      <<"\t a_mipR "          << a_mipR          
	      << std::endl;
  }
  initL1 = false;
  
}

IsolatedTracksHcalScale::~IsolatedTracksHcalScale() {}

void IsolatedTracksHcalScale::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  bField = bFieldH.product();

  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();

  /*  
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology); 
  const CaloTopology *caloTopology = theCaloTopology.product();
  
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<IdealGeometryRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();

  // Retrieve the good/bad ECAL channels from the DB
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus);
  const EcalChannelStatus* theEcalChStatus = ecalChStatus.product();

  // Retrieve trigger tower map

  //const edm::ESHandle<EcalTrigTowerConstituentsMap> hTtmap;
  edm::ESHandle<EcalTrigTowerConstituentsMap> hTtmap;
  iSetup.get<IdealGeometryRecord>().get(hTtmap);
  const EcalTrigTowerConstituentsMap& ttMap = *hTtmap;
  */

  clearTreeVectors();

  t_RunNo = iEvent.id().run();
  t_EvtNo = iEvent.id().event();
  t_Lumi  = iEvent.luminosityBlock();
  t_Bunch = iEvent.bunchCrossing();

  nEventProc++;

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByLabel("generalTracks", trkCollection);

  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByLabel("offlinePrimaryVertices",recVtxs);  

  // Get the beamspot
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByLabel("offlineBeamSpot", beamSpotH);

  math::XYZPoint leadPV(0,0,0);
  if (recVtxs->size()>0 && !((*recVtxs)[0].isFake())) {
    leadPV = math::XYZPoint( (*recVtxs)[0].x(),(*recVtxs)[0].y(), (*recVtxs)[0].z() );
  } else if (beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }

  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDirections, true);
  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
  
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByLabel("ecalRecHit","EcalRecHitsEB",barrelRecHitsHandle);
  iEvent.getByLabel("ecalRecHit","EcalRecHitsEE",endcapRecHitsHandle);

  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByLabel("hbhereco",hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());

  //get Handles to SimTracks and SimHits
  edm::Handle<edm::SimTrackContainer> SimTk;
  edm::SimTrackContainer::const_iterator simTrkItr;
  edm::Handle<edm::SimVertexContainer> SimVtx;
  edm::SimVertexContainer::const_iterator vtxItr = SimVtx->begin();

  //get Handles to PCaloHitContainers of eb/ee/hbhe
  edm::Handle<edm::PCaloHitContainer> pcaloeb;
  edm::Handle<edm::PCaloHitContainer> pcaloee;
  edm::Handle<edm::PCaloHitContainer> pcalohh;

  //associates tracker rechits/simhits to a track
  TrackerHitAssociator* associate=0;
 
  if (doMC) {
    iEvent.getByLabel("g4SimHits",SimTk);
    iEvent.getByLabel("g4SimHits",SimVtx);
    iEvent.getByLabel("g4SimHits", "EcalHitsEB", pcaloeb);
    iEvent.getByLabel("g4SimHits", "EcalHitsEE", pcaloee);
    iEvent.getByLabel("g4SimHits", "HcalHits", pcalohh);
    associate = new TrackerHitAssociator::TrackerHitAssociator(iEvent);
  }
 
  unsigned int nTracks=0;
  for (trkDetItr = trkCaloDirections.begin(),nTracks=0; trkDetItr != trkCaloDirections.end(); trkDetItr++,nTracks++){
    const reco::Track* pTrack = &(*(trkDetItr->trkItr));
    if (spr::goodTrack(pTrack,leadPV,selectionParameters,true) && trkDetItr->okECAL && trkDetItr->okHCAL) {
      int                nRH_eMipDR=0, nRH_eDR=0, nNearTRKs=0, nRecHitsCone=-99;
      double             distFromHotCell = -99.0;
      int                ietaHotCell=-99, iphiHotCell=-99;
      GlobalPoint        gposHotCell(0.,0.,0.);
      std::vector<DetId> coneRecHitDetIds;

      double hCone = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, trkDetItr->pointECAL,
                                     a_coneR, trkDetItr->directionHCAL, nRecHitsCone,
                                     coneRecHitDetIds, distFromHotCell,
                                     ietaHotCell, iphiHotCell, gposHotCell);

      double conehmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR, nNearTRKs, false);

      double eMipDR  = spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
				       trkDetItr->pointHCAL, trkDetItr->pointECAL,
				       a_mipR, trkDetItr->directionECAL, nRH_eMipDR);
      double eECALDR = spr::eCone_ecal(geo, barrelRecHitsHandle,  endcapRecHitsHandle,
				       trkDetItr->pointHCAL, trkDetItr->pointECAL,
				       a_neutIsoR, trkDetItr->directionECAL, nRH_eDR);

      HcalDetId closestCell = (HcalDetId)(trkDetItr->detIdHCAL);

      
      // Fill the tree Branches here 
      t_trackP                ->push_back( pTrack->p() );
      t_trackPt               ->push_back( pTrack->pt() );
      t_trackEta              ->push_back( pTrack->momentum().eta() );
      t_trackPhi              ->push_back( pTrack->momentum().phi() );
      t_trackHcalEta          ->push_back( closestCell.ieta() );
      t_trackHcalPhi          ->push_back( closestCell.iphi() );
      t_hCone                 ->push_back( hCone);
      t_conehmaxNearP         ->push_back( conehmaxNearP);
      t_eMipDR                ->push_back( eMipDR);
      t_eECALDR               ->push_back( eECALDR);
    
    }
  }

  //  delete associate;
  if (associate) delete associate;
  tree->Fill();
}

void IsolatedTracksHcalScale::beginJob() {

  nEventProc=0;


  //////Book Tree
  tree = fs->make<TTree>("tree", "tree");
  tree->SetAutoSave(10000);

  tree->Branch("t_RunNo"              ,&t_RunNo               ,"t_RunNo/I");
  tree->Branch("t_Lumi"               ,&t_Lumi                ,"t_Lumi/I");
  tree->Branch("t_Bunch"              ,&t_Bunch               ,"t_Bunch/I");

  t_trackP              = new std::vector<double>();
  t_trackPt             = new std::vector<double>();
  t_trackEta            = new std::vector<double>();
  t_trackPhi            = new std::vector<double>();
  t_trackHcalEta        = new std::vector<double>();
  t_trackHcalPhi        = new std::vector<double>();
  t_hCone               = new std::vector<double>();
  t_conehmaxNearP       = new std::vector<double>();
  t_eMipDR              = new std::vector<double>();
  t_eECALDR             = new std::vector<double>();

  tree->Branch("t_trackP",            "vector<double>", &t_trackP            );
  tree->Branch("t_trackPt",           "vector<double>", &t_trackPt           );
  tree->Branch("t_trackEta",          "vector<double>", &t_trackEta          );
  tree->Branch("t_trackPhi",          "vector<double>", &t_trackPhi          );
  tree->Branch("t_trackHcalEta",      "vector<double>", &t_trackHcalEta      );
  tree->Branch("t_trackHcalPhi",      "vector<double>", &t_trackHcalPhi      );
  tree->Branch("t_hCone",             "vector<double>", &t_hCone     );
  tree->Branch("t_conehmaxNearP",     "vector<double>", &t_conehmaxNearP     );
  tree->Branch("t_eMipDR",            "vector<double>", &t_eMipDR     );
  tree->Branch("t_eECALDR",           "vector<double>", &t_eECALDR     );

}

void IsolatedTracksHcalScale::endJob() {

  std::cout << "Number of Events Processed " << nEventProc << std::endl;
}

void IsolatedTracksHcalScale::clearTreeVectors() {

  t_trackP            ->clear();
  t_trackPt           ->clear();
  t_trackEta          ->clear();
  t_trackPhi          ->clear();
  t_trackHcalEta      ->clear();
  t_trackHcalPhi      ->clear();
  t_hCone             ->clear();
  t_conehmaxNearP     ->clear();
  t_eMipDR            ->clear();
  t_eECALDR           ->clear();


}


//define this as a plug-in
DEFINE_FWK_MODULE(IsolatedTracksHcalScale);


#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
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
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

IsolatedTracksHcalScale::IsolatedTracksHcalScale(const edm::ParameterSet& iConfig) :
   trackerHitAssociatorConfig_(consumesCollector()) {

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
  selectionParameters.maxDpOverP      = iConfig.getUntrackedParameter<double>("MaxDpOverP",  0.1);
  selectionParameters.minOuterHit     = iConfig.getUntrackedParameter<int>("MinOuterHit", 4);
  selectionParameters.minLayerCrossed = iConfig.getUntrackedParameter<int>("MinLayerCrossed", 8);
  selectionParameters.maxInMiss       = iConfig.getUntrackedParameter<int>("MaxInMiss", 0);
  selectionParameters.maxOutMiss      = iConfig.getUntrackedParameter<int>("MaxOutMiss", 0);
  a_coneR                             = iConfig.getUntrackedParameter<double>("ConeRadius",34.98);
  a_charIsoR                          = a_coneR + 28.9;
  a_neutIsoR                          = a_charIsoR*0.726;
  a_mipR                              = iConfig.getUntrackedParameter<double>("ConeRadiusMIP",14.0);
  tMinE_                              = iConfig.getUntrackedParameter<double>("TimeMinCutECAL", -500.);
  tMaxE_                              = iConfig.getUntrackedParameter<double>("TimeMaxCutECAL",  500.);
  
  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_   = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_       = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  tok_EB_       = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  tok_EE_       = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEE"));
  tok_hbhe_     = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  tok_simTk_    = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  tok_simVtx_   = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
  tok_caloEB_   = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEB"));
  tok_caloEE_   = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEE"));
  tok_caloHH_   = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "HcalHits"));

  if (myverbose>=0) {
    std::cout <<"Parameters read from config file \n" 
	      <<" doMC "              << doMC
	      <<"\t myverbose "       << myverbose        
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
	      <<"\t a_neutIsoR "      << a_neutIsoR          
	      <<"\t a_mipR "          << a_mipR 
              <<"\t time Range ("     << tMinE_ << ":" << tMaxE_ << ")"
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

  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology); 
  const CaloTopology *caloTopology = theCaloTopology.product();

  // Retrieve the good/bad ECAL channels from the DB
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus);
  const EcalChannelStatus* theEcalChStatus = ecalChStatus.product();

  clearTreeVectors();

  nEventProc++;

  t_RunNo = iEvent.id().run();
  t_EvtNo = iEvent.id().event();
  t_Lumi  = iEvent.luminosityBlock();
  t_Bunch = iEvent.bunchCrossing();
  if (myverbose>0) std::cout << nEventProc << " Run " << t_RunNo << " Event " << t_EvtNo << " Lumi " << t_Lumi << " Bunch " << t_Bunch << std::endl;

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);

  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_,recVtxs);

  // Get the beamspot
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);

  math::XYZPoint leadPV(0,0,0);
  if (recVtxs->size()>0 && !((*recVtxs)[0].isFake())) {
    leadPV = math::XYZPoint( (*recVtxs)[0].x(),(*recVtxs)[0].y(), (*recVtxs)[0].z() );
  } else if (beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }

  if (myverbose>0) {
    std::cout << "Primary Vertex " << leadPV;
    if (beamSpotH.isValid()) std::cout << " Beam Spot " << beamSpotH->position();
    std::cout << std::endl;
  }

  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDirections, (myverbose>2));
  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
  
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EB_,barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_,endcapRecHitsHandle);

  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());

  //get Handles to SimTracks and SimHits
  edm::Handle<edm::SimTrackContainer> SimTk;
  edm::SimTrackContainer::const_iterator simTrkItr;
  edm::Handle<edm::SimVertexContainer> SimVtx;

  //get Handles to PCaloHitContainers of eb/ee/hbhe
  edm::Handle<edm::PCaloHitContainer> pcaloeb;
  edm::Handle<edm::PCaloHitContainer> pcaloee;
  edm::Handle<edm::PCaloHitContainer> pcalohh;

  //associates tracker rechits/simhits to a track
  std::unique_ptr<TrackerHitAssociator> associate;
 
  if (doMC) {
    iEvent.getByToken(tok_simTk_,SimTk);
    iEvent.getByToken(tok_simVtx_,SimVtx);
    iEvent.getByToken(tok_caloEB_, pcaloeb);
    iEvent.getByToken(tok_caloEE_, pcaloee);
    iEvent.getByToken(tok_caloHH_, pcalohh);
    associate.reset(new TrackerHitAssociator(iEvent, trackerHitAssociatorConfig_));
  }
 
  unsigned int nTracks=0;
  for (trkDetItr = trkCaloDirections.begin(),nTracks=0; trkDetItr != trkCaloDirections.end(); trkDetItr++,nTracks++){
    const reco::Track* pTrack = &(*(trkDetItr->trkItr));
    if (spr::goodTrack(pTrack,leadPV,selectionParameters,(myverbose>2)) && trkDetItr->okECAL && trkDetItr->okHCAL) {
      int                nRH_eMipDR=0, nRH_eDR=0, nNearTRKs=0, nRecHitsCone=-99;
      double             distFromHotCell=-99.0, distFromHotCell2=-99.0;
      int                ietaHotCell=-99, iphiHotCell=-99;
      int                ietaHotCell2=-99, iphiHotCell2=-99;
      GlobalPoint        gposHotCell(0.,0.,0.), gposHotCell2(0.,0.,0.);
      std::vector<DetId> coneRecHitDetIds, coneRecHitDetIds2;
      std::pair<double, bool> e11x11_20SigP, e15x15_20SigP;
      double hCone = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, 
				     trkDetItr->pointECAL,
                                     a_coneR, trkDetItr->directionHCAL, 
				     nRecHitsCone, coneRecHitDetIds,
				     distFromHotCell, ietaHotCell, iphiHotCell,
				     gposHotCell, -1);
      double hConeHB = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, 
				       trkDetItr->pointECAL,
				       a_coneR, trkDetItr->directionHCAL, 
				       nRecHitsCone, coneRecHitDetIds,
				       distFromHotCell, ietaHotCell,
				       iphiHotCell, gposHotCell,
				       (int)(HcalBarrel));
      double eHCALDR = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, 
				       trkDetItr->pointECAL, a_charIsoR, 
				       trkDetItr->directionHCAL, nRecHitsCone,
				       coneRecHitDetIds2, distFromHotCell2,
				       ietaHotCell2, iphiHotCell2, gposHotCell2,
				       -1);
      double eHCALDRHB = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, 
					 trkDetItr->pointECAL, a_charIsoR, 
					 trkDetItr->directionHCAL, nRecHitsCone,
					 coneRecHitDetIds2, distFromHotCell2,
					 ietaHotCell2, iphiHotCell2,
					 gposHotCell2, (int)(HcalBarrel));

      double conehmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR, nNearTRKs, (myverbose>3));

      double eMipDR  = spr::eCone_ecal(geo, barrelRecHitsHandle, 
				       endcapRecHitsHandle,trkDetItr->pointHCAL,
				       trkDetItr->pointECAL, a_mipR, 
				       trkDetItr->directionECAL, nRH_eMipDR);
      double eECALDR = spr::eCone_ecal(geo, barrelRecHitsHandle,  
				       endcapRecHitsHandle,trkDetItr->pointHCAL,
				       trkDetItr->pointECAL, a_neutIsoR, 
				       trkDetItr->directionECAL, nRH_eDR);
      double eMipDR_1= spr::eCone_ecal(geo, barrelRecHitsHandle, 
				       endcapRecHitsHandle,trkDetItr->pointHCAL,
				       trkDetItr->pointECAL, a_mipR, 
				       trkDetItr->directionECAL, nRH_eMipDR,
				       0.030, 0.150);
      double eECALDR_1=spr::eCone_ecal(geo, barrelRecHitsHandle,
				       endcapRecHitsHandle,trkDetItr->pointHCAL,
				       trkDetItr->pointECAL, a_neutIsoR,
				       trkDetItr->directionECAL, nRH_eDR,
				       0.030, 0.150);
      double eMipDR_2= spr::eCone_ecal(geo, barrelRecHitsHandle,
				       endcapRecHitsHandle,trkDetItr->pointHCAL,
				       trkDetItr->pointECAL, a_mipR,
				       trkDetItr->directionECAL, nRH_eMipDR,
				       0.060, 0.300);
      double eECALDR_2=spr::eCone_ecal(geo, barrelRecHitsHandle,
				       endcapRecHitsHandle,trkDetItr->pointHCAL,
				       trkDetItr->pointECAL, a_neutIsoR,
				       trkDetItr->directionECAL, nRH_eDR,
				       0.060, 0.300);

      HcalDetId closestCell = (HcalDetId)(trkDetItr->detIdHCAL);

      edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
      iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);

      e11x11_20SigP = spr::eECALmatrix(trkDetItr->detIdECAL,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,sevlv.product(),5,5,   0.060,  0.300, tMinE_,tMaxE_);
      e15x15_20SigP = spr::eECALmatrix(trkDetItr->detIdECAL,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,sevlv.product(),7,7,   0.060,  0.300, tMinE_,tMaxE_);
      
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
      t_eHCALDR               ->push_back( eHCALDR);
      t_e11x11_20Sig          ->push_back( e11x11_20SigP.first );
      t_e15x15_20Sig          ->push_back( e15x15_20SigP.first );
      t_eMipDR_1              ->push_back( eMipDR_1);
      t_eECALDR_1             ->push_back( eECALDR_1);
      t_eMipDR_2              ->push_back( eMipDR_2);
      t_eECALDR_2             ->push_back( eECALDR_2);
      t_hConeHB               ->push_back( hConeHB);
      t_eHCALDRHB             ->push_back( eHCALDRHB);

      if (myverbose > 0) {
	std::cout << "Track p " << pTrack->p() << " pt " << pTrack->pt()
		  << " eta " << pTrack->momentum().eta() << " phi "
		  << pTrack->momentum().phi() << " ieta/iphi ("
		  << closestCell.ieta() << ", " << closestCell.iphi() 
		  << ") Energy in cone " << hCone << " Charge Isolation "
		  << conehmaxNearP << " eMIP (" << eMipDR << ", "
		  << eMipDR_1 << ", " << eMipDR_2 << ")"
		  << " Neutral isolation (ECAL) (" << eECALDR-eMipDR << ", "
		  << eECALDR_1-eMipDR_1 << ", " << eECALDR_2-eMipDR_2 << ")"
		  << " (ECAL NxN) " << e15x15_20SigP.first-e11x11_20SigP.first
		  << " (HCAL) " << eHCALDR-hCone << std::endl;
      }

      if (doMC) {
	int nSimHits = -999;
	double hsim;
	std::map<std::string, double> hsimInfo;
	std::vector<int> multiplicity;
        hsim = spr::eCone_hcal(geo, pcalohh, trkDetItr->pointHCAL, 
			       trkDetItr->pointECAL, a_coneR, 
			       trkDetItr->directionHCAL, nSimHits);
	hsimInfo = spr::eHCALSimInfoCone(iEvent, pcalohh, SimTk, SimVtx, 
					 pTrack, *associate, geo, 
					 trkDetItr->pointHCAL, 
					 trkDetItr->pointECAL, a_coneR,
					 trkDetItr->directionHCAL,
					 multiplicity);

        t_hsimInfoMatched   ->push_back(hsimInfo["eMatched"   ]);
        t_hsimInfoRest      ->push_back(hsimInfo["eRest"      ]);
        t_hsimInfoPhoton    ->push_back(hsimInfo["eGamma"     ]);
        t_hsimInfoNeutHad   ->push_back(hsimInfo["eNeutralHad"]);
        t_hsimInfoCharHad   ->push_back(hsimInfo["eChargedHad"]);
        t_hsimInfoPdgMatched->push_back(hsimInfo["pdgMatched" ]);
        t_hsimInfoTotal     ->push_back(hsimInfo["eTotal"     ]);

        t_hsimInfoNMatched  ->push_back(multiplicity.at(0));
        t_hsimInfoNTotal    ->push_back(multiplicity.at(1));
        t_hsimInfoNNeutHad  ->push_back(multiplicity.at(2));
        t_hsimInfoNCharHad  ->push_back(multiplicity.at(3));
        t_hsimInfoNPhoton   ->push_back(multiplicity.at(4));
        t_hsimInfoNRest     ->push_back(multiplicity.at(5));

        t_hsim              ->push_back(hsim                   );
        t_nSimHits          ->push_back(nSimHits               );

	if (myverbose > 0) {
	  std::cout << "Matched (E) " << hsimInfo["eMatched"] << " (N) "
		    << multiplicity.at(0) << " Rest (E) " << hsimInfo["eRest"] 
		    << " (N) " << multiplicity.at(1) << " Gamma (E) "
		    << hsimInfo["eGamma"] << " (N) "  << multiplicity.at(2) 
		    << " Neutral Had (E) " << hsimInfo["eNeutralHad"]  
		    << " (N) "  << multiplicity.at(3) << " Charged Had (E) "
		    << hsimInfo["eChargedHad"] << " (N) " << multiplicity.at(4)
		    << " Total (E) " << hsimInfo["eTotal"] << " (N) "
		    << multiplicity.at(5) << " PDG " << hsimInfo["pdgMatched"] 
		    << " Total E " << hsim << " NHit " << nSimHits <<std::endl;
	}
      }
    }
  }

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
  t_eHCALDR             = new std::vector<double>();
  t_e11x11_20Sig        = new std::vector<double>();
  t_e15x15_20Sig        = new std::vector<double>();
  t_eMipDR_1            = new std::vector<double>();
  t_eECALDR_1           = new std::vector<double>();
  t_eMipDR_2            = new std::vector<double>();
  t_eECALDR_2           = new std::vector<double>();
  t_hConeHB             = new std::vector<double>();
  t_eHCALDRHB           = new std::vector<double>();

  tree->Branch("t_trackP",            "vector<double>", &t_trackP           );
  tree->Branch("t_trackPt",           "vector<double>", &t_trackPt          );
  tree->Branch("t_trackEta",          "vector<double>", &t_trackEta         );
  tree->Branch("t_trackPhi",          "vector<double>", &t_trackPhi         );
  tree->Branch("t_trackHcalEta",      "vector<double>", &t_trackHcalEta     );
  tree->Branch("t_trackHcalPhi",      "vector<double>", &t_trackHcalPhi     );
  tree->Branch("t_hCone",             "vector<double>", &t_hCone            );
  tree->Branch("t_conehmaxNearP",     "vector<double>", &t_conehmaxNearP    );
  tree->Branch("t_eMipDR",            "vector<double>", &t_eMipDR           );
  tree->Branch("t_eECALDR",           "vector<double>", &t_eECALDR          );
  tree->Branch("t_eHCALDR",           "vector<double>", &t_eHCALDR          );
  tree->Branch("t_e11x11_20Sig",      "vector<double>", &t_e11x11_20Sig     );
  tree->Branch("t_e15x15_20Sig",      "vector<double>", &t_e15x15_20Sig     );
  tree->Branch("t_eMipDR_1",          "vector<double>", &t_eMipDR_1         );
  tree->Branch("t_eECALDR_1",         "vector<double>", &t_eECALDR_1        );
  tree->Branch("t_eMipDR_2",          "vector<double>", &t_eMipDR_2         );
  tree->Branch("t_eECALDR_2",         "vector<double>", &t_eECALDR_2        );
  tree->Branch("t_hConeHB",           "vector<double>", &t_hConeHB          );
  tree->Branch("t_eHCALDRHB",         "vector<double>", &t_eHCALDRHB        );

  if (doMC) {
    t_hsimInfoMatched    = new std::vector<double>();
    t_hsimInfoRest       = new std::vector<double>();
    t_hsimInfoPhoton     = new std::vector<double>();
    t_hsimInfoNeutHad    = new std::vector<double>();
    t_hsimInfoCharHad    = new std::vector<double>();
    t_hsimInfoPdgMatched = new std::vector<double>();
    t_hsimInfoTotal      = new std::vector<double>();
    t_hsimInfoNMatched   = new std::vector<int>();
    t_hsimInfoNTotal     = new std::vector<int>();
    t_hsimInfoNNeutHad   = new std::vector<int>();
    t_hsimInfoNCharHad   = new std::vector<int>();
    t_hsimInfoNPhoton    = new std::vector<int>();
    t_hsimInfoNRest      = new std::vector<int>();
    t_hsim               = new std::vector<double>();
    t_nSimHits           = new std::vector<int>();

    tree->Branch("t_hsimInfoMatched",    "vector<double>", &t_hsimInfoMatched    );
    tree->Branch("t_hsimInfoRest",       "vector<double>", &t_hsimInfoRest    );
    tree->Branch("t_hsimInfoPhoton",     "vector<double>", &t_hsimInfoPhoton    );
    tree->Branch("t_hsimInfoNeutHad",    "vector<double>", &t_hsimInfoNeutHad    );
    tree->Branch("t_hsimInfoCharHad",    "vector<double>", &t_hsimInfoCharHad    );
    tree->Branch("t_hsimInfoPdgMatched", "vector<double>", &t_hsimInfoPdgMatched );
    tree->Branch("t_hsimInfoTotal",      "vector<double>", &t_hsimInfoTotal    );
    tree->Branch("t_hsimInfoNMatched",   "vector<int>",    &t_hsimInfoNMatched    );
    tree->Branch("t_hsimInfoNTotal",     "vector<int>",    &t_hsimInfoNTotal    );
    tree->Branch("t_hsimInfoNNeutHad",   "vector<int>",    &t_hsimInfoNNeutHad    );
    tree->Branch("t_hsimInfoNCharHad",   "vector<int>",    &t_hsimInfoNCharHad    );
    tree->Branch("t_hsimInfoNPhoton",    "vector<int>",    &t_hsimInfoNPhoton    );
    tree->Branch("t_hsimInfoNRest",      "vector<int>",    &t_hsimInfoNRest    );
    tree->Branch("t_hsim",               "vector<double>", &t_hsim    );
    tree->Branch("t_nSimHits",           "vector<int>",    &t_nSimHits    );
  }
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
  t_eHCALDR           ->clear();
  t_e11x11_20Sig      ->clear();
  t_e15x15_20Sig      ->clear();
  t_eMipDR_1          ->clear();
  t_eECALDR_1         ->clear();
  t_eMipDR_2          ->clear();
  t_eECALDR_2         ->clear();
  t_hConeHB           ->clear();
  t_eHCALDRHB         ->clear();

  if (doMC) {
    t_hsimInfoMatched    ->clear();
    t_hsimInfoRest       ->clear();
    t_hsimInfoPhoton     ->clear();
    t_hsimInfoNeutHad    ->clear();
    t_hsimInfoCharHad    ->clear();
    t_hsimInfoPdgMatched ->clear();
    t_hsimInfoTotal      ->clear();
    t_hsimInfoNMatched   ->clear();
    t_hsimInfoNTotal     ->clear();
    t_hsimInfoNNeutHad   ->clear();
    t_hsimInfoNCharHad   ->clear();
    t_hsimInfoNPhoton    ->clear();
    t_hsimInfoNRest      ->clear();
    t_hsim               ->clear();
    t_nSimHits           ->clear();
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(IsolatedTracksHcalScale);

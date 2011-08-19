// -*- C++ -*
//
// Package:    IsolatedTracksNxN
// Class:      IsolatedTracksNxN
// 
/**\class IsolatedTracksNxN IsolatedTracksNxN.cc Calibration/IsolatedParticles/plugins/IsolatedTracksNxN.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seema Sharma
//         Created:  Mon Aug 10 15:30:40 CST 2009
//
//

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "Calibration/IsolatedParticles/plugins/IsolatedTracksNxN.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrixExtra.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

IsolatedTracksNxN::IsolatedTracksNxN(const edm::ParameterSet& iConfig) {

  //now do what ever initialization is needed
  doMC                   = iConfig.getUntrackedParameter<bool>  ("DoMC", false); 
  myverbose_             = iConfig.getUntrackedParameter<int>   ("Verbosity", 5          );
  pvTracksPtMin_         = iConfig.getUntrackedParameter<double>("PVTracksPtMin", 1.0    );
  debugTrks_             = iConfig.getUntrackedParameter<int>   ("DebugTracks"           );
  printTrkHitPattern_    = iConfig.getUntrackedParameter<bool>  ("PrintTrkHitPattern"    );
  minTrackP_             = iConfig.getUntrackedParameter<double>("minTrackP", 1.0        );
  maxTrackEta_           = iConfig.getUntrackedParameter<double>("maxTrackEta", 5.0      );
  debugL1Info_           = iConfig.getUntrackedParameter<bool>  ("DebugL1Info", false    );
  L1extraTauJetSource_   = iConfig.getParameter<edm::InputTag>  ("L1extraTauJetSource"   );
  L1extraCenJetSource_   = iConfig.getParameter<edm::InputTag>  ("L1extraCenJetSource"   );
  L1extraFwdJetSource_   = iConfig.getParameter<edm::InputTag>  ("L1extraFwdJetSource"   );
  L1extraMuonSource_     = iConfig.getParameter<edm::InputTag>  ("L1extraMuonSource"     );
  L1extraIsoEmSource_    = iConfig.getParameter<edm::InputTag>  ("L1extraIsoEmSource"    );
  L1extraNonIsoEmSource_ = iConfig.getParameter<edm::InputTag>  ("L1extraNonIsoEmSource" );
  L1GTReadoutRcdSource_  = iConfig.getParameter<edm::InputTag>  ("L1GTReadoutRcdSource"  );
  L1GTObjectMapRcdSource_= iConfig.getParameter<edm::InputTag>  ("L1GTObjectMapRcdSource");
  JetSrc_                = iConfig.getParameter<edm::InputTag>  ("JetSource");
  JetExtender_           = iConfig.getParameter<edm::InputTag>  ("JetExtender");
  tMinE_                 = iConfig.getUntrackedParameter<double>("TimeMinCutECAL", -500.);
  tMaxE_                 = iConfig.getUntrackedParameter<double>("TimeMaxCutECAL",  500.);
  tMinH_                 = iConfig.getUntrackedParameter<double>("TimeMinCutHCAL", -500.);
  tMaxH_                 = iConfig.getUntrackedParameter<double>("TimeMaxCutHCAL",  500.);

  //edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  //parameters_.loadParameters( parameters );
  //trackAssociator_ =  new TrackDetectorAssociator();
  //trackAssociator_->useDefaultPropagator();

  if(myverbose_>=0) {
    std::cout <<"Parameters read from config file \n" 
	      <<" doMC         "      << doMC
	      <<"\t myverbose_ "      << myverbose_        
	      <<"\t minTrackP_ "      << minTrackP_     
	      << "\t maxTrackEta_ " << maxTrackEta_    
	      << "\t tMinE_ " << tMinE_
	      << "\t tMaxE_ " << tMaxE_
	      << "\t tMinH_ " << tMinH_
	      << "\t tMaxH_ " << tMaxH_      << "\n"
	      << std::endl;
  }

  initL1 = false;

}

IsolatedTracksNxN::~IsolatedTracksNxN() {
  //delete  trackAssociator_;
}

void IsolatedTracksNxN::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //  std::cout<< tMinE_ << " " << tMaxE_ << " HCAL " << tMinH_ << " " << tMaxH_ << std::endl;

  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  bField = bFieldH.product();

  clearTreeVectors();

  t_RunNo = iEvent.id().run();
  t_EvtNo = iEvent.id().event();
  t_Lumi  = iEvent.luminosityBlock();
  t_Bunch = iEvent.bunchCrossing();

  nEventProc++;

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByLabel("generalTracks", trkCollection);
  reco::TrackCollection::const_iterator trkItr;
  if(debugTrks_>1){
    std::cout << "Track Collection: " << std::endl;
    std::cout << "Number of Tracks " << trkCollection->size() << std::endl;
  }
  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);

  //===================== save L1 Trigger information =======================
  // get L1TriggerReadout records
  edm::Handle<L1GlobalTriggerReadoutRecord> gtRecord;
  iEvent.getByLabel(L1GTReadoutRcdSource_,  gtRecord);
  
  // sanity check on L1 Trigger Records
  if (!gtRecord.isValid()) {
    std::cout << "\nL1GlobalTriggerReadoutRecord with \n \nnot found"
      "\n  --> returning false by default!\n" << std::endl;
  }

  edm::ESHandle<L1GtTriggerMenu> gtOMRec;
  iSetup.get<L1GtTriggerMenuRcd>().get(gtOMRec) ;

  // L1 decision word  
  const DecisionWord dWord = gtRecord->decisionWord();  
  unsigned int numberTriggerBits= dWord.size();

  // just print the L1Bit number and AlgoName in first event
  /*
  if ( !initL1){
    initL1=true;
    std::cout << "\n  Number of Trigger bits " << numberTriggerBits << "\n\n";
    std::cout << "\tBit \t L1 Algorithm " << std::endl;

    // get ObjectMaps from ObjectMapRecord
    const std::vector<L1GlobalTriggerObjectMap>& objMapVec =  gtOMRec->gtObjectMap();
    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
	 itMap != objMapVec.end(); ++itMap) {

      // Get trigger bits
      int itrig = (*itMap).algoBitNumber();

      // Get trigger names
      algoBitToName[itrig] = (*itMap).algoName();
      std::cout  << "\t" << itrig << "\t" << algoBitToName[itrig] << std::endl;      

      // store the algoNames as bin labels of a histogram
      h_L1AlgoNames->GetXaxis()->SetBinLabel(itrig+1, algoBitToName[itrig].c_str());
    } // end of for loop    
  } // end of initL1
  */

  if ( !initL1){
    initL1=true;
    //  std::cout << "\n  Number of Trigger bits " << numberTriggerBits << "\n\n";
    //std::cout << "\tBit \t L1 Algorithm " << std::endl;
    int itrig=0; 
    for (CItAlgo algo = (*gtOMRec.product()).gtAlgorithmMap().begin(); algo!=(*gtOMRec.product()).gtAlgorithmMap().end(); ++algo, itrig++) {
      //      std::cout << "Name: " << (algo->second).algoName() << " Alias: " << (algo->second).algoAlias() << std::endl;
      algoBitToName[itrig] = (algo->second).algoName() ;
    }
  } // end of initL1
  
  for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {  
    bool accept = dWord[iBit];
    //if (accept) h_L1Results2->Fill(float(iBit));
    
    // fill the trigger map
    typedef std::map<std::string,bool>::value_type valType;
    trig_iter=l1TriggerMap.find(algoBitToName[iBit]);
    if (trig_iter==l1TriggerMap.end())
      l1TriggerMap.insert(valType(algoBitToName[iBit],accept));
    else
      trig_iter->second=accept;
  }


  // save L1 decision for each event 
  for (unsigned int iBit = 0; iBit < numberTriggerBits; ++iBit) {  

    bool accept = dWord[iBit];
    t_L1Decision->push_back(accept);

    // fill the trigger map
    if(debugL1Info_) std::cout << "Bit " << iBit << " " << algoBitToName[iBit] << " " << accept << std::endl;

    if(accept) h_L1AlgoNames->Fill(iBit);
  }

  //===================

  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByLabel("offlinePrimaryVertices",recVtxs);
  
  std::vector<reco::Track> svTracks;
  math::XYZPoint leadPV(0,0,0);
  double sumPtMax = -1.0;
  for(unsigned int ind=0; ind < recVtxs->size(); ind++) {
    if (!((*recVtxs)[ind].isFake())) {
     
      double vtxTrkSumPt=0.0, vtxTrkSumPtWt=0.0;
      int    vtxTrkNWt =0;
      double vtxTrkSumPtHP=0.0, vtxTrkSumPtHPWt=0.0;
      int    vtxTrkNHP =0, vtxTrkNHPWt =0;

      reco::Vertex::trackRef_iterator vtxTrack = (*recVtxs)[ind].tracks_begin();
      for (vtxTrack = (*recVtxs)[ind].tracks_begin(); vtxTrack!=(*recVtxs)[ind].tracks_end(); vtxTrack++) {

	bool trkQuality  = (*vtxTrack)->quality(trackQuality_);

	if((*vtxTrack)->pt()<pvTracksPtMin_) continue;
	
	vtxTrkSumPt += (*vtxTrack)->pt();
	if( trkQuality ) {
	  vtxTrkSumPtHP += (*vtxTrack)->pt();
	  vtxTrkNHP++;
	}

	double weight = (*recVtxs)[ind].trackWeight(*vtxTrack);
	h_PVTracksWt ->Fill(weight);
	if( weight>0.5) {
	  vtxTrkSumPtWt += (*vtxTrack)->pt();
	  vtxTrkNWt++;
	  if( trkQuality ) {
	    vtxTrkSumPtHPWt += (*vtxTrack)->pt();
	    vtxTrkNHPWt++;
	  }
	}
      }

      if(vtxTrkSumPt>sumPtMax) {
	sumPtMax = vtxTrkSumPt;
	leadPV = math::XYZPoint( (*recVtxs)[ind].x(),(*recVtxs)[ind].y(), (*recVtxs)[ind].z() );
      } 

      t_PVx            ->push_back( (*recVtxs)[ind].x() );
      t_PVy            ->push_back( (*recVtxs)[ind].y() );
      t_PVz            ->push_back( (*recVtxs)[ind].z() );
      t_PVisValid      ->push_back( (*recVtxs)[ind].isValid() );
      t_PVNTracks      ->push_back( (*recVtxs)[ind].tracksSize() );
      t_PVndof         ->push_back( (*recVtxs)[ind].ndof() );
      t_PVNTracksWt    ->push_back( vtxTrkNWt );
      t_PVTracksSumPt  ->push_back( vtxTrkSumPt );
      t_PVTracksSumPtWt->push_back( vtxTrkSumPtWt );

      t_PVNTracksHP      ->push_back( vtxTrkNHP );
      t_PVNTracksHPWt    ->push_back( vtxTrkNHPWt );
      t_PVTracksSumPtHP  ->push_back( vtxTrkSumPtHP );
      t_PVTracksSumPtHPWt->push_back( vtxTrkSumPtHPWt );
      if(myverbose_==4) {
	std::cout<<"PV "<<ind<<" isValid "<<(*recVtxs)[ind].isValid()<<" isFake "<<(*recVtxs)[ind].isFake()
		 <<" hasRefittedTracks() "<<ind<<" "<<(*recVtxs)[ind].hasRefittedTracks()
		 <<" refittedTrksSize "<<(*recVtxs)[ind].refittedTracks().size()
		 <<"  tracksSize() "<<(*recVtxs)[ind].tracksSize()<<" sumPt "<<vtxTrkSumPt
		 <<std::endl;
      }
    }
  }

  // Get the beamspot
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByLabel("offlineBeamSpot", beamSpotH);
  math::XYZPoint bspot;
  bspot = ( beamSpotH.isValid() ) ? beamSpotH->position() : math::XYZPoint(0, 0, 0);

  //===================

  // L1Taus 
  edm::Handle<l1extra::L1JetParticleCollection> l1TauHandle;
  iEvent.getByLabel(L1extraTauJetSource_,l1TauHandle);
  l1extra::L1JetParticleCollection::const_iterator itr;
  for(itr = l1TauHandle->begin(); itr != l1TauHandle->end(); ++itr ) {
    t_L1TauJetPt      ->push_back( itr->pt() );
    t_L1TauJetEta     ->push_back( itr->eta() );
    t_L1TauJetPhi     ->push_back( itr->phi() );
    if(debugL1Info_) {
      std::cout << "tauJ p/pt  " << itr->momentum() << " " << itr->pt() 
		<< "  eta/phi " << itr->eta() << " " << itr->phi()
		<< std::endl;
    }
  }

  // L1 Central Jets
  edm::Handle<l1extra::L1JetParticleCollection> l1CenJetHandle;
  iEvent.getByLabel(L1extraCenJetSource_,l1CenJetHandle);
  for( itr = l1CenJetHandle->begin();  itr != l1CenJetHandle->end(); ++itr ) {
    t_L1CenJetPt    ->push_back( itr->pt() );
    t_L1CenJetEta   ->push_back( itr->eta() );
    t_L1CenJetPhi   ->push_back( itr->phi() );
    if(debugL1Info_) {
      std::cout << "cenJ p/pt     " << itr->momentum() << " " << itr->pt() 
		<< "  eta/phi " << itr->eta() << " " << itr->phi()
		<< std::endl;
    }
  }
  
  // L1 Forward Jets
  edm::Handle<l1extra::L1JetParticleCollection> l1FwdJetHandle;
  iEvent.getByLabel(L1extraFwdJetSource_,l1FwdJetHandle);
  for( itr = l1FwdJetHandle->begin();  itr != l1FwdJetHandle->end(); ++itr ) {
    t_L1FwdJetPt    ->push_back( itr->pt() );
    t_L1FwdJetEta   ->push_back( itr->eta() );
    t_L1FwdJetPhi   ->push_back( itr->phi() );
    if(debugL1Info_) {
      std::cout << "fwdJ p/pt     " << itr->momentum() << " " << itr->pt() 
		<< "  eta/phi " << itr->eta() << " " << itr->phi()
		<< std::endl;
    }
  }

  // L1 Isolated EM onjects
  l1extra::L1EmParticleCollection::const_iterator itrEm;
  edm::Handle<l1extra::L1EmParticleCollection> l1IsoEmHandle ;
  iEvent.getByLabel(L1extraIsoEmSource_, l1IsoEmHandle);
  for( itrEm = l1IsoEmHandle->begin();  itrEm != l1IsoEmHandle->end(); ++itrEm ) {
    t_L1IsoEMPt     ->push_back(  itrEm->pt() );
    t_L1IsoEMEta    ->push_back(  itrEm->eta() );
    t_L1IsoEMPhi    ->push_back(  itrEm->phi() );
    if(debugL1Info_) {
      std::cout << "isoEm p/pt    " << itrEm->momentum() << " " << itrEm->pt() 
		<< "  eta/phi " << itrEm->eta() << " " << itrEm->phi()
		<< std::endl;
    }
  }
  
  // L1 Non-Isolated EM onjects
  edm::Handle<l1extra::L1EmParticleCollection> l1NonIsoEmHandle ;
  iEvent.getByLabel(L1extraNonIsoEmSource_, l1NonIsoEmHandle);
  for( itrEm = l1NonIsoEmHandle->begin();  itrEm != l1NonIsoEmHandle->end(); ++itrEm ) {
    t_L1NonIsoEMPt  ->push_back( itrEm->pt() );
    t_L1NonIsoEMEta ->push_back( itrEm->eta() );
    t_L1NonIsoEMPhi ->push_back( itrEm->phi() );
    if(debugL1Info_) {
      std::cout << "nonIsoEm p/pt " << itrEm->momentum() << " " << itrEm->pt() 
		<< "  eta/phi " << itrEm->eta() << " " << itrEm->phi()
		<< std::endl;
    }
  }
  
  // L1 Muons
  l1extra::L1MuonParticleCollection::const_iterator itrMu;
  edm::Handle<l1extra::L1MuonParticleCollection> l1MuHandle ;
  iEvent.getByLabel(L1extraMuonSource_, l1MuHandle);
  for( itrMu = l1MuHandle->begin();  itrMu != l1MuHandle->end(); ++itrMu ) {
    t_L1MuonPt      ->push_back( itrMu->pt() );
    t_L1MuonEta     ->push_back( itrMu->eta() );
    t_L1MuonPhi     ->push_back( itrMu->phi() );
    if(debugL1Info_) {
      std::cout << "l1muon p/pt   " << itrMu->momentum() << " " << itrMu->pt() 
		<< "  eta/phi " << itrMu->eta() << " " << itrMu->phi()
		<< std::endl;
    }
  }
  //=====================================================================

  edm::Handle<reco::CaloJetCollection> jets;
  iEvent.getByLabel(JetSrc_,jets);
  edm::Handle<reco::JetExtendedAssociation::Container> jetExtender;
  iEvent.getByLabel(JetExtender_,jetExtender);

  for(unsigned int ijet=0;ijet<(*jets).size();ijet++) {
    t_jetPt       ->push_back( (*jets)[ijet].pt()     );
    t_jetEta      ->push_back( (*jets)[ijet].eta()    );
    t_jetPhi      ->push_back( (*jets)[ijet].phi()    );
    //t_nTrksJetVtx  ->push_back( reco::JetExtendedAssociation::tracksAtVertexNumber(*jetExtender,(*jets)[ijet]) );
    //t_nTrksJetCalo ->push_back( reco::JetExtendedAssociation::tracksAtCaloNumber  (*jetExtender,(*jets)[ijet])   );
    t_nTrksJetVtx  ->push_back( -1.0);
    t_nTrksJetCalo ->push_back( -1.0);
  }

  //=====================================================================

  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology); 
  const CaloTopology *caloTopology = theCaloTopology.product();
  
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<IdealGeometryRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();
  
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByLabel("ecalRecHit","EcalRecHitsEB",barrelRecHitsHandle);
  iEvent.getByLabel("ecalRecHit","EcalRecHitsEE",endcapRecHitsHandle);

  // Retrieve the good/bad ECAL channels from the DB
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus);
  const EcalChannelStatus* theEcalChStatus = ecalChStatus.product();

  // Retrieve trigger tower map
  //const edm::ESHandle<EcalTrigTowerConstituentsMap> hTtmap;
  edm::ESHandle<EcalTrigTowerConstituentsMap> hTtmap;
  iSetup.get<IdealGeometryRecord>().get(hTtmap);
  const EcalTrigTowerConstituentsMap& ttMap = *hTtmap;

  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByLabel("hbhereco",hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());

  //get Handles to SimTracks and SimHits
  edm::Handle<edm::SimTrackContainer> SimTk;
  if (doMC) iEvent.getByLabel("g4SimHits",SimTk);
  edm::SimTrackContainer::const_iterator simTrkItr;

  edm::Handle<edm::SimVertexContainer> SimVtx;
  if (doMC) iEvent.getByLabel("g4SimHits",SimVtx);
  edm::SimVertexContainer::const_iterator vtxItr = SimVtx->begin();

  //get Handles to PCaloHitContainers of eb/ee/hbhe
  edm::Handle<edm::PCaloHitContainer> pcaloeb;
  if (doMC) iEvent.getByLabel("g4SimHits", "EcalHitsEB", pcaloeb);

  edm::Handle<edm::PCaloHitContainer> pcaloee;
  if (doMC) iEvent.getByLabel("g4SimHits", "EcalHitsEE", pcaloee);

  edm::Handle<edm::PCaloHitContainer> pcalohh;
  if (doMC) iEvent.getByLabel("g4SimHits", "HcalHits", pcalohh);
  
  //associates tracker rechits/simhits to a track
  TrackerHitAssociator* associate=0;
  if (doMC) associate = new TrackerHitAssociator(iEvent);
  
  std::vector<int>  ifGood(trkCollection->size(), 1);

  h_nTracks->Fill(trkCollection->size());
  
  int nTracks        = 0;
  t_nTracks          = 0;

  t_nTracks = trkCollection->size();

  //decide the goodness of each track for this track collection
  for( trkItr = trkCollection->begin(),nTracks=0; trkItr != trkCollection->end(); ++trkItr, nTracks++){
    const reco::Track* pTrack = &(*trkItr);
    bool   trkQuality  = pTrack->quality(trackQuality_);
    if( !trkQuality ) ifGood[nTracks]=0;
  }

  // get the list of DetIds closest to the impact point of track on surface calorimeters
  std::vector<spr::propagatedTrackID> trkCaloDets;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDets, false);
  std::vector<spr::propagatedTrackID>::const_iterator trkDetItr;

  if(myverbose_>2) {
    for(trkDetItr = trkCaloDets.begin(); trkDetItr != trkCaloDets.end(); trkDetItr++){
      std::cout<<trkDetItr->trkItr->p()<<" "<<trkDetItr->trkItr->eta()<<" "<<trkDetItr->okECAL<<" ";
      if(trkDetItr->detIdECAL.subdetId() == EcalBarrel) std::cout << (EBDetId)trkDetItr->detIdECAL <<" ";
      else                                              std::cout << (EEDetId)trkDetItr->detIdECAL <<" ";
      std::cout<<trkDetItr->okHCAL<<" ";
      if(trkDetItr->okHCAL) std::cout<<(HcalDetId)trkDetItr->detIdHCAL;
      std::cout << std::endl;
    }
  }

  for(trkDetItr = trkCaloDets.begin(),nTracks=0; trkDetItr != trkCaloDets.end(); trkDetItr++,nTracks++){
    const reco::Track* pTrack = &(*(trkDetItr->trkItr));
    
    const reco::HitPattern& hitp = pTrack->hitPattern();
    int nLayersCrossed = hitp.trackerLayersWithMeasurement() ;        
    int nOuterHits     = hitp.stripTOBLayersWithMeasurement()+hitp.stripTECLayersWithMeasurement() ;
    const reco::HitPattern& hitpIn  = pTrack->trackerExpectedHitsInner();
    const reco::HitPattern& hitpOut = pTrack->trackerExpectedHitsOuter();
    
    double eta1        = pTrack->momentum().eta();
    double phi1        = pTrack->momentum().phi();
    double etaEcal1    = trkDetItr->etaECAL;
    double phiEcal1    = trkDetItr->phiECAL;
    double etaHcal1    = trkDetItr->etaHCAL;
    double phiHcal1    = trkDetItr->phiHCAL;
    double pt1         = pTrack->pt();
    double p1          = pTrack->p();
    double dxy1        = pTrack->dxy();
    double dz1         = pTrack->dz();
    double dxybs1      = beamSpotH.isValid() ? pTrack->dxy(bspot) : pTrack->dxy();	
    double dzbs1       = beamSpotH.isValid() ? pTrack->dz(bspot)  : pTrack->dz();	
    double dxypv1      = (recVtxs->size()>0 && !((*recVtxs)[0].isFake()))  ? pTrack->dxy(leadPV) : pTrack->dxy();	
    double dzpv1       = (recVtxs->size()>0 && !((*recVtxs)[0].isFake()))  ? pTrack->dz(leadPV)  : pTrack->dz();	
    double chisq1      = pTrack->normalizedChi2();

    h_recEtaPt_0->Fill(eta1, pt1);
    h_recEtaP_0 ->Fill(eta1, p1);
    h_recPt_0   ->Fill(pt1);
    h_recP_0    ->Fill(p1);
    h_recEta_0  ->Fill(eta1);
    h_recPhi_0  ->Fill(phi1);
    
    if(ifGood[nTracks] && nLayersCrossed>7 ) {
      h_recEtaPt_1->Fill(eta1, pt1);
      h_recEtaP_1 ->Fill(eta1, p1);
      h_recPt_1   ->Fill(pt1);
      h_recP_1    ->Fill(p1);


      h_recEta_1  ->Fill(eta1);
      h_recPhi_1  ->Fill(phi1);
    }
    
    if( ! ifGood[nTracks] ) continue;
    if( pt1>2.0 && nLayersCrossed>7) {
      t_trackPAll             ->push_back( p1 );
      t_trackEtaAll           ->push_back( eta1 );
      t_trackPhiAll           ->push_back( phi1 );
      t_trackPtAll            ->push_back( pt1 );
      t_trackDxyAll           ->push_back( dxy1 );	
      t_trackDzAll            ->push_back( dz1 );	
      t_trackDxyPVAll         ->push_back( dxypv1 );	
      t_trackDzPVAll          ->push_back( dzpv1 );	
      t_trackChiSqAll         ->push_back( chisq1 );	
    }
    if (doMC) {
      edm::SimTrackContainer::const_iterator matchedSimTrkAll = spr::matchedSimTrack(iEvent, SimTk, SimVtx, pTrack, *associate, false); 
      if( matchedSimTrkAll != SimTk->end())     t_trackPdgIdAll->push_back( matchedSimTrkAll->type() );
    }
    
    if( pt1>minTrackP_ && std::abs(eta1)<maxTrackEta_ && trkDetItr->okECAL) { 
      
      double maxNearP31x31=999.0, maxNearP25x25=999.0, maxNearP21x21=999.0, maxNearP15x15=999.0;
      //double maxNearP13x13=999.0, maxNearP11x11=999.0, maxNearP9x9  =999.0, maxNearP7x7  =999.0;

      maxNearP31x31 = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology, 15,15);
      maxNearP25x25 = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology, 12,12);
      maxNearP21x21 = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology, 10,10);
      maxNearP15x15 = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology,  7, 7);
      //maxNearP13x13 = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology,  6, 6);
      //maxNearP11x11 = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology,  5, 5);
      //maxNearP9x9   = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology,  4, 4);
      //maxNearP7x7   = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology,  3, 3);

      int iTrkEtaBin=-1, iTrkMomBin=-1;
      for(unsigned int ieta=0; ieta<NEtaBins; ieta++) {
        if(std::abs(eta1)>genPartEtaBins[ieta] && std::abs(eta1)<genPartEtaBins[ieta+1] ) iTrkEtaBin = ieta;
      }
      for(unsigned int ipt=0;  ipt<NPBins;   ipt++) {
        if( p1>genPartPBins[ipt] &&  p1<genPartPBins[ipt+1] )  iTrkMomBin = ipt;
      }
      if( iTrkMomBin>=0 && iTrkEtaBin>=0 ) {
	h_maxNearP31x31[iTrkMomBin][iTrkEtaBin]->Fill( maxNearP31x31 );
	h_maxNearP25x25[iTrkMomBin][iTrkEtaBin]->Fill( maxNearP25x25 );
	h_maxNearP21x21[iTrkMomBin][iTrkEtaBin]->Fill( maxNearP21x21 );
	h_maxNearP15x15[iTrkMomBin][iTrkEtaBin]->Fill( maxNearP15x15 );
      }
      if( maxNearP31x31<0.0 && nLayersCrossed>7 && nOuterHits>4) {
	h_recEtaPt_2->Fill(eta1, pt1);
	h_recEtaP_2 ->Fill(eta1, p1);
	h_recPt_2   ->Fill(pt1);
	h_recP_2    ->Fill(p1);
	h_recEta_2  ->Fill(eta1);
	h_recPhi_2  ->Fill(phi1);
      }
      
      // if isolated in 7x7 then store the further quantities
      //if( maxNearP7x7<0.0) {
      if( maxNearP31x31<0.0) {
	
	// get the matching simTrack
	double simTrackP = -1;
	if (doMC) {
	  edm::SimTrackContainer::const_iterator matchedSimTrk = spr::matchedSimTrack(iEvent, SimTk, SimVtx, pTrack, *associate, false);
	  if( matchedSimTrk != SimTk->end() )simTrackP = matchedSimTrk->momentum().P();
	}
	// get ECal Tranverse Profile
	//double e3x3=0, e5x5=0, e7x7=0, e9x9=0, e11x11=0, e13x13=0, e15x15=0, e21x21=0, e25x25=0, e31x31=0;
	//double e7x7_10Sig=0, e9x9_10Sig=0, e11x11_10Sig=0, e15x15_10Sig=0;
	//double e7x7_15Sig=0, e9x9_15Sig=0, e11x11_15Sig=0, e15x15_15Sig=0;
	//double e7x7_20Sig=0, e9x9_20Sig=0, e11x11_20Sig=0, e15x15_20Sig=0;
	//double e7x7_25Sig=0, e9x9_25Sig=0, e11x11_25Sig=0, e15x15_25Sig=0;
	//double e7x7_30Sig=0, e9x9_30Sig=0, e11x11_30Sig=0, e15x15_30Sig=0;

	std::pair<double, bool>  e3x3P, e5x5P, e7x7P, e9x9P, e11x11P, e13x13P, e15x15P, e21x21P, e25x25P, e31x31P;
	std::pair<double, bool>  e7x7_10SigP, e9x9_10SigP, e11x11_10SigP, e15x15_10SigP;
	std::pair<double, bool>  e7x7_15SigP, e9x9_15SigP, e11x11_15SigP, e15x15_15SigP;
	std::pair<double, bool>  e7x7_20SigP, e9x9_20SigP, e11x11_20SigP, e15x15_20SigP;
	std::pair<double, bool>  e7x7_25SigP, e9x9_25SigP, e11x11_25SigP, e15x15_25SigP;
	std::pair<double, bool>  e7x7_30SigP, e9x9_30SigP, e11x11_30SigP, e15x15_30SigP;

	spr::caloSimInfo simInfo3x3,   simInfo5x5,   simInfo7x7,   simInfo9x9;
	spr::caloSimInfo simInfo11x11, simInfo13x13, simInfo15x15, simInfo21x21, simInfo25x25, simInfo31x31;
	double trkEcalEne=0;

	const DetId isoCell = trkDetItr->detIdECAL;
	/*
	e3x3         = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,1,1,   -100.0, -100.0);
	e5x5         = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,2,2,   -100.0, -100.0);
	e7x7         = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,3,3,   -100.0, -100.0);
	e9x9         = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,4,4,   -100.0, -100.0);
	e11x11       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,5,5,   -100.0, -100.0);
	e13x13       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,6,6,   -100.0, -100.0);
	e15x15       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,7,7,   -100.0, -100.0);
	e21x21       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,10,10, -100.0, -100.0);
	e25x25       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,12,12, -100.0, -100.0);
	e31x31       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,15,15, -100.0, -100.0);
	
	e7x7_10Sig   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,3,3,   0.030,  0.150);
	e9x9_10Sig   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,4,4,   0.030,  0.150);
	e11x11_10Sig = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,5,5,   0.030,  0.150);
	e15x15_10Sig = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,7,7,   0.030,  0.150);
	
	e7x7_15Sig   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,3,3,   0.045,  0.225);
	e9x9_15Sig   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,4,4,   0.045,  0.225);
	e11x11_15Sig = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,5,5,   0.045,  0.225);
	e15x15_15Sig = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,7,7,   0.045,  0.225);
	
	e7x7_20Sig   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,3,3,   0.060,  0.300);
	e9x9_20Sig   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,4,4,   0.060,  0.300);
	e11x11_20Sig = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,5,5,   0.060,  0.300);
	e15x15_20Sig = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,7,7,   0.060,  0.300);
	
	e7x7_25Sig   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,3,3,   0.075,  0.375);
	e9x9_25Sig   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,4,4,   0.075,  0.375);
	e11x11_25Sig = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,5,5,   0.075,  0.375);
	e15x15_25Sig = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,7,7,   0.075,  0.375);
	
	e7x7_30Sig   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,3,3,   0.090,  0.450);
	e9x9_30Sig   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,4,4,   0.090,  0.450);
	e11x11_30Sig = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,5,5,   0.090,  0.450);
	e15x15_30Sig = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, geo, caloTopology,7,7,   0.090,  0.450);

	std::cout << "default ecal rechit " << std::endl;
	std::cout<<"e3x3 "<<e3x3<<" e9x9 "<<e9x9<<" e15x15 " << e15x15 << " e31x31 "<<e31x31<<std::endl;
	std::cout<<"e7x7_10Sig "<<e7x7_10Sig<<" e11x11_10Sig "<<e11x11_10Sig<<" e15x15_10Sig "<<e15x15_10Sig<<std::endl;
	std::pair<double, bool> eJunk1 = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,3,3,   0.060,  0.300, false, true);
	std::pair<double, bool> eJunk2 = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,3,3,   0.20,  0.45, true, true);
	*/

	//e3x3P         = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,1,1,   -100.0, -100.0);
	//e5x5P         = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,2,2,   -100.0, -100.0);
	e7x7P         = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,3,3,   -100.0, -100.0, tMinE_,tMaxE_);
	e9x9P         = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,4,4,   -100.0, -100.0, tMinE_,tMaxE_);
	e11x11P       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,5,5,   -100.0, -100.0, tMinE_,tMaxE_);
	//e13x13P       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,6,6,   -100.0, -100.0);
	e15x15P       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,7,7,   -100.0, -100.0, tMinE_,tMaxE_);
	//e21x21P       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,10,10, -100.0, -100.0);
	//e25x25P       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,12,12, -100.0, -100.0);
	//e31x31P       = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,15,15, -100.0, -100.0);
	
	e7x7_10SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,3,3,   0.030,  0.150, tMinE_,tMaxE_);
	e9x9_10SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,4,4,   0.030,  0.150, tMinE_,tMaxE_);
	e11x11_10SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,5,5,   0.030,  0.150, tMinE_,tMaxE_);
	e15x15_10SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,7,7,   0.030,  0.150, tMinE_,tMaxE_);

	/*	
	e7x7_15SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,3,3,   0.045,  0.225);
	e9x9_15SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,4,4,   0.045,  0.225);
	e11x11_15SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,5,5,   0.045,  0.225);
	e15x15_15SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,7,7,   0.045,  0.225);
	*/
	e7x7_15SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology, ttMap, 3,3, 0.20,0.45, tMinE_,tMaxE_);
	e9x9_15SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology, ttMap, 4,4, 0.20,0.45, tMinE_,tMaxE_);
	e11x11_15SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology, ttMap, 5,5, 0.20,0.45, tMinE_,tMaxE_);
	e15x15_15SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology, ttMap, 7,7, 0.20,0.45, tMinE_,tMaxE_, false);
	
	e7x7_20SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,3,3,   0.060,  0.300, tMinE_,tMaxE_);
	e9x9_20SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,4,4,   0.060,  0.300, tMinE_,tMaxE_);
	e11x11_20SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,5,5,   0.060,  0.300, tMinE_,tMaxE_);
	e15x15_20SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,7,7,   0.060,  0.300, tMinE_,tMaxE_);
	
	e7x7_25SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,3,3,   0.075,  0.375, tMinE_,tMaxE_);
	e9x9_25SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,4,4,   0.075,  0.375, tMinE_,tMaxE_);
	e11x11_25SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,5,5,   0.075,  0.375, tMinE_,tMaxE_);
	e15x15_25SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,7,7,   0.075,  0.375, tMinE_,tMaxE_);
	
	e7x7_30SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,3,3,   0.090,  0.450, tMinE_,tMaxE_);
	e9x9_30SigP   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,4,4,   0.090,  0.450, tMinE_,tMaxE_);
	e11x11_30SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,5,5,   0.090,  0.450, tMinE_,tMaxE_);
	e15x15_30SigP = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,7,7,   0.090,  0.450, tMinE_,tMaxE_);
	if(myverbose_ == 2) {
	  std::cout << "clean  ecal rechit " << std::endl;
	  std::cout<<"e3x3 "<<e3x3P.first<<" e9x9 "<<e9x9P.first<<" e15x15 " << e15x15P.first << " e31x31 "<<e31x31P.first<<std::endl;
	  std::cout<<"e7x7_10Sig "<<e7x7_10SigP.first<<" e11x11_10Sig "<<e11x11_10SigP.first<<" e15x15_10Sig "<<e15x15_10SigP.first<<std::endl;
	}
	
	if (doMC) {
	  // check the energy from SimHits
	  spr::eECALSimInfo(iEvent,isoCell,geo,caloTopology,pcaloeb,pcaloee,SimTk,SimVtx,pTrack, *associate, 1,1, simInfo3x3);
	  spr::eECALSimInfo(iEvent,isoCell,geo,caloTopology,pcaloeb,pcaloee,SimTk,SimVtx,pTrack, *associate, 2,2, simInfo5x5);
	  spr::eECALSimInfo(iEvent,isoCell,geo,caloTopology,pcaloeb,pcaloee,SimTk,SimVtx,pTrack, *associate, 3,3, simInfo7x7);
	  spr::eECALSimInfo(iEvent,isoCell,geo,caloTopology,pcaloeb,pcaloee,SimTk,SimVtx,pTrack, *associate, 4,4, simInfo9x9);
	  spr::eECALSimInfo(iEvent,isoCell,geo,caloTopology,pcaloeb,pcaloee,SimTk,SimVtx,pTrack, *associate, 5,5, simInfo11x11);
	  spr::eECALSimInfo(iEvent,isoCell,geo,caloTopology,pcaloeb,pcaloee,SimTk,SimVtx,pTrack, *associate, 6,6, simInfo13x13);
	  spr::eECALSimInfo(iEvent,isoCell,geo,caloTopology,pcaloeb,pcaloee,SimTk,SimVtx,pTrack, *associate, 7,7, simInfo15x15, 150.0,false);
	  spr::eECALSimInfo(iEvent,isoCell,geo,caloTopology,pcaloeb,pcaloee,SimTk,SimVtx,pTrack, *associate, 10,10, simInfo21x21);
	  spr::eECALSimInfo(iEvent,isoCell,geo,caloTopology,pcaloeb,pcaloee,SimTk,SimVtx,pTrack, *associate, 12,12, simInfo25x25);
	  spr::eECALSimInfo(iEvent,isoCell,geo,caloTopology,pcaloeb,pcaloee,SimTk,SimVtx,pTrack, *associate, 15,15, simInfo31x31);
	  
	  trkEcalEne   = spr::eCaloSimInfo(iEvent, geo, pcaloeb,pcaloee, SimTk, SimVtx, pTrack, *associate, 150.0, false);
	   if(myverbose_ == 1) {
	    std::cout << "Track momentum " << pt1 << std::endl;

	    std::cout << "ecal siminfo " << std::endl;
	    std::cout << "simInfo3x3: " << "eTotal " << simInfo3x3.eTotal << " eMatched " << simInfo3x3.eMatched << " eRest " << simInfo3x3.eRest << " eGamma "<<simInfo3x3.eGamma << " eNeutralHad " << simInfo3x3.eNeutralHad << " eChargedHad " << simInfo3x3.eChargedHad << std::endl;
	    std::cout << "simInfo5x5: " << "eTotal " << simInfo5x5.eTotal << " eMatched " << simInfo5x5.eMatched << " eRest " << simInfo5x5.eRest << " eGamma "<<simInfo5x5.eGamma << " eNeutralHad " << simInfo5x5.eNeutralHad << " eChargedHad " << simInfo5x5.eChargedHad << std::endl;
	    std::cout << "simInfo7x7: " << "eTotal " << simInfo7x7.eTotal << " eMatched " << simInfo7x7.eMatched << " eRest " << simInfo7x7.eRest << " eGamma "<<simInfo7x7.eGamma << " eNeutralHad " << simInfo7x7.eNeutralHad << " eChargedHad " << simInfo7x7.eChargedHad << std::endl;
	    std::cout << "simInfo9x9: " << "eTotal " << simInfo9x9.eTotal << " eMatched " << simInfo9x9.eMatched << " eRest " << simInfo9x9.eRest << " eGamma "<<simInfo9x9.eGamma << " eNeutralHad " << simInfo9x9.eNeutralHad << " eChargedHad " << simInfo9x9.eChargedHad << std::endl;
	    std::cout << "simInfo11x11: " << "eTotal " << simInfo11x11.eTotal << " eMatched " << simInfo11x11.eMatched << " eRest " << simInfo11x11.eRest << " eGamma "<<simInfo11x11.eGamma << " eNeutralHad " << simInfo11x11.eNeutralHad << " eChargedHad " << simInfo11x11.eChargedHad << std::endl;
	    std::cout << "simInfo15x15: " << "eTotal " << simInfo15x15.eTotal << " eMatched " << simInfo15x15.eMatched << " eRest " << simInfo15x15.eRest << " eGamma "<<simInfo15x15.eGamma << " eNeutralHad " << simInfo15x15.eNeutralHad << " eChargedHad " << simInfo15x15.eChargedHad << std::endl;
	    std::cout << "simInfo31x31: " << "eTotal " << simInfo31x31.eTotal << " eMatched " << simInfo31x31.eMatched << " eRest " << simInfo31x31.eRest << " eGamma "<<simInfo31x31.eGamma << " eNeutralHad " << simInfo31x31.eNeutralHad << " eChargedHad " << simInfo31x31.eChargedHad << std::endl;
	    std::cout << "trkEcalEne" << trkEcalEne << std::endl;

	  }
	}

	// =======  Get HCAL information 
	double hcalScale=1.0;
	if( std::abs(pTrack->eta())<1.4 ) {
	  hcalScale=120.0;
	} else {
	  hcalScale=135.0;
	}
	
	double maxNearHcalP3x3=-1, maxNearHcalP5x5=-1, maxNearHcalP7x7=-1;
	maxNearHcalP3x3  = spr::chargeIsolationHcal(nTracks, trkCaloDets, theHBHETopology, 1,1);
	maxNearHcalP5x5  = spr::chargeIsolationHcal(nTracks, trkCaloDets, theHBHETopology, 2,2);
	maxNearHcalP7x7  = spr::chargeIsolationHcal(nTracks, trkCaloDets, theHBHETopology, 3,3);

	double h3x3=0,    h5x5=0,    h7x7=0;
	double h3x3Sig=0,    h5x5Sig=0,    h7x7Sig=0;
	double trkHcalEne = 0;
	spr::caloSimInfo hsimInfo3x3, hsimInfo5x5, hsimInfo7x7;
	
	if(trkDetItr->okHCAL) {
	  const DetId ClosestCell(trkDetItr->detIdHCAL);
	  // bool includeHO=false, bool algoNew=true, bool debug=false
	  h3x3 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,1,1, false, true, -100.0, -100.0, -100.0, -100.0, tMinH_,tMaxH_);  
	  h5x5 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,2,2, false, true, -100.0, -100.0, -100.0, -100.0, tMinH_,tMaxH_);  
	  h7x7 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,3,3, false, true, -100.0, -100.0, -100.0, -100.0, tMinH_,tMaxH_);  
	  h3x3Sig = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,1,1, false, true, 0.7, 0.8, -100.0, -100.0, tMinH_,tMaxH_);  
	  h5x5Sig = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,2,2, false, true, 0.7, 0.8, -100.0, -100.0, tMinH_,tMaxH_);  
	  h7x7Sig = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,3,3, false, true, 0.7, 0.8, -100.0, -100.0, tMinH_,tMaxH_);  
	  if(myverbose_==2) {
	    std::cout << "HCAL 3x3 " << h3x3 << " " << h3x3Sig << " 5x5 " <<  h5x5 << " " << h5x5Sig << " 7x7 " << h7x7 << " " << h7x7Sig << std::endl;
	  }
	  
	  if (doMC) {
	    spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell, geo,pcalohh, SimTk, SimVtx, pTrack, *associate, 1,1, hsimInfo3x3);
	    spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell, geo,pcalohh, SimTk, SimVtx, pTrack, *associate, 2,2, hsimInfo5x5);
	    spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell, geo,pcalohh, SimTk, SimVtx, pTrack, *associate, 3,3, hsimInfo7x7, 150.0, false,false);
	    trkHcalEne  = spr::eCaloSimInfo(iEvent, geo,pcalohh, SimTk, SimVtx, pTrack, *associate);
	    if(myverbose_ == 1) {
	      std::cout << "Hcal siminfo " << std::endl;
	      std::cout << "hsimInfo3x3: " << "eTotal " << hsimInfo3x3.eTotal << " eMatched " << hsimInfo3x3.eMatched << " eRest " << hsimInfo3x3.eRest << " eGamma "<<hsimInfo3x3.eGamma << " eNeutralHad " << hsimInfo3x3.eNeutralHad << " eChargedHad " << hsimInfo3x3.eChargedHad << std::endl;
	      std::cout << "hsimInfo5x5: " << "eTotal " << hsimInfo5x5.eTotal << " eMatched " << hsimInfo5x5.eMatched << " eRest " << hsimInfo5x5.eRest << " eGamma "<<hsimInfo5x5.eGamma << " eNeutralHad " << hsimInfo5x5.eNeutralHad << " eChargedHad " << hsimInfo5x5.eChargedHad << std::endl;
	      std::cout << "hsimInfo7x7: " << "eTotal " << hsimInfo7x7.eTotal << " eMatched " << hsimInfo7x7.eMatched << " eRest " << hsimInfo7x7.eRest << " eGamma "<<hsimInfo7x7.eGamma << " eNeutralHad " << hsimInfo7x7.eNeutralHad << " eChargedHad " << hsimInfo7x7.eChargedHad << std::endl;
	      std::cout << "trkHcalEne " << trkHcalEne << std::endl;
	    }
	  }
	  
	  // debug the ecal and hcal matrix
	  if(myverbose_==4) {
	    std::cout<<"Run "<<iEvent.id().run()<<"  Event "<<iEvent.id().event()<<std::endl; 
	    std::vector<std::pair<DetId,double> > v7x7 = spr::eHCALmatrixCell(theHBHETopology, ClosestCell, hbhe,3,3, false, false);
	    double sumv=0.0;
	    
	    for(unsigned int iv=0; iv<v7x7.size(); iv++) { 
	      sumv += v7x7[iv].second;
	    }
	    std::cout<<"h7x7 "<<h7x7<<" v7x7 "<<sumv << " in " << v7x7.size() <<std::endl;
	    for(unsigned int iv=0; iv<v7x7.size(); iv++) { 
	      HcalDetId id = v7x7[iv].first;
	      std::cout << " Cell " << iv << " 0x" << std::hex << v7x7[iv].first() << std::dec << " " << id << " Energy " << v7x7[iv].second << std::endl;
	    }
	  }
	  
	}

	
	// ====================================================================================================

	// get diff between track outermost hit position and the propagation point at outermost surface of tracker	
	std::pair<math::XYZPoint,double> point2_TK0 = spr::propagateTrackerEnd( pTrack, bField, false);
	math::XYZPoint diff(pTrack->outerPosition().X()-point2_TK0.first.X(), 
			    pTrack->outerPosition().Y()-point2_TK0.first.Y(), 
			    pTrack->outerPosition().Z()-point2_TK0.first.Z() );
	double trackOutPosOutHitDr = diff.R();
	double trackL              = point2_TK0.second;
	//std::cout<<" propagted "<<point2_TK0.first<<" "<< point2_TK0.first.eta()<<" "<<point2_TK0.first.phi()<<std::endl;
	//std::cout<<" outerPosition() "<< pTrack->outerPosition() << " "<< pTrack->outerPosition().eta()<< " " << pTrack->outerPosition().phi()<< std::endl;
	//std::cout<<"diff " << diff << " diffR " <<diff.R()<<" diffR/L "<<diff.R()/point2_TK0.second <<std::endl;

	for(unsigned int ind=0;ind<recVtxs->size();ind++) {
	  if (!((*recVtxs)[ind].isFake())) {
	    reco::Vertex::trackRef_iterator vtxTrack = (*recVtxs)[ind].tracks_begin();
	    if( DeltaR(eta1,phi1, (*vtxTrack)->eta(),(*vtxTrack)->phi()) < 0.01 ) t_trackPVIdx ->push_back( ind );
	    else                                                                  t_trackPVIdx ->push_back( -1 );
	  }
	}

	// Fill the tree Branches here 
	t_trackP                ->push_back( p1 );
	t_trackPt               ->push_back( pt1 );
	t_trackEta              ->push_back( eta1 );
	t_trackPhi              ->push_back( phi1 );
	t_trackEcalEta          ->push_back( etaEcal1 );
	t_trackEcalPhi          ->push_back( phiEcal1 );
	t_trackHcalEta          ->push_back( etaHcal1 );
	t_trackHcalPhi          ->push_back( phiHcal1 );
	t_trackDxy              ->push_back( dxy1 );	
	t_trackDz               ->push_back( dz1 );	
	t_trackDxyBS            ->push_back( dxybs1 );	
	t_trackDzBS             ->push_back( dzbs1 );	
	t_trackDxyPV            ->push_back( dxypv1 );	
	t_trackDzPV             ->push_back( dzpv1 );	
	t_trackChiSq            ->push_back( chisq1 );	
	t_trackNOuterHits       ->push_back( nOuterHits );
	t_NLayersCrossed        ->push_back( nLayersCrossed );
	
	t_trackHitsTOB          ->push_back( hitp.stripTOBLayersWithMeasurement()       ); 
	t_trackHitsTEC          ->push_back( hitp.stripTECLayersWithMeasurement()       );
	t_trackHitInMissTOB     ->push_back( hitpIn.stripTOBLayersWithoutMeasurement()  ); 
	t_trackHitInMissTEC     ->push_back( hitpIn.stripTECLayersWithoutMeasurement()  );  
	t_trackHitInMissTIB     ->push_back( hitpIn.stripTIBLayersWithoutMeasurement()  );  
	t_trackHitInMissTID     ->push_back( hitpIn.stripTIDLayersWithoutMeasurement()  );
	t_trackHitInMissTIBTID  ->push_back( hitpIn.stripTIBLayersWithoutMeasurement() + hitpIn.stripTIDLayersWithoutMeasurement() );  

	t_trackHitOutMissTOB    ->push_back( hitpOut.stripTOBLayersWithoutMeasurement() );
	t_trackHitOutMissTEC    ->push_back( hitpOut.stripTECLayersWithoutMeasurement() ); 
	t_trackHitOutMissTIB    ->push_back( hitpOut.stripTIBLayersWithoutMeasurement() );
	t_trackHitOutMissTID    ->push_back( hitpOut.stripTIDLayersWithoutMeasurement() );
	t_trackHitOutMissTOBTEC ->push_back( hitpOut.stripTOBLayersWithoutMeasurement() + hitpOut.stripTECLayersWithoutMeasurement() );

	t_trackHitInMeasTOB     ->push_back( hitpIn.stripTOBLayersWithMeasurement()  ); 
	t_trackHitInMeasTEC     ->push_back( hitpIn.stripTECLayersWithMeasurement()  );  
	t_trackHitInMeasTIB     ->push_back( hitpIn.stripTIBLayersWithMeasurement()  );  
	t_trackHitInMeasTID     ->push_back( hitpIn.stripTIDLayersWithMeasurement()  );
	t_trackHitOutMeasTOB    ->push_back( hitpOut.stripTOBLayersWithMeasurement() );
	t_trackHitOutMeasTEC    ->push_back( hitpOut.stripTECLayersWithMeasurement() ); 
	t_trackHitOutMeasTIB    ->push_back( hitpOut.stripTIBLayersWithMeasurement() );
	t_trackHitOutMeasTID    ->push_back( hitpOut.stripTIDLayersWithMeasurement() );
	t_trackOutPosOutHitDr   ->push_back( trackOutPosOutHitDr                     );
	t_trackL                ->push_back( trackL                                  );

	t_maxNearP31x31         ->push_back( maxNearP31x31 );
	//t_maxNearP25x25         ->push_back( maxNearP25x25 );
	t_maxNearP21x21         ->push_back( maxNearP21x21 );
	//t_maxNearP15x15         ->push_back( maxNearP15x15 );
	//t_maxNearP13x13         ->push_back( maxNearP13x13 );
	//t_maxNearP11x11         ->push_back( maxNearP11x11 );
	//t_maxNearP9x9           ->push_back( maxNearP9x9   );
	//t_maxNearP7x7           ->push_back( maxNearP7x7   );
	
	t_ecalSpike11x11        ->push_back( e11x11P.second );
	//t_e3x3                  ->push_back( e3x3P.first );
	//t_e5x5                  ->push_back( e5x5P.first );
	t_e7x7                  ->push_back( e7x7P.first );
	t_e9x9                  ->push_back( e9x9P.first );
	t_e11x11                ->push_back( e11x11P.first );
	//t_e13x13                ->push_back( e13x13P.first );
	t_e15x15                ->push_back( e15x15P.first );
	//t_e21x21                ->push_back( e21x21P.first );
	//t_e25x25                ->push_back( e25x25P.first );
	//t_e31x31                ->push_back( e31x31P.first );

	t_e7x7_10Sig            ->push_back( e7x7_10SigP.first   ); 
	t_e9x9_10Sig            ->push_back( e9x9_10SigP.first   ); 
	t_e11x11_10Sig          ->push_back( e11x11_10SigP.first ); 
	t_e15x15_10Sig          ->push_back( e15x15_10SigP.first );
	t_e7x7_15Sig            ->push_back( e7x7_15SigP.first   ); 
	t_e9x9_15Sig            ->push_back( e9x9_15SigP.first   ); 
	t_e11x11_15Sig          ->push_back( e11x11_15SigP.first ); 
	t_e15x15_15Sig          ->push_back( e15x15_15SigP.first );
	t_e7x7_20Sig            ->push_back( e7x7_20SigP.first   ); 
	t_e9x9_20Sig            ->push_back( e9x9_20SigP.first   ); 
	t_e11x11_20Sig          ->push_back( e11x11_20SigP.first ); 
	t_e15x15_20Sig          ->push_back( e15x15_20SigP.first );
	t_e7x7_25Sig            ->push_back( e7x7_25SigP.first   ); 
	t_e9x9_25Sig            ->push_back( e9x9_25SigP.first   ); 
	t_e11x11_25Sig          ->push_back( e11x11_25SigP.first ); 
	t_e15x15_25Sig          ->push_back( e15x15_25SigP.first );
	t_e7x7_30Sig            ->push_back( e7x7_30SigP.first   ); 
	t_e9x9_30Sig            ->push_back( e9x9_30SigP.first   ); 
	t_e11x11_30Sig          ->push_back( e11x11_30SigP.first ); 
	t_e15x15_30Sig          ->push_back( e15x15_30SigP.first );

	if (doMC) {
	  //t_esim3x3               ->push_back( simInfo3x3.eTotal );
	  //t_esim5x5               ->push_back( simInfo5x5.eTotal );
	  t_esim7x7               ->push_back( simInfo7x7.eTotal );
	  t_esim9x9               ->push_back( simInfo9x9.eTotal );
	  t_esim11x11             ->push_back( simInfo11x11.eTotal );
	  //t_esim13x13             ->push_back( simInfo13x13.eTotal );
	  t_esim15x15             ->push_back( simInfo15x15.eTotal );
	  //t_esim21x21             ->push_back( simInfo21x21.eTotal );
	  //t_esim25x25             ->push_back( simInfo25x25.eTotal );
	  //t_esim31x31             ->push_back( simInfo31x31.eTotal );
	
	  //t_esim3x3Matched        ->push_back( simInfo3x3.eMatched );
	  //t_esim5x5Matched        ->push_back( simInfo5x5.eMatched );
	  t_esim7x7Matched        ->push_back( simInfo7x7.eMatched );
	  t_esim9x9Matched        ->push_back( simInfo9x9.eMatched );
	  t_esim11x11Matched      ->push_back( simInfo11x11.eMatched );
	  //t_esim13x13Matched      ->push_back( simInfo13x13.eMatched );
	  t_esim15x15Matched      ->push_back( simInfo15x15.eMatched );
	  //t_esim21x21Matched      ->push_back( simInfo21x21.eMatched );
	  //t_esim25x25Matched      ->push_back( simInfo25x25.eMatched );
	  //t_esim31x31Matched      ->push_back( simInfo31x31.eMatched );
	
	  //t_esim3x3Rest           ->push_back( simInfo3x3.eRest );
	  //t_esim5x5Rest           ->push_back( simInfo5x5.eRest );
	  t_esim7x7Rest           ->push_back( simInfo7x7.eRest );
	  t_esim9x9Rest           ->push_back( simInfo9x9.eRest );
	  t_esim11x11Rest         ->push_back( simInfo11x11.eRest );
	  //t_esim13x13Rest         ->push_back( simInfo13x13.eRest );
	  t_esim15x15Rest         ->push_back( simInfo15x15.eRest );
	  //t_esim21x21Rest         ->push_back( simInfo21x21.eRest );
	  //t_esim25x25Rest         ->push_back( simInfo25x25.eRest );
	  //t_esim31x31Rest         ->push_back( simInfo31x31.eRest );
	
	  //t_esim3x3Photon         ->push_back( simInfo3x3.eGamma );
	  //t_esim5x5Photon         ->push_back( simInfo5x5.eGamma );
	  t_esim7x7Photon         ->push_back( simInfo7x7.eGamma );
	  t_esim9x9Photon         ->push_back( simInfo9x9.eGamma );
	  t_esim11x11Photon       ->push_back( simInfo11x11.eGamma );
	  //t_esim13x13Photon       ->push_back( simInfo13x13.eGamma );
	  t_esim15x15Photon       ->push_back( simInfo15x15.eGamma );
	  //t_esim21x21Photon       ->push_back( simInfo21x21.eGamma );
	  //t_esim25x25Photon       ->push_back( simInfo25x25.eGamma );
	  //t_esim31x31Photon       ->push_back( simInfo31x31.eGamma );
	
	  //t_esim3x3NeutHad        ->push_back( simInfo3x3.eNeutralHad );
	  //t_esim5x5NeutHad        ->push_back( simInfo5x5.eNeutralHad );
	  t_esim7x7NeutHad        ->push_back( simInfo7x7.eNeutralHad );
	  t_esim9x9NeutHad        ->push_back( simInfo9x9.eNeutralHad );
	  t_esim11x11NeutHad      ->push_back( simInfo11x11.eNeutralHad );
	  //t_esim13x13NeutHad      ->push_back( simInfo13x13.eNeutralHad );
	  t_esim15x15NeutHad      ->push_back( simInfo15x15.eNeutralHad );
	  //t_esim21x21NeutHad      ->push_back( simInfo21x21.eNeutralHad );
	  //t_esim25x25NeutHad      ->push_back( simInfo25x25.eNeutralHad );
	  //t_esim31x31NeutHad      ->push_back( simInfo31x31.eNeutralHad );
	
	  //t_esim3x3CharHad        ->push_back( simInfo3x3.eChargedHad );
	  //t_esim5x5CharHad        ->push_back( simInfo5x5.eChargedHad );
	  t_esim7x7CharHad        ->push_back( simInfo7x7.eChargedHad );
	  t_esim9x9CharHad        ->push_back( simInfo9x9.eChargedHad );
	  t_esim11x11CharHad      ->push_back( simInfo11x11.eChargedHad );
	  //t_esim13x13CharHad      ->push_back( simInfo13x13.eChargedHad );
	  t_esim15x15CharHad      ->push_back( simInfo15x15.eChargedHad );
	  //t_esim21x21CharHad      ->push_back( simInfo21x21.eChargedHad );
	  //t_esim25x25CharHad      ->push_back( simInfo25x25.eChargedHad );
	  //t_esim31x31CharHad      ->push_back( simInfo31x31.eChargedHad );
	
	  t_trkEcalEne            ->push_back( trkEcalEne );
	  t_simTrackP             ->push_back( simTrackP );
	  t_esimPdgId             ->push_back( simInfo11x11.pdgMatched );
	}

	t_maxNearHcalP3x3       ->push_back( maxNearHcalP3x3 );
	t_maxNearHcalP5x5       ->push_back( maxNearHcalP5x5 );
	t_maxNearHcalP7x7       ->push_back( maxNearHcalP7x7 );
	
	t_h3x3                  ->push_back( h3x3 );
	t_h5x5                  ->push_back( h5x5 );
	t_h7x7                  ->push_back( h7x7 );
	t_h3x3Sig               ->push_back( h3x3Sig );
	t_h5x5Sig               ->push_back( h5x5Sig );
	t_h7x7Sig               ->push_back( h7x7Sig );
	
	t_infoHcal              ->push_back( trkDetItr->okHCAL );
	if (doMC) {
	  t_trkHcalEne            ->push_back( hcalScale*trkHcalEne );
	
	  t_hsim3x3               ->push_back( hcalScale*hsimInfo3x3.eTotal );
	  t_hsim5x5               ->push_back( hcalScale*hsimInfo5x5.eTotal );
	  t_hsim7x7               ->push_back( hcalScale*hsimInfo7x7.eTotal );
	
	  t_hsim3x3Matched        ->push_back( hcalScale*hsimInfo3x3.eMatched );
	  t_hsim5x5Matched        ->push_back( hcalScale*hsimInfo5x5.eMatched );
	  t_hsim7x7Matched        ->push_back( hcalScale*hsimInfo7x7.eMatched );
	
	  t_hsim3x3Rest           ->push_back( hcalScale*hsimInfo3x3.eRest );
	  t_hsim5x5Rest           ->push_back( hcalScale*hsimInfo5x5.eRest );
	  t_hsim7x7Rest           ->push_back( hcalScale*hsimInfo7x7.eRest );
	
	  t_hsim3x3Photon         ->push_back( hcalScale*hsimInfo3x3.eGamma );
	  t_hsim5x5Photon         ->push_back( hcalScale*hsimInfo5x5.eGamma );
	  t_hsim7x7Photon         ->push_back( hcalScale*hsimInfo7x7.eGamma );
	
	  t_hsim3x3NeutHad        ->push_back( hcalScale*hsimInfo3x3.eNeutralHad );
	  t_hsim5x5NeutHad        ->push_back( hcalScale*hsimInfo5x5.eNeutralHad );
	  t_hsim7x7NeutHad        ->push_back( hcalScale*hsimInfo7x7.eNeutralHad );
	
	  t_hsim3x3CharHad        ->push_back( hcalScale*hsimInfo3x3.eChargedHad );
	  t_hsim5x5CharHad        ->push_back( hcalScale*hsimInfo5x5.eChargedHad );
	  t_hsim7x7CharHad        ->push_back( hcalScale*hsimInfo7x7.eChargedHad );
	}
	/*
	if(hcalScale*hsimInfo3x3.eTotal > 50.0) {

	  std::cout << "Loosely Iso Track : eta " << eta1 << " Rec Mom " << p1 << " SimMom " << simTrackP << " h3x3 " << h3x3 << std::endl;

	  std::cout <<"Closest cell Hcal (atHCAL) " << (HcalDetId)ClosestCell << std::endl;

	  std::cout <<"trkHcalEne, etotal, matched, rest " <<hcalScale*trkHcalEne<<std::setw(15)<<hcalScale*hsimInfo3x3.eTotal
		    <<std::setw(15)<<hcalScale*hsimInfo3x3.eMatched<<std::setw(15)<<hcalScale*hsimInfo3x3.eRest
		    <<std::endl;
	  unsigned int nn = t_trkHcalEne->size();
	  std::cout <<"in Tree                           " << (*t_trkHcalEne)[nn-1] <<std::setw(15)<< (*t_hsim3x3)[nn-1]
		    <<std::setw(15)<< (*t_hsim3x3Matched)[nn-1] <<std::setw(15)<< (*t_hsim3x3Rest)[nn-1]
		    << std::endl;

	  std::cout << "debug output \n" << std::endl;
	  spr::caloSimInfo hsimInfo3x3_debug;
	  spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell, geo,pcalohh, SimTk, SimVtx, pTrack, *associate, 1,1, hsimInfo3x3_debug, 150.0, true);

	}
	*/


      } // if loosely isolated track
    } // check p1/eta
  } // loop over track collection
  
  //if (doMC) delete associate;
  
  tree->Fill();

}

// ----- method called once each job just before starting event loop ----
void IsolatedTracksNxN::beginJob() {

  nEventProc=0;

  //  double tempgen_TH[21] = { 1.0,  2.0,  3.0,  4.0,  5.0, 
 double tempgen_TH[16] = { 0.0,  1.0,  2.0,  3.0,  4.0,  
			   5.0,  6.0,  7.0,  9.0, 11.0, 
			  15.0, 20.0, 30.0, 50.0, 75.0, 100.0};

  for(int i=0; i<16; i++)  genPartPBins[i]  = tempgen_TH[i];

  double tempgen_Eta[4] = {0.0, 1.131, 1.653, 2.172};

  for(int i=0; i<4; i++) genPartEtaBins[i] = tempgen_Eta[i];

  BookHistograms();
}

// ----- method called once each job just after ending the event loop ----
void IsolatedTracksNxN::endJob() {

  std::cout << "Number of Events Processed " << nEventProc << std::endl;
  if( h_L1AlgoNames ){
    int nbins=h_L1AlgoNames->GetNbinsX(); 
    for (int ibin=0; ibin<nbins; ++ibin){
      double cont = (double)h_L1AlgoNames->GetBinContent(ibin+1);
      if( cont>0 ) {
	trig_iter=l1TriggerMap.find(algoBitToName[ibin]);
	const char* trigName =  trig_iter->first.c_str();
	h_L1AlgoNames->GetXaxis()->SetBinLabel(ibin+1,trigName);
	std::cout<<"===============> "<<ibin+1<<" "<<trigName<<" "<<(int)cont<< std::endl;
      }
    }
  }
}

//===========================================================================================================


//===========================================================================================================




//---
void IsolatedTracksNxN::clearTreeVectors() {

  t_PVx               ->clear();
  t_PVy               ->clear();
  t_PVz               ->clear();
  t_PVisValid         ->clear();
  t_PVndof            ->clear();
  t_PVNTracks         ->clear();
  t_PVNTracksWt       ->clear();
  t_PVTracksSumPt     ->clear();
  t_PVTracksSumPtWt   ->clear();
  t_PVNTracksHP       ->clear();
  t_PVNTracksHPWt     ->clear();
  t_PVTracksSumPtHP   ->clear();
  t_PVTracksSumPtHPWt ->clear();

  t_L1Decision        ->clear();
  t_L1CenJetPt        ->clear();
  t_L1CenJetEta       ->clear();    
  t_L1CenJetPhi       ->clear();
  t_L1FwdJetPt        ->clear();
  t_L1FwdJetEta       ->clear();
  t_L1FwdJetPhi       ->clear();
  t_L1TauJetPt        ->clear();
  t_L1TauJetEta       ->clear();     
  t_L1TauJetPhi       ->clear();
  t_L1MuonPt          ->clear();
  t_L1MuonEta         ->clear();     
  t_L1MuonPhi         ->clear();
  t_L1IsoEMPt         ->clear();
  t_L1IsoEMEta        ->clear();
  t_L1IsoEMPhi        ->clear();
  t_L1NonIsoEMPt      ->clear();
  t_L1NonIsoEMEta     ->clear();
  t_L1NonIsoEMPhi     ->clear();
  t_L1METPt           ->clear();
  t_L1METEta          ->clear();
  t_L1METPhi          ->clear();

  t_jetPt             ->clear();
  t_jetEta            ->clear();
  t_jetPhi            ->clear();
  t_nTrksJetCalo      ->clear();  
  t_nTrksJetVtx       ->clear();

  t_trackPAll         ->clear();
  t_trackEtaAll       ->clear();
  t_trackPhiAll       ->clear();
  t_trackPdgIdAll     ->clear();
  t_trackPtAll        ->clear();
  t_trackDxyAll       ->clear();
  t_trackDzAll        ->clear();
  t_trackDxyPVAll     ->clear();
  t_trackDzPVAll      ->clear();
  t_trackChiSqAll     ->clear();

  t_trackP            ->clear();
  t_trackPt           ->clear();
  t_trackEta          ->clear();
  t_trackPhi          ->clear();
  t_trackEcalEta      ->clear();
  t_trackEcalPhi      ->clear();
  t_trackHcalEta      ->clear();
  t_trackHcalPhi      ->clear();
  t_NLayersCrossed    ->clear();
  t_trackNOuterHits   ->clear();
  t_trackDxy          ->clear();
  t_trackDxyBS        ->clear();
  t_trackDz           ->clear();
  t_trackDzBS         ->clear();
  t_trackDxyPV        ->clear();
  t_trackDzPV         ->clear();
  t_trackChiSq        ->clear();
  t_trackPVIdx        ->clear();
  t_trackHitsTOB          ->clear(); 
  t_trackHitsTEC          ->clear();
  t_trackHitInMissTOB     ->clear(); 
  t_trackHitInMissTEC     ->clear();  
  t_trackHitInMissTIB     ->clear();  
  t_trackHitInMissTID     ->clear();
  t_trackHitInMissTIBTID  ->clear();  
  t_trackHitOutMissTOB    ->clear();
  t_trackHitOutMissTEC    ->clear(); 
  t_trackHitOutMissTIB    ->clear();
  t_trackHitOutMissTID    ->clear(); 
  t_trackHitOutMissTOBTEC ->clear();

  t_trackHitInMeasTOB     ->clear(); 
  t_trackHitInMeasTEC     ->clear();  
  t_trackHitInMeasTIB     ->clear();  
  t_trackHitInMeasTID     ->clear();
  t_trackHitOutMeasTOB    ->clear();
  t_trackHitOutMeasTEC    ->clear(); 
  t_trackHitOutMeasTIB    ->clear();
  t_trackHitOutMeasTID    ->clear();
  t_trackOutPosOutHitDr   ->clear();
  t_trackL                ->clear();

  t_maxNearP31x31     ->clear();
  t_maxNearP25x25     ->clear();
  t_maxNearP21x21     ->clear();
  t_maxNearP15x15     ->clear();
  t_maxNearP13x13     ->clear();
  t_maxNearP11x11     ->clear();
  t_maxNearP9x9       ->clear();
  t_maxNearP7x7       ->clear();

  t_ecalSpike11x11    ->clear();
  t_e3x3              ->clear();
  t_e5x5              ->clear();
  t_e7x7              ->clear();
  t_e9x9              ->clear();
  t_e11x11            ->clear();
  t_e13x13            ->clear();
  t_e15x15            ->clear();
  t_e21x21            ->clear();
  t_e25x25            ->clear();
  t_e31x31            ->clear();
  //  t_e11x11Xtals       ->clear();

  t_e7x7_10Sig        ->clear();
  t_e9x9_10Sig        ->clear();
  t_e11x11_10Sig      ->clear();
  t_e15x15_10Sig      ->clear();
  t_e7x7_15Sig        ->clear();
  t_e9x9_15Sig        ->clear();
  t_e11x11_15Sig      ->clear();
  t_e15x15_15Sig      ->clear();
  t_e7x7_20Sig        ->clear();
  t_e9x9_20Sig        ->clear();
  t_e11x11_20Sig      ->clear();
  t_e15x15_20Sig      ->clear();
  t_e7x7_25Sig        ->clear();
  t_e9x9_25Sig        ->clear();
  t_e11x11_25Sig      ->clear();
  t_e15x15_25Sig      ->clear();
  t_e7x7_30Sig        ->clear();
  t_e9x9_30Sig        ->clear();
  t_e11x11_30Sig      ->clear();
  t_e15x15_30Sig      ->clear();

  if (doMC) {
    t_simTrackP         ->clear();
    t_esimPdgId         ->clear();
    t_trkEcalEne        ->clear();

    t_esim3x3           ->clear();
    t_esim5x5           ->clear();
    t_esim7x7           ->clear();
    t_esim9x9           ->clear();
    t_esim11x11         ->clear();
    t_esim13x13         ->clear();
    t_esim15x15         ->clear();
    t_esim21x21         ->clear();
    t_esim25x25         ->clear();
    t_esim31x31         ->clear();
  
    t_esim3x3Matched    ->clear();
    t_esim5x5Matched    ->clear();
    t_esim7x7Matched    ->clear();
    t_esim9x9Matched    ->clear();
    t_esim11x11Matched  ->clear();
    t_esim13x13Matched  ->clear();
    t_esim15x15Matched  ->clear();
    t_esim21x21Matched  ->clear();
    t_esim25x25Matched  ->clear();
    t_esim31x31Matched  ->clear();

    t_esim3x3Rest       ->clear();
    t_esim5x5Rest       ->clear();
    t_esim7x7Rest       ->clear();
    t_esim9x9Rest       ->clear();
    t_esim11x11Rest     ->clear();
    t_esim13x13Rest     ->clear();
    t_esim15x15Rest     ->clear();
    t_esim21x21Rest     ->clear();
    t_esim25x25Rest     ->clear();
    t_esim31x31Rest     ->clear();

    t_esim3x3Photon     ->clear();
    t_esim5x5Photon     ->clear();
    t_esim7x7Photon     ->clear();
    t_esim9x9Photon     ->clear();
    t_esim11x11Photon   ->clear();
    t_esim13x13Photon   ->clear();
    t_esim15x15Photon   ->clear();
    t_esim21x21Photon   ->clear();
    t_esim25x25Photon   ->clear();
    t_esim31x31Photon   ->clear();

    t_esim3x3NeutHad    ->clear();
    t_esim5x5NeutHad    ->clear();
    t_esim7x7NeutHad    ->clear();
    t_esim9x9NeutHad    ->clear();
    t_esim11x11NeutHad  ->clear();
    t_esim13x13NeutHad  ->clear();
    t_esim15x15NeutHad  ->clear();
    t_esim21x21NeutHad  ->clear();
    t_esim25x25NeutHad  ->clear();
    t_esim31x31NeutHad  ->clear();

    t_esim3x3CharHad    ->clear();
    t_esim5x5CharHad    ->clear();
    t_esim7x7CharHad    ->clear();
    t_esim9x9CharHad    ->clear();
    t_esim11x11CharHad  ->clear();
    t_esim13x13CharHad  ->clear();
    t_esim15x15CharHad  ->clear();
    t_esim21x21CharHad  ->clear();
    t_esim25x25CharHad  ->clear();
    t_esim31x31CharHad  ->clear();
  }

  t_maxNearHcalP3x3   ->clear();
  t_maxNearHcalP5x5   ->clear();
  t_maxNearHcalP7x7   ->clear();

  t_h3x3              ->clear();
  t_h5x5              ->clear();
  t_h7x7              ->clear();
  t_h3x3Sig           ->clear();
  t_h5x5Sig           ->clear();
  t_h7x7Sig           ->clear();

  t_infoHcal          ->clear();

  if (doMC) {
    t_trkHcalEne        ->clear();

    t_hsim3x3           ->clear();
    t_hsim5x5           ->clear();
    t_hsim7x7           ->clear();
    t_hsim3x3Matched    ->clear();
    t_hsim5x5Matched    ->clear();
    t_hsim7x7Matched    ->clear();
    t_hsim3x3Rest       ->clear();
    t_hsim5x5Rest       ->clear();
    t_hsim7x7Rest       ->clear();
    t_hsim3x3Photon     ->clear();
    t_hsim5x5Photon     ->clear();
    t_hsim7x7Photon     ->clear();
    t_hsim3x3NeutHad    ->clear();
    t_hsim5x5NeutHad    ->clear();
    t_hsim7x7NeutHad    ->clear();
    t_hsim3x3CharHad    ->clear();
    t_hsim5x5CharHad    ->clear();
    t_hsim7x7CharHad    ->clear();
  }
}

void IsolatedTracksNxN::BookHistograms(){

  char hname[100], htit[100];

  TFileDirectory dir = fs->mkdir("nearMaxTrackP");

  for(unsigned int ieta=0; ieta<NEtaBins; ieta++) {
    double lowEta=-5.0, highEta= 5.0;
    lowEta  = genPartEtaBins[ieta];
    highEta = genPartEtaBins[ieta+1];

    for(unsigned int ipt=0; ipt<NPBins; ipt++) {
      double lowP=0.0, highP=300.0;
      lowP    = genPartPBins[ipt];
      highP   = genPartPBins[ipt+1];
      sprintf(hname, "h_maxNearP31x31_ptBin%i_etaBin%i",ipt, ieta);
      sprintf(htit,  "maxNearP in 31x31 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_maxNearP31x31[ipt][ieta] = dir.make<TH1F>(hname, htit, 220, -2.0, 20.0);
      h_maxNearP31x31[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_maxNearP25x25_ptBin%i_etaBin%i",ipt, ieta);
      sprintf(htit,  "maxNearP in 25x25 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_maxNearP25x25[ipt][ieta] = dir.make<TH1F>(hname, htit, 220, -2.0, 20.0);
      h_maxNearP25x25[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_maxNearP21x21_ptBin%i_etaBin%i",ipt, ieta);
      sprintf(htit,  "maxNearP in 21x21 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_maxNearP21x21[ipt][ieta] = dir.make<TH1F>(hname, htit, 220, -2.0, 20.0);
      h_maxNearP21x21[ipt][ieta] ->Sumw2();
      sprintf(hname, "h_maxNearP15x15_ptBin%i_etaBin%i",ipt, ieta);
      sprintf(htit,  "maxNearP in 15x15 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP );
      h_maxNearP15x15[ipt][ieta] = dir.make<TH1F>(hname, htit, 220, -2.0, 20.0);
      h_maxNearP15x15[ipt][ieta] ->Sumw2();
     }
  }

  h_L1AlgoNames = fs->make<TH1I>("h_L1AlgoNames", "h_L1AlgoNames:Bin Labels", 128, -0.5, 127.5);  

  // Reconstructed Tracks

  h_PVTracksWt = fs->make<TH1F>("h_PVTracksWt", "h_PVTracksWt", 600, -0.1, 1.1);

  h_nTracks = fs->make<TH1F>("h_nTracks", "h_nTracks", 1000, -0.5, 999.5);

  sprintf(hname, "h_recEtaPt_0");
  sprintf(htit,  "h_recEtaPt (all tracks Eta vs pT)");
  h_recEtaPt_0 = fs->make<TH2F>(hname, htit, 30, -3.0,3.0, 15, genPartPBins);

  sprintf(hname, "h_recEtaP_0");
  sprintf(htit,  "h_recEtaP (all tracks Eta vs pT)");
  h_recEtaP_0 = fs->make<TH2F>(hname, htit, 30, -3.0,3.0, 15, genPartPBins);

  h_recPt_0  = fs->make<TH1F>("h_recPt_0",  "Pt (all tracks)",  15, genPartPBins);
  h_recP_0   = fs->make<TH1F>("h_recP_0",   "P  (all tracks)",  15, genPartPBins);
  h_recEta_0 = fs->make<TH1F>("h_recEta_0", "Eta (all tracks)", 60, -3.0,   3.0);
  h_recPhi_0 = fs->make<TH1F>("h_recPhi_0", "Phi (all tracks)", 100, -3.2,   3.2);
  //-------------------------
  sprintf(hname, "h_recEtaPt_1");
  sprintf(htit,  "h_recEtaPt (all good tracks Eta vs pT)");
  h_recEtaPt_1 = fs->make<TH2F>(hname, htit, 30, -3.0,3.0, 15, genPartPBins);

  sprintf(hname, "h_recEtaP_1");
  sprintf(htit,  "h_recEtaP (all good tracks Eta vs pT)");
  h_recEtaP_1 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 15, genPartPBins);

  h_recPt_1  = fs->make<TH1F>("h_recPt_1",  "Pt (all good tracks)",  15, genPartPBins);
  h_recP_1   = fs->make<TH1F>("h_recP_1",   "P  (all good tracks)",  15, genPartPBins);
  h_recEta_1 = fs->make<TH1F>("h_recEta_1", "Eta (all good tracks)", 60, -3.0,   3.0);
  h_recPhi_1 = fs->make<TH1F>("h_recPhi_1", "Phi (all good tracks)", 100, -3.2,   3.2);
  //-------------------------
  sprintf(hname, "h_recEtaPt_2");
  sprintf(htit,  "h_recEtaPt (charge isolation Eta vs pT)");
  h_recEtaPt_2 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 15, genPartPBins);

  sprintf(hname, "h_recEtaP_2");
  sprintf(htit,  "h_recEtaP (charge isolation Eta vs pT)");
  h_recEtaP_2 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 15, genPartPBins);
  
  h_recPt_2  = fs->make<TH1F>("h_recPt_2",  "Pt (charge isolation)",  15, genPartPBins);
  h_recP_2   = fs->make<TH1F>("h_recP_2",   "P  (charge isolation)",  15, genPartPBins);
  h_recEta_2 = fs->make<TH1F>("h_recEta_2", "Eta (charge isolation)", 60, -3.0,   3.0);
  h_recPhi_2 = fs->make<TH1F>("h_recPhi_2", "Phi (charge isolation)", 100, -3.2,   3.2);


  tree = fs->make<TTree>("tree", "tree");
  tree->SetAutoSave(10000);


  tree->Branch("t_EvtNo"              ,&t_EvtNo               ,"t_EvtNo/I");
  tree->Branch("t_RunNo"              ,&t_RunNo               ,"t_RunNo/I");
  tree->Branch("t_Lumi"               ,&t_Lumi                ,"t_Lumi/I");
  tree->Branch("t_Bunch"              ,&t_Bunch               ,"t_Bunch/I");


  t_PVx               = new std::vector<double>();
  t_PVy               = new std::vector<double>();
  t_PVz               = new std::vector<double>();
  t_PVisValid         = new std::vector<double>();
  t_PVndof            = new std::vector<double>(); 
  t_PVNTracks         = new std::vector<double>();
  t_PVNTracksWt       = new std::vector<double>();
  t_PVTracksSumPt     = new std::vector<double>();
  t_PVTracksSumPtWt   = new std::vector<double>();
  t_PVNTracksHP       = new std::vector<double>();
  t_PVNTracksHPWt     = new std::vector<double>();
  t_PVTracksSumPtHP   = new std::vector<double>();
  t_PVTracksSumPtHPWt = new std::vector<double>();

  tree->Branch("PVx"                  ,"vector<double>"      ,&t_PVx);
  tree->Branch("PVy"                  ,"vector<double>"      ,&t_PVy);
  tree->Branch("PVz"                  ,"vector<double>"      ,&t_PVz);
  tree->Branch("PVisValid"            ,"vector<double>"      ,&t_PVisValid);
  tree->Branch("PVndof"               ,"vector<double>"      ,&t_PVndof);
  tree->Branch("PVNTracks"            ,"vector<double>"      ,&t_PVNTracks);
  tree->Branch("PVNTracksWt"          ,"vector<double>"      ,&t_PVNTracksWt);
  tree->Branch("t_PVTracksSumPt"      ,"vector<double>"      ,&t_PVTracksSumPt);
  tree->Branch("t_PVTracksSumPtWt"    ,"vector<double>"      ,&t_PVTracksSumPtWt);
  tree->Branch("PVNTracksHP"          ,"vector<double>"      ,&t_PVNTracksHP);
  tree->Branch("PVNTracksHPWt"        ,"vector<double>"      ,&t_PVNTracksHPWt);
  tree->Branch("t_PVTracksSumPtHP"    ,"vector<double>"      ,&t_PVTracksSumPtHP);
  tree->Branch("t_PVTracksSumPtHPWt"  ,"vector<double>"      ,&t_PVTracksSumPtHPWt);

  //----- L1Trigger information
  t_L1Decision        = new std::vector<int>();
  t_L1CenJetPt        = new std::vector<double>();
  t_L1CenJetEta       = new std::vector<double>();    
  t_L1CenJetPhi       = new std::vector<double>();
  t_L1FwdJetPt        = new std::vector<double>();
  t_L1FwdJetEta       = new std::vector<double>();
  t_L1FwdJetPhi       = new std::vector<double>();
  t_L1TauJetPt        = new std::vector<double>();
  t_L1TauJetEta       = new std::vector<double>();     
  t_L1TauJetPhi       = new std::vector<double>();
  t_L1MuonPt          = new std::vector<double>();
  t_L1MuonEta         = new std::vector<double>();     
  t_L1MuonPhi         = new std::vector<double>();
  t_L1IsoEMPt         = new std::vector<double>();
  t_L1IsoEMEta        = new std::vector<double>();
  t_L1IsoEMPhi        = new std::vector<double>();
  t_L1NonIsoEMPt      = new std::vector<double>();
  t_L1NonIsoEMEta     = new std::vector<double>();
  t_L1NonIsoEMPhi     = new std::vector<double>();
  t_L1METPt           = new std::vector<double>();
  t_L1METEta          = new std::vector<double>();
  t_L1METPhi          = new std::vector<double>();
  
  tree->Branch("t_L1Decision",        "vector<int>",    &t_L1Decision);
  tree->Branch("t_L1CenJetPt",        "vector<double>", &t_L1CenJetPt);
  tree->Branch("t_L1CenJetEta",       "vector<double>", &t_L1CenJetEta);
  tree->Branch("t_L1CenJetPhi",       "vector<double>", &t_L1CenJetPhi);
  tree->Branch("t_L1FwdJetPt",        "vector<double>", &t_L1FwdJetPt);
  tree->Branch("t_L1FwdJetEta",       "vector<double>", &t_L1FwdJetEta);
  tree->Branch("t_L1FwdJetPhi",       "vector<double>", &t_L1FwdJetPhi);
  tree->Branch("t_L1TauJetPt",        "vector<double>", &t_L1TauJetPt);
  tree->Branch("t_L1TauJetEta",       "vector<double>", &t_L1TauJetEta);     
  tree->Branch("t_L1TauJetPhi",       "vector<double>", &t_L1TauJetPhi);
  tree->Branch("t_L1MuonPt",          "vector<double>", &t_L1MuonPt);
  tree->Branch("t_L1MuonEta",         "vector<double>", &t_L1MuonEta);
  tree->Branch("t_L1MuonPhi",         "vector<double>", &t_L1MuonPhi);
  tree->Branch("t_L1IsoEMPt",         "vector<double>", &t_L1IsoEMPt);
  tree->Branch("t_L1IsoEMEta",        "vector<double>", &t_L1IsoEMEta);
  tree->Branch("t_L1IsoEMPhi",        "vector<double>", &t_L1IsoEMPhi);
  tree->Branch("t_L1NonIsoEMPt",      "vector<double>", &t_L1NonIsoEMPt);
  tree->Branch("t_L1NonIsoEMEta",     "vector<double>", &t_L1NonIsoEMEta);
  tree->Branch("t_L1NonIsoEMPhi",     "vector<double>", &t_L1NonIsoEMPhi);
  tree->Branch("t_L1METPt",           "vector<double>", &t_L1METPt);
  tree->Branch("t_L1METEta",          "vector<double>", &t_L1METEta);
  tree->Branch("t_L1METPhi",          "vector<double>", &t_L1METPhi);

  t_jetPt            = new std::vector<double>();
  t_jetEta           = new std::vector<double>();
  t_jetPhi           = new std::vector<double>();
  t_nTrksJetCalo     = new std::vector<double>();  
  t_nTrksJetVtx      = new std::vector<double>();
  tree->Branch("t_jetPt",             "vector<double>",&t_jetPt);
  tree->Branch("t_jetEta",            "vector<double>",&t_jetEta);
  tree->Branch("t_jetPhi",            "vector<double>",&t_jetPhi);
  tree->Branch("t_nTrksJetCalo",      "vector<double>",&t_nTrksJetCalo);  
  tree->Branch("t_nTrksJetVtx",       "vector<double>",&t_nTrksJetVtx);

  t_trackPAll        = new std::vector<double>();
  t_trackEtaAll      = new std::vector<double>();
  t_trackPhiAll      = new std::vector<double>();
  t_trackPdgIdAll    = new std::vector<double>();
  t_trackPtAll       = new std::vector<double>();
  t_trackDxyAll      = new std::vector<double>();
  t_trackDzAll       = new std::vector<double>();
  t_trackDxyPVAll    = new std::vector<double>();
  t_trackDzPVAll     = new std::vector<double>();
  t_trackChiSqAll    = new std::vector<double>();
  tree->Branch("t_trackPAll",         "vector<double>", &t_trackPAll    );
  tree->Branch("t_trackPhiAll",       "vector<double>", &t_trackPhiAll  );
  tree->Branch("t_trackEtaAll",       "vector<double>", &t_trackEtaAll  );
  tree->Branch("t_trackPtAll",        "vector<double>", &t_trackPtAll    );
  tree->Branch("t_trackDxyAll",       "vector<double>", &t_trackDxyAll    );
  tree->Branch("t_trackDzAll",        "vector<double>", &t_trackDzAll    );
  tree->Branch("t_trackDxyPVAll",     "vector<double>", &t_trackDxyPVAll    );
  tree->Branch("t_trackDzPVAll",      "vector<double>", &t_trackDzPVAll    );
  tree->Branch("t_trackChiSqAll",     "vector<double>", &t_trackChiSqAll    );
  //tree->Branch("t_trackPdgIdAll",     "vector<double>", &t_trackPdgIdAll);

  t_trackP              = new std::vector<double>();
  t_trackPt             = new std::vector<double>();
  t_trackEta            = new std::vector<double>();
  t_trackPhi            = new std::vector<double>();
  t_trackEcalEta        = new std::vector<double>();
  t_trackEcalPhi        = new std::vector<double>();
  t_trackHcalEta        = new std::vector<double>();
  t_trackHcalPhi        = new std::vector<double>();
  t_trackNOuterHits     = new std::vector<int>();
  t_NLayersCrossed      = new std::vector<int>();
  t_trackDxy            = new std::vector<double>();
  t_trackDxyBS          = new std::vector<double>();
  t_trackDz             = new std::vector<double>();
  t_trackDzBS           = new std::vector<double>();
  t_trackDxyPV          = new std::vector<double>();
  t_trackDzPV           = new std::vector<double>();
  t_trackPVIdx          = new std::vector<int>();
  t_trackChiSq          = new std::vector<double>();
  t_trackHitsTOB        = new std::vector<int>(); 
  t_trackHitsTEC        = new std::vector<int>();
  t_trackHitInMissTOB   = new std::vector<int>(); 
  t_trackHitInMissTEC   = new std::vector<int>();  
  t_trackHitInMissTIB   = new std::vector<int>();  
  t_trackHitInMissTID   = new std::vector<int>();
  t_trackHitInMissTIBTID= new std::vector<int>();  

  t_trackHitOutMissTOB   = new std::vector<int>();
  t_trackHitOutMissTEC   = new std::vector<int>(); 
  t_trackHitOutMissTIB   = new std::vector<int>();
  t_trackHitOutMissTID   = new std::vector<int>();
  t_trackHitOutMissTOBTEC= new std::vector<int>();
  t_trackHitInMeasTOB    = new std::vector<int>(); 
  t_trackHitInMeasTEC    = new std::vector<int>();  
  t_trackHitInMeasTIB    = new std::vector<int>();  
  t_trackHitInMeasTID    = new std::vector<int>();
  t_trackHitOutMeasTOB   = new std::vector<int>();
  t_trackHitOutMeasTEC   = new std::vector<int>(); 
  t_trackHitOutMeasTIB   = new std::vector<int>();
  t_trackHitOutMeasTID   = new std::vector<int>();
  t_trackOutPosOutHitDr  =new std::vector<double>();
  t_trackL               =new std::vector<double>();

  tree->Branch("t_trackP",            "vector<double>", &t_trackP            );
  tree->Branch("t_trackPt",           "vector<double>", &t_trackPt           );
  tree->Branch("t_trackEta",          "vector<double>", &t_trackEta          );
  tree->Branch("t_trackPhi",          "vector<double>", &t_trackPhi          );
  tree->Branch("t_trackEcalEta",      "vector<double>", &t_trackEcalEta      );
  tree->Branch("t_trackEcalPhi",      "vector<double>", &t_trackEcalPhi      );
  tree->Branch("t_trackHcalEta",      "vector<double>", &t_trackHcalEta      );
  tree->Branch("t_trackHcalPhi",      "vector<double>", &t_trackHcalPhi      );

  tree->Branch("t_trackNOuterHits",      "vector<int>",    &t_trackNOuterHits   );
  tree->Branch("t_NLayersCrossed",       "vector<int>",    &t_NLayersCrossed    );
  tree->Branch("t_trackHitsTOB",         "vector<int>",   &t_trackHitsTOB      ); 
  tree->Branch("t_trackHitsTEC",         "vector<int>",   &t_trackHitsTEC      );
  tree->Branch("t_trackHitInMissTOB",    "vector<int>",   &t_trackHitInMissTOB ); 
  tree->Branch("t_trackHitInMissTEC",    "vector<int>",   &t_trackHitInMissTEC );  
  tree->Branch("t_trackHitInMissTIB",    "vector<int>",   &t_trackHitInMissTIB );  
  tree->Branch("t_trackHitInMissTID",    "vector<int>",   &t_trackHitInMissTID );
  tree->Branch("t_trackHitInMissTIBTID", "vector<int>",   &t_trackHitInMissTIBTID );  
  tree->Branch("t_trackHitOutMissTOB",   "vector<int>",   &t_trackHitOutMissTOB);
  tree->Branch("t_trackHitOutMissTEC",   "vector<int>",   &t_trackHitOutMissTEC); 
  tree->Branch("t_trackHitOutMissTIB",   "vector<int>",   &t_trackHitOutMissTIB);
  tree->Branch("t_trackHitOutMissTID",   "vector<int>",   &t_trackHitOutMissTID);
  tree->Branch("t_trackHitOutMissTOBTEC","vector<int>",   &t_trackHitOutMissTOBTEC);
  tree->Branch("t_trackHitInMeasTOB",    "vector<int>",   &t_trackHitInMeasTOB ); 
  tree->Branch("t_trackHitInMeasTEC",    "vector<int>",   &t_trackHitInMeasTEC );  
  tree->Branch("t_trackHitInMeasTIB",    "vector<int>",   &t_trackHitInMeasTIB );  
  tree->Branch("t_trackHitInMeasTID",    "vector<int>",   &t_trackHitInMeasTID );
  tree->Branch("t_trackHitOutMeasTOB",   "vector<int>",   &t_trackHitOutMeasTOB);
  tree->Branch("t_trackHitOutMeasTEC",   "vector<int>",   &t_trackHitOutMeasTEC); 
  tree->Branch("t_trackHitOutMeasTIB",   "vector<int>",   &t_trackHitOutMeasTIB);
  tree->Branch("t_trackHitOutMeasTID",   "vector<int>",   &t_trackHitOutMeasTID);
  tree->Branch("t_trackOutPosOutHitDr",  "vector<double>", &t_trackOutPosOutHitDr);
  tree->Branch("t_trackL",               "vector<double>", &t_trackL);

  tree->Branch("t_trackDxy",          "vector<double>", &t_trackDxy     );
  tree->Branch("t_trackDxyBS",        "vector<double>", &t_trackDxyBS   );
  tree->Branch("t_trackDz",           "vector<double>", &t_trackDz      );
  tree->Branch("t_trackDzBS",         "vector<double>", &t_trackDzBS    );
  tree->Branch("t_trackDxyPV",        "vector<double>", &t_trackDxyPV   );
  tree->Branch("t_trackDzPV",         "vector<double>", &t_trackDzPV    );
  tree->Branch("t_trackChiSq",        "vector<double>", &t_trackChiSq   );
  tree->Branch("t_trackPVIdx",        "vector<int>",    &t_trackPVIdx   );

  t_maxNearP31x31     = new std::vector<double>();
  t_maxNearP25x25     = new std::vector<double>();
  t_maxNearP21x21     = new std::vector<double>();
  t_maxNearP15x15     = new std::vector<double>();
  t_maxNearP13x13     = new std::vector<double>();
  t_maxNearP11x11     = new std::vector<double>();
  t_maxNearP9x9       = new std::vector<double>();
  t_maxNearP7x7       = new std::vector<double>();

  tree->Branch("t_maxNearP31x31",     "vector<double>", &t_maxNearP31x31);
  //tree->Branch("t_maxNearP25x25",     "vector<double>", &t_maxNearP25x25);
  tree->Branch("t_maxNearP21x21",     "vector<double>", &t_maxNearP21x21);
  //tree->Branch("t_maxNearP15x15",     "vector<double>", &t_maxNearP15x15);
  //tree->Branch("t_maxNearP13x13",     "vector<double>", &t_maxNearP13x13);
  //tree->Branch("t_maxNearP11x11",     "vector<double>", &t_maxNearP11x11);
  //tree->Branch("t_maxNearP9x9",       "vector<double>", &t_maxNearP9x9);
  //tree->Branch("t_maxNearP7x7",       "vector<double>", &t_maxNearP7x7);

  t_ecalSpike11x11    = new std::vector<int>();
  t_e3x3              = new std::vector<double>();
  t_e5x5              = new std::vector<double>();
  t_e7x7              = new std::vector<double>();
  t_e9x9              = new std::vector<double>();
  t_e11x11            = new std::vector<double>();
  t_e13x13            = new std::vector<double>();
  t_e15x15            = new std::vector<double>();
  t_e21x21            = new std::vector<double>();
  t_e25x25            = new std::vector<double>();
  t_e31x31            = new std::vector<double>();
  //t_e11x11Xtals       = new std::vector<std::vector<double> >(); 

  tree->Branch("t_ecalSpike11x11",    "vector<int>",    &t_ecalSpike11x11);
  //tree->Branch("t_e3x3",              "vector<double>", &t_e3x3);
  //tree->Branch("t_e5x5",              "vector<double>", &t_e5x5);
  tree->Branch("t_e7x7",              "vector<double>", &t_e7x7);
  tree->Branch("t_e9x9",              "vector<double>", &t_e9x9);
  tree->Branch("t_e11x11",            "vector<double>", &t_e11x11);
  //tree->Branch("t_e13x13",            "vector<double>", &t_e13x13);
  tree->Branch("t_e15x15",            "vector<double>", &t_e15x15);
  //tree->Branch("t_e21x21",            "vector<double>", &t_e21x21);
  //tree->Branch("t_e25x25",            "vector<double>", &t_e25x25);
  //tree->Branch("t_e31x31",            "vector<double>", &t_e31x31);
  //tree->Branch("t_e11x11Xtals",       "vector<vector<double> >", &t_e11x11Xtals);

  t_e7x7_10Sig        = new std::vector<double>();
  t_e9x9_10Sig        = new std::vector<double>();
  t_e11x11_10Sig      = new std::vector<double>();
  t_e15x15_10Sig      = new std::vector<double>();
  t_e7x7_15Sig        = new std::vector<double>();
  t_e9x9_15Sig        = new std::vector<double>();
  t_e11x11_15Sig      = new std::vector<double>();
  t_e15x15_15Sig      = new std::vector<double>();
  t_e7x7_20Sig        = new std::vector<double>();
  t_e9x9_20Sig        = new std::vector<double>();
  t_e11x11_20Sig      = new std::vector<double>();
  t_e15x15_20Sig      = new std::vector<double>();
  t_e7x7_25Sig        = new std::vector<double>();
  t_e9x9_25Sig        = new std::vector<double>();
  t_e11x11_25Sig      = new std::vector<double>();
  t_e15x15_25Sig      = new std::vector<double>();
  t_e7x7_30Sig        = new std::vector<double>();
  t_e9x9_30Sig        = new std::vector<double>();
  t_e11x11_30Sig      = new std::vector<double>();
  t_e15x15_30Sig      = new std::vector<double>();

  tree->Branch("t_e7x7_10Sig"        ,"vector<double>", &t_e7x7_10Sig);
  tree->Branch("t_e9x9_10Sig"        ,"vector<double>", &t_e9x9_10Sig);
  tree->Branch("t_e11x11_10Sig"      ,"vector<double>", &t_e11x11_10Sig);
  tree->Branch("t_e15x15_10Sig"      ,"vector<double>", &t_e15x15_10Sig);
  tree->Branch("t_e7x7_15Sig"        ,"vector<double>", &t_e7x7_15Sig);
  tree->Branch("t_e9x9_15Sig"        ,"vector<double>", &t_e9x9_15Sig);
  tree->Branch("t_e11x11_15Sig"      ,"vector<double>", &t_e11x11_15Sig);
  tree->Branch("t_e15x15_15Sig"      ,"vector<double>", &t_e15x15_15Sig);
  tree->Branch("t_e7x7_20Sig"        ,"vector<double>", &t_e7x7_20Sig);
  tree->Branch("t_e9x9_20Sig"        ,"vector<double>", &t_e9x9_20Sig);
  tree->Branch("t_e11x11_20Sig"      ,"vector<double>", &t_e11x11_20Sig);
  tree->Branch("t_e15x15_20Sig"      ,"vector<double>", &t_e15x15_20Sig);
  tree->Branch("t_e7x7_25Sig"        ,"vector<double>", &t_e7x7_25Sig);
  tree->Branch("t_e9x9_25Sig"        ,"vector<double>", &t_e9x9_25Sig);
  tree->Branch("t_e11x11_25Sig"      ,"vector<double>", &t_e11x11_25Sig);
  tree->Branch("t_e15x15_25Sig"      ,"vector<double>", &t_e15x15_25Sig);
  tree->Branch("t_e7x7_30Sig"        ,"vector<double>", &t_e7x7_30Sig);
  tree->Branch("t_e9x9_30Sig"        ,"vector<double>", &t_e9x9_30Sig);
  tree->Branch("t_e11x11_30Sig"      ,"vector<double>", &t_e11x11_30Sig);
  tree->Branch("t_e15x15_30Sig"      ,"vector<double>", &t_e15x15_30Sig);

  if (doMC) {
    t_esim3x3              = new std::vector<double>();
    t_esim5x5              = new std::vector<double>();
    t_esim7x7              = new std::vector<double>();
    t_esim9x9              = new std::vector<double>();
    t_esim11x11            = new std::vector<double>();
    t_esim13x13            = new std::vector<double>();
    t_esim15x15            = new std::vector<double>();
    t_esim21x21            = new std::vector<double>();
    t_esim25x25            = new std::vector<double>();
    t_esim31x31            = new std::vector<double>();

    t_esim3x3Matched       = new std::vector<double>();
    t_esim5x5Matched       = new std::vector<double>();
    t_esim7x7Matched       = new std::vector<double>();
    t_esim9x9Matched       = new std::vector<double>();
    t_esim11x11Matched     = new std::vector<double>();
    t_esim13x13Matched     = new std::vector<double>();
    t_esim15x15Matched     = new std::vector<double>();
    t_esim21x21Matched     = new std::vector<double>();
    t_esim25x25Matched     = new std::vector<double>();
    t_esim31x31Matched     = new std::vector<double>();

    t_esim3x3Rest          = new std::vector<double>();
    t_esim5x5Rest          = new std::vector<double>();
    t_esim7x7Rest          = new std::vector<double>();
    t_esim9x9Rest          = new std::vector<double>();
    t_esim11x11Rest        = new std::vector<double>();
    t_esim13x13Rest        = new std::vector<double>();
    t_esim15x15Rest        = new std::vector<double>();
    t_esim21x21Rest        = new std::vector<double>();
    t_esim25x25Rest        = new std::vector<double>();
    t_esim31x31Rest        = new std::vector<double>();

    t_esim3x3Photon        = new std::vector<double>();
    t_esim5x5Photon        = new std::vector<double>();
    t_esim7x7Photon        = new std::vector<double>();
    t_esim9x9Photon        = new std::vector<double>();
    t_esim11x11Photon      = new std::vector<double>();
    t_esim13x13Photon      = new std::vector<double>();
    t_esim15x15Photon      = new std::vector<double>();
    t_esim21x21Photon      = new std::vector<double>();
    t_esim25x25Photon      = new std::vector<double>();
    t_esim31x31Photon      = new std::vector<double>();

    t_esim3x3NeutHad       = new std::vector<double>();
    t_esim5x5NeutHad       = new std::vector<double>();
    t_esim7x7NeutHad       = new std::vector<double>();
    t_esim9x9NeutHad       = new std::vector<double>();
    t_esim11x11NeutHad     = new std::vector<double>();
    t_esim13x13NeutHad     = new std::vector<double>();
    t_esim15x15NeutHad     = new std::vector<double>();
    t_esim21x21NeutHad     = new std::vector<double>();
    t_esim25x25NeutHad     = new std::vector<double>();
    t_esim31x31NeutHad     = new std::vector<double>();

    t_esim3x3CharHad       = new std::vector<double>();
    t_esim5x5CharHad       = new std::vector<double>();
    t_esim7x7CharHad       = new std::vector<double>();
    t_esim9x9CharHad       = new std::vector<double>();
    t_esim11x11CharHad     = new std::vector<double>();
    t_esim13x13CharHad     = new std::vector<double>();
    t_esim15x15CharHad     = new std::vector<double>();
    t_esim21x21CharHad     = new std::vector<double>();
    t_esim25x25CharHad     = new std::vector<double>();
    t_esim31x31CharHad     = new std::vector<double>();

    t_trkEcalEne           = new std::vector<double>();
    t_simTrackP            = new std::vector<double>();
    t_esimPdgId            = new std::vector<double>();

    //tree->Branch("t_esim3x3",             "vector<double>", &t_esim3x3);
    //tree->Branch("t_esim5x5",             "vector<double>", &t_esim5x5);
    tree->Branch("t_esim7x7",             "vector<double>", &t_esim7x7);
    tree->Branch("t_esim9x9",             "vector<double>", &t_esim9x9);
    tree->Branch("t_esim11x11",           "vector<double>", &t_esim11x11);
    //tree->Branch("t_esim13x13",           "vector<double>", &t_esim13x13);
    tree->Branch("t_esim15x15",           "vector<double>", &t_esim15x15);
    //tree->Branch("t_esim21x21",           "vector<double>", &t_esim21x21);
    //tree->Branch("t_esim25x25",           "vector<double>", &t_esim25x25);
    //tree->Branch("t_esim31x31",           "vector<double>", &t_esim31x31);

    //tree->Branch("t_esim3x3Matched",      "vector<double>", &t_esim3x3Matched);
    //tree->Branch("t_esim5x5Matched",      "vector<double>", &t_esim5x5Matched);
    tree->Branch("t_esim7x7Matched",      "vector<double>", &t_esim7x7Matched);
    tree->Branch("t_esim9x9Matched",      "vector<double>", &t_esim9x9Matched);
    tree->Branch("t_esim11x11Matched",    "vector<double>", &t_esim11x11Matched);
    //tree->Branch("t_esim13x13Matched",    "vector<double>", &t_esim13x13Matched);
    tree->Branch("t_esim15x15Matched",    "vector<double>", &t_esim15x15Matched);
    //tree->Branch("t_esim21x21Matched",    "vector<double>", &t_esim21x21Matched);
    //tree->Branch("t_esim25x25Matched",    "vector<double>", &t_esim25x25Matched);
    //tree->Branch("t_esim31x31Matched",    "vector<double>", &t_esim31x31Matched);

    //tree->Branch("t_esim3x3Rest",         "vector<double>", &t_esim3x3Rest);
    //tree->Branch("t_esim5x5Rest",         "vector<double>", &t_esim5x5Rest);
    tree->Branch("t_esim7x7Rest",         "vector<double>", &t_esim7x7Rest);
    tree->Branch("t_esim9x9Rest",         "vector<double>", &t_esim9x9Rest);
    tree->Branch("t_esim11x11Rest",       "vector<double>", &t_esim11x11Rest);
    //tree->Branch("t_esim13x13Rest",       "vector<double>", &t_esim13x13Rest);
    tree->Branch("t_esim15x15Rest",       "vector<double>", &t_esim15x15Rest);
    //tree->Branch("t_esim21x21Rest",       "vector<double>", &t_esim21x21Rest);
    //tree->Branch("t_esim25x25Rest",       "vector<double>", &t_esim25x25Rest);
    //tree->Branch("t_esim31x31Rest",       "vector<double>", &t_esim31x31Rest);

    //tree->Branch("t_esim3x3Photon",       "vector<double>", &t_esim3x3Photon);
    //tree->Branch("t_esim5x5Photon",       "vector<double>", &t_esim5x5Photon);
    tree->Branch("t_esim7x7Photon",       "vector<double>", &t_esim7x7Photon);
    tree->Branch("t_esim9x9Photon",       "vector<double>", &t_esim9x9Photon);
    tree->Branch("t_esim11x11Photon",     "vector<double>", &t_esim11x11Photon);
    //tree->Branch("t_esim13x13Photon",     "vector<double>", &t_esim13x13Photon);
    tree->Branch("t_esim15x15Photon",     "vector<double>", &t_esim15x15Photon);
    //tree->Branch("t_esim21x21Photon",     "vector<double>", &t_esim21x21Photon);
    //tree->Branch("t_esim25x25Photon",     "vector<double>", &t_esim25x25Photon);
    //tree->Branch("t_esim31x31Photon",     "vector<double>", &t_esim31x31Photon);

    //tree->Branch("t_esim3x3NeutHad",      "vector<double>", &t_esim3x3NeutHad);
    //tree->Branch("t_esim5x5NeutHad",      "vector<double>", &t_esim5x5NeutHad);
    tree->Branch("t_esim7x7NeutHad",      "vector<double>", &t_esim7x7NeutHad);
    tree->Branch("t_esim9x9NeutHad",      "vector<double>", &t_esim9x9NeutHad);
    tree->Branch("t_esim11x11NeutHad",    "vector<double>", &t_esim11x11NeutHad);
    //tree->Branch("t_esim13x13NeutHad",    "vector<double>", &t_esim13x13NeutHad);
    tree->Branch("t_esim15x15NeutHad",    "vector<double>", &t_esim15x15NeutHad);
    //tree->Branch("t_esim21x21NeutHad",    "vector<double>", &t_esim21x21NeutHad);
    //tree->Branch("t_esim25x25NeutHad",    "vector<double>", &t_esim25x25NeutHad);
    //tree->Branch("t_esim31x31NeutHad",    "vector<double>", &t_esim31x31NeutHad);

    //tree->Branch("t_esim3x3CharHad",      "vector<double>", &t_esim3x3CharHad);
    //tree->Branch("t_esim5x5CharHad",      "vector<double>", &t_esim5x5CharHad);
    tree->Branch("t_esim7x7CharHad",      "vector<double>", &t_esim7x7CharHad);
    tree->Branch("t_esim9x9CharHad",      "vector<double>", &t_esim9x9CharHad);
    tree->Branch("t_esim11x11CharHad",    "vector<double>", &t_esim11x11CharHad);
    //tree->Branch("t_esim13x13CharHad",    "vector<double>", &t_esim13x13CharHad);
    tree->Branch("t_esim15x15CharHad",    "vector<double>", &t_esim15x15CharHad);
    //tree->Branch("t_esim21x21CharHad",    "vector<double>", &t_esim21x21CharHad);
    //tree->Branch("t_esim25x25CharHad",    "vector<double>", &t_esim25x25CharHad);
    //tree->Branch("t_esim31x31CharHad",    "vector<double>", &t_esim31x31CharHad);

    tree->Branch("t_trkEcalEne",          "vector<double>", &t_trkEcalEne);
    tree->Branch("t_simTrackP",           "vector<double>", &t_simTrackP);
    tree->Branch("t_esimPdgId",           "vector<double>", &t_esimPdgId);
  }

  t_maxNearHcalP3x3      = new std::vector<double>();
  t_maxNearHcalP5x5      = new std::vector<double>();
  t_maxNearHcalP7x7      = new std::vector<double>();
  t_h3x3                 = new std::vector<double>();
  t_h5x5                 = new std::vector<double>();
  t_h7x7                 = new std::vector<double>();
  t_h3x3Sig              = new std::vector<double>();
  t_h5x5Sig              = new std::vector<double>();
  t_h7x7Sig              = new std::vector<double>();
  t_infoHcal             = new std::vector<int>();

  if (doMC) {
    t_trkHcalEne           = new std::vector<double>();
    t_hsim3x3              = new std::vector<double>();
    t_hsim5x5              = new std::vector<double>();
    t_hsim7x7              = new std::vector<double>();
    t_hsim3x3Matched       = new std::vector<double>();
    t_hsim5x5Matched       = new std::vector<double>();
    t_hsim7x7Matched       = new std::vector<double>();
    t_hsim3x3Rest          = new std::vector<double>();
    t_hsim5x5Rest          = new std::vector<double>();
    t_hsim7x7Rest          = new std::vector<double>();
    t_hsim3x3Photon        = new std::vector<double>();
    t_hsim5x5Photon        = new std::vector<double>();
    t_hsim7x7Photon        = new std::vector<double>();
    t_hsim3x3NeutHad       = new std::vector<double>();
    t_hsim5x5NeutHad       = new std::vector<double>();
    t_hsim7x7NeutHad       = new std::vector<double>();
    t_hsim3x3CharHad       = new std::vector<double>();
    t_hsim5x5CharHad       = new std::vector<double>();
    t_hsim7x7CharHad       = new std::vector<double>();
  }

  tree->Branch("t_maxNearHcalP3x3",     "vector<double>", &t_maxNearHcalP3x3);
  tree->Branch("t_maxNearHcalP5x5",     "vector<double>", &t_maxNearHcalP5x5);
  tree->Branch("t_maxNearHcalP7x7",     "vector<double>", &t_maxNearHcalP7x7);
  tree->Branch("t_h3x3",                "vector<double>", &t_h3x3);
  tree->Branch("t_h5x5",                "vector<double>", &t_h5x5);
  tree->Branch("t_h7x7",                "vector<double>", &t_h7x7);
  tree->Branch("t_h3x3Sig",             "vector<double>", &t_h3x3Sig);
  tree->Branch("t_h5x5Sig",             "vector<double>", &t_h5x5Sig);
  tree->Branch("t_h7x7Sig",             "vector<double>", &t_h7x7Sig);
  tree->Branch("t_infoHcal",            "vector<int>",    &t_infoHcal);

  if (doMC) {
    tree->Branch("t_trkHcalEne",          "vector<double>", &t_trkHcalEne);
    tree->Branch("t_hsim3x3",             "vector<double>", &t_hsim3x3);
    tree->Branch("t_hsim5x5",             "vector<double>", &t_hsim5x5);
    tree->Branch("t_hsim7x7",             "vector<double>", &t_hsim7x7);
    tree->Branch("t_hsim3x3Matched",      "vector<double>", &t_hsim3x3Matched);
    tree->Branch("t_hsim5x5Matched",      "vector<double>", &t_hsim5x5Matched);
    tree->Branch("t_hsim7x7Matched",      "vector<double>", &t_hsim7x7Matched);
    tree->Branch("t_hsim3x3Rest",         "vector<double>", &t_hsim3x3Rest);
    tree->Branch("t_hsim5x5Rest",         "vector<double>", &t_hsim5x5Rest);
    tree->Branch("t_hsim7x7Rest",         "vector<double>", &t_hsim7x7Rest);
    tree->Branch("t_hsim3x3Photon",       "vector<double>", &t_hsim3x3Photon);
    tree->Branch("t_hsim5x5Photon",       "vector<double>", &t_hsim5x5Photon);
    tree->Branch("t_hsim7x7Photon",       "vector<double>", &t_hsim7x7Photon);
    tree->Branch("t_hsim3x3NeutHad",      "vector<double>", &t_hsim3x3NeutHad);
    tree->Branch("t_hsim5x5NeutHad",      "vector<double>", &t_hsim5x5NeutHad);
    tree->Branch("t_hsim7x7NeutHad",      "vector<double>", &t_hsim7x7NeutHad);
    tree->Branch("t_hsim3x3CharHad",      "vector<double>", &t_hsim3x3CharHad);
    tree->Branch("t_hsim5x5CharHad",      "vector<double>", &t_hsim5x5CharHad);
    tree->Branch("t_hsim7x7CharHad",      "vector<double>", &t_hsim7x7CharHad);
  }
  tree->Branch("t_nTracks",             &t_nTracks,       "t_nTracks/I");

}


double IsolatedTracksNxN::DeltaPhi(double v1, double v2) {
  // Computes the correctly normalized phi difference
  // v1, v2 = phi of object 1 and 2
  
  double pi    = 3.141592654;
  double twopi = 6.283185307;
  
  double diff = std::abs(v2 - v1);
  double corr = twopi - diff;
  if (diff < pi){ return diff;} else { return corr;} 
}

double IsolatedTracksNxN::DeltaR(double eta1, double phi1, double eta2, double phi2) {
  double deta = eta1 - eta2;
  double dphi = DeltaPhi(phi1, phi2);
  return std::sqrt(deta*deta + dphi*dphi);
}

void IsolatedTracksNxN::printTrack(const reco::Track* pTrack) {
  
  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality);

  std::cout << " Reference Point " << pTrack->referencePoint() <<"\n"
	    << " TrackMmentum " << pTrack->momentum()
	    << " (pt,eta,phi)(" << pTrack->pt()<<","<<pTrack->eta()<<","<<pTrack->phi()<<")"
	    << " p " << pTrack->p() << "\n"
	    << " Normalized chi2 " << pTrack->normalizedChi2() <<"  charge " << pTrack->charge()
	    << " qoverp() " << pTrack->qoverp() <<"+-" << pTrack->qoverpError()
	    << " d0 " << pTrack->d0() << "\n"
	    << " NValidHits " << pTrack->numberOfValidHits() << "  NLostHits " << pTrack->numberOfLostHits()
	    << " TrackQuality " << pTrack->qualityName(trackQuality_) << " " << pTrack->quality(trackQuality_) 
	    << std::endl;
  
  if( printTrkHitPattern_ ) {
    const reco::HitPattern& p = pTrack->hitPattern();
    const reco::HitPattern& p1 = pTrack->trackerExpectedHitsInner();
    const reco::HitPattern& p2 = pTrack->trackerExpectedHitsOuter();

    std::cout<<"default " << std::endl;
    for (int i=0; i<p.numberOfHits(); i++) {
      p.printHitPattern(i, std::cout);
    }
    std::cout<<"trackerExpectedHitsInner() " << std::endl;
    for (int i=0; i<p1.numberOfHits(); i++) {
      p1.printHitPattern(i, std::cout);
    }
    std::cout<<"trackerExpectedHitsOuter() " << std::endl;
    for (int i=0; i<p2.numberOfHits(); i++) {
      p2.printHitPattern(i, std::cout);
    }


    std::cout << "\n \t trackerLayersWithMeasurement() "     << p.trackerLayersWithMeasurement() 
	      << "\n \t pixelLayersWithMeasurement() "       << p.pixelLayersWithMeasurement() 
	      << "\n \t stripLayersWithMeasurement() "       << p.stripLayersWithMeasurement()  
	      << "\n \t pixelBarrelLayersWithMeasurement() " << p.pixelBarrelLayersWithMeasurement()
	      << "\n \t pixelEndcapLayersWithMeasurement() " << p.pixelEndcapLayersWithMeasurement()
	      << "\n \t stripTIBLayersWithMeasurement() "    << p.stripTIBLayersWithMeasurement()
	      << "\n \t stripTIDLayersWithMeasurement() "    << p.stripTIDLayersWithMeasurement()
	      << "\n \t stripTOBLayersWithMeasurement() "    << p.stripTOBLayersWithMeasurement()
	      << "\n \t stripTECLayersWithMeasurement() "    << p.stripTECLayersWithMeasurement()
	      << std::endl;

  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsolatedTracksNxN);

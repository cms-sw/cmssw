// -*- C++ -*
//
// Package:    IsolatedTracksTree
// Class:      IsolatedTracksTree
// 
/**\class IsolatedTracksTree IsolatedTracksTree.cc Analysis/IsolatedTracksTree/src/IsolatedTracksTree.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seema Sharma
//         Created:  Mon Aug 10 15:30:40 CST 2009
// $Id: IsolatedTracksTree.cc,v 1.1 2009/11/05 21:02:30 sunanda Exp $
//
//

#include "Calibration/IsolatedParticles/plugins/IsolatedTracksTree.h"

IsolatedTracksTree::IsolatedTracksTree(const edm::ParameterSet& iConfig) {

  //now do what ever initialization is needed
  myverbose_           = iConfig.getUntrackedParameter<int>( "Verbosity", 5 );

  useJetTrigger_       = iConfig.getUntrackedParameter<bool>( "useJetTrigger", false);
  drLeadJetVeto_       = iConfig.getUntrackedParameter<double>( "drLeadJetVeto",  1.2 );
  ptMinLeadJet_        = iConfig.getUntrackedParameter<double>( "ptMinLeadJet",  15.0 );

  debugTrks_           = iConfig.getUntrackedParameter<int>("DebugTracks");
  printTrkHitPattern_  = iConfig.getUntrackedParameter<bool>("PrintTrkHitPattern");
  
  minTrackP_           = iConfig.getUntrackedParameter<double>( "minTrackP", 1.0);
  maxTrackEta_         = iConfig.getUntrackedParameter<double>( "maxTrackEta", 5.0);
  maxNearTrackPT_      = iConfig.getUntrackedParameter<double>( "maxNearTrackPT", 1.0);

  debugEcalSimInfo_    = iConfig.getUntrackedParameter<int>("DebugEcalSimInfo");
  applyEcalIsolation_  = iConfig.getUntrackedParameter<bool>("ApplyEcalIsolation");

  L1extraTauJetSource_ = iConfig.getParameter<edm::InputTag>("L1extraTauJetSource");
  L1extraCenJetSource_ = iConfig.getParameter<edm::InputTag>("L1extraCenJetSource");
  L1extraFwdJetSource_ = iConfig.getParameter<edm::InputTag>("L1extraFwdJetSource");

  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
  trackAssociator_ =  new TrackDetectorAssociator();
  trackAssociator_->useDefaultPropagator();

  if(myverbose_>=0) {
    std::cout <<"Parameters read from config file \n" 
	      <<"myverbose_ "        << myverbose_        << "\t useJetTrigger_ "  << useJetTrigger_ << "\n"
	      <<"drLeadJetVeto_ "  << drLeadJetVeto_  
	      <<"minTrackP_ "     << minTrackP_     << "\t maxTrackEta_ "    << maxTrackEta_    << "\n"
	      <<"maxNearTrackPT_ " << maxNearTrackPT_ 
	      << std::endl;
  }
}

IsolatedTracksTree::~IsolatedTracksTree() {
  delete  trackAssociator_;
}

void IsolatedTracksTree::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  nEventProc++;

  // check the L1 objects
  bool   L1Pass = false;
  double leadL1JetPT=-999, leadL1JetEta=-999,  leadL1JetPhi=-999;
  if( !useJetTrigger_ ) {
    L1Pass = true;
  } else {
    edm::Handle<l1extra::L1JetParticleCollection> l1TauHandle;
    iEvent.getByLabel(L1extraTauJetSource_,l1TauHandle);
    l1extra::L1JetParticleCollection::const_iterator itr;
    for(itr = l1TauHandle->begin(); itr != l1TauHandle->end(); ++itr ) {
      if( itr->pt()>leadL1JetPT ) {
	leadL1JetPT  = itr->pt();
	leadL1JetEta = itr->eta();
	leadL1JetPhi = itr->phi();
      }
      /*
      std::cout << "tau p/pt " << itr->momentum() << " " << itr->pt() 
		<< "  eta/phi " << itr->eta() << " " << itr->phi()
		<< std::endl;
      */
    }
    edm::Handle<l1extra::L1JetParticleCollection> l1CenJetHandle;
    iEvent.getByLabel(L1extraCenJetSource_,l1CenJetHandle);
    for( itr = l1CenJetHandle->begin();  itr != l1CenJetHandle->end(); ++itr ) {
      if( itr->pt()>leadL1JetPT ) {
	leadL1JetPT  = itr->pt();
	leadL1JetEta = itr->eta();
	leadL1JetPhi = itr->phi();
      }
      /*
      std::cout << "jet p/pt " << itr->momentum() << " " << itr->pt() 
		<< "  eta/phi " << itr->eta() << " " << itr->phi()
		<< std::endl;
      */
    }
    edm::Handle<l1extra::L1JetParticleCollection> l1FwdJetHandle;
    iEvent.getByLabel(L1extraFwdJetSource_,l1FwdJetHandle);
    for( itr = l1FwdJetHandle->begin();  itr != l1FwdJetHandle->end(); ++itr ) {
      if( itr->pt()>leadL1JetPT ) {
	leadL1JetPT  = itr->pt();
	leadL1JetEta = itr->eta();
	leadL1JetPhi = itr->phi();
      }
      /*
      std::cout << "jet p/pt " << itr->momentum() << " " << itr->pt() 
		<< "  eta/phi " << itr->eta() << " " << itr->phi()
		<< std::endl;
      */
    }
    if(leadL1JetPT>ptMinLeadJet_) L1Pass = true;
  }

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  const CaloSubdetectorGeometry* gEB = geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  const CaloSubdetectorGeometry* gEE = geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
  const CaloSubdetectorGeometry* gHB = geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
  //  const CaloSubdetectorGeometry* gHE = geo->getSubdetectorGeometry(DetId::Hcal,HcalEndcap);
  
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology); 
  const CaloSubdetectorTopology* theEBTopology   = theCaloTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel);
  const CaloSubdetectorTopology* theEETopology   = theCaloTopology->getSubdetectorTopology(DetId::Ecal,EcalEndcap);
  
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<IdealGeometryRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();
  
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByLabel("ecalRecHit","EcalRecHitsEB",barrelRecHitsHandle);
  iEvent.getByLabel("ecalRecHit","EcalRecHitsEE",endcapRecHitsHandle);
  
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByLabel("hbhereco",hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByLabel("generalTracks", trkCollection);
  reco::TrackCollection::const_iterator trkItr;
  if(debugTrks_>1){
    std::cout << "Track Collection: " << std::endl;
    std::cout << "Number of Tracks " << trkCollection->size() << std::endl;
  }
  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);
  
  //get Handles to SimTracks and SimHits
  edm::Handle<edm::SimTrackContainer> SimTk;
  iEvent.getByLabel("g4SimHits",SimTk);
  edm::SimTrackContainer::const_iterator simTrkItr;

  edm::Handle<edm::SimVertexContainer> SimVtx;
  iEvent.getByLabel("g4SimHits",SimVtx);
  edm::SimVertexContainer::const_iterator vtxItr = SimVtx->begin();

  //get Handles to PCaloHitContainers of eb/ee/hbhe
  edm::Handle<edm::PCaloHitContainer> pcaloeb;
  iEvent.getByLabel("g4SimHits", "EcalHitsEB", pcaloeb);

  edm::Handle<edm::PCaloHitContainer> pcaloee;
  iEvent.getByLabel("g4SimHits", "EcalHitsEE", pcaloee);

  edm::Handle<edm::PCaloHitContainer> pcalohh;
  iEvent.getByLabel("g4SimHits", "HcalHits", pcalohh);

  
  std::vector<int>  ifGood(trkCollection->size(), 1);

  //  std::vector<int>  isChargedIso(trkCollection->size(), 1);

  h_nTracks->Fill(trkCollection->size());
  
  bool printdone = false;

  int nTracks        = 0;
  t_nTracks          = 0;
  t_nTracksAwayL1    = 0;
  t_nTracksIsoBy5GeV = 0;

  if(L1Pass) {

    t_nTracks = trkCollection->size();

    if(myverbose_>5) std::cout << "leadL1JetPT " << leadL1JetPT << " L1Pass " << L1Pass << std::endl;

    TrackerHitAssociator* associate = new TrackerHitAssociator::TrackerHitAssociator(iEvent);

    //decide the goodness of each track for this track collection
    for( trkItr = trkCollection->begin(),nTracks=0; trkItr != trkCollection->end(); ++trkItr, nTracks++){
      const reco::Track* pTrack = &(*trkItr);
      bool   trkQuality  = pTrack->quality(trackQuality_);

      //const reco::HitPattern& hitp = pTrack->hitPattern();
      //int nLayersCrossed = hitp.trackerLayersWithMeasurement() ;        
      //if( !trkQuality || nLayersCrossed<8 ) ifGood[nTracks]=0;
      if( !trkQuality ) ifGood[nTracks]=0;
    }
    
    for( trkItr = trkCollection->begin(),nTracks=0; trkItr != trkCollection->end(); ++trkItr, nTracks++){
    
      const reco::Track* pTrack = &(*trkItr);
      
      if(debugTrks_>0) {std::cout << " Track index " << nTracks << " ";  printTrack(pTrack); }

      const reco::HitPattern& hitp = pTrack->hitPattern();
      int nLayersCrossed = hitp.trackerLayersWithMeasurement() ;        
      int nOuterHits     = hitp.stripTOBLayersWithMeasurement()+hitp.stripTECLayersWithMeasurement() ;

      double eta1        = pTrack->momentum().eta();
      double phi1        = pTrack->momentum().phi();
      double pt1         = pTrack->pt();
      double p1          = pTrack->p();

      h_recEtaPt_0->Fill(eta1, pt1);
      h_recEtaP_0->Fill(eta1, p1);
      h_recPt_0->Fill(pt1);
      h_recP_0->Fill(p1);
      h_recEta_0->Fill(eta1);
      h_recPhi_0->Fill(phi1);
      
      if(ifGood[nTracks] && nLayersCrossed>7 ) {
	h_recEtaPt_1->Fill(eta1, pt1);
	h_recEtaP_1->Fill(eta1, p1);
	h_recPt_1->Fill(pt1);
	h_recP_1->Fill(p1);
	h_recEta_1->Fill(eta1);
	h_recPhi_1->Fill(phi1);
      }
    
      // check the charge isolation by propagating tracks to ecal surface
      // find the impact point on ecal surface
      const FreeTrajectoryState fts1 = trackAssociator_->getFreeTrajectoryState(iSetup, *pTrack);
      TrackDetMatchInfo info1 = trackAssociator_->associate(iEvent, iSetup, fts1, parameters_);
      const GlobalPoint point1(info1.trkGlobPosAtEcal.x(),info1.trkGlobPosAtEcal.y(),info1.trkGlobPosAtEcal.z());
      //std::cout << "Ecal point1 " << point1 << std::endl;

      if( ifGood[nTracks] && pt1>minTrackP_ && std::abs(eta1)<maxTrackEta_ && info1.isGoodEcal) { 

	bool useRegion = true;
	if( useJetTrigger_ ) {
	  double dphi = DeltaPhi(phi1, leadL1JetPhi);
	  double deta = eta1 - leadL1JetEta;
	  double dr =  sqrt(dphi*dphi + deta*deta);
	  if(dr<drLeadJetVeto_) useRegion = false;
	  //if (useRegion && nOuterHits>4) std::cout << "nTracks " <<  nTracks <<"  dr " << dr << std::endl;
	}
	//if( !useRegion ) continue;
	if( useRegion ) {
      
	  t_nTracksAwayL1++;
	  
	  double maxNearP = 999.0, maxNearP31x31=999.0, maxNearP21x21=999.0, maxNearP15x15=999.0;
	  if(std::abs(point1.eta())<1.479) {
	    const DetId isoCell = gEB->getClosestCell(point1);
	    CaloNavigator<DetId> theNavigator(isoCell,theEBTopology);
	    //maxNearP = chargeIsolation(iEvent, iSetup, theNavigator, trkItr, trkCollection, barrelRecHitsHandle, endcapRecHitsHandle, gEB, gEE, 15,15);
	    maxNearP31x31 = chargeIsolation(iEvent, iSetup, theNavigator, trkItr, trkCollection, barrelRecHitsHandle, endcapRecHitsHandle, gEB, gEE, 15,15);
	    maxNearP21x21 = chargeIsolation(iEvent, iSetup, theNavigator, trkItr, trkCollection, barrelRecHitsHandle, endcapRecHitsHandle, gEB, gEE, 10,10);
	    maxNearP15x15 = chargeIsolation(iEvent, iSetup, theNavigator, trkItr, trkCollection, barrelRecHitsHandle, endcapRecHitsHandle, gEB, gEE, 7,7);
	    maxNearP = maxNearP15x15;
	  } else {
	    const DetId isoCell = gEE->getClosestCell(point1);
	    CaloNavigator<DetId> theNavigator(isoCell,theEETopology);
	    //maxNearP = chargeIsolation(iEvent, iSetup, theNavigator, trkItr, trkCollection, barrelRecHitsHandle, endcapRecHitsHandle, gEB, gEE, 15,15);
	    maxNearP31x31 = chargeIsolation(iEvent, iSetup, theNavigator, trkItr, trkCollection, barrelRecHitsHandle, endcapRecHitsHandle, gEB, gEE, 15,15);
	    maxNearP21x21 = chargeIsolation(iEvent, iSetup, theNavigator, trkItr, trkCollection, barrelRecHitsHandle, endcapRecHitsHandle, gEB, gEE, 10,10);
	    maxNearP15x15 = chargeIsolation(iEvent, iSetup, theNavigator, trkItr, trkCollection, barrelRecHitsHandle, endcapRecHitsHandle, gEB, gEE, 7,7);
	    maxNearP = maxNearP15x15;
	  }
	  
	  //std::cout << "maxNearP " << maxNearP << " isChargedIso[nTracks] " << isChargedIso[nTracks] << std::endl;
	  //std::cout << "maxNearP " << maxNearP << " maxNearP31x31 " << maxNearP31x31 << " maxNearP21x21 " << maxNearP21x21 << " maxNearP15x15 " << maxNearP15x15 << std::endl;
	  
	  
	  //if( isChargedIso[nTracks]>0 ) {
	  //if(nOuterHits>4) std::cout << "nTracks " << nTracks << " " << p1 << " " << eta1 << " " << maxNearP << std::endl;

	  if( maxNearP31x31<1.0 && nLayersCrossed>7 && nOuterHits>4) {
	    h_recEtaPt_2->Fill(eta1, pt1);
	    h_recEtaP_2->Fill(eta1, p1);
	    h_recPt_2->Fill(pt1);
	    h_recP_2->Fill(p1);
	    h_recEta_2->Fill(eta1);
	    h_recPhi_2->Fill(phi1);
	  }
	  
	  if( maxNearP<5.0) {
	    
	  // get the matching simTrack
	    edm::SimTrackContainer::const_iterator matchedSimTrk = spr::matchedSimTrack(iEvent, SimTk, SimVtx, pTrack, *associate, false);
	    double simTrackP = matchedSimTrk->momentum().P();
	    //std::cout << "Rec Mom " << p1 << " SimMom " << simTrackP << std::endl;
	    
	    // get ECal Tranverse Profile
	    double e3x3=0,    e5x5=0,    e7x7=0,    e9x9=0,    e11x11=0,    e13x13=0,    e15x15=0,   e25x25=0;
	    double esim3x3=0, esim5x5=0, esim7x7=0, esim9x9=0, esim11x11=0, esim13x13=0, esim15x15=0,esim25x25=0;
	    double trkEcalEne=0;
	    std::map<std::string, double> simInfo3x3, simInfo5x5, simInfo7x7, simInfo9x9, simInfo11x11, simInfo13x13, simInfo15x15, simInfo25x25;
	    
	    if(std::abs(point1.eta())<1.479) {
	      const DetId isoCell = gEB->getClosestCell(point1);
	      CaloNavigator<DetId> theNavigator(isoCell,theEBTopology);
	      
	      e3x3         = spr::eECALmatrix(theNavigator,barrelRecHitsHandle,1,1);
	      e5x5         = spr::eECALmatrix(theNavigator,barrelRecHitsHandle,2,2);
	      e7x7         = spr::eECALmatrix(theNavigator,barrelRecHitsHandle,3,3);
	      e9x9         = spr::eECALmatrix(theNavigator,barrelRecHitsHandle,4,4);
	      e11x11       = spr::eECALmatrix(theNavigator,barrelRecHitsHandle,5,5);
	      e13x13       = spr::eECALmatrix(theNavigator,barrelRecHitsHandle,6,6);
	      e15x15       = spr::eECALmatrix(theNavigator,barrelRecHitsHandle,7,7);
	      e25x25       = spr::eECALmatrix(theNavigator,barrelRecHitsHandle,12,12);
	      
	      // check the energy from SimHits
	      simInfo3x3   = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloeb, SimTk, SimVtx, pTrack, *associate, 1,1);
	      simInfo5x5   = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloeb, SimTk, SimVtx, pTrack, *associate, 2,2);
	      simInfo7x7   = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloeb, SimTk, SimVtx, pTrack, *associate, 3,3);
	      simInfo9x9   = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloeb, SimTk, SimVtx, pTrack, *associate, 4,4);
	      simInfo11x11 = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloeb, SimTk, SimVtx, pTrack, *associate, 5,5);
	      simInfo13x13 = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloeb, SimTk, SimVtx, pTrack, *associate, 6,6);
	      simInfo15x15 = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloeb, SimTk, SimVtx, pTrack, *associate, 7,7);
	      simInfo25x25 = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloeb, SimTk, SimVtx, pTrack, *associate, 12,12);
	      
	      trkEcalEne   = spr::eCaloSimInfo(iEvent, geo, pcaloeb, SimTk, SimVtx, pTrack, *associate);

	      esim3x3      = simInfo3x3["eTotal"];
	      esim5x5      = simInfo5x5["eTotal"];
	      esim7x7      = simInfo7x7["eTotal"];
	      esim9x9      = simInfo9x9["eTotal"];
	      esim11x11    = simInfo11x11["eTotal"];
	      esim13x13    = simInfo13x13["eTotal"];
	      esim15x15    = simInfo15x15["eTotal"];
	      esim25x25    = simInfo25x25["eTotal"];
	    
	      // debug it
	      // ------------------------------------------------------------------------------------------------
	      if ( myverbose_>10 && simInfo15x15["eRest"]>0 ) {
		
		std::cout << "esim15x15 "    << esim15x15 << "  eMatched " << simInfo15x15["eMatched"] 
			  << " pdgMatched "  << simInfo15x15["pdgMatched"] <<" eGamma " << simInfo15x15["eGamma"] 
			  << " eNeutralHad " << simInfo15x15["eNeutralHad"] << " eChargedHad " 
			  << simInfo15x15["eChargedHad"]
			  << " eRest " << simInfo15x15["eRest"] << " eTotal " << simInfo15x15["eTotal"] 
			  << " trkEcalEne " << trkEcalEne << std::endl;
		
		edm::SimTrackContainer::const_iterator tkInfo = spr::matchedSimTrack(iEvent, SimTk, SimVtx, pTrack, *associate, true);
		std::vector<edm::PCaloHitContainer::const_iterator> hittemp = spr::hitECALmatrix(theNavigator,pcaloeb,7,7);
		std::cout << "trkMom " << pTrack->p() << " eta " << pTrack->eta() << " phi " << pTrack->phi() << std::endl;
		
		std::map<std::string, double> debug15x15 = spr::eCaloSimInfo(geo, pcaloeb, SimTk, SimVtx, hittemp, tkInfo, 150, true);
		trkEcalEne   = spr::eCaloSimInfo(iEvent, geo, pcaloeb, SimTk, SimVtx, pTrack, *associate, 150, true);
		
		if (!printdone) {
		  printdone = true;
		  std::cout << "Printing sim Tracks " << std::endl;
		  std::cout << "No. of SimVertices " << SimVtx->size() << " SimTk Size " << SimTk->size() << std::endl;
		  int ii=0;
		  for(edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr!= SimTk->end(); simTrkItr++, ii++){
		    std::cout <<"SimTrack " << ii <<" TrackId " <<  simTrkItr->trackId() << " GenPartIdx " << simTrkItr->genpartIndex() 
			      << " VertexIdx " << simTrkItr->vertIndex() << " Type " << simTrkItr->type() << std::endl;
		  }
		  ii=0;
		  for(edm::SimVertexContainer::const_iterator simVtxItr = SimVtx->begin(); simVtxItr!= SimVtx->end(); simVtxItr++, ii++){
		    std::cout << "SimVertex " << ii << " VtxParentIdx " << simVtxItr->parentIndex() 
			      << "  position " << simVtxItr->position() << std::endl;
		  }
		}
	      } // just printing 
	      // ------------------------------------------------------------------------------------------------
	      
	    } else {
	      const DetId isoCell = gEE->getClosestCell(point1);
	      CaloNavigator<DetId> theNavigator(isoCell,theEETopology);
	      
	      //---	  std::cout <<"Closest cell EE " << (EEDetId)isoCell  << std::endl;
	      
	      e3x3         = spr::eECALmatrix(theNavigator,endcapRecHitsHandle,1,1);
	      e5x5         = spr::eECALmatrix(theNavigator,endcapRecHitsHandle,2,2);
	      e7x7         = spr::eECALmatrix(theNavigator,endcapRecHitsHandle,3,3);
	      e9x9         = spr::eECALmatrix(theNavigator,endcapRecHitsHandle,4,4);
	      e11x11       = spr::eECALmatrix(theNavigator,endcapRecHitsHandle,5,5);
	      e13x13       = spr::eECALmatrix(theNavigator,endcapRecHitsHandle,6,6);
	      e15x15       = spr::eECALmatrix(theNavigator,endcapRecHitsHandle,7,7);
	      e25x25       = spr::eECALmatrix(theNavigator,endcapRecHitsHandle,12,12);
	      
	      // check the energy from SimHits
	      simInfo3x3   = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloee, SimTk, SimVtx, pTrack, *associate, 1,1);
	      simInfo5x5   = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloee, SimTk, SimVtx, pTrack, *associate, 2,2);
	      simInfo7x7   = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloee, SimTk, SimVtx, pTrack, *associate, 3,3);
	      simInfo9x9   = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloee, SimTk, SimVtx, pTrack, *associate, 4,4);
	      simInfo11x11 = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloee, SimTk, SimVtx, pTrack, *associate, 5,5);
	      simInfo13x13 = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloee, SimTk, SimVtx, pTrack, *associate, 6,6);
	      simInfo15x15 = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloee, SimTk, SimVtx, pTrack, *associate, 7,7);
              simInfo25x25 = spr::eECALSimInfo(iEvent, theNavigator,geo,pcaloee, SimTk, SimVtx, pTrack, *associate, 12,12);
	      
	      trkEcalEne   = spr::eCaloSimInfo(iEvent, geo, pcaloeb, SimTk, SimVtx, pTrack, *associate);

	      esim3x3      = simInfo3x3["eTotal"];
	      esim5x5      = simInfo5x5["eTotal"];
	      esim7x7      = simInfo7x7["eTotal"];
	      esim9x9      = simInfo9x9["eTotal"];
	      esim11x11    = simInfo11x11["eTotal"];
	      esim13x13    = simInfo13x13["eTotal"];
	      esim15x15    = simInfo15x15["eTotal"];
	      esim25x25    = simInfo25x25["eTotal"];
	      
	      // debug it
	      // ------------------------------------------------------------------------------------------------
	      if(myverbose_>10 && simInfo15x15["eRest"]>0 ) {
		
		std::cout << "esim15x15 "    << esim15x15 << "  eMatched " << simInfo15x15["eMatched"] 
			  << " pdgMatched "  << simInfo15x15["pdgMatched"] <<" eGamma " << simInfo15x15["eGamma"] 
			  << " eNeutralHad " << simInfo15x15["eNeutralHad"] << " eChargedHad " 
			  << simInfo15x15["eChargedHad"]
			  << " eRest " << simInfo15x15["eRest"] << " eTotal " << simInfo15x15["eTotal"] 
			  << "\n" << std::endl;
		
		edm::SimTrackContainer::const_iterator tkInfo = spr::matchedSimTrack(iEvent, SimTk, SimVtx, pTrack, *associate, true);
		std::vector< edm::PCaloHitContainer::const_iterator> hittemp = spr::hitECALmatrix(theNavigator,pcaloee,7,7);
		std::cout << "trkMom " << pTrack->p() << " eta " << pTrack->eta() << " phi " << pTrack->phi() << std::endl;

		std::map<std::string, double> debug15x15 = spr::eCaloSimInfo(geo, pcaloeb, SimTk, SimVtx, hittemp, tkInfo, 150, true);
		trkEcalEne   = spr::eCaloSimInfo(iEvent, geo, pcaloeb, SimTk, SimVtx, pTrack, *associate, 150, true);
		
		if (!printdone) {
		  printdone = true;
		  std::cout << "Printing sim Tracks " << std::endl;
		  std::cout << "No. of SimVertices " << SimVtx->size() << " SimTk Size " << SimTk->size() << std::endl;
		  int ii=0;
		  for(edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr!= SimTk->end(); simTrkItr++, ii++){
		    std::cout <<"SimTrack " << ii <<" TrackId " <<  simTrkItr->trackId() << " GenPartIdx " << simTrkItr->genpartIndex() 
			      << " VertexIdx " << simTrkItr->vertIndex() << " Type " << simTrkItr->type() << std::endl;
		  }
		  ii=0;
		  for(edm::SimVertexContainer::const_iterator simVtxItr = SimVtx->begin(); simVtxItr!= SimVtx->end(); simVtxItr++, ii++){
		    std::cout << "SimVertex " << ii << " VtxParentIdx " << simVtxItr->parentIndex() 
			      << "  position " << simVtxItr->position() << std::endl;
		  }
		}
	      } // just some printing
	      // ------------------------------------------------------------------------------------------------
	    } // if EB or EE
	    
	    bool ecalIsolation = true;
	    if( applyEcalIsolation_ && (std::abs(e15x15-e11x11)>1.0 || std::abs(e25x25-e15x15)>2.0) ) 
	      ecalIsolation = false;
	    
	    double h3x3       = 0, h5x5    = 0, h7x7    = 0;
	    double hsim3x3    = 0, hsim5x5 = 0, hsim7x7 = 0;
	    double trkHcalEne = 0;
	  
	    std::map<std::string, double> hsimInfo3x3, hsimInfo5x5, hsimInfo7x7;
	    
	    const DetId ClosestCell = gHB->getClosestCell(point1);
	    // std::cout << eta1 << " " << (HcalDetId) ClosestCell << " ";

	    /*
	    if( std::abs(pTrack->eta())<1.4 ) {
	    const DetId ClosestCell = gHB->getClosestCell(point1);
	    std::cout << " Using SubDet Geometry " << (HcalDetId) ClosestCell << std::endl;
	    } else {
	      const DetId ClosestCell = gHE->getClosestCell(point1);
	      std::cout << " Using SubDet Geometry " << (HcalDetId) ClosestCell << std::endl;
	    }
	    */

	    double hcalScale=1.0;
	    if( std::abs(pTrack->eta())<1.4 ) {
	      hcalScale=120.0;
	    } else {
	      hcalScale=135.0;
	    }
	    
	    // bool includeHO=false, bool algoNew=true, bool debug=false
	    h3x3 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,1,1, false, true, false);  
	    h5x5 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,2,2, false, true, false);  
	    h7x7 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,3,3, false, true, false);  
	    
	    hsimInfo3x3  = spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell, geo,pcalohh, SimTk, SimVtx, pTrack, *associate, 1,1);
	    hsimInfo5x5  = spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell, geo,pcalohh, SimTk, SimVtx, pTrack, *associate, 2,2);
	    hsimInfo7x7  = spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell, geo,pcalohh, SimTk, SimVtx, pTrack, *associate, 3,3);
	    
	    trkHcalEne   = spr::eCaloSimInfo(iEvent, geo,pcalohh, SimTk, SimVtx, pTrack, *associate);

	    hsim3x3    = hsimInfo3x3["eTotal"];
	    hsim5x5    = hsimInfo5x5["eTotal"];
	    hsim7x7    = hsimInfo7x7["eTotal"];

	    if(myverbose_>10){
	      if(esim15x15<100.001  &&  std::abs(pTrack->eta())<1.1 ) {
		std::cout<<" nTracks " << nTracks << "  esim15x15 " << esim15x15 
			 <<" hsim5x5 " << hsim5x5 << " eta " << pTrack->eta()
			 <<" stripTOBLayersWithMeasurement " << hitp.stripTOBLayersWithMeasurement()
			 << std::endl;
		//printTrack(pTrack);
	      }
	    }
	    

	    GlobalPoint hpoint1(info1.trkGlobPosAtHcal.x(),info1.trkGlobPosAtHcal.y(),info1.trkGlobPosAtHcal.z());
	    double x=info1.trkGlobPosAtHcal.x(); 
	    double y=info1.trkGlobPosAtHcal.y();
	    double z=info1.trkGlobPosAtHcal.z();
	    
	    
	    if( x==0 && y==0 && z==0) {
	      //std::cout << "Hcal point1 (before) " << hpoint1 << std::endl;
	      hpoint1 = point1;
	      //std::cout << "Hcal point1 (after)  " << hpoint1 << std::endl;
	    }
	    
	    const DetId ClosestCell_1 = gHB->getClosestCell(hpoint1);
	    //std::cout <<"Closest cell Hcal (atHCAL) " << (HcalDetId)ClosestCell_1 << std::endl;
	    /*
	    if( x==0 && y==0 && z==0) {
	      HcalDetId hcd   = (HcalDetId)ClosestCell;
	      HcalDetId hcd_1 = (HcalDetId)ClosestCell_1;
	      
	      std::cout << "idx p,eta,phi of track " << nTracks << " " << p1 << " " << eta1 << " " << phi1 << " " << pt1 << std::endl;
	      std::cout <<" stripTOB, TEC " << hitp.stripTOBLayersWithMeasurement()<<" " << hitp.stripTECLayersWithMeasurement()
		      << "  infoGoodness " <<  info1.isGoodEcal << " " << info1.isGoodHcal << std::endl;
	      std::cout << "ecalImpactPoint " << hcd.ieta() << " " << hcd.iphi() 
			<< "  hcalImpactPoint " <<  hcd_1.ieta() << " " << hcd_1.iphi() << "\n"
			<< std::endl;
	    }
	    */
	    //HBHERecHitCollection::const_iterator hbheItr_1 = Hithbhe.find(ClosestCell_1);
	    //	  CaloNavigator<DetId> theNavigator_1(ClosestCell_1,theHBHETopology);
	    
	    double h3x3_1=0,    h5x5_1=0,    h7x7_1=0;
	    double hsim3x3_1=0, hsim5x5_1=0, hsim7x7_1=0;
	    double trkHcalEne_1; 
	    double maxNearHcalP3x3_1=999, maxNearHcalP5x5_1=999, maxNearHcalP7x7_1=999;

	    std::map<std::string, double> hsimInfo3x3_1, hsimInfo5x5_1, hsimInfo7x7_1;
	    

	    
	    // bool includeHO=false, bool algoNew=true, bool debug=false
	    h3x3_1 = spr::eHCALmatrix(theHBHETopology, ClosestCell_1, hbhe,1,1, false, true, false);  
	    h5x5_1 = spr::eHCALmatrix(theHBHETopology, ClosestCell_1, hbhe,2,2, false, true, false);  
	    h7x7_1 = spr::eHCALmatrix(theHBHETopology, ClosestCell_1, hbhe,3,3, false, true, false);  
	    
	    hsimInfo3x3_1 = spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell_1, geo,pcalohh, SimTk, SimVtx, pTrack, *associate, 1,1);
	    hsimInfo5x5_1 = spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell_1, geo,pcalohh, SimTk, SimVtx, pTrack, *associate, 2,2);
	    hsimInfo7x7_1 = spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell_1, geo,pcalohh, SimTk, SimVtx, pTrack, *associate, 3,3);

	    trkHcalEne_1  = spr::eCaloSimInfo(iEvent, geo,pcalohh, SimTk, SimVtx, pTrack, *associate);
	    
	    hsim3x3_1     = hsimInfo3x3_1["eTotal"];
	    hsim5x5_1     = hsimInfo5x5_1["eTotal"];
	    hsim7x7_1     = hsimInfo7x7_1["eTotal"];
	    
	    maxNearHcalP3x3_1  = chargeIsolationHcal(iEvent, iSetup, trkItr, trkCollection, ClosestCell_1, theHBHETopology, gHB,1,1);
	    maxNearHcalP5x5_1  = chargeIsolationHcal(iEvent, iSetup, trkItr, trkCollection, ClosestCell_1, theHBHETopology, gHB,2,2);
	    maxNearHcalP7x7_1  = chargeIsolationHcal(iEvent, iSetup, trkItr, trkCollection, ClosestCell_1, theHBHETopology, gHB,3,3);

	    /*
	    std::cout <<"~~~~~~~~~~~~~~~~ Old & New Algos ~~~~~~~~~~~~~~" << std::endl;
	    double oldh5x5 = spr::eHCALmatrix(theHBHETopology, ClosestCell_1, hbhe,2,2, false, false, true);  
	    std::cout << "-------------------" << std::endl;
	    double newh5x5 = spr::eHCALmatrix(theHBHETopology, ClosestCell_1, hbhe,2,2, false, true,  true);  
	    std::cout <<"~~~~~~~~~~~~~~~~ Old & New Algos ~~~~~~~~~~~~~~ \n" << std::endl;
	    */

	    if(t_nTracksIsoBy5GeV<200 ) {
	      
	      t_infoHcal[t_nTracksIsoBy5GeV]         = info1.isGoodHcal;
	      
	      t_maxNearP[t_nTracksIsoBy5GeV]         = maxNearP;
	      t_maxNearP15x15[t_nTracksIsoBy5GeV]    = maxNearP15x15;
	      t_maxNearP21x21[t_nTracksIsoBy5GeV]    = maxNearP21x21;
	      t_maxNearP31x31[t_nTracksIsoBy5GeV]    = maxNearP31x31;

	      t_trackP[t_nTracksIsoBy5GeV]           = p1;
	      t_trackPt[t_nTracksIsoBy5GeV]          = pt1;
	      t_trackEta[t_nTracksIsoBy5GeV]         = eta1;
	      t_trackPhi[t_nTracksIsoBy5GeV]         = phi1;
	      t_trackNOuterHits[t_nTracksIsoBy5GeV]  = nOuterHits;
	      t_NLayersCrossed[t_nTracksIsoBy5GeV]   = nLayersCrossed;

	      t_e3x3[t_nTracksIsoBy5GeV]             = e3x3;
	      t_e5x5[t_nTracksIsoBy5GeV]             = e5x5;
	      t_e7x7[t_nTracksIsoBy5GeV]             = e7x7;
	      t_e9x9[t_nTracksIsoBy5GeV]             = e9x9;
	      t_e11x11[t_nTracksIsoBy5GeV]           = e11x11;
	      t_e13x13[t_nTracksIsoBy5GeV]           = e13x13;
	      t_e15x15[t_nTracksIsoBy5GeV]           = e15x15;
	      t_e25x25[t_nTracksIsoBy5GeV]           = e25x25;
	      
	      t_simTrackP[t_nTracksIsoBy5GeV]        = simTrackP;
	      
	      t_trkEcalEne[t_nTracksIsoBy5GeV]       = trkEcalEne;
	      
	      t_esim3x3[t_nTracksIsoBy5GeV]          = esim3x3;
	      t_esim5x5[t_nTracksIsoBy5GeV]          = esim5x5;
	      t_esim7x7[t_nTracksIsoBy5GeV]          = esim7x7;
	      t_esim9x9[t_nTracksIsoBy5GeV]          = esim9x9;
	      t_esim11x11[t_nTracksIsoBy5GeV]        = esim11x11;
	      t_esim13x13[t_nTracksIsoBy5GeV]        = esim13x13;
	      t_esim15x15[t_nTracksIsoBy5GeV]        = esim15x15;
	      t_esim25x25[t_nTracksIsoBy5GeV]        = esim25x25;
	      
	      t_esim3x3PdgId[t_nTracksIsoBy5GeV]     = simInfo3x3["pdgMatched"];
	      t_esim5x5PdgId[t_nTracksIsoBy5GeV]     = simInfo5x5["pdgMatched"];
	      t_esim7x7PdgId[t_nTracksIsoBy5GeV]     = simInfo7x7["pdgMatched"];
	      t_esim9x9PdgId[t_nTracksIsoBy5GeV]     = simInfo9x9["pdgMatched"];
	      t_esim11x11PdgId[t_nTracksIsoBy5GeV]   = simInfo11x11["pdgMatched"];
	      t_esim13x13PdgId[t_nTracksIsoBy5GeV]   = simInfo13x13["pdgMatched"];
	      t_esim15x15PdgId[t_nTracksIsoBy5GeV]   = simInfo15x15["pdgMatched"];

	      t_esim3x3Matched[t_nTracksIsoBy5GeV]   = simInfo3x3["eMatched"];
	      t_esim5x5Matched[t_nTracksIsoBy5GeV]   = simInfo5x5["eMatched"];
	      t_esim7x7Matched[t_nTracksIsoBy5GeV]   = simInfo7x7["eMatched"];
	      t_esim9x9Matched[t_nTracksIsoBy5GeV]   = simInfo9x9["eMatched"];
	      t_esim11x11Matched[t_nTracksIsoBy5GeV] = simInfo11x11["eMatched"];
	      t_esim13x13Matched[t_nTracksIsoBy5GeV] = simInfo13x13["eMatched"];
	      t_esim15x15Matched[t_nTracksIsoBy5GeV] = simInfo15x15["eMatched"];
	      
	      t_esim3x3Rest[t_nTracksIsoBy5GeV]      = simInfo3x3["eRest"];
	      t_esim5x5Rest[t_nTracksIsoBy5GeV]      = simInfo5x5["eRest"];
	      t_esim7x7Rest[t_nTracksIsoBy5GeV]      = simInfo7x7["eRest"];
	      t_esim9x9Rest[t_nTracksIsoBy5GeV]      = simInfo9x9["eRest"];
	      t_esim11x11Rest[t_nTracksIsoBy5GeV]    = simInfo11x11["eRest"];
	      t_esim13x13Rest[t_nTracksIsoBy5GeV]    = simInfo13x13["eRest"];
	      t_esim15x15Rest[t_nTracksIsoBy5GeV]    = simInfo15x15["eRest"];
	    
	      t_esim3x3Photon[t_nTracksIsoBy5GeV]    = simInfo3x3["eGamma"];
	      t_esim5x5Photon[t_nTracksIsoBy5GeV]    = simInfo5x5["eGamma"];
	      t_esim7x7Photon[t_nTracksIsoBy5GeV]    = simInfo7x7["eGamma"];
	      t_esim9x9Photon[t_nTracksIsoBy5GeV]    = simInfo9x9["eGamma"];
	      t_esim11x11Photon[t_nTracksIsoBy5GeV]  = simInfo11x11["eGamma"];
	      t_esim13x13Photon[t_nTracksIsoBy5GeV]  = simInfo13x13["eGamma"];
	      t_esim15x15Photon[t_nTracksIsoBy5GeV]  = simInfo15x15["eGamma"];
	      
	      t_esim3x3NeutHad[t_nTracksIsoBy5GeV]   = simInfo3x3["eNeutralHad"];
	      t_esim5x5NeutHad[t_nTracksIsoBy5GeV]   = simInfo5x5["eNeutralHad"];
	      t_esim7x7NeutHad[t_nTracksIsoBy5GeV]   = simInfo7x7["eNeutralHad"];
	      t_esim9x9NeutHad[t_nTracksIsoBy5GeV]   = simInfo9x9["eNeutralHad"];
	      t_esim11x11NeutHad[t_nTracksIsoBy5GeV] = simInfo11x11["eNeutralHad"];
	      t_esim13x13NeutHad[t_nTracksIsoBy5GeV] = simInfo13x13["eNeutralHad"];
	      t_esim15x15NeutHad[t_nTracksIsoBy5GeV] = simInfo15x15["eNeutralHad"];
	      
	      t_esim3x3CharHad[t_nTracksIsoBy5GeV]   = simInfo3x3["eChargedHad"];
	      t_esim5x5CharHad[t_nTracksIsoBy5GeV]   = simInfo5x5["eChargedHad"];
	      t_esim7x7CharHad[t_nTracksIsoBy5GeV]   = simInfo7x7["eChargedHad"];
	      t_esim9x9CharHad[t_nTracksIsoBy5GeV]   = simInfo9x9["eChargedHad"];
	      t_esim11x11CharHad[t_nTracksIsoBy5GeV] = simInfo11x11["eChargedHad"];
	      t_esim13x13CharHad[t_nTracksIsoBy5GeV] = simInfo13x13["eChargedHad"];
	      t_esim15x15CharHad[t_nTracksIsoBy5GeV] = simInfo15x15["eChargedHad"];
	      
	      
	      t_trkHcalEne[t_nTracksIsoBy5GeV]       = hcalScale*trkHcalEne;
	      
	      t_h3x3[t_nTracksIsoBy5GeV]             = h3x3;
	      t_h5x5[t_nTracksIsoBy5GeV]             = h5x5;
	      t_h7x7[t_nTracksIsoBy5GeV]             = h7x7;

	      t_maxNearHcalP3x3_1[t_nTracksIsoBy5GeV]= maxNearHcalP3x3_1;
	      t_maxNearHcalP5x5_1[t_nTracksIsoBy5GeV]= maxNearHcalP5x5_1;
	      t_maxNearHcalP7x7_1[t_nTracksIsoBy5GeV]= maxNearHcalP7x7_1;
		
	      t_hsim3x3[t_nTracksIsoBy5GeV]          = hcalScale*hsim3x3;
	      t_hsim5x5[t_nTracksIsoBy5GeV]          = hcalScale*hsim5x5;
	      t_hsim7x7[t_nTracksIsoBy5GeV]          = hcalScale*hsim7x7;
	      
	      t_hsim3x3Matched[t_nTracksIsoBy5GeV]   = hcalScale*hsimInfo3x3["eMatched"];
	      t_hsim5x5Matched[t_nTracksIsoBy5GeV]   = hcalScale*hsimInfo5x5["eMatched"];
	      t_hsim7x7Matched[t_nTracksIsoBy5GeV]   = hcalScale*hsimInfo7x7["eMatched"];
	      
	      t_hsim3x3Rest[t_nTracksIsoBy5GeV]      = hcalScale*hsimInfo3x3["eRest"];
  	      t_hsim5x5Rest[t_nTracksIsoBy5GeV]      = hcalScale*hsimInfo5x5["eRest"];
  	      t_hsim7x7Rest[t_nTracksIsoBy5GeV]      = hcalScale*hsimInfo7x7["eRest"];
 	      
	      t_hsim3x3Photon[t_nTracksIsoBy5GeV]    = hcalScale*hsimInfo3x3["eGamma"];
	      t_hsim5x5Photon[t_nTracksIsoBy5GeV]    = hcalScale*hsimInfo5x5["eGamma"];
	      t_hsim7x7Photon[t_nTracksIsoBy5GeV]    = hcalScale*hsimInfo7x7["eGamma"];
	      
	      t_hsim3x3NeutHad[t_nTracksIsoBy5GeV]   = hcalScale*hsimInfo3x3["eNeutralHad"];
	      t_hsim5x5NeutHad[t_nTracksIsoBy5GeV]   = hcalScale*hsimInfo5x5["eNeutralHad"];
	      t_hsim7x7NeutHad[t_nTracksIsoBy5GeV]   = hcalScale*hsimInfo7x7["eNeutralHad"];
	      
	      t_hsim3x3CharHad[t_nTracksIsoBy5GeV]   = hcalScale*hsimInfo3x3["eChargedHad"];
	      t_hsim5x5CharHad[t_nTracksIsoBy5GeV]   = hcalScale*hsimInfo5x5["eChargedHad"];
	      t_hsim7x7CharHad[t_nTracksIsoBy5GeV]   = hcalScale*hsimInfo7x7["eChargedHad"];
	      
	      t_trkHcalEne_1[t_nTracksIsoBy5GeV]     = hcalScale*trkHcalEne_1;
	      
	      t_h3x3_1[t_nTracksIsoBy5GeV]           = h3x3_1;
	      t_h5x5_1[t_nTracksIsoBy5GeV]           = h5x5_1;
	      t_h7x7_1[t_nTracksIsoBy5GeV]           = h7x7_1;
	      
	      t_hsim3x3_1[t_nTracksIsoBy5GeV]        = hcalScale*hsim3x3_1;
	      t_hsim5x5_1[t_nTracksIsoBy5GeV]        = hcalScale*hsim5x5_1;
	      t_hsim7x7_1[t_nTracksIsoBy5GeV]        = hcalScale*hsim7x7_1;
	      
	      t_hsim3x3Matched_1[t_nTracksIsoBy5GeV] = hcalScale*hsimInfo3x3_1["eMatched"];
	      t_hsim5x5Matched_1[t_nTracksIsoBy5GeV] = hcalScale*hsimInfo5x5_1["eMatched"];
	      t_hsim7x7Matched_1[t_nTracksIsoBy5GeV] = hcalScale*hsimInfo7x7_1["eMatched"];
	      
	      t_hsim3x3Rest_1[t_nTracksIsoBy5GeV]    = hcalScale*hsimInfo3x3_1["eRest"];
	      t_hsim5x5Rest_1[t_nTracksIsoBy5GeV]    = hcalScale*hsimInfo5x5_1["eRest"];
	      t_hsim7x7Rest_1[t_nTracksIsoBy5GeV]    = hcalScale*hsimInfo7x7_1["eRest"];
 	      
	      t_hsim3x3Photon_1[t_nTracksIsoBy5GeV]  = hcalScale*hsimInfo3x3_1["eGamma"];
	      t_hsim5x5Photon_1[t_nTracksIsoBy5GeV]  = hcalScale*hsimInfo5x5_1["eGamma"];
	      t_hsim7x7Photon_1[t_nTracksIsoBy5GeV]  = hcalScale*hsimInfo7x7_1["eGamma"];
	      
	      t_hsim3x3NeutHad_1[t_nTracksIsoBy5GeV] = hcalScale*hsimInfo3x3_1["eNeutralHad"];
	      t_hsim5x5NeutHad_1[t_nTracksIsoBy5GeV] = hcalScale*hsimInfo5x5_1["eNeutralHad"];
	      t_hsim7x7NeutHad_1[t_nTracksIsoBy5GeV] = hcalScale*hsimInfo7x7_1["eNeutralHad"];
	      
	      t_hsim3x3CharHad_1[t_nTracksIsoBy5GeV] = hcalScale*hsimInfo3x3_1["eChargedHad"];
	      t_hsim5x5CharHad_1[t_nTracksIsoBy5GeV] = hcalScale*hsimInfo5x5_1["eChargedHad"];
	      t_hsim7x7CharHad_1[t_nTracksIsoBy5GeV] = hcalScale*hsimInfo7x7_1["eChargedHad"];
	    }
	    
	    t_nTracksIsoBy5GeV++;
	  } // if loosely isolated track
	}
      }
    } // loop over track collection
    
    delete associate;
  } // if L1Pass

  t_nEvtProc = nEventProc;

  tree->Fill();

}

// ----- method called once each job just before starting event loop ----
void IsolatedTracksTree::beginJob() {

  t_nEvtProc=0;
  nEventProc=0;

  //  double tempgen_TH[21] = { 1.0,  2.0,  3.0,  4.0,  5.0, 
  double tempgen_TH[22] = { 0.0,  1.0,  2.0,  3.0,  4.0,  
			    5.0,  6.0,  7.0,  8.0,  9.0, 
			    10.0, 12.0, 15.0, 20.0, 25.0, 
			    30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100};

  for(int i=0; i<22; i++)  genPartPBins[i]  = tempgen_TH[i];

  double tempgen_Eta[5] = {0.0, 0.5, 1.1, 1.7, 2.0};

  for(int i=0; i<5; i++) genPartEtaBins[i] = tempgen_Eta[i];

  BookHistograms();
}

// ----- method called once each job just after ending the event loop ----
void IsolatedTracksTree::endJob() {

  std::cout << "Number of Events Processed " << nEventProc << std::endl;
  
}

double IsolatedTracksTree::chargeIsolation(const edm::Event& iEvent, const edm::EventSetup& iSetup, CaloNavigator<DetId>& theNavigator, reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle, edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle, const CaloSubdetectorGeometry* gEB, const CaloSubdetectorGeometry* gEE, int ieta, int iphi) {
  
  double maxNearP = -1.0;
  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);

  //,const DetId anyCell,
  reco::TrackCollection::const_iterator trkItr2;
  for( trkItr2 = trkCollection->begin(); trkItr2 != trkCollection->end(); ++trkItr2){

    const reco::Track* pTrack2 = &(*trkItr2);

    bool   trkQuality  = pTrack2->quality(trackQuality_);
    //const reco::HitPattern& hitp = pTrack2->hitPattern();
    //int nLayersCrossed = hitp.trackerLayersWithMeasurement() ;        

    //if( (trkItr2 != trkItr) && trkQuality && nLayersCrossed>7 )  {
    if( (trkItr2 != trkItr) && trkQuality )  {
      
      const FreeTrajectoryState fts2 = trackAssociator_->getFreeTrajectoryState(iSetup, *pTrack2);
      TrackDetMatchInfo info2 = trackAssociator_->associate(iEvent, iSetup, fts2, parameters_);
      const GlobalPoint point2(info2.trkGlobPosAtEcal.x(),info2.trkGlobPosAtEcal.y(),info2.trkGlobPosAtEcal.z());

      if( info2.isGoodEcal ) {
	if( std::abs(point2.eta())<1.479) {
	  const DetId anyCell = gEB->getClosestCell(point2);
	  int isChargedIso = chargeIsolation(theNavigator,anyCell,ieta, iphi);
	  if(isChargedIso==0) {
	    if(maxNearP<pTrack2->p()) maxNearP=pTrack2->p();
	  }
	} else {
	  const DetId anyCell = gEE->getClosestCell(point2);
	  int isChargedIso = chargeIsolation(theNavigator,anyCell,ieta, iphi);
	  if(isChargedIso==0) {
	    if(maxNearP<pTrack2->p()) maxNearP=pTrack2->p();
	  }
	}
      } //info2.isGoodEcal
    }
  }
  return maxNearP;
}

int IsolatedTracksTree::chargeIsolation(CaloNavigator<DetId>& navigator,const DetId anyCell, int ieta, int iphi){

  int isIsolated = 1;

  DetId thisDet;

  for (int dx = -ieta; dx < ieta+1; ++dx) {
    for (int dy = -iphi; dy < iphi+1; ++dy) {

      thisDet = navigator.offsetBy(dx, dy);
      navigator.home();
      
      if (thisDet != DetId(0)) {
	if (thisDet == anyCell) {
	  isIsolated = 0;
	  return isIsolated;
	}
      }
    }
  }
  return isIsolated;
}

double IsolatedTracksTree::chargeIsolationHcal(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
					       reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection,
					       const DetId ClosestCell, const HcalTopology* topology, const CaloSubdetectorGeometry* gHB, 
					       int ieta, int iphi, bool debug) {

  std::vector<DetId> dets(1,ClosestCell);

   if(debug) std::cout << (HcalDetId) ClosestCell << std::endl;

  std::vector<DetId> vdets = spr::matrixHCALIds(dets, topology, ieta, iphi); //, debug);
  std::vector<DetId>::iterator it;  
  
  if(debug) {
    for(unsigned int i=0; i<vdets.size(); i++) {
      std::cout << "HcalDetId in " <<2*ieta+1 << "x" << 2*iphi+1 << " " << (HcalDetId) vdets[i] << std::endl;
    }
  }

  double maxNearP = -1.0;
  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);
  
  reco::TrackCollection::const_iterator trkItr2;
  for( trkItr2 = trkCollection->begin(); trkItr2 != trkCollection->end(); ++trkItr2){
    
    const reco::Track* pTrack2 = &(*trkItr2);
    
    bool   trkQuality  = pTrack2->quality(trackQuality_);
    //const reco::HitPattern& hitp = pTrack2->hitPattern();
    //int nLayersCrossed = hitp.trackerLayersWithMeasurement() ;        
    
    //if( (trkItr2 != trkItr) && trkQuality && nLayersCrossed>7 )  {
    if( (trkItr2 != trkItr) && trkQuality )  {
      const FreeTrajectoryState fts2 = trackAssociator_->getFreeTrajectoryState(iSetup, *pTrack2);
      TrackDetMatchInfo info2 = trackAssociator_->associate(iEvent, iSetup, fts2, parameters_);
      const GlobalPoint point2(info2.trkGlobPosAtHcal.x(),info2.trkGlobPosAtHcal.y(),info2.trkGlobPosAtHcal.z());

      if(debug){
	std::cout << "Track2 (p,eta,phi) " << pTrack2->p() << " " << pTrack2->eta() << " " << pTrack2->phi() << std::endl;
      }

      if( info2.isGoodHcal ) {
	const DetId anyCell = gHB->getClosestCell(point2);
	//it = vdets.find(vdets.begin(), vdets.end(),anyCell);
	for(unsigned int i=0; i<vdets.size(); i++) {
	  if(anyCell == vdets[i]) {
	    if(maxNearP<pTrack2->p())  maxNearP=pTrack2->p();
	    break;
	  }
	}
	if(debug){
	  std::cout << "maxNearP " << maxNearP << " thisCell " << (HcalDetId)anyCell 
		    << " (" << info2.trkGlobPosAtHcal.x()<<","<< info2.trkGlobPosAtHcal.y()<<","<< info2.trkGlobPosAtHcal.z()<<")"
		    << std::endl;
	}

      }
    }
  }  
  
  return maxNearP;
}
  


//---
void IsolatedTracksTree::BookHistograms(){

  char hname[100], htit[100];

  // Reconstructed Tracks

  h_nTracks = fs->make<TH1F>("h_nTracks", "h_nTracks", 1000, -0.5, 999.5);

  sprintf(hname, "h_recEtaPt_0");
  sprintf(htit,  "h_recEtaPt (all tracks Eta vs pT)");
  h_recEtaPt_0 = fs->make<TH2F>(hname, htit, 30, -3.0,3.0, 20, genPartPBins);

  sprintf(hname, "h_recEtaP_0");
  sprintf(htit,  "h_recEtaP (all tracks Eta vs pT)");
  h_recEtaP_0 = fs->make<TH2F>(hname, htit, 30, -3.0,3.0, 20, genPartPBins);

  h_recPt_0  = fs->make<TH1F>("h_recPt_0",  "Pt (all tracks)",  20, genPartPBins);
  h_recP_0   = fs->make<TH1F>("h_recP_0",   "P  (all tracks)",  20, genPartPBins);
  h_recEta_0 = fs->make<TH1F>("h_recEta_0", "Eta (all tracks)", 60, -3.0,   3.0);
  h_recPhi_0 = fs->make<TH1F>("h_recPhi_0", "Phi (all tracks)", 100, -3.2,   3.2);
  //-------------------------
  sprintf(hname, "h_recEtaPt_1");
  sprintf(htit,  "h_recEtaPt (all good tracks Eta vs pT)");
  h_recEtaPt_1 = fs->make<TH2F>(hname, htit, 30, -3.0,3.0, 20, genPartPBins);

  sprintf(hname, "h_recEtaP_1");
  sprintf(htit,  "h_recEtaP (all good tracks Eta vs pT)");
  h_recEtaP_1 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 20, genPartPBins);

  h_recPt_1  = fs->make<TH1F>("h_recPt_1",  "Pt (all good tracks)",  20, genPartPBins);
  h_recP_1   = fs->make<TH1F>("h_recP_1",   "P  (all good tracks)",  20, genPartPBins);
  h_recEta_1 = fs->make<TH1F>("h_recEta_1", "Eta (all good tracks)", 60, -3.0,   3.0);
  h_recPhi_1 = fs->make<TH1F>("h_recPhi_1", "Phi (all good tracks)", 100, -3.2,   3.2);
  //-------------------------
  sprintf(hname, "h_recEtaPt_2");
  sprintf(htit,  "h_recEtaPt (charge isolation Eta vs pT)");
  h_recEtaPt_2 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 20, genPartPBins);

  sprintf(hname, "h_recEtaP_2");
  sprintf(htit,  "h_recEtaP (charge isolation Eta vs pT)");
  h_recEtaP_2 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 20, genPartPBins);
  
  h_recPt_2  = fs->make<TH1F>("h_recPt_2",  "Pt (charge isolation)",  20, genPartPBins);
  h_recP_2   = fs->make<TH1F>("h_recP_2",   "P  (charge isolation)",  20, genPartPBins);
  h_recEta_2 = fs->make<TH1F>("h_recEta_2", "Eta (charge isolation)", 60, -3.0,   3.0);
  h_recPhi_2 = fs->make<TH1F>("h_recPhi_2", "Phi (charge isolation)", 100, -3.2,   3.2);


  tree = fs->make<TTree>("tree", "tree");
  tree->SetAutoSave(10000);
 
  tree->Branch("t_nEvtProc",         &t_nEvtProc,         "t_nEvtProc/I");

  tree->Branch("t_nTracks",          &t_nTracks,          "t_nTracks/I");
  tree->Branch("t_nTracksAwayL1",    &t_nTracksAwayL1,    "t_nTracksAwayL1/I");
  
  tree->Branch("t_nTracksIsoBy5GeV", &t_nTracksIsoBy5GeV, "t_nTracksIsoBy5GeV/I");

  tree->Branch("t_infoHcal",          t_infoHcal,         "t_infoHcal[t_nTracksIsoBy5GeV]/I");

  tree->Branch("t_trackP",            t_trackP,           "t_trackP[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_trackPt",           t_trackPt,          "t_trackPt[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_trackEta",          t_trackEta,         "t_trackEta[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_trackPhi",          t_trackPhi,         "t_trackPhi[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_trackNOuterHits",   t_trackNOuterHits,  "t_trackNOuterHits[t_nTracksIsoBy5GeV]/I");
  tree->Branch("t_NLayersCrossed",    t_NLayersCrossed,   "t_NLayersCrossed[t_nTracksIsoBy5GeV]/I");

  tree->Branch("t_maxNearP",          t_maxNearP,         "t_maxNearP[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_maxNearP15x15",     t_maxNearP15x15,    "t_maxNearP15x15[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_maxNearP21x21",     t_maxNearP21x21,    "t_maxNearP21x21[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_maxNearP31x31",     t_maxNearP31x31,    "t_maxNearP31x31[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_e3x3",              t_e3x3,             "t_e3x3[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_e5x5",              t_e5x5,             "t_e5x5[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_e7x7",              t_e7x7,             "t_e7x7[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_e9x9",              t_e9x9,             "t_e9x9[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_e11x11",            t_e11x11,           "t_e11x11[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_e13x13",            t_e13x13,           "t_e13x13[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_e15x15",            t_e15x15,           "t_e15x15[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_e25x25",            t_e25x25,           "t_e25x25[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_simTrackP",         t_simTrackP,        "t_simTrackP[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_trkEcalEne",        t_trkEcalEne,       "t_trkEcalEne[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim3x3",           t_esim3x3,          "t_esim3x3[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim5x5",           t_esim5x5,          "t_esim5x5[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim7x7",           t_esim7x7,          "t_esim7x7[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim9x9",           t_esim9x9,          "t_esim9x9[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim11x11",         t_esim11x11,        "t_esim11x11[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim13x13",         t_esim13x13,        "t_esim13x13[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim15x15",         t_esim15x15,        "t_esim15x15[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim25x25",         t_esim25x25,        "t_esim25x25[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_esim3x3PdgId",      t_esim3x3PdgId,     "t_esim3x3PdgId[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim5x5PdgId",      t_esim5x5PdgId,     "t_esim5x5PdgId[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim7x7PdgId",      t_esim7x7PdgId,     "t_esim7x7PdgId[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim9x9PdgId",      t_esim9x9PdgId,     "t_esim9x9PdgId[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim11x11PdgId",    t_esim11x11PdgId,   "t_esim11x11PdgId[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim13x13PdgId",    t_esim13x13PdgId,   "t_esim13x13PdgId[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim15x15PdgId",    t_esim15x15PdgId,   "t_esim15x15PdgId[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_esim3x3Matched",    t_esim3x3Matched,   "t_esim3x3Matched[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim5x5Matched",    t_esim5x5Matched,   "t_esim5x5Matched[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim7x7Matched",    t_esim7x7Matched,   "t_esim7x7Matched[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim9x9Matched",    t_esim9x9Matched,   "t_esim9x9Matched[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim11x11Matched",  t_esim11x11Matched, "t_esim11x11Matched[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim13x13Matched",  t_esim13x13Matched, "t_esim13x13Matched[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim15x15Matched",  t_esim15x15Matched, "t_esim15x15Matched[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_esim3x3Rest",       t_esim3x3Rest,      "t_esim3x3Rest[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim5x5Rest",       t_esim5x5Rest,      "t_esim5x5Rest[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim7x7Rest",       t_esim7x7Rest,      "t_esim7x7Rest[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim9x9Rest",       t_esim9x9Rest,      "t_esim9x9Rest[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim11x11Rest",     t_esim11x11Rest,    "t_esim11x11Rest[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim13x13Rest",     t_esim13x13Rest,    "t_esim13x13Rest[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim15x15Rest",     t_esim15x15Rest,    "t_esim15x15Rest[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_esim3x3Photon",     t_esim3x3Photon,    "t_esim3x3Photon[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim5x5Photon",     t_esim5x5Photon,    "t_esim5x5Photon[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim7x7Photon",     t_esim7x7Photon,    "t_esim7x7Photon[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim9x9Photon",     t_esim9x9Photon,    "t_esim9x9Photon[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim11x11Photon",   t_esim11x11Photon,  "t_esim11x11Photon[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim13x13Photon",   t_esim13x13Photon,  "t_esim13x13Photon[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim15x15Photon",   t_esim15x15Photon,  "t_esim15x15Photon[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_esim3x3NeutHad",    t_esim3x3NeutHad,   "t_esim3x3NeutHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim5x5NeutHad",    t_esim5x5NeutHad,   "t_esim5x5NeutHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim7x7NeutHad",    t_esim7x7NeutHad,   "t_esim7x7NeutHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim9x9NeutHad",    t_esim9x9NeutHad,   "t_esim9x9NeutHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim11x11NeutHad",  t_esim11x11NeutHad, "t_esim11x11NeutHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim13x13NeutHad",  t_esim13x13NeutHad, "t_esim13x13NeutHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim15x15NeutHad",  t_esim15x15NeutHad, "t_esim15x15NeutHad[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_esim3x3CharHad",    t_esim3x3CharHad,   "t_esim3x3CharHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim5x5CharHad",    t_esim5x5CharHad,   "t_esim5x5CharHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim7x7CharHad",    t_esim7x7CharHad,   "t_esim7x7CharHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim9x9CharHad",    t_esim9x9CharHad,   "t_esim9x9CharHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim11x11CharHad",  t_esim11x11CharHad, "t_esim11x11CharHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim13x13CharHad",  t_esim13x13CharHad, "t_esim13x13CharHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_esim15x15CharHad",  t_esim15x15CharHad, "t_esim15x15CharHad[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_trkHcalEne",        t_trkHcalEne,       "t_trkHcalEne[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_h3x3",              t_h3x3,             "t_h3x3[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_h5x5",              t_h5x5,             "t_h5x5[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3",           t_hsim3x3,          "t_hsim3x3[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5",           t_hsim5x5,          "t_hsim5x5[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7",           t_hsim7x7,          "t_hsim7x7[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3Matched",    t_hsim3x3Matched,   "t_hsim3x3Matched[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5Matched",    t_hsim5x5Matched,   "t_hsim5x5Matched[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7Matched",    t_hsim7x7Matched,   "t_hsim7x7Matched[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3Rest",       t_hsim3x3Rest,      "t_hsim3x3Rest[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5Rest",       t_hsim5x5Rest,      "t_hsim5x5Rest[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7Rest",       t_hsim7x7Rest,      "t_hsim7x7Rest[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3Photon",     t_hsim3x3Photon,    "t_hsim3x3Photon[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5Photon",     t_hsim5x5Photon,    "t_hsim5x5Photon[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7Photon",     t_hsim7x7Photon,    "t_hsim7x7Photon[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3NeutHad",    t_hsim3x3NeutHad,   "t_hsim3x3NeutHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5NeutHad",    t_hsim5x5NeutHad,   "t_hsim5x5NeutHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7NeutHad",    t_hsim7x7NeutHad,   "t_hsim7x7NeutHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3CharHad",    t_hsim3x3CharHad,   "t_hsim3x3CharHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5CharHad",    t_hsim5x5CharHad,   "t_hsim5x5CharHad[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7CharHad",    t_hsim7x7CharHad,   "t_hsim7x7CharHad[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_trkHcalEne_1",      t_trkHcalEne_1,     "t_trkHcalEne_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_h3x3_1",            t_h3x3_1,           "t_h3x3_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_h5x5_1",            t_h5x5_1,           "t_h5x5_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_h7x7_1",            t_h7x7_1,           "t_h7x7_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_maxNearHcalP3x3_1", t_maxNearHcalP3x3_1,"t_maxNearHcalP3x3_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_maxNearHcalP5x5_1", t_maxNearHcalP5x5_1,"t_maxNearHcalP5x5_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_maxNearHcalP7x7_1", t_maxNearHcalP7x7_1,"t_maxNearHcalP7x7_1[t_nTracksIsoBy5GeV]/D");

  tree->Branch("t_hsim3x3_1",         t_hsim3x3_1,        "t_hsim3x3_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5_1",         t_hsim5x5_1,        "t_hsim5x5_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7_1",         t_hsim7x7_1,        "t_hsim7x7_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3Matched_1",  t_hsim3x3Matched_1, "t_hsim3x3Matched_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5Matched_1",  t_hsim5x5Matched_1, "t_hsim5x5Matched_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7Matched_1",  t_hsim7x7Matched_1, "t_hsim7x7Matched_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3Rest_1",     t_hsim3x3Rest_1,    "t_hsim3x3Rest_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5Rest_1",     t_hsim5x5Rest_1,    "t_hsim5x5Rest_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7Rest_1",     t_hsim7x7Rest_1,    "t_hsim7x7Rest_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3Photon_1",   t_hsim3x3Photon_1,  "t_hsim3x3Photon_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5Photon_1",   t_hsim5x5Photon_1,  "t_hsim5x5Photon_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7Photon_1",   t_hsim7x7Photon_1,  "t_hsim7x7Photon_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3NeutHad_1",  t_hsim3x3NeutHad_1, "t_hsim3x3NeutHad_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5NeutHad_1",  t_hsim5x5NeutHad_1, "t_hsim5x5NeutHad_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7NeutHad_1",  t_hsim7x7NeutHad_1, "t_hsim7x7NeutHad_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim3x3CharHad_1",  t_hsim3x3CharHad_1, "t_hsim3x3CharHad_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim5x5CharHad_1",  t_hsim5x5CharHad_1, "t_hsim5x5CharHad_1[t_nTracksIsoBy5GeV]/D");
  tree->Branch("t_hsim7x7CharHad_1",  t_hsim7x7CharHad_1, "t_hsim7x7CharHad_1[t_nTracksIsoBy5GeV]/D");
}


double IsolatedTracksTree::DeltaPhi(double v1, double v2) {
  // Computes the correctly normalized phi difference
  // v1, v2 = phi of object 1 and 2
  
  double pi    = 3.141592654;
  double twopi = 6.283185307;
  
  double diff = std::abs(v2 - v1);
  double corr = twopi - diff;
  if (diff < pi){ return diff;} else { return corr;} 
}

double IsolatedTracksTree::DeltaR(double eta1, double phi1, double eta2, double phi2) {
  double deta = eta1 - eta2;
  double dphi = DeltaPhi(phi1, phi2);
  return std::sqrt(deta*deta + dphi*dphi);
}

void IsolatedTracksTree::printTrack(const reco::Track* pTrack) {
  
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
    
    for (int i=0; i<p.numberOfHits(); i++) {
      p.printHitPattern(i, std::cout);
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
DEFINE_FWK_MODULE(IsolatedTracksTree);

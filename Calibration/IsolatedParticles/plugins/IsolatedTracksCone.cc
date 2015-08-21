// -*- C++ -*-
//
// Package:    IsolatedTracksCone
// Class:      IsolatedTracksCone
// 

/**\class IsolatedTracksCone IsolatedTracksCone.cc
 // Analysis/IsolatedTracksCone/src/IsolatedTracksCone.cc

 Description: <one line class summary>

 Implementation:
 <Notes on implementation>
*/
//
// Original Author: Jim Hirschauer (adaptation of Seema Sharma's
// IsolatedTracksNew)
//         Created:  Thu Nov  6 15:30:40 CST 2008
//
//
#define _DEBUG_QUIET

#include "Calibration/IsolatedParticles/plugins/IsolatedTracksCone.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

IsolatedTracksCone::IsolatedTracksCone(const edm::ParameterSet& iConfig) :
   trackerHitAssociatorConfig_(consumesCollector()) {

  //now do what ever initialization is needed
  doMC            = iConfig.getUntrackedParameter<bool>  ("DoMC", false); 
  myverbose_      = iConfig.getUntrackedParameter<int>( "Verbosity", 5 );
  useJetTrigger_  = iConfig.getUntrackedParameter<bool>( "useJetTrigger", false);
  drLeadJetVeto_  = iConfig.getUntrackedParameter<double>( "drLeadJetVeto",  1.2 );
  ptMinLeadJet_   = iConfig.getUntrackedParameter<double>( "ptMinLeadJet",  15.0 );

  debugTrks_          = iConfig.getUntrackedParameter<int>("DebugTracks");
  printTrkHitPattern_ = iConfig.getUntrackedParameter<bool>("PrintTrkHitPattern");
  
  minTrackP_      = iConfig.getUntrackedParameter<double>( "minTrackP", 10.0);
  maxTrackEta_    = iConfig.getUntrackedParameter<double>( "maxTrackEta", 5.0);
  maxNearTrackP_  = iConfig.getUntrackedParameter<double>( "maxNearTrackP", 1.0);

  debugEcalSimInfo_   = iConfig.getUntrackedParameter<int>("DebugEcalSimInfo");

  applyEcalIsolation_ = iConfig.getUntrackedParameter<bool>("ApplyEcalIsolation");

  tok_L1extTauJet_ = consumes<l1extra::L1JetParticleCollection>(iConfig.getParameter<edm::InputTag>("L1extraTauJetSource"));
  tok_L1extCenJet_ = consumes<l1extra::L1JetParticleCollection>(iConfig.getParameter<edm::InputTag>("L1extraCenJetSource"));
  tok_L1extFwdJet_ = consumes<l1extra::L1JetParticleCollection>(iConfig.getParameter<edm::InputTag>("L1extraFwdJetSource"));

  // hard coded collection access
  tok_EB_   = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  tok_EE_   = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEE"));
  tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_simTk_    = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  tok_simVtx_   = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
  tok_caloEB_   = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEB"));
  tok_caloEE_   = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEE"));
  tok_caloHH_   = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "HcalHits"));
  tok_trigger_  = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults","","HLT"));

  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  edm::ConsumesCollector iC = consumesCollector();
  parameters_.loadParameters( parameters, iC );
  trackAssociator_ =  new TrackDetectorAssociator();
  trackAssociator_->useDefaultPropagator();

  if(myverbose_>=0) {
    std::cout <<"Parameters read from config file \n" 
	      << "myverbose_ "          << myverbose_     << "\n"     
	      << "useJetTrigger_ "      << useJetTrigger_ << "\n"
	      << "drLeadJetVeto_ "      << drLeadJetVeto_ << "\n"
	      << "minTrackP_ "         << minTrackP_    << "\n"
	      << "maxTrackEta_ "        << maxTrackEta_   << "\n"
	      << "maxNearTrackP_ "      << maxNearTrackP_ 
	      << std::endl;
  }
}

IsolatedTracksCone::~IsolatedTracksCone() {
  delete  trackAssociator_;
}

void IsolatedTracksCone::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup) {
  
  unsigned int irun = (unsigned int)iEvent.id().run();
  unsigned int ilum = (unsigned int)iEvent.getLuminosityBlock().luminosityBlock();
  unsigned int ievt = (unsigned int)iEvent.id().event();

  
  
  clearTrackVectors();
  
  // check the L1 objects
  bool   L1Pass = false;
  leadL1JetPT=-999, leadL1JetEta=-999,  leadL1JetPhi=-999;
  if( !useJetTrigger_ ) {
    L1Pass = true;
  } else {
    edm::Handle<l1extra::L1JetParticleCollection> l1TauHandle;
    iEvent.getByToken(tok_L1extTauJet_,l1TauHandle);
    l1extra::L1JetParticleCollection::const_iterator itr;
    for(itr = l1TauHandle->begin(); itr != l1TauHandle->end(); ++itr ) 
    {
      if( itr->pt()>leadL1JetPT ) {
	leadL1JetPT  = itr->pt();
	leadL1JetEta = itr->eta();
	leadL1JetPhi = itr->phi();
      }
    }
    edm::Handle<l1extra::L1JetParticleCollection> l1CenJetHandle;
    iEvent.getByToken(tok_L1extCenJet_,l1CenJetHandle);
    for( itr = l1CenJetHandle->begin();  itr != l1CenJetHandle->end(); ++itr ) 
    {
      if( itr->pt()>leadL1JetPT ) {
	leadL1JetPT  = itr->pt();
	leadL1JetEta = itr->eta();
	leadL1JetPhi = itr->phi();
      }
    }
    edm::Handle<l1extra::L1JetParticleCollection> l1FwdJetHandle;
    iEvent.getByToken(tok_L1extFwdJet_,l1FwdJetHandle);
    for( itr = l1FwdJetHandle->begin();  itr != l1FwdJetHandle->end(); ++itr ) 
    {
      if( itr->pt()>leadL1JetPT ) {
	leadL1JetPT  = itr->pt();
	leadL1JetEta = itr->eta();
	leadL1JetPhi = itr->phi();
      }
    }
    if(leadL1JetPT>ptMinLeadJet_) 
    {
      L1Pass = true;
    }
    
  }
  

  ////////////////////////////
  // Break now if L1Pass is false
  ////////////////////////////
  if (!L1Pass) {
    nEVT_failL1++;
  //     //    std::cout << "L1Pass is false : " << L1Pass << std::endl;
  //     return;  
  }
  
  ///////////////////////////////////////////////
  // Get the collection handles
  ///////////////////////////////////////////////
  

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  const CaloSubdetectorGeometry* gEB = 
    geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  const CaloSubdetectorGeometry* gEE = 
    geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
  const CaloSubdetectorGeometry* gHB = 
    geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
  const CaloSubdetectorGeometry* gHE = 
    geo->getSubdetectorGeometry(DetId::Hcal,HcalEndcap);
  
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology); 
  const CaloTopology *caloTopology = theCaloTopology.product();
  //  const CaloSubdetectorTopology* theEBTopology   = theCaloTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel);
  //  const CaloSubdetectorTopology* theEETopology   = theCaloTopology->getSubdetectorTopology(DetId::Ecal,EcalEndcap);
  
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();
  
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EB_,barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_,endcapRecHitsHandle);

  // Retrieve the good/bad ECAL channels from the DB
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus);
  const EcalChannelStatus* theEcalChStatus = ecalChStatus.product();
  
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_,hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);
  reco::TrackCollection::const_iterator trkItr;
  if(debugTrks_>1){
    std::cout << "Track Collection: " << std::endl;
    std::cout << "Number of Tracks " << trkCollection->size() << std::endl;
  }
  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality_=
    reco::TrackBase::qualityByName(theTrackQuality);
  
  //get Handles to SimTracks and SimHits
  edm::Handle<edm::SimTrackContainer> SimTk;
  if (doMC) iEvent.getByToken(tok_simTk_,SimTk);
  edm::SimTrackContainer::const_iterator simTrkItr;

  edm::Handle<edm::SimVertexContainer> SimVtx;
  if (doMC) iEvent.getByToken(tok_simVtx_,SimVtx);

  //get Handles to PCaloHitContainers of eb/ee/hbhe
  edm::Handle<edm::PCaloHitContainer> pcaloeb;
  if (doMC) iEvent.getByToken(tok_caloEB_, pcaloeb);

  edm::Handle<edm::PCaloHitContainer> pcaloee;
  if (doMC) iEvent.getByToken(tok_caloEE_, pcaloee);

  edm::Handle<edm::PCaloHitContainer> pcalohh;
  if (doMC) iEvent.getByToken(tok_caloHH_, pcalohh);
  
  
  
  /////////////////////////////////////////////////////////
  // Get HLT_IsoTrackHB/HE Information
  /////////////////////////////////////////////////////////
    
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken( tok_trigger_, triggerResults);

  

  std::vector<int> v_hlTriggers;
  int hltHB(-99);
  int hltHE(-99);
  int hltL1Jet15		    	(-99);
  int hltJet30		    	(-99);
  int hltJet50		    	(-99);
  int hltJet80		    	(-99);
  int hltJet110		    	(-99);
  int hltJet140		    	(-99);
  int hltJet180		    	(-99);
  int hltL1SingleEG5		(-99);
  int hltZeroBias		(-99);
  int hltMinBiasHcal		(-99);
  int hltMinBiasEcal		(-99);
  int hltMinBiasPixel	    	(-99);
  int hltSingleIsoTau30_Trk5	(-99);
  int hltDoubleLooseIsoTau15_Trk5(-99);

  if (triggerResults.isValid()) {
    
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
    // TriggerNames class  triggerNames.init(*triggerResults);
    

    for (unsigned int i=0; i<triggerResults->size(); i++){
      //      std::cout << "triggerNames.triggerName(" << i << ") = " << triggerNames.triggerName(i) << std::endl;
      if (triggerNames.triggerName(i) == "HLT_IsoTrackHE_1E31") hltHE = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_IsoTrackHB_1E31") hltHB = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_L1Jet15"		    	) hltL1Jet15		    	 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_Jet30"		    	) hltJet30		    	 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_Jet50"		    	) hltJet50		    	 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_Jet80"		    	) hltJet80		    	 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_Jet110"		    	) hltJet110		    	 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_Jet140"		    	) hltJet140		    	 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_Jet180"		    	) hltJet180		    	 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_L1SingleEG5"		) hltL1SingleEG5		 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_ZeroBias"		        ) hltZeroBias		         = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_MinBiasHcal"		) hltMinBiasHcal		 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_MinBiasEcal"		) hltMinBiasEcal		 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_MinBiasPixel"	    	) hltMinBiasPixel	    	 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_SingleIsoTau30_Trk5"	) hltSingleIsoTau30_Trk5	 = triggerResults->accept(i);
      if (triggerNames.triggerName(i) == "HLT_DoubleLooseIsoTau15_Trk5" ) hltDoubleLooseIsoTau15_Trk5    = triggerResults->accept(i);
    }
  }
  
    

  
  ////////////////////////////
  // Primary loop over tracks
  ////////////////////////////
  std::unique_ptr<TrackerHitAssociator> associate;
  if (doMC) associate.reset(new TrackerHitAssociator(iEvent, trackerHitAssociatorConfig_));
  

  nTRK      = 0;
  nRawTRK   = 0;
  nFailPt   = 0;
  nFailEta  = 0;
  nFailHighPurityQaul = 0;
  nMissEcal = 0;
  nMissHcal = 0;

  for( trkItr = trkCollection->begin(); 
       trkItr != trkCollection->end(); ++trkItr)
  {

    nRawTRK++;

    const reco::Track* pTrack = &(*trkItr);

    /////////////////////////////////////////
    // Check for min Pt and max Eta P
    /////////////////////////////////////////

    bool trkQual  = pTrack->quality(trackQuality_);
    bool goodPt   = pTrack->p()>minTrackP_;
    bool goodEta  = std::abs(pTrack->momentum().eta())<maxTrackEta_;

    double eta1       = pTrack->momentum().eta();
    double phi1       = pTrack->momentum().phi();
    double pt1        = pTrack->pt();
    double p1         = pTrack->p();


    if (!goodEta){
      nFailEta++;
    }
    if (!goodPt){
      nFailPt++;
    }
    if (!trkQual){
      nFailHighPurityQaul++;
    }
    
    hRawPt ->Fill(pt1 );
    hRawP  ->Fill(p1  );
    hRawEta->Fill(eta1);
    hRawPhi->Fill(phi1);
      
    if( !goodEta || !goodPt || !trkQual ) continue; // Skip to next track
    
    ////////////////////////////////////////////
    // Find track trajectory
    ////////////////////////////////////////////
    
      
    const FreeTrajectoryState fts1 = 
      trackAssociator_->getFreeTrajectoryState(iSetup, *pTrack);
    
      
    TrackDetMatchInfo info1 = 
      trackAssociator_->associate(iEvent, iSetup, fts1, parameters_);
    
      

    ////////////////////////////////////////////
    // First confirm track makes it to Hcal
    ////////////////////////////////////////////

    if (info1.trkGlobPosAtHcal.x()==0 && 
	info1.trkGlobPosAtHcal.y()==0 && 
	info1.trkGlobPosAtHcal.z()==0) 
    {
      nMissHcal++;
      continue;      
    }
    
    const GlobalPoint hpoint1(info1.trkGlobPosAtHcal.x(),
			      info1.trkGlobPosAtHcal.y(),
			      info1.trkGlobPosAtHcal.z());

      

    ////////////////////////////
    // Get basic quantities
    ////////////////////////////

    const reco::HitPattern& hitp = pTrack->hitPattern();
    int nLayersCrossed = hitp.trackerLayersWithMeasurement();        
    int nOuterHits     = hitp.stripTOBLayersWithMeasurement()
      +hitp.stripTECLayersWithMeasurement() ;

    
    double simP = 0;
    if (doMC) {
      edm::SimTrackContainer::const_iterator matchedSimTrk = 
	spr::matchedSimTrack(iEvent, SimTk, SimVtx, pTrack, *associate, false);
      simP = matchedSimTrk->momentum().P();
    }
    ////////////////////////////////////////////
    // Get Ecal Point
    ////////////////////////////////////////////

    const GlobalPoint point1(info1.trkGlobPosAtEcal.x(),
			     info1.trkGlobPosAtEcal.y(),
			     info1.trkGlobPosAtEcal.z());
    

    // Sanity check that track hits Ecal

    if (info1.trkGlobPosAtEcal.x()==0 && 
	info1.trkGlobPosAtEcal.y()==0 && 
	info1.trkGlobPosAtEcal.z()==0) 
    {
      std::cout << "Track doesn't reach Ecal." << std::endl;
      nMissEcal++;
      continue;
    }

    // Get Track Momentum - make sure you have latest version of TrackDetMatchInfo
    
    GlobalVector trackMomAtEcal = info1.trkMomAtEcal;
    GlobalVector trackMomAtHcal = info1.trkMomAtHcal;

    /////////////////////////////////////////////////////////
    // If using Jet trigger, get distance from leading jet
    /////////////////////////////////////////////////////////

    double drFromLeadJet = 999.0;
    if( useJetTrigger_ ) {
      double dphi = DeltaPhi(phi1, leadL1JetPhi);
      double deta = eta1 - leadL1JetEta;
      drFromLeadJet =  sqrt(dphi*dphi + deta*deta);
    }
    

    ///////////////////////////////////////////////////////
    // Define Arrays for sizes of Charge, Neut Iso Radii and
    // Clustering Cone Radius.
    ///////////////////////////////////////////////////////

    const int a_size = 7;
    double a_coneR[a_size];
    double a_charIsoR[a_size];
    double a_neutIsoR[a_size];

    a_coneR[0] = 17.49; // = area of 2x2
    a_coneR[1] = 26.23; // = area of 3x3
    a_coneR[2] = 30.61;
    a_coneR[3] = 34.98; // = area of 4x4
    a_coneR[4] = 39.35;
    a_coneR[5] = 43.72; // = area of 5x5
    a_coneR[6] = 52.46; // = area of 6x6
    
    for (int i=0; i<a_size; i++){
      a_charIsoR[i] = a_coneR[i]+28.9; // 28.9 gives 55.1 for 3x3 benchmark 
      a_neutIsoR[i] = a_charIsoR[i]*0.726; // Ecal radius = 0.726*Hcal radius
    }
    
    ///////////////////////////////////////////////////////
    // Do Neutral Iso in radius on Ecal surface.
    ///////////////////////////////////////////////////////
  
    // NxN cluster
    double e3x3=-999.0;
    double trkEcalEne =-999.0;
    edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
    iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);

    if(std::abs(point1.eta())<1.479) {
      const DetId isoCell = gEB->getClosestCell(point1);
      e3x3   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology, sevlv.product(),1,1).first;  
      trkEcalEne   = spr::eCaloSimInfo(iEvent, geo, pcaloeb, pcaloee, SimTk, SimVtx, pTrack, *associate);
    } else {
      const DetId isoCell = gEE->getClosestCell(point1);
      e3x3   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology, sevlv.product(),1,1).first;
      trkEcalEne   = spr::eCaloSimInfo(iEvent, geo, pcaloeb, pcaloee, SimTk, SimVtx, pTrack, *associate);
    }

    // Cone cluster

    // Set up array of cone sizes for MIP cut
    const int a_mip_size = 5;
    double a_mipR[a_mip_size];
    a_mipR[0] = 3.84; // = area of 3x3 ecal
    a_mipR[1] = 14.0;
    a_mipR[2] = 19.0;
    a_mipR[3] = 24.0;
    a_mipR[4] = 9.0;  // = from standard analyzer

    std::vector<double> v_eDR;
    for (int i = 0 ; i < a_size ; i++){
      int nRH_eDR = 0;

      // Cone in ecal
      double eDR = spr::eCone_ecal(geo, 
				   barrelRecHitsHandle, 
				   endcapRecHitsHandle, 
				   hpoint1, point1, 
				   a_neutIsoR[i],  
				   trackMomAtEcal, nRH_eDR);
      v_eDR.push_back(eDR);
      
    }

    std::vector<double> v_eMipDR;
    for (int i = 0 ; i < a_mip_size ; i++){
      int nRH_eMipDR = 0;
      double eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, 
				      endcapRecHitsHandle, 
				      hpoint1, point1, 
				      a_mipR[i], trackMomAtEcal, nRH_eMipDR);
      
      v_eMipDR.push_back(eMipDR);
    }
    
	    
    ////////////////////////////////////////////
    // Do charge isolation in radius at Hcal surface for 5 different
    // radii defined above in a_charIso
    ////////////////////////////////////////////
    
    std::vector<double> v_hmaxNearP_goodTrk;
    std::vector<double> v_hmaxNearP        ;
    std::vector<int>    v_hnNearTRKs       ;
    std::vector<int>    v_hnLayers_maxNearP;
    std::vector<int>    v_htrkQual_maxNearP;

    std::vector<double> v_cone_hmaxNearP_goodTrk;
    std::vector<double> v_cone_hmaxNearP        ;
    std::vector<int>    v_cone_hnNearTRKs       ;
    std::vector<int>    v_cone_hnLayers_maxNearP;
    std::vector<int>    v_cone_htrkQual_maxNearP;

    for (int i = 0 ; i < a_size ; i++){

      double hmaxNearP         = -999.0;
      int    hnNearTRKs        =  0;
      int    hnLayers_maxNearP =  0;
      int    htrkQual_maxNearP = -1; 
      double hmaxNearP_goodTrk = -999.0;

      double conehmaxNearP         = -999.0;
      int    conehnNearTRKs        =  0;
      int    conehnLayers_maxNearP =  0;
      int    conehtrkQual_maxNearP = -1; 
      double conehmaxNearP_goodTrk = -999.0;

      conehmaxNearP = spr::coneChargeIsolation(iEvent, iSetup, 
					       trkItr, trkCollection, 
					       *trackAssociator_, parameters_, 
					       theTrackQuality, conehnNearTRKs,
					       conehnLayers_maxNearP,
					       conehtrkQual_maxNearP, 
					       conehmaxNearP_goodTrk, 
					       hpoint1, trackMomAtHcal, 
					       a_charIsoR[i]); 

      v_hmaxNearP_goodTrk.push_back(hmaxNearP_goodTrk);
      v_hmaxNearP        .push_back(hmaxNearP        );
      v_hnNearTRKs       .push_back(hnNearTRKs       );
      v_hnLayers_maxNearP.push_back(hnLayers_maxNearP);
      v_htrkQual_maxNearP.push_back(htrkQual_maxNearP);

      v_cone_hmaxNearP_goodTrk.push_back(conehmaxNearP_goodTrk);
      v_cone_hmaxNearP        .push_back(conehmaxNearP        );
      v_cone_hnNearTRKs       .push_back(conehnNearTRKs       );
      v_cone_hnLayers_maxNearP.push_back(conehnLayers_maxNearP);
      v_cone_htrkQual_maxNearP.push_back(conehtrkQual_maxNearP);
      
    }
    
    double h3x3=-999.0, h5x5=-999.0;
    double hsim3x3=-999.0, hsim5x5=-999.0, trkHcalEne=-999.0;
    std::map<std::string, double> hsimInfo3x3, hsimInfo5x5;
    double distFromHotCell_h3x3 = -99.;
    int ietaFromHotCell_h3x3    = -99;
    int iphiFromHotCell_h3x3    = -99;
    double distFromHotCell_h5x5 = -99.;
    int ietaFromHotCell_h5x5    = -99;
    int iphiFromHotCell_h5x5    = -99;

    GlobalPoint gPosHotCell_h3x3(0.,0.,0.);
    GlobalPoint gPosHotCell_h5x5(0.,0.,0.);

    int nRH_h3x3(0), nRH_h5x5(0);

    // Hcal Energy Clustering
    
    // Get closetcell for ietaFromHotCell and iphiFromHotCell
    DetId ClosestCell;
    if( std::abs(pTrack->eta())<1.392 ) {
      ClosestCell = gHB->getClosestCell(hpoint1);
    } else {
      ClosestCell = gHE->getClosestCell(hpoint1);
    }
    // Transform into HcalDetId so that I can get ieta, iphi later.
    HcalDetId ClosestCell_HcalDetId(ClosestCell.rawId());

    // Using NxN Cluster
    std::vector<int>    v_RH_h3x3_ieta;
    std::vector<int>    v_RH_h3x3_iphi;
    std::vector<double> v_RH_h3x3_ene;
    std::vector<int>    v_RH_h5x5_ieta;
    std::vector<int>    v_RH_h5x5_iphi;
    std::vector<double> v_RH_h5x5_ene;
    
    
    h3x3 = spr::eHCALmatrix(geo, theHBHETopology, ClosestCell, hbhe,1,1,
			    nRH_h3x3, v_RH_h3x3_ieta, v_RH_h3x3_iphi, v_RH_h3x3_ene, 
			    gPosHotCell_h3x3);
    distFromHotCell_h3x3 = spr::getDistInPlaneTrackDir(hpoint1, trackMomAtHcal, gPosHotCell_h3x3);
    
    h5x5 = spr::eHCALmatrix(geo, theHBHETopology, ClosestCell, hbhe,2,2,
			    nRH_h5x5, v_RH_h5x5_ieta, v_RH_h5x5_iphi,  v_RH_h5x5_ene, 
			    gPosHotCell_h5x5);
    distFromHotCell_h5x5 = spr::getDistInPlaneTrackDir(hpoint1, trackMomAtHcal, gPosHotCell_h5x5);
    

    //     double heta = (double)hpoint1.eta();
    //     double hphi = (double)hpoint1.phi();
    std::vector<int> multiplicity3x3;
    std::vector<int> multiplicity5x5;
    if (doMC) {
      hsim3x3   = spr::eHCALmatrix(theHBHETopology, ClosestCell, 
				   pcalohh,1,1);
      hsim5x5   = spr::eHCALmatrix(theHBHETopology, ClosestCell,
				   pcalohh,2,2);

      hsimInfo3x3 = spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell, pcalohh, SimTk, SimVtx, pTrack, *associate, 1,1, multiplicity3x3);
      hsimInfo5x5 = spr::eHCALSimInfo(iEvent, theHBHETopology, ClosestCell, pcalohh, SimTk, SimVtx, pTrack, *associate, 2,2, multiplicity5x5);
    
      // Get energy from all simhits in hcal associated with iso track
      trkHcalEne   = spr::eCaloSimInfo(iEvent, geo, pcalohh, SimTk, SimVtx, pTrack, *associate);
    }

    // Finally for cones of varying radii.
    std::vector<double> v_hsimInfoConeMatched;
    std::vector<double> v_hsimInfoConeRest   ;
    std::vector<double> v_hsimInfoConePhoton  ;
    std::vector<double> v_hsimInfoConeNeutHad;
    std::vector<double> v_hsimInfoConeCharHad;
    std::vector<double> v_hsimInfoConePdgMatched;
    std::vector<double> v_hsimInfoConeTotal  ;

    std::vector<int> v_hsimInfoConeNMatched;
    std::vector<int> v_hsimInfoConeNTotal  ;
    std::vector<int> v_hsimInfoConeNNeutHad;
    std::vector<int> v_hsimInfoConeNCharHad;
    std::vector<int> v_hsimInfoConeNPhoton ;
    std::vector<int> v_hsimInfoConeNRest   ;

    std::vector<double> v_hsimCone           ;
    std::vector<double> v_hCone              ;
    
    std::vector<int>    v_nRecHitsCone       ;
    std::vector<int>    v_nSimHitsCone       ;

    std::vector<double> v_distFromHotCell;
    std::vector<int> v_ietaFromHotCell;
    std::vector<int> v_iphiFromHotCell;
    GlobalPoint gposHotCell(0.,0.,0.);
    
    
    std::vector<int>    v_RH_r26_ieta;
    std::vector<int>    v_RH_r26_iphi;
    std::vector<double> v_RH_r26_ene;
    std::vector<int>    v_RH_r44_ieta;
    std::vector<int>    v_RH_r44_iphi;
    std::vector<double> v_RH_r44_ene;
    
    
	
    for (int i = 0 ; i < a_size ; i++){

      
      std::map<std::string, double> hsimInfoCone;
      double hsimCone = -999.0, hCone = -999.0;
      double distFromHotCell = -99.0;
      int ietaFromHotCell = -99;
      int iphiFromHotCell = -99;
      int ietaHotCell = -99;
      int iphiHotCell = -99;
      int nRecHitsCone = -999;
      int nSimHitsCone = -999;

      std::vector<int> multiplicityCone;
      std::vector<DetId> coneRecHitDetIds;
      if (doMC) 
	hsimCone = spr::eCone_hcal(geo, pcalohh, hpoint1, point1, 
				   a_coneR[i], trackMomAtHcal, nSimHitsCone);
      
      // If needed, get ieta and iphi of rechits for cones of 23.25
      // and for hitmap for debugging
      bool makeHitmaps = false;
      if (a_coneR[i] == 26.23 && makeHitmaps)
      {
	
	hCone = spr::eCone_hcal(geo, hbhe, hpoint1, point1, 
				a_coneR[i], trackMomAtHcal,nRecHitsCone,
				v_RH_r26_ieta, v_RH_r26_iphi,  v_RH_r26_ene, 
				coneRecHitDetIds, distFromHotCell, 
				ietaHotCell, iphiHotCell, gposHotCell);
      } 
      else if (a_coneR[i] == 43.72 && makeHitmaps)
      {
	
	hCone = spr::eCone_hcal(geo, hbhe, hpoint1, point1, 
				a_coneR[i], trackMomAtHcal,nRecHitsCone,
				v_RH_r44_ieta, v_RH_r44_iphi,  v_RH_r44_ene, 
				coneRecHitDetIds, distFromHotCell, 
				ietaHotCell, iphiHotCell, gposHotCell);
      } 
      else 
      {
	
	hCone = spr::eCone_hcal(geo, hbhe, hpoint1, point1, 
				a_coneR[i], trackMomAtHcal, nRecHitsCone, 
				coneRecHitDetIds, distFromHotCell, 
				ietaHotCell, iphiHotCell, gposHotCell);
      }

      
      
      if (ietaHotCell != 99){
	ietaFromHotCell = ietaHotCell-ClosestCell_HcalDetId.ieta();
	iphiFromHotCell = iphiHotCell-ClosestCell_HcalDetId.iphi();
      }
      
      // SimHits NOT matched to RecHits
      if (doMC) {
	hsimInfoCone = spr::eHCALSimInfoCone(iEvent,pcalohh, SimTk, SimVtx, pTrack, *associate, geo, hpoint1, point1, a_coneR[i], trackMomAtHcal, multiplicityCone);      
      
      
	  
	// SimHits matched to RecHits
	//       hsimInfoCone = spr::eHCALSimInfoCone(iEvent,pcalohh, SimTk, SimVtx, 
	//        					   pTrack, *associate, 
	//        					   geo, hpoint1, point1, 
	//        					   a_coneR[i], trackMomAtHcal, 
	//        					   multiplicityCone,
	//        					   coneRecHitDetIds);      
      
	v_hsimInfoConeMatched   .push_back(hsimInfoCone["eMatched"   ]);
	v_hsimInfoConeRest      .push_back(hsimInfoCone["eRest"      ]);
	v_hsimInfoConePhoton    .push_back(hsimInfoCone["eGamma"     ]);
	v_hsimInfoConeNeutHad   .push_back(hsimInfoCone["eNeutralHad"]);
	v_hsimInfoConeCharHad   .push_back(hsimInfoCone["eChargedHad"]);
	v_hsimInfoConePdgMatched.push_back(hsimInfoCone["pdgMatched" ]);
	v_hsimInfoConeTotal     .push_back(hsimInfoCone["eTotal"     ]);

	v_hsimInfoConeNMatched  .push_back(multiplicityCone.at(0));

	v_hsimInfoConeNTotal      .push_back(multiplicityCone.at(1));
	v_hsimInfoConeNNeutHad    .push_back(multiplicityCone.at(2));
	v_hsimInfoConeNCharHad    .push_back(multiplicityCone.at(3));
	v_hsimInfoConeNPhoton     .push_back(multiplicityCone.at(4));
	v_hsimInfoConeNRest       .push_back(multiplicityCone.at(5));

	v_hsimCone                .push_back(hsimCone                   );
	v_nSimHitsCone            .push_back(nSimHitsCone               );
      }
      v_hCone                   .push_back(hCone                      );
      v_nRecHitsCone            .push_back(nRecHitsCone               );
      
      v_distFromHotCell         .push_back(distFromHotCell            );
      v_ietaFromHotCell         .push_back(ietaFromHotCell            );
      v_iphiFromHotCell         .push_back(iphiFromHotCell            );
      
	  
    }
    
 
    ////////////////////////////////////////////
    // Fill Vectors that go into root file
    ////////////////////////////////////////////

    t_v_hnNearTRKs        ->push_back(v_hnNearTRKs                 );
    t_v_hnLayers_maxNearP ->push_back(v_hnLayers_maxNearP          );
    t_v_htrkQual_maxNearP ->push_back(v_htrkQual_maxNearP          );
    t_v_hmaxNearP_goodTrk ->push_back(v_hmaxNearP_goodTrk          ); 
    t_v_hmaxNearP         ->push_back(v_hmaxNearP                  );    

    t_v_cone_hnNearTRKs        ->push_back(v_cone_hnNearTRKs       );
    t_v_cone_hnLayers_maxNearP ->push_back(v_cone_hnLayers_maxNearP);
    t_v_cone_htrkQual_maxNearP ->push_back(v_cone_htrkQual_maxNearP);
    t_v_cone_hmaxNearP_goodTrk ->push_back(v_cone_hmaxNearP_goodTrk); 
    t_v_cone_hmaxNearP         ->push_back(v_cone_hmaxNearP        );    

    //    t_hScale            ->push_back(hScale                     );
    t_trkNOuterHits     ->push_back(nOuterHits                 );
    t_trkNLayersCrossed ->push_back(nLayersCrossed             );
    t_dtFromLeadJet     ->push_back(drFromLeadJet              );
    t_trkP              ->push_back(p1                         );
    t_trkPt             ->push_back(pt1                        );
    t_trkEta            ->push_back(eta1                       );
    t_trkPhi            ->push_back(phi1                       );

    t_e3x3              ->push_back(e3x3                       );
    t_v_eDR             ->push_back(v_eDR                      );
    t_v_eMipDR          ->push_back(v_eMipDR                   );

    t_h3x3              ->push_back(h3x3                       );
    t_h5x5              ->push_back(h5x5                       );
    t_nRH_h3x3          ->push_back(nRH_h3x3                   );
    t_nRH_h5x5          ->push_back(nRH_h5x5                   );
    
    t_v_RH_h3x3_ieta    ->push_back(v_RH_h3x3_ieta);
    t_v_RH_h3x3_iphi    ->push_back(v_RH_h3x3_iphi);
    t_v_RH_h3x3_ene     ->push_back(v_RH_h3x3_ene);
    t_v_RH_h5x5_ieta    ->push_back(v_RH_h5x5_ieta);
    t_v_RH_h5x5_iphi    ->push_back(v_RH_h5x5_iphi);
    t_v_RH_h5x5_ene     ->push_back(v_RH_h5x5_ene);

    if (doMC) {
      t_simP              ->push_back(simP                       );
      t_hsim3x3           ->push_back(hsim3x3                    );
      t_hsim5x5           ->push_back(hsim5x5                    );

      t_hsim3x3Matched    ->push_back(hsimInfo3x3["eMatched"]    );
      t_hsim5x5Matched    ->push_back(hsimInfo5x5["eMatched"]    );
      t_hsim3x3Rest       ->push_back(hsimInfo3x3["eRest"]       );
      t_hsim5x5Rest       ->push_back(hsimInfo5x5["eRest"]       );
      t_hsim3x3Photon     ->push_back(hsimInfo3x3["eGamma"]      );
      t_hsim5x5Photon     ->push_back(hsimInfo5x5["eGamma"]      );
      t_hsim3x3NeutHad    ->push_back(hsimInfo3x3["eNeutralHad"] );
      t_hsim5x5NeutHad    ->push_back(hsimInfo5x5["eNeutralHad"] );
      t_hsim3x3CharHad    ->push_back(hsimInfo3x3["eChargedHad"] );
      t_hsim5x5CharHad    ->push_back(hsimInfo5x5["eChargedHad"] );
      t_hsim3x3Total      ->push_back(hsimInfo3x3["eTotal"]      );
      t_hsim5x5Total      ->push_back(hsimInfo5x5["eTotal"]      );
      t_hsim3x3PdgMatched ->push_back(hsimInfo3x3["pdgMatched"]  );
      t_hsim5x5PdgMatched ->push_back(hsimInfo5x5["pdgMatched"]  );

      t_hsim3x3NMatched   ->push_back(multiplicity3x3.at(0));
      t_hsim3x3NTotal     ->push_back(multiplicity3x3.at(1));
      t_hsim3x3NNeutHad   ->push_back(multiplicity3x3.at(2));
      t_hsim3x3NCharHad   ->push_back(multiplicity3x3.at(3));
      t_hsim3x3NPhoton    ->push_back(multiplicity3x3.at(4));
      t_hsim3x3NRest      ->push_back(multiplicity3x3.at(5));

      t_hsim5x5NMatched   ->push_back(multiplicity5x5.at(0));
      t_hsim5x5NTotal     ->push_back(multiplicity5x5.at(1));
      t_hsim5x5NNeutHad   ->push_back(multiplicity5x5.at(2));
      t_hsim5x5NCharHad   ->push_back(multiplicity5x5.at(3));
      t_hsim5x5NPhoton    ->push_back(multiplicity5x5.at(4));
      t_hsim5x5NRest      ->push_back(multiplicity5x5.at(5));
    }
    
    t_distFromHotCell_h3x3->push_back(distFromHotCell_h3x3);
    t_ietaFromHotCell_h3x3->push_back(ietaFromHotCell_h3x3);
    t_iphiFromHotCell_h3x3->push_back(iphiFromHotCell_h3x3);
    t_distFromHotCell_h5x5->push_back(distFromHotCell_h5x5);
    t_ietaFromHotCell_h5x5->push_back(ietaFromHotCell_h5x5);
    t_iphiFromHotCell_h5x5->push_back(iphiFromHotCell_h5x5);
    
    if (doMC) {
      t_trkHcalEne                ->push_back(trkHcalEne                 );
      t_trkEcalEne                ->push_back(trkEcalEne                 );

      t_v_hsimInfoConeMatched     ->push_back(v_hsimInfoConeMatched   );
      t_v_hsimInfoConeRest        ->push_back(v_hsimInfoConeRest      );
      t_v_hsimInfoConePhoton      ->push_back(v_hsimInfoConePhoton   );
      t_v_hsimInfoConeNeutHad     ->push_back(v_hsimInfoConeNeutHad   );
      t_v_hsimInfoConeCharHad     ->push_back(v_hsimInfoConeCharHad   );
      t_v_hsimInfoConePdgMatched  ->push_back(v_hsimInfoConePdgMatched);  
      t_v_hsimInfoConeTotal       ->push_back(v_hsimInfoConeTotal     );

      t_v_hsimInfoConeNMatched    ->push_back(v_hsimInfoConeNMatched  );
      t_v_hsimInfoConeNTotal      ->push_back(v_hsimInfoConeNTotal    );
      t_v_hsimInfoConeNNeutHad    ->push_back(v_hsimInfoConeNNeutHad  );
      t_v_hsimInfoConeNCharHad    ->push_back(v_hsimInfoConeNCharHad  );
      t_v_hsimInfoConeNPhoton     ->push_back(v_hsimInfoConeNPhoton    );
      t_v_hsimInfoConeNRest       ->push_back(v_hsimInfoConeNRest     );  

      t_v_hsimCone    ->push_back(v_hsimCone    );
      t_v_hCone       ->push_back(v_hCone       );
      t_v_nRecHitsCone->push_back(v_nRecHitsCone);
      t_v_nSimHitsCone->push_back(v_nSimHitsCone);
    }

    
    t_v_distFromHotCell->push_back(v_distFromHotCell);
    t_v_ietaFromHotCell->push_back(v_ietaFromHotCell);
    t_v_iphiFromHotCell->push_back(v_iphiFromHotCell);
    
    t_v_RH_r26_ieta    ->push_back(v_RH_r26_ieta);
    t_v_RH_r26_iphi    ->push_back(v_RH_r26_iphi);
    t_v_RH_r26_ene     ->push_back(v_RH_r26_ene);
    t_v_RH_r44_ieta    ->push_back(v_RH_r44_ieta);
    t_v_RH_r44_iphi    ->push_back(v_RH_r44_iphi);
    t_v_RH_r44_ene     ->push_back(v_RH_r44_ene);

    

    t_v_hlTriggers    ->push_back(v_hlTriggers);
    t_hltHB ->push_back(hltHB);
    t_hltHE ->push_back(hltHE);
    t_hltL1Jet15                 ->push_back(hltL1Jet15		    	);
    t_hltJet30		    	 ->push_back(hltJet30		    	);
    t_hltJet50		    	 ->push_back(hltJet50		    	);
    t_hltJet80		    	 ->push_back(hltJet80		    	);
    t_hltJet110		    	 ->push_back(hltJet110		    	);
    t_hltJet140		    	 ->push_back(hltJet140		    	);
    t_hltJet180		    	 ->push_back(hltJet180		    	);
    t_hltL1SingleEG5		 ->push_back(hltL1SingleEG5		);
    t_hltZeroBias		 ->push_back(hltZeroBias		);
    t_hltMinBiasHcal		 ->push_back(hltMinBiasHcal		);
    t_hltMinBiasEcal		 ->push_back(hltMinBiasEcal		);
    t_hltMinBiasPixel	    	 ->push_back(hltMinBiasPixel	    	);
    t_hltSingleIsoTau30_Trk5	 ->push_back(hltSingleIsoTau30_Trk5	);
    t_hltDoubleLooseIsoTau15_Trk5->push_back(hltDoubleLooseIsoTau15_Trk5);

    t_irun->push_back(irun);
    t_ievt->push_back(ievt);
    t_ilum->push_back(ilum);
    
    nTRK++;
    
    
  } // Loop over track collection
  
    //  std::cout << "nEVT= " << nEVT << std::endl;
  
  ntp->Fill();
  nEVT++;
  
}
  


void IsolatedTracksCone::beginJob(const edm::EventSetup&) {

  //   hbScale = 120.0;
  //   heScale = 135.0;
  nEVT=0;
  nEVT_failL1=0;
  nTRK=0;
  
  double tempgen_TH[22] = { 0.0,  1.0,  2.0,  3.0,  4.0,  
			    5.0,  6.0,  7.0,  8.0,  9.0, 
			    10.0, 12.0, 15.0, 20.0, 25.0, 
			    30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100};
  for(int i=0; i<22; i++)  genPartPBins[i]  = tempgen_TH[i];


  double tempgen_Eta[5] = {0.0, 0.5, 1.1, 1.7, 2.0};
  for(int i=0; i<5; i++) genPartEtaBins[i] = tempgen_Eta[i];

  t_v_hnNearTRKs           = new std::vector<std::vector<int> >   (); 
  t_v_hnLayers_maxNearP    = new std::vector<std::vector<int> >   (); 
  t_v_htrkQual_maxNearP    = new std::vector<std::vector<int> >  (); 
  t_v_hmaxNearP_goodTrk    = new std::vector<std::vector<double> >();
  t_v_hmaxNearP            = new std::vector<std::vector<double> >();
			       				    
  t_v_cone_hnNearTRKs           = new std::vector<std::vector<int> >   (); 
  t_v_cone_hnLayers_maxNearP    = new std::vector<std::vector<int> >   (); 
  t_v_cone_htrkQual_maxNearP    = new std::vector<std::vector<int> >  (); 
  t_v_cone_hmaxNearP_goodTrk    = new std::vector<std::vector<double> >();
  t_v_cone_hmaxNearP            = new std::vector<std::vector<double> >();
			       				    
  //  t_hScale             = new std::vector<double>();
  t_trkNOuterHits      = new std::vector<double>();
  t_trkNLayersCrossed  = new std::vector<double>();
  t_dtFromLeadJet      = new std::vector<double>();
  t_trkP               = new std::vector<double>();
  t_trkPt              = new std::vector<double>();
  t_trkEta             = new std::vector<double>();
  t_trkPhi             = new std::vector<double>();

  t_e3x3               = new std::vector<double>();
  t_v_eDR              = new std::vector<std::vector<double> >();
  t_v_eMipDR           = new std::vector<std::vector<double> >();

  t_h3x3               = new std::vector<double>();
  t_h5x5               = new std::vector<double>();

  t_nRH_h3x3           = new std::vector<double>();
  t_nRH_h5x5           = new std::vector<double>();

  if (doMC) {
    t_simP               = new std::vector<double>();
    t_hsim3x3            = new std::vector<double>();
    t_hsim5x5            = new std::vector<double>();

    t_hsim3x3Matched     = new std::vector<double>();
    t_hsim5x5Matched     = new std::vector<double>();
    t_hsim3x3Rest        = new std::vector<double>();
    t_hsim5x5Rest        = new std::vector<double>();
    t_hsim3x3Photon      = new std::vector<double>();
    t_hsim5x5Photon      = new std::vector<double>();
    t_hsim3x3NeutHad     = new std::vector<double>();
    t_hsim5x5NeutHad     = new std::vector<double>();
    t_hsim3x3CharHad     = new std::vector<double>();
    t_hsim5x5CharHad     = new std::vector<double>();
    t_hsim3x3PdgMatched  = new std::vector<double>();
    t_hsim5x5PdgMatched  = new std::vector<double>();
    t_hsim3x3Total       = new std::vector<double>();
    t_hsim5x5Total       = new std::vector<double>();

    t_hsim3x3NMatched  = new std::vector<int>();
    t_hsim3x3NTotal    = new std::vector<int>();
    t_hsim3x3NNeutHad  = new std::vector<int>();
    t_hsim3x3NCharHad  = new std::vector<int>();
    t_hsim3x3NPhoton   = new std::vector<int>();
    t_hsim3x3NRest     = new std::vector<int>();

    t_hsim5x5NMatched  = new std::vector<int>();
    t_hsim5x5NTotal    = new std::vector<int>();
    t_hsim5x5NNeutHad  = new std::vector<int>();
    t_hsim5x5NCharHad  = new std::vector<int>();
    t_hsim5x5NPhoton   = new std::vector<int>();
    t_hsim5x5NRest     = new std::vector<int>();

    t_trkHcalEne         = new std::vector<double>();
    t_trkEcalEne         = new std::vector<double>();
  }

  t_distFromHotCell_h3x3  = new std::vector<double>();
  t_ietaFromHotCell_h3x3  = new std::vector<int>();
  t_iphiFromHotCell_h3x3  = new std::vector<int>();
  t_distFromHotCell_h5x5  = new std::vector<double>();
  t_ietaFromHotCell_h5x5  = new std::vector<int>();
  t_iphiFromHotCell_h5x5  = new std::vector<int>();

  if (doMC) {
    t_v_hsimInfoConeMatched   = new std::vector<std::vector<double> >();
    t_v_hsimInfoConeRest      = new std::vector<std::vector<double> >();
    t_v_hsimInfoConePhoton     = new std::vector<std::vector<double> >();
    t_v_hsimInfoConeNeutHad   = new std::vector<std::vector<double> >();
    t_v_hsimInfoConeCharHad   = new std::vector<std::vector<double> >();
    t_v_hsimInfoConePdgMatched= new std::vector<std::vector<double> >();
    t_v_hsimInfoConeTotal     = new std::vector<std::vector<double> >();

    t_v_hsimInfoConeNMatched  = new std::vector<std::vector<int> >();
    t_v_hsimInfoConeNTotal    = new std::vector<std::vector<int> >();
    t_v_hsimInfoConeNNeutHad  = new std::vector<std::vector<int> >();
    t_v_hsimInfoConeNCharHad  = new std::vector<std::vector<int> >();
    t_v_hsimInfoConeNPhoton    = new std::vector<std::vector<int> >();
    t_v_hsimInfoConeNRest     = new std::vector<std::vector<int> >();

    t_v_hsimCone    = new std::vector<std::vector<double> >();
  }

  t_v_hCone       = new std::vector<std::vector<double> >();
  t_v_nRecHitsCone= new std::vector<std::vector<int> >();
  t_v_nSimHitsCone= new std::vector<std::vector<int> >();

  t_v_distFromHotCell= new std::vector<std::vector<double> >();
  t_v_ietaFromHotCell= new std::vector<std::vector<int> >();
  t_v_iphiFromHotCell= new std::vector<std::vector<int> >();

  t_v_RH_h3x3_ieta = new std::vector<std::vector<int> >();
  t_v_RH_h3x3_iphi = new std::vector<std::vector<int> >();
  t_v_RH_h3x3_ene  = new std::vector<std::vector<double> >();
  t_v_RH_h5x5_ieta = new std::vector<std::vector<int> >();
  t_v_RH_h5x5_iphi = new std::vector<std::vector<int> >();
  t_v_RH_h5x5_ene  = new std::vector<std::vector<double> >();
  t_v_RH_r26_ieta  = new std::vector<std::vector<int> >();
  t_v_RH_r26_iphi  = new std::vector<std::vector<int> >();
  t_v_RH_r26_ene   = new std::vector<std::vector<double> >();
  t_v_RH_r44_ieta  = new std::vector<std::vector<int> >();
  t_v_RH_r44_iphi  = new std::vector<std::vector<int> >();
  t_v_RH_r44_ene   = new std::vector<std::vector<double> >();


  t_v_hlTriggers    = new std::vector<std::vector<int> >();

  t_hltHE                         = new std::vector<int>();
  t_hltHB                         = new std::vector<int>();
  t_hltL1Jet15		          = new std::vector<int>();
  t_hltJet30		          = new std::vector<int>();
  t_hltJet50		          = new std::vector<int>();
  t_hltJet80		          = new std::vector<int>();
  t_hltJet110		          = new std::vector<int>();
  t_hltJet140		          = new std::vector<int>();
  t_hltJet180		          = new std::vector<int>();
  t_hltL1SingleEG5	          = new std::vector<int>();
  t_hltZeroBias		          = new std::vector<int>();
  t_hltMinBiasHcal	          = new std::vector<int>();
  t_hltMinBiasEcal	          = new std::vector<int>();
  t_hltMinBiasPixel	          = new std::vector<int>();
  t_hltSingleIsoTau30_Trk5        = new std::vector<int>();
  t_hltDoubleLooseIsoTau15_Trk5   = new std::vector<int>();
  

  t_irun = new std::vector<unsigned int>();
  t_ievt = new std::vector<unsigned int>();
  t_ilum = new std::vector<unsigned int>();

  BuildTree();
}

void IsolatedTracksCone::clearTrackVectors() {

  t_v_hnNearTRKs          ->clear();   
  t_v_hnLayers_maxNearP   ->clear();   
  t_v_htrkQual_maxNearP   ->clear();   
  t_v_hmaxNearP_goodTrk   ->clear();
  t_v_hmaxNearP           ->clear();

  t_v_cone_hnNearTRKs          ->clear();   
  t_v_cone_hnLayers_maxNearP   ->clear();   
  t_v_cone_htrkQual_maxNearP   ->clear();   
  t_v_cone_hmaxNearP_goodTrk   ->clear();
  t_v_cone_hmaxNearP           ->clear();

  //  t_hScale             ->clear();
  t_trkNOuterHits      ->clear();
  t_trkNLayersCrossed  ->clear();
  t_dtFromLeadJet      ->clear();
  t_trkP               ->clear();
  t_trkPt              ->clear();
  t_trkEta             ->clear();
  t_trkPhi             ->clear();
  t_e3x3               ->clear();
  t_v_eDR              ->clear();
  t_v_eMipDR           ->clear();
  t_h3x3               ->clear();
  t_h5x5               ->clear();
  t_nRH_h3x3           ->clear();
  t_nRH_h5x5           ->clear();

  if (doMC) {
    t_simP               ->clear();
    t_hsim3x3            ->clear();
    t_hsim5x5            ->clear();
    t_hsim3x3Matched     ->clear();
    t_hsim5x5Matched     ->clear();
    t_hsim3x3Rest        ->clear();
    t_hsim5x5Rest        ->clear();
    t_hsim3x3Photon      ->clear();
    t_hsim5x5Photon      ->clear();
    t_hsim3x3NeutHad     ->clear();
    t_hsim5x5NeutHad     ->clear();
    t_hsim3x3CharHad     ->clear();
    t_hsim5x5CharHad     ->clear();
    t_hsim3x3PdgMatched  ->clear();
    t_hsim5x5PdgMatched  ->clear();
    t_hsim3x3Total       ->clear();
    t_hsim5x5Total       ->clear();

    t_hsim3x3NMatched    ->clear();
    t_hsim3x3NTotal      ->clear();
    t_hsim3x3NNeutHad    ->clear();
    t_hsim3x3NCharHad    ->clear();
    t_hsim3x3NPhoton      ->clear();
    t_hsim3x3NRest       ->clear();

    t_hsim5x5NMatched    ->clear();
    t_hsim5x5NTotal      ->clear();
    t_hsim5x5NNeutHad    ->clear();
    t_hsim5x5NCharHad    ->clear();
    t_hsim5x5NPhoton      ->clear();
    t_hsim5x5NRest       ->clear();

    t_trkHcalEne         ->clear();
    t_trkEcalEne         ->clear();
  }

  t_distFromHotCell_h3x3  ->clear();
  t_ietaFromHotCell_h3x3  ->clear();
  t_iphiFromHotCell_h3x3  ->clear();
  t_distFromHotCell_h5x5  ->clear();
  t_ietaFromHotCell_h5x5  ->clear();
  t_iphiFromHotCell_h5x5  ->clear();

  if (doMC) {
    t_v_hsimInfoConeMatched   ->clear();
    t_v_hsimInfoConeRest      ->clear();
    t_v_hsimInfoConePhoton     ->clear();
    t_v_hsimInfoConeNeutHad   ->clear();
    t_v_hsimInfoConeCharHad   ->clear();
    t_v_hsimInfoConePdgMatched->clear();  
    t_v_hsimInfoConeTotal     ->clear();

    t_v_hsimInfoConeNMatched   ->clear();
    t_v_hsimInfoConeNRest      ->clear();
    t_v_hsimInfoConeNPhoton     ->clear();
    t_v_hsimInfoConeNNeutHad   ->clear();
    t_v_hsimInfoConeNCharHad   ->clear();
    t_v_hsimInfoConeNTotal     ->clear();

    t_v_hsimCone    ->clear();
  }

  t_v_hCone       ->clear();
  t_v_nRecHitsCone->clear();
  t_v_nSimHitsCone->clear();

  t_v_distFromHotCell->clear();
  t_v_ietaFromHotCell->clear();
  t_v_iphiFromHotCell->clear();
  
  t_v_RH_h3x3_ieta ->clear();
  t_v_RH_h3x3_iphi ->clear();
  t_v_RH_h3x3_ene ->clear();
  t_v_RH_h5x5_ieta ->clear();
  t_v_RH_h5x5_iphi ->clear();
  t_v_RH_h5x5_ene ->clear();
  t_v_RH_r26_ieta  ->clear();
  t_v_RH_r26_iphi  ->clear();
  t_v_RH_r26_ene  ->clear();
  t_v_RH_r44_ieta  ->clear();
  t_v_RH_r44_iphi  ->clear();
  t_v_RH_r44_ene  ->clear();

  t_v_hlTriggers  ->clear();
  t_hltHB->clear();
  t_hltHE->clear();
  t_hltL1Jet15		     ->clear();
  t_hltJet30		     ->clear();
  t_hltJet50		     ->clear();
  t_hltJet80		     ->clear();
  t_hltJet110		     ->clear();
  t_hltJet140		     ->clear();
  t_hltJet180		     ->clear();
  t_hltL1SingleEG5	     ->clear();
  t_hltZeroBias		     ->clear();
  t_hltMinBiasHcal	     ->clear();
  t_hltMinBiasEcal	     ->clear();
  t_hltMinBiasPixel	     ->clear();
  t_hltSingleIsoTau30_Trk5     ->clear();
  t_hltDoubleLooseIsoTau15_Trk5->clear();
  
  t_irun->clear();
  t_ievt->clear();
  t_ilum->clear();



}

// ----- method called once each job just after ending the event loop ----
void IsolatedTracksCone::endJob() {

  std::cout << "Number of Events Processed " << nEVT << " failed L1 "
	    << nEVT_failL1 << std::endl;
  
}


void IsolatedTracksCone::BuildTree(){


  hRawPt  = fs->make<TH1F>("hRawPt ", "hRawPt ",  100,  0.0, 100.0);
  hRawP   = fs->make<TH1F>("hRawP  ", "hRawP  ",  100,  0.0, 100.0);
  hRawEta = fs->make<TH1F>("hRawEta", "hRawEta",   15,  0.0,   3.0);
  hRawPhi = fs->make<TH1F>("hRawPhi", "hRawPhi",  100, -3.2,   3.2);

  ntp = fs->make<TTree>("ntp", "ntp");

  
  // Counters
  ntp->Branch("nEVT"        , &nEVT        , "nEVT/I"        );
  ntp->Branch("leadL1JetPT" , &leadL1JetPT , "leadL1JetPT/D" );
  ntp->Branch("leadL1JetEta", &leadL1JetEta, "leadL1JetEta/D");
  ntp->Branch("leadL1JetPhi", &leadL1JetPhi, "leadL1JetPhi/D");
  ntp->Branch("nTRK",         &nTRK,         "nTRK/I");
  ntp->Branch("nRawTRK"            , &nRawTRK            ,"nRawTRK/I"            );
  ntp->Branch("nFailHighPurityQaul", &nFailHighPurityQaul,"nFailHighPurityQaul/I");
  ntp->Branch("nFailPt"            , &nFailPt            ,"nFailPt/I"            );
  ntp->Branch("nFailEta"           , &nFailEta           ,"nFailEta/I"           );
  ntp->Branch("nMissEcal"          , &nMissEcal          ,"nMissEcal/I"          );
  ntp->Branch("nMissHcal"          , &nMissHcal          ,"nMissHcal/I"          );

  ntp->Branch("hnNearTRKs"          ,"vector<vector<int> >   ",&t_v_hnNearTRKs          );
  ntp->Branch("hnLayers_maxNearP"   ,"vector<vector<int> >   ",&t_v_hnLayers_maxNearP   );
  ntp->Branch("htrkQual_maxNearP"   ,"vector<vector<int> >   ",&t_v_htrkQual_maxNearP   );
  ntp->Branch("hmaxNearP_goodTrk"   ,"vector<vector<double> >",&t_v_hmaxNearP_goodTrk   );
  ntp->Branch("hmaxNearP"           ,"vector<vector<double> >",&t_v_hmaxNearP           );

  ntp->Branch("cone_hnNearTRKs"       ,"vector<vector<int> >   ",&t_v_cone_hnNearTRKs       );
  ntp->Branch("cone_hnLayers_maxNearP","vector<vector<int> >   ",&t_v_cone_hnLayers_maxNearP);
  ntp->Branch("cone_htrkQual_maxNearP","vector<vector<int> >   ",&t_v_cone_htrkQual_maxNearP);
  ntp->Branch("cone_hmaxNearP_goodTrk","vector<vector<double> >",&t_v_cone_hmaxNearP_goodTrk);
  ntp->Branch("cone_hmaxNearP"        ,"vector<vector<double> >",&t_v_cone_hmaxNearP        );
				      				    
  //  ntp->Branch("hScale"           , "vector<double>", &t_hScale           );
  ntp->Branch("trkNOuterHits"    , "vector<double>", &t_trkNOuterHits    );
  ntp->Branch("trkNLayersCrossed", "vector<double>", &t_trkNLayersCrossed);
  ntp->Branch("drFromLeadJet"    , "vector<double>", &t_dtFromLeadJet    );
  ntp->Branch("trkP"             , "vector<double>", &t_trkP             );
  ntp->Branch("trkPt"            , "vector<double>", &t_trkPt            );
  ntp->Branch("trkEta"           , "vector<double>", &t_trkEta           );
  ntp->Branch("trkPhi"           , "vector<double>", &t_trkPhi           );
  ntp->Branch("e3x3"             , "vector<double>", &t_e3x3             );

  ntp->Branch("e3x3"          , "vector<double>"         , &t_e3x3 );
  ntp->Branch("v_eDR"         , "vector<vector<double> >", &t_v_eDR);
  ntp->Branch("v_eMipDR"      , "vector<vector<double> >", &t_v_eMipDR);

  ntp->Branch("h3x3"             , "vector<double>", &t_h3x3             );
  ntp->Branch("h5x5"             , "vector<double>", &t_h5x5             );
  ntp->Branch("nRH_h3x3"         , "vector<double>", &t_nRH_h3x3         );
  ntp->Branch("nRH_h5x5"         , "vector<double>", &t_nRH_h5x5         );

  if (doMC) {
    ntp->Branch("simP"             , "vector<double>", &t_simP             );
    ntp->Branch("hsim3x3"          , "vector<double>", &t_hsim3x3          );
    ntp->Branch("hsim5x5"          , "vector<double>", &t_hsim5x5          );

    ntp->Branch("hsim3x3Matched"   , "vector<double>", &t_hsim3x3Matched   );
    ntp->Branch("hsim5x5Matched"   , "vector<double>", &t_hsim5x5Matched   );
    ntp->Branch("hsim3x3Rest"      , "vector<double>", &t_hsim3x3Rest      );
    ntp->Branch("hsim5x5Rest"      , "vector<double>", &t_hsim5x5Rest      );
    ntp->Branch("hsim3x3Photon"    , "vector<double>", &t_hsim3x3Photon    );
    ntp->Branch("hsim5x5Photon"    , "vector<double>", &t_hsim5x5Photon    );
    ntp->Branch("hsim3x3NeutHad"   , "vector<double>", &t_hsim3x3NeutHad   );
    ntp->Branch("hsim5x5NeutHad"   , "vector<double>", &t_hsim5x5NeutHad   );
    ntp->Branch("hsim3x3CharHad"   , "vector<double>", &t_hsim3x3CharHad   );
    ntp->Branch("hsim5x5CharHad"   , "vector<double>", &t_hsim5x5CharHad   );
    ntp->Branch("hsim3x3PdgMatched", "vector<double>", &t_hsim3x3PdgMatched);
    ntp->Branch("hsim5x5PdgMatched", "vector<double>", &t_hsim5x5PdgMatched);
    ntp->Branch("hsim3x3Total"     , "vector<double>", &t_hsim3x3Total     );
    ntp->Branch("hsim5x5Total"     , "vector<double>", &t_hsim5x5Total     );

    ntp->Branch("hsim3x3NMatched"   , "vector<int>", &t_hsim3x3NMatched   );
    ntp->Branch("hsim3x3NRest"      , "vector<int>", &t_hsim3x3NRest      );
    ntp->Branch("hsim3x3NPhoton"    , "vector<int>", &t_hsim3x3NPhoton    );
    ntp->Branch("hsim3x3NNeutHad"   , "vector<int>", &t_hsim3x3NNeutHad   );
    ntp->Branch("hsim3x3NCharHad"   , "vector<int>", &t_hsim3x3NCharHad   );
    ntp->Branch("hsim3x3NTotal"     , "vector<int>", &t_hsim3x3NTotal     );

    ntp->Branch("hsim5x5NMatched"   , "vector<int>", &t_hsim5x5NMatched   );
    ntp->Branch("hsim5x5NRest"      , "vector<int>", &t_hsim5x5NRest      );
    ntp->Branch("hsim5x5NPhoton"    , "vector<int>", &t_hsim5x5NPhoton    );
    ntp->Branch("hsim5x5NNeutHad"   , "vector<int>", &t_hsim5x5NNeutHad   );
    ntp->Branch("hsim5x5NCharHad"   , "vector<int>", &t_hsim5x5NCharHad   );
    ntp->Branch("hsim5x5NTotal"     , "vector<int>", &t_hsim5x5NTotal     );

    ntp->Branch("trkHcalEne"       , "vector<double>", &t_trkHcalEne       );
    ntp->Branch("trkEcalEne"       , "vector<double>", &t_trkEcalEne       );
  }

  ntp->Branch("distFromHotCell_h3x3", "vector<double>", &t_distFromHotCell_h3x3);
  ntp->Branch("ietaFromHotCell_h3x3", "vector<int>", &t_ietaFromHotCell_h3x3);
  ntp->Branch("iphiFromHotCell_h3x3", "vector<int>", &t_iphiFromHotCell_h3x3);
  ntp->Branch("distFromHotCell_h5x5", "vector<double>", &t_distFromHotCell_h5x5);
  ntp->Branch("ietaFromHotCell_h5x5", "vector<int>", &t_ietaFromHotCell_h5x5);
  ntp->Branch("iphiFromHotCell_h5x5", "vector<int>", &t_iphiFromHotCell_h5x5);

  if (doMC) {
    ntp->Branch("v_hsimInfoConeMatched"   ,"vector<vector<double> >",&t_v_hsimInfoConeMatched   );
    ntp->Branch("v_hsimInfoConeRest"      ,"vector<vector<double> >",&t_v_hsimInfoConeRest      );
    ntp->Branch("v_hsimInfoConePhoton"     ,"vector<vector<double> >",&t_v_hsimInfoConePhoton     );
    ntp->Branch("v_hsimInfoConeNeutHad"   ,"vector<vector<double> >",&t_v_hsimInfoConeNeutHad   );
    ntp->Branch("v_hsimInfoConeCharHad"   ,"vector<vector<double> >",&t_v_hsimInfoConeCharHad   );
    ntp->Branch("v_hsimInfoConePdgMatched","vector<vector<double> >",&t_v_hsimInfoConePdgMatched);
    ntp->Branch("v_hsimInfoConeTotal"     ,"vector<vector<double> >",&t_v_hsimInfoConeTotal     );

    ntp->Branch("v_hsimInfoConeNMatched"  ,"vector<vector<int> >"   ,&t_v_hsimInfoConeNMatched   );
    ntp->Branch("v_hsimInfoConeNRest"     ,"vector<vector<int> >"   ,&t_v_hsimInfoConeNRest      );
    ntp->Branch("v_hsimInfoConeNPhoton"    ,"vector<vector<int> >"   ,&t_v_hsimInfoConeNPhoton     );
    ntp->Branch("v_hsimInfoConeNNeutHad"  ,"vector<vector<int> >"   ,&t_v_hsimInfoConeNNeutHad   );
    ntp->Branch("v_hsimInfoConeNCharHad"  ,"vector<vector<int> >"   ,&t_v_hsimInfoConeNCharHad   );
    ntp->Branch("v_hsimInfoConeNTotal"    ,"vector<vector<int> >"   ,&t_v_hsimInfoConeNTotal     );

    ntp->Branch("v_hsimCone"     ,"vector<vector<double> >",&t_v_hsimCone    );
  }

  ntp->Branch("v_hCone"        ,"vector<vector<double> >",&t_v_hCone       );
  ntp->Branch("v_nRecHitsCone" ,"vector<vector<int> >"   ,&t_v_nRecHitsCone);
  ntp->Branch("v_nSimHitsCone" ,"vector<vector<int> >"   ,&t_v_nSimHitsCone);

  ntp->Branch("v_distFromHotCell"        ,"vector<vector<double> >",&t_v_distFromHotCell       );
  ntp->Branch("v_ietaFromHotCell"        ,"vector<vector<int> >",&t_v_ietaFromHotCell       );
  ntp->Branch("v_iphiFromHotCell"        ,"vector<vector<int> >",&t_v_iphiFromHotCell       );

  ntp->Branch("v_RH_h3x3_ieta" ,"vector<vector<int> >"   ,&t_v_RH_h3x3_ieta);
  ntp->Branch("v_RH_h3x3_iphi" ,"vector<vector<int> >"   ,&t_v_RH_h3x3_iphi);
  ntp->Branch("v_RH_h3x3_ene"  ,"vector<vector<double> >",&t_v_RH_h3x3_ene );
  ntp->Branch("v_RH_h5x5_ieta" ,"vector<vector<int> >"   ,&t_v_RH_h5x5_ieta);
  ntp->Branch("v_RH_h5x5_iphi" ,"vector<vector<int> >"   ,&t_v_RH_h5x5_iphi);
  ntp->Branch("v_RH_h5x5_ene"  ,"vector<vector<double> >",&t_v_RH_h5x5_ene );
  ntp->Branch("v_RH_r26_ieta"  ,"vector<vector<int> >"   ,&t_v_RH_r26_ieta );
  ntp->Branch("v_RH_r26_iphi"  ,"vector<vector<int> >"   ,&t_v_RH_r26_iphi );
  ntp->Branch("v_RH_r26_ene"   ,"vector<vector<double> >",&t_v_RH_r26_ene  );
  ntp->Branch("v_RH_r44_ieta"  ,"vector<vector<int> >"   ,&t_v_RH_r44_ieta );
  ntp->Branch("v_RH_r44_iphi"  ,"vector<vector<int> >"   ,&t_v_RH_r44_iphi );
  ntp->Branch("v_RH_r44_ene"   ,"vector<vector<double> >",&t_v_RH_r44_ene  );

  ntp->Branch("v_hlTriggers" ,"vector<vector<int> >",&t_v_hlTriggers);
  ntp->Branch("v_hltHB" ,"vector<int>",&t_hltHB);
  ntp->Branch("v_hltHE" ,"vector<int>",&t_hltHE);
  ntp->Branch("v_hltL1Jet15" ,"vector<int>",&t_hltL1Jet15		     );
  ntp->Branch("v_hltJet30" ,"vector<int>",&t_hltJet30		     );
  ntp->Branch("v_hltJet50" ,"vector<int>",&t_hltJet50		     );
  ntp->Branch("v_hltJet80" ,"vector<int>",&t_hltJet80		     );
  ntp->Branch("v_hltJet110" ,"vector<int>",&t_hltJet110		     );
  ntp->Branch("v_hltJet140" ,"vector<int>",&t_hltJet140		     );
  ntp->Branch("v_hltJet180" ,"vector<int>",&t_hltJet180		     );
  ntp->Branch("v_hltL1SingleEG5" ,"vector<int>",&t_hltL1SingleEG5	     );
  ntp->Branch("v_hltZeroBias" ,"vector<int>",&t_hltZeroBias		     );
  ntp->Branch("v_hltMinBiasHcal" ,"vector<int>",&t_hltMinBiasHcal	     );
  ntp->Branch("v_hltMinBiasEcal" ,"vector<int>",&t_hltMinBiasEcal	     );
  ntp->Branch("v_hltMinBiasPixel" ,"vector<int>",&t_hltMinBiasPixel	     );
  ntp->Branch("v_hltSingleIsoTau30_Trk5" ,"vector<int>",&t_hltSingleIsoTau30_Trk5     );
  ntp->Branch("v_hltDoubleLooseIsoTau15_Trk5" ,"vector<int>",&t_hltDoubleLooseIsoTau15_Trk5);

  ntp->Branch("irun" ,"vector<unsigned int>", &t_irun);
  ntp->Branch("ievt" ,"vector<unsigned int>", &t_ievt);
  ntp->Branch("ilum" ,"vector<unsigned int>", &t_ilum);
   
}


void IsolatedTracksCone::printTrack(const reco::Track* pTrack) {
  
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
    
    for (int i=0; i<p.numberOfHits(reco::HitPattern::TRACK_HITS); i++) {
      p.printHitPattern(reco::HitPattern::TRACK_HITS, i, std::cout);
    }
  }

}

double IsolatedTracksCone::DeltaPhi(double v1, double v2) {
  // Computes the correctly normalized phi difference
  // v1, v2 = phi of object 1 and 2
  
  double pi    = 3.141592654;
  double twopi = 6.283185307;
  
  double diff = std::abs(v2 - v1);
  double corr = twopi - diff;
  if (diff < pi){ return diff;} else { return corr;} 
}


double IsolatedTracksCone::DeltaR(double eta1, double phi1, 
				  double eta2, double phi2) {
  double deta = eta1 - eta2;
  double dphi = DeltaPhi(phi1, phi2);
  return std::sqrt(deta*deta + dphi*dphi);
}


//define this as a plug-in
DEFINE_FWK_MODULE(IsolatedTracksCone);
	  
	  
	  
	  
	  

	  
	  






  

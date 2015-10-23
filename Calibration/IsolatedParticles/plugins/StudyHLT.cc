#include "StudyHLT.h"

//Triggers
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

StudyHLT::StudyHLT(const edm::ParameterSet& iConfig) : nRun(0) {
  verbosity                           = iConfig.getUntrackedParameter<int>("Verbosity",0);
  trigNames                           = iConfig.getUntrackedParameter<std::vector<std::string> >("Triggers");
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
  minTrackP                           =  iConfig.getUntrackedParameter<double>("MinTrackP", 1.0);
  maxTrackEta                         =  iConfig.getUntrackedParameter<double>("MaxTrackEta", 2.5);
  tMinE_                              = iConfig.getUntrackedParameter<double>("TimeMinCutECAL", -500.);
  tMaxE_                              = iConfig.getUntrackedParameter<double>("TimeMaxCutECAL",  500.);
  tMinH_                              = iConfig.getUntrackedParameter<double>("TimeMinCutHCAL", -500.);
  tMaxH_                              = iConfig.getUntrackedParameter<double>("TimeMaxCutHCAL",  500.);
  isItAOD                             = iConfig.getUntrackedParameter<bool>("IsItAOD", false);
  triggerEvent_                       = edm::InputTag("hltTriggerSummaryAOD","","HLT");
 theTriggerResultsLabel               = edm::InputTag("TriggerResults","","HLT");

  // define tokens for access
  tok_lumi      = consumes<LumiDetails, edm::InLumi>(edm::InputTag("lumiProducer"));
  tok_trigEvt   = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes   = consumes<edm::TriggerResults>(theTriggerResultsLabel);
  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_   = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  if (isItAOD) {
    tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));
    tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("reducedHcalRecHits", "hbhereco"));
  } else {
    tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
    tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEE"));
    tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  }
  
  edm::LogInfo("IsoTrack") << "Verbosity " << verbosity << " with " 
			   << trigNames.size() << " triggers:";
  for (unsigned int k=0; k<trigNames.size(); ++k)
    edm::LogInfo("IsoTrack") << " [" << k << "] " << trigNames[k];
  edm::LogInfo("IsoTrack") << "TrackQuality " << theTrackQuality << " Minpt "
			   << selectionParameters.minPt << " maxDxy " 
			   << selectionParameters.maxDxyPV << " maxDz "
			   << selectionParameters.maxDzPV << " maxChi2 "
			   << selectionParameters.maxChi2 << " maxDp/p "
			   << selectionParameters.maxDpOverP << " minOuterHit "
			   << selectionParameters.minOuterHit << " minLayerCrossed "
			   << selectionParameters.minLayerCrossed << " maxInMiss "
			   << selectionParameters.maxInMiss << " maxOutMiss " 
			   << selectionParameters.maxOutMiss << " minTrackP "
			   << minTrackP << " maxTrackEta " << maxTrackEta 
			   << " tMinE_ " << tMinE_ << " tMaxE " << tMaxE_ 
			   << " tMinH_ " << tMinH_ << " tMaxH_ " << tMaxH_ 
			   << " isItAOD " << isItAOD;

  double pBins[nPBin+1] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,9.0,11.0,15.0,20.0};
  int    etaBins[nEtaBin+1] = {1, 7, 13, 17, 23};
  int    pvBins[nPVBin+1] = {1, 2, 3, 5, 100};
  for (int i=0; i<=nPBin; ++i)   pBin[i]   = pBins[i];
  for (int i=0; i<=nEtaBin; ++i) etaBin[i] = etaBins[i];
  for (int i=0; i<=nPVBin; ++i)  pvBin[i]  = pvBins[i];
}

StudyHLT::~StudyHLT() {}

void StudyHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
if (verbosity > 0) 
  edm::LogInfo("IsoTrack") << "Event starts===================================="; 
  int RunNo = iEvent.id().run();
  int EvtNo = iEvent.id().event();
  int Lumi  = iEvent.luminosityBlock();
  int Bunch = iEvent.bunchCrossing();
  
  edm::Handle<LumiDetails> Lumid;
  iEvent.getLuminosityBlock().getByToken(tok_lumi,Lumid);
  
  std::string newNames[5]={"HLT","PixelTracks_Multiplicity","HLT_Physics_","HLT_JetE","HLT_ZeroBias"};
  int         newAccept[5];
  for (int i=0; i<5; ++i) newAccept[i] = 0;
  float mybxlumi=-1;
  if (Lumid.isValid()) 
    mybxlumi=Lumid->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing())*6.37;
  
  if (verbosity > 0)
    edm::LogInfo("IsoTrack") << "RunNo " << RunNo << " EvtNo " << EvtNo 
			     << " Lumi " << Lumi << " Bunch " << Bunch 
			     << " mybxlumi " << mybxlumi;
  
  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt,triggerEventHandle);
  
  bool ok(false);
  if (trigNames.size() < 1) {
    ok = true;
  } else if (!triggerEventHandle.isValid()) {
    edm::LogWarning("IsoTrack") << "Error! Can't get the product "
				<< triggerEvent_.label();
  } else {
    triggerEvent = *(triggerEventHandle.product());
    
    /////////////////////////////TriggerResults
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByToken(tok_trigRes, triggerResults);

    if (triggerResults.isValid()) {
      h_nHLT->Fill(triggerResults->size());
      h_nHLTvsRN->Fill(RunNo, triggerResults->size());

      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);      
      const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
      for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
	//        unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[iHLT]);
	//        const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(triggerindx));
        int ipos=-1;
	std::string newtriggerName = truncate_str(triggerNames_[iHLT]);
	for (unsigned int i=0; i<HLTNames.size(); ++i) {
	  if (newtriggerName == HLTNames[i]) {
	    ipos = i+1;
	    break;
	  }
	}
	if (ipos < 0) {
	  HLTNames.push_back(newtriggerName);
	  ipos = (int)(HLTNames.size());
	  if (ipos <= h_HLTAccept->GetNbinsX())
	    h_HLTAccept->GetXaxis()->SetBinLabel(ipos,newtriggerName.c_str());
	}
	if ((int)(iHLT+1) > h_HLTAccepts[nRun]->GetNbinsX()) {
	  edm::LogInfo("IsoTrack") << "Wrong trigger " << RunNo << " Event " 
				   << EvtNo << " Hlt " << iHLT;
	} else {
	  if (firstEvent)  h_HLTAccepts[nRun]->GetXaxis()->SetBinLabel(iHLT+1, newtriggerName.c_str());
	}
	int hlt    = triggerResults->accept(iHLT);
	if (hlt) {
	  h_HLTAccepts[nRun]->Fill(iHLT+1);
	  h_HLTAccept->Fill(ipos);
	}
	for (unsigned int i=0; i<trigNames.size(); ++i) {
	  if (newtriggerName.find(trigNames[i].c_str())!=std::string::npos) {
	    if (verbosity%10 > 0)  
	      edm::LogInfo("IsoTrack") << newtriggerName;
	    if (hlt > 0) ok = true;
	  }
	}
	for (int i=0; i<5; ++i) {
	  if (newtriggerName.find(newNames[i].c_str())!=std::string::npos) {
	    if (verbosity%10 > 0)
	      edm::LogInfo("IsoTrack") << "[" << i << "] " << newNames[i] 
				       << " : " << newtriggerName;
	    if (hlt > 0) newAccept[i] = 1;
	  }
	}
      }
      int iflg(0), indx(1);
      for (int i=0; i<5; ++i) {
	iflg += (indx*newAccept[i]); indx *= 2;
      }
      h_HLTCorr->Fill(iflg);
    }
  }

  //Look at the tracks  
  if (ok) {
    h_goodRun->Fill(RunNo);
    // get handles to calogeometry and calotopology
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    const CaloGeometry* geo = pG.product();
  
    edm::ESHandle<CaloTopology> theCaloTopology;
    iSetup.get<CaloTopologyRecord>().get(theCaloTopology); 
    const CaloTopology *caloTopology = theCaloTopology.product();
  
    edm::ESHandle<HcalTopology> htopo;
    iSetup.get<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* theHBHETopology = htopo.product();
 
    edm::ESHandle<MagneticField> bFieldH;
    iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
    const MagneticField *bField = bFieldH.product();

    edm::ESHandle<EcalChannelStatus> ecalChStatus;
    iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus);
    const EcalChannelStatus* theEcalChStatus = ecalChStatus.product();

    edm::Handle<reco::VertexCollection> recVtxs;
    iEvent.getByToken(tok_recVtx_,recVtxs);
    int                                   ntrk(0), ngoodPV(0), nPV(-1);
    for (unsigned int ind=0; ind<recVtxs->size(); ind++) {
      if (!((*recVtxs)[ind].isFake()) && (*recVtxs)[ind].ndof() > 4) ngoodPV++;
    }
    for (int i=0; i<nPVBin; ++i) {
      if (ngoodPV >= pvBin[i] && ngoodPV < pvBin[i+1]) {
	nPV = i; break;
      }
    }
    if ((verbosity/10)%10 > 0) 
      edm::LogInfo("IsoTrack") << "Number of vertices: " << recVtxs->size() 
			       << " Good " << ngoodPV << " Bin " << nPV;
    h_numberPV->Fill((int)(recVtxs->size()));
    h_goodPV->Fill(ngoodPV);
    
    
    edm::Handle<reco::TrackCollection> trkCollection;
    iEvent.getByToken(tok_genTrack_, trkCollection);
    reco::TrackCollection::const_iterator trkItr;
    for (trkItr=trkCollection->begin(); trkItr != trkCollection->end(); ++trkItr,++ntrk) {
      const reco::Track* pTrack = &(*trkItr);
      double pt1         = pTrack->pt();
      double p1          = pTrack->p();
      double eta1        = pTrack->momentum().eta();
      double phi1        = pTrack->momentum().phi();
      bool quality       = pTrack->quality(selectionParameters.minQuality);
      fillTrack(0, pt1,p1,eta1,phi1);
      if (quality) fillTrack(1, pt1,p1,eta1,phi1);
    }
    h_ntrk[0]->Fill(ntrk);

    std::vector<spr::propagatedTrackID> trkCaloDets;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDets, ((verbosity/100)%10 > 0));
    std::vector<spr::propagatedTrackID>::const_iterator trkDetItr;
    for (trkDetItr = trkCaloDets.begin(),ntrk=0; trkDetItr != trkCaloDets.end(); trkDetItr++,ntrk++) {
      const reco::Track* pTrack = &(*(trkDetItr->trkItr));
      double pt1         = pTrack->pt();
      double p1          = pTrack->p();
      double eta1        = pTrack->momentum().eta();
      double phi1        = pTrack->momentum().phi();
      if ((verbosity/10)%10 > 0) 
	edm::LogInfo("IsoTrack") << "track: p " << p1 << " pt " << pt1 
				 << " eta " << eta1 << " phi " << phi1 
				 << " okEcal " << trkDetItr->okECAL;
      fillTrack(2, pt1,p1,eta1,phi1);
      if (pt1>minTrackP && std::abs(eta1)<maxTrackEta && trkDetItr->okECAL) { 
	fillTrack(3, pt1,p1,eta1,phi1);
	double maxNearP31x31 = spr::chargeIsolationEcal(ntrk, trkCaloDets, geo, caloTopology, 15, 15, ((verbosity/1000)%10 > 0));
	
	edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
	iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  
	edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
	edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
	iEvent.getByToken(tok_EB_,barrelRecHitsHandle);
	iEvent.getByToken(tok_EE_,endcapRecHitsHandle);
	// get ECal Tranverse Profile
	std::pair<double, bool>  e7x7P, e11x11P, e15x15P;
	const DetId isoCell = trkDetItr->detIdECAL;
	e7x7P   = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,sevlv.product(),3,3, 0.030, 0.150, tMinE_,tMaxE_, ((verbosity/10000)%10 > 0));
	e11x11P = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,sevlv.product(),5,5, 0.030, 0.150, tMinE_,tMaxE_, ((verbosity/10000)%10 > 0));
	e15x15P = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,sevlv.product(),7,7, 0.030, 0.150, tMinE_,tMaxE_, ((verbosity/10000)%10 > 0));

	double  maxNearHcalP7x7 = spr::chargeIsolationHcal(ntrk, trkCaloDets, theHBHETopology, 3,3, ((verbosity/1000)%10 > 0));
	int    ieta(0);
	double h3x3(0), h5x5(0), h7x7(0);
	fillIsolation(0, maxNearP31x31,e11x11P.first,e15x15P.first);
	if ((verbosity/10)%10 > 0) 
	  edm::LogInfo("IsoTrack") << "Accepted Tracks reaching Ecal maxNearP31x31 " 
				   << maxNearP31x31 << " e11x11P " 
				   << e11x11P.first << " e15x15P " 
				   << e15x15P.first << " okHCAL " 
				   << trkDetItr->okHCAL;

	if (trkDetItr->okHCAL) {
	  edm::Handle<HBHERecHitCollection> hbhe;
	  iEvent.getByToken(tok_hbhe_, hbhe);
	  const DetId ClosestCell(trkDetItr->detIdHCAL);
	  ieta = ((HcalDetId)(ClosestCell)).ietaAbs();
	  h3x3 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,1,1, false, true, 0.7, 0.8, -100.0, -100.0, tMinH_,tMaxH_, ((verbosity/10000)%10 > 0));  
	  h5x5 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,2,2, false, true, 0.7, 0.8, -100.0, -100.0, tMinH_,tMaxH_, ((verbosity/10000)%10 > 0) );  
	  h7x7 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,3,3, false, true, 0.7, 0.8, -100.0, -100.0, tMinH_,tMaxH_, ((verbosity/10000)%10 > 0) );  
	  fillIsolation(1, maxNearHcalP7x7,h5x5,h7x7);
	  if ((verbosity/10)%10 > 0)
	    edm::LogInfo("IsoTrack") << "Tracks Reaching Hcal maxNearHcalP7x7/h5x5/h7x7 " 
				     << maxNearHcalP7x7 << "/" << h5x5 << "/" << h7x7; 
	}
	if (maxNearP31x31 < 0) {
	  fillTrack(4, pt1,p1,eta1,phi1);
	  fillEnergy(0,ieta,p1,e7x7P.first,h3x3,e11x11P.first,h5x5);
	  if (maxNearHcalP7x7 < 0) {
	    fillTrack(5, pt1,p1,eta1,phi1);
	    fillEnergy(1,ieta,p1,e7x7P.first,h3x3,e11x11P.first,h5x5);
	    if (e11x11P.second && e15x15P.second && (e15x15P.first-e11x11P.first)<2.0) {
	      fillTrack(6, pt1,p1,eta1,phi1);
	      fillEnergy(2,ieta,p1,e7x7P.first,h3x3,e11x11P.first,h5x5);
	      if (h7x7-h5x5 < 2.0) {
		fillTrack(7, pt1,p1,eta1,phi1);
		fillEnergy(3,ieta,p1,e7x7P.first,h3x3,e11x11P.first,h5x5);
		if (nPV >= 0) {
		  fillTrack(nPV+8, pt1,p1,eta1,phi1);
		  fillEnergy(nPV+4,ieta,p1,e7x7P.first,h3x3,e11x11P.first,h5x5);
		}
	      }
	    }
	  }
	}
      }
    }
    h_ntrk[1]->Fill(ntrk);
  }
  firstEvent = false;
}

void StudyHLT::beginJob() {
  h_nHLT        = fs->make<TH1I>("h_nHLT" , "size of trigger Names", 1000, 0, 1000);
  h_HLTAccept   = fs->make<TH1I>("h_HLTAccept", "HLT Accepts for all runs", 500, 0, 500);
  for (int i=1; i<=500; ++i) h_HLTAccept->GetXaxis()->SetBinLabel(i," ");
  h_nHLTvsRN    = fs->make<TH2I>("h_nHLTvsRN" , "size of trigger Names vs RunNo", 2168, 190949, 193116, 100, 400, 500);
  h_HLTCorr     = fs->make<TH1I>("h_HLTCorr", "Correlation among different paths", 100, 0, 100);
  h_numberPV    = fs->make<TH1I>("h_numberPV", "Number of Primary Vertex", 100, 0, 100);
  h_goodPV      = fs->make<TH1I>("h_goodPV", "Number of good Primary Vertex", 100, 0, 100);
  h_goodRun     = fs->make<TH1I>("h_goodRun","Number of accepted events for Run", 4000, 190000, 1940000);
  char hname[50], htit[400];
  std::string CollectionNames[2] = {"Reco", "Propagated"};
  for (unsigned int i=0; i<2; i++) {
    sprintf(hname, "h_nTrk_%s", CollectionNames[i].c_str());
    sprintf(htit, "Number of %s tracks", CollectionNames[i].c_str());
    h_ntrk[i] = fs->make<TH1I>(hname, htit, 500, 0, 500);
  }
  std::string TrkNames[8]       = {"All", "Quality", "NoIso", "okEcal", "EcalCharIso", "HcalCharIso", "EcalNeutIso", "HcalNeutIso"};
  for (unsigned int i=0; i<8+nPVBin; i++) {
    if (i < 8) {
      sprintf(hname, "h_pt_%s", TrkNames[i].c_str());
      sprintf(htit, "p_{T} of %s tracks", TrkNames[i].c_str());
    } else {
      sprintf(hname, "h_pt_%s_%d", TrkNames[7].c_str(), i-8);
      sprintf(htit, "p_{T} of %s tracks (PV=%d:%d)", TrkNames[7].c_str(), pvBin[i-8], pvBin[i-7]-1);
    }
    h_pt[i]   = fs->make<TH1D>(hname, htit, 400, 0, 200.0);
    h_pt[i]->Sumw2();

    if (i < 8) {
      sprintf(hname, "h_p_%s", TrkNames[i].c_str());
      sprintf(htit, "Momentum of %s tracks", TrkNames[i].c_str());
    } else {
      sprintf(hname, "h_p_%s_%d", TrkNames[7].c_str(), i-8);
      sprintf(htit, "Momentum of %s tracks (PV=%d:%d)", TrkNames[7].c_str(), pvBin[i-8], pvBin[i-7]-1);
    }
    h_p[i]    = fs->make<TH1D>(hname, htit, 400, 0, 200.0);
    h_p[i]->Sumw2();

    if (i < 8) {
      sprintf(hname, "h_eta_%s", TrkNames[i].c_str());
      sprintf(htit, "Eta of %s tracks", TrkNames[i].c_str());
    } else {
      sprintf(hname, "h_eta_%s_%d", TrkNames[7].c_str(), i-8);
      sprintf(htit, "Eta of %s tracks (PV=%d:%d)", TrkNames[7].c_str(), pvBin[i-8], pvBin[i-7]-1);
    }
    h_eta[i]  = fs->make<TH1D>(hname, htit, 60, -3.0, 3.0);
    h_eta[i]->Sumw2();

    if (i < 8) {
      sprintf(hname, "h_phi_%s", TrkNames[i].c_str());
      sprintf(htit, "Phi of %s tracks", TrkNames[i].c_str());
    } else {
      sprintf(hname, "h_phi_%s_%d", TrkNames[7].c_str(), i-8);
      sprintf(htit, "Phi of %s tracks (PV=%d:%d)", TrkNames[7].c_str(), pvBin[i-8], pvBin[i-7]-1);
    }
    h_phi[i]  = fs->make<TH1D>(hname, htit, 100, -3.15, 3.15);
    h_phi[i]->Sumw2();
  }
  std::string IsolationNames[2] = {"Ecal", "Hcal"};
  for (unsigned int i=0; i<2; i++) {
    sprintf(hname, "h_maxNearP_%s", IsolationNames[i].c_str());
    sprintf(htit, "Energy in ChargeIso region for %s", IsolationNames[i].c_str());
    h_maxNearP[i] = fs->make<TH1D>(hname, htit, 120, -1.5, 10.5);
    h_maxNearP[i]->Sumw2(); 

    sprintf(hname, "h_ene1_%s", IsolationNames[i].c_str());
    sprintf(htit, "Energy in smaller cone for %s", IsolationNames[i].c_str());
    h_ene1[i]     = fs->make<TH1D>(hname, htit, 400, 0.0, 200.0);
    h_ene1[i]->Sumw2();

    sprintf(hname, "h_ene2_%s", IsolationNames[i].c_str());
    sprintf(htit, "Energy in bigger cone for %s", IsolationNames[i].c_str());
    h_ene2[i]     = fs->make<TH1D>(hname, htit, 400, 0.0, 200.0);
    h_ene2[i]->Sumw2(); 

    sprintf(hname, "h_ediff_%s", IsolationNames[i].c_str());
    sprintf(htit, "Energy in NeutralIso region for %s", IsolationNames[i].c_str());
    h_ediff[i]      = fs->make<TH1D>(hname, htit, 100, -0.5, 19.5);
    h_ediff[i]->Sumw2();
  }
  std::string energyNames[6]={"E_{7x7}", "H_{3x3}", "(E_{7x7}+H_{3x3})",
			      "E_{11x11}", "H_{5x5}", "{E_{11x11}+H_{5x5})"};
  for (int i=0; i<4+nPVBin; ++i) {
    for (int ip=0; ip<nPBin; ++ip) {
      for (int ie=0; ie<nEtaBin; ++ie) {
	for (int j=0; j<6; ++j) {
	  sprintf(hname, "h_energy_%d_%d_%d_%d", i, ip, ie, j);
	  if (i < 4) {
	    sprintf(htit,"%s/p (p=%4.1f:%4.1f; i#eta=%d:%d) for tracks with %s",
		    energyNames[j].c_str(),pBin[ip],pBin[ip+1],etaBin[ie],
		    (etaBin[ie+1]-1), TrkNames[i+4].c_str());
	  } else {
	    sprintf(htit,"%s/p (p=%4.1f:%4.1f; i#eta=%d:%d, PV=%d:%d) for tracks with %s",
		    energyNames[j].c_str(),pBin[ip],pBin[ip+1],etaBin[ie],
		    (etaBin[ie+1]-1), pvBin[i-4], pvBin[i-3],
		    TrkNames[7].c_str());
	  }
	  h_energy[i][ip][ie][j] = fs->make<TH1D>(hname, htit, 500, -0.1, 4.9);
	  h_energy[i][ip][ie][j]->Sumw2();
	}
      }
    }
  }
}

void StudyHLT::endJob() {}

// ------------ method called when starting to processes a run  ------------
void StudyHLT::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  char hname[100], htit[400];
  edm::LogInfo("IsoTrack")  << "Run[" << nRun << "] " << iRun.run() << " hltconfig.init " 
			    << hltConfig_.init(iRun,iSetup,"HLT",changed);
  sprintf(hname, "h_HLTAccepts_%i", iRun.run());
  sprintf(htit, "HLT Accepts for Run No %i", iRun.run());
  TH1I *hnew = fs->make<TH1I>(hname, htit, 500, 0, 500);
  for (int i=1; i<=500; ++i) hnew->GetXaxis()->SetBinLabel(i," ");
  h_HLTAccepts.push_back(hnew);
  edm::LogInfo("IsoTrack") << "beginrun " << iRun.run();
  firstEvent = true;
}

// ------------ method called when ending the processing of a run  ------------
void StudyHLT::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun++;
  edm::LogInfo("IsoTrack") << "endrun[" << nRun << "] " << iRun.run();
}

// ------------ method called when starting to processes a luminosity block  ------------
void StudyHLT::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method called when ending the processing of a luminosity block  ------------
void StudyHLT::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

void StudyHLT::fillTrack(int i, double pt, double p, double eta, double phi){
  h_pt[i]->Fill(pt);
  h_p[i]->Fill(p);
  h_eta[i]->Fill(eta);
  h_phi[i]->Fill(phi);
}

void StudyHLT::fillIsolation(int i, double emaxnearP, double eneutIso1, double eneutIso2){
  h_maxNearP[i]->Fill(emaxnearP);
  h_ene1[i]->Fill(eneutIso1);
  h_ene2[i]->Fill(eneutIso2);
  h_ediff[i]->Fill(eneutIso2-eneutIso1);
}

void StudyHLT::fillEnergy(int flag, int ieta, double p, double enEcal1,
			  double enHcal1, double enEcal2, double enHcal2) {
  int ip(-1), ie(-1);
  for (int i=0; i<nPBin; ++i) {
    if (p >= pBin[i] && p < pBin[i+1]) { ip = i; break; }
  }
  for (int i=0; i<nEtaBin; ++i) {
    if (ieta >= etaBin[i] && ieta < etaBin[i+1]) { ie = i; break; }
  }
  if (ip >= 0 && ie >= 0 && enEcal1 > 0.02 && enHcal1 > 0.1) {
    h_energy[flag][ip][ie][0]->Fill(enEcal1/p);
    h_energy[flag][ip][ie][1]->Fill(enHcal1/p);
    h_energy[flag][ip][ie][2]->Fill((enEcal1+enHcal1)/p);
    h_energy[flag][ip][ie][3]->Fill(enEcal2/p);
    h_energy[flag][ip][ie][4]->Fill(enHcal2/p);
    h_energy[flag][ip][ie][5]->Fill((enEcal2+enHcal2)/p);
  }
}

std::string StudyHLT::truncate_str(const std::string& str){
  std::string truncated_str(str);
  int length = str.length();
  for (int i=0; i<length-2; i++){
    if (str[i]=='_' && str[i+1]=='v' && isdigit(str.at(i+2))) {
      int z = i+1;
      truncated_str = str.substr(0,z);
    } 
  }
  return(truncated_str);
}

DEFINE_FWK_MODULE(StudyHLT);


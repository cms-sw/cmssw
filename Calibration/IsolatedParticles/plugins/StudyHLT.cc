#include "StudyHLT.h"

//Triggers
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

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
}

StudyHLT::~StudyHLT(){
}

void StudyHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (verbosity > 0) std::cout << "Event starts====================================" << std::endl;
  int RunNo = iEvent.id().run();
  int EvtNo = iEvent.id().event();
  int Lumi  = iEvent.luminosityBlock();
  int Bunch = iEvent.bunchCrossing();
  
  edm::InputTag lumiProducer("LumiProducer", "", "RECO");
  edm::Handle<LumiDetails> Lumid;
  iEvent.getLuminosityBlock().getByLabel("lumiProducer",Lumid); 
  
  float mybxlumi=-1;
  if (Lumid.isValid()) 
    mybxlumi=Lumid->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing())*6.37;
  
  if (verbosity > 0)
    std::cout << "RunNo " << RunNo << " EvtNo " << EvtNo << " Lumi " << Lumi 
	      << " Bunch " << Bunch << " mybxlumi " << mybxlumi << std::endl;
  
  edm::InputTag triggerEvent_ ("hltTriggerSummaryAOD","","HLT");
  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByLabel(triggerEvent_,triggerEventHandle);
  
  if (!triggerEventHandle.isValid())
    std::cout << "Error! Can't get the product "<< triggerEvent_.label() << std::endl;
  else {
    triggerEvent = *(triggerEventHandle.product());
    
    /////////////////////////////TriggerResults
    edm::InputTag theTriggerResultsLabel ("TriggerResults","","HLT");
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByLabel( theTriggerResultsLabel, triggerResults);

    if (triggerResults.isValid()) {
      h_nHLT->Fill(triggerResults->size());
      h_nHLTvsRN->Fill(RunNo, triggerResults->size());

      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);      
      const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
      bool ok(false);
      for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
	//        unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[iHLT]);
	//        const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(triggerindx));
	int ipos=-1;
	for (unsigned int i=0; i<HLTNames.size(); ++i) {
	  if (triggerNames_[iHLT] == HLTNames[i]) {
	    ipos = i;
	    break;
	  }
	}
	if (ipos < 0) {
	  ipos = (int)(HLTNames.size()+1);
	  HLTNames.push_back(triggerNames_[iHLT]);
	  h_HLTAccept->GetXaxis()->SetBinLabel(ipos+1,triggerNames_[iHLT].c_str());
	}
	if (firstEvent)  h_HLTAccepts[nRun]->GetXaxis()->SetBinLabel(iHLT+1, triggerNames_[iHLT].c_str());
	int hlt    = triggerResults->accept(iHLT);
	if (hlt) {
	  h_HLTAccepts[nRun]->Fill(iHLT+1);
	  h_HLTAccept->Fill(ipos+1);
	}
	if (iHLT >= 499) std::cout << "Wrong trigger " << RunNo << " Event " << EvtNo << " Hlt " << iHLT << std::endl;
	for (unsigned int i=0; i<trigNames.size(); ++i) {
	  if (triggerNames_[iHLT].find(trigNames[i].c_str())!=std::string::npos) {
	    if(verbosity)  std::cout << triggerNames_[iHLT] << std::endl;
	    if (hlt > 0) ok = true;
	  }
	}
      }
      if (ok) {
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
 
	edm::ESHandle<MagneticField> bFieldH;
	iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
	const MagneticField *bField = bFieldH.product();

	edm::ESHandle<EcalChannelStatus> ecalChStatus;
	iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus);
	const EcalChannelStatus* theEcalChStatus = ecalChStatus.product();

	edm::Handle<reco::TrackCollection> trkCollection;
	iEvent.getByLabel("generalTracks", trkCollection);
	reco::TrackCollection::const_iterator trkItr;
	int                                   ntrk = 0;
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
	spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDets, false);
	std::vector<spr::propagatedTrackID>::const_iterator trkDetItr;
	for (trkDetItr = trkCaloDets.begin(),ntrk=0; trkDetItr != trkCaloDets.end(); trkDetItr++,ntrk++) {
	  const reco::Track* pTrack = &(*(trkDetItr->trkItr));
	  double pt1         = pTrack->pt();
	  double p1          = pTrack->p();
	  double eta1        = pTrack->momentum().eta();
	  double phi1        = pTrack->momentum().phi();
	  if(verbosity > 0) std::cout << "track (p/pt/eta/phi/okEcal) : " << p1 << "/" << pt1 << "/" << eta1 << "/" << phi1 <<  "/" << trkDetItr->okECAL << std::endl;
	  fillTrack(2, pt1,p1,eta1,phi1);

	  if (pt1>minTrackP && std::abs(eta1)<maxTrackEta && trkDetItr->okECAL) { 
	    fillTrack(3, pt1,p1,eta1,phi1);
	    double maxNearP31x31 = spr::chargeIsolationEcal(ntrk, trkCaloDets, geo, caloTopology, 15, 15);

	    edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
	    iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  
	    edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
	    edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
	    iEvent.getByLabel("ecalRecHit","EcalRecHitsEB",barrelRecHitsHandle);
	    iEvent.getByLabel("ecalRecHit","EcalRecHitsEE",endcapRecHitsHandle);
	    // get ECal Tranverse Profile
	    std::pair<double, bool>  e11x11P, e15x15P;
	    const DetId isoCell = trkDetItr->detIdECAL;
	    e11x11P = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,sevlv.product(),5,5,   0.060,  0.300, tMinE_,tMaxE_);
	    e15x15P = spr::eECALmatrix(isoCell,barrelRecHitsHandle,endcapRecHitsHandle, *theEcalChStatus, geo, caloTopology,sevlv.product(),7,7,   0.060,  0.300, tMinE_,tMaxE_);

	    double  maxNearHcalP7x7 = spr::chargeIsolationHcal(ntrk, trkCaloDets, theHBHETopology, 3,3);
	    double h5x5=0,    h7x7=0;
	    fillIsolation(0, maxNearP31x31,e11x11P.first,e15x15P.first);
	    if(verbosity > 0) std::cout << "Accepted Tracks reaching Ecal maxNearP31x31/e11x11P/e15x15P/okHCAL " << maxNearP31x31 << "/" << e11x11P.first << "/" << e15x15P.first << "/" << trkDetItr->okHCAL << std::endl;

	    if (trkDetItr->okHCAL) {
	      edm::Handle<HBHERecHitCollection> hbhe;
	      iEvent.getByLabel("hbhereco", hbhe);

	      const DetId ClosestCell(trkDetItr->detIdHCAL);
	      // bool includeHO=false, bool algoNew=true, bool debug=false
	      h5x5 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,2,2, false, true, -100.0, -100.0, -100.0, -100.0, tMinH_,tMaxH_);  
	      h7x7 = spr::eHCALmatrix(theHBHETopology, ClosestCell, hbhe,3,3, false, true, -100.0, -100.0, -100.0, -100.0, tMinH_,tMaxH_);  
	      fillIsolation(1, maxNearHcalP7x7,h5x5,h7x7);
	      if(verbosity) std::cout << "Tracks Reaching Hcal maxNearHcalP7x7/h5x5/h7x7 " << maxNearHcalP7x7 << "/" << h5x5 << "/" << h7x7 << std::endl;
	    }
	    if (maxNearP31x31 < 0) {
	      fillTrack(4, pt1,p1,eta1,phi1);
	      if (maxNearHcalP7x7 < 0) {
		fillTrack(5, pt1,p1,eta1,phi1);
		if (e11x11P.second && e15x15P.second && (e15x15P.first-e11x11P.first)<2.0) {
		  fillTrack(6, pt1,p1,eta1,phi1);
		  if (h7x7-h5x5 < 2.0) {
		    fillTrack(7, pt1,p1,eta1,phi1);
		  }
		}
	      }
	    }
	  }
	}
	h_ntrk[1]->Fill(ntrk);
      }
    }
  }
  firstEvent = false;
}

void StudyHLT::beginJob() {
  h_nHLT        = fs->make<TH1I>("h_nHLT" , "size of trigger Names", 1000, 0, 1000);
  h_HLTAccept   = fs->make<TH1I>("h_HLTAccept", "HLT Accepts for all runs", 500, 0, 500);
  h_nHLTvsRN    = fs->make<TH2I>("h_nHLTvsRN" , "size of trigger Names vs RunNo", 2168, 190949, 193116, 100, 400, 500);

  char hname[50], htit[100];
  std::string CollectionNames[2] = {"Reco", "Propagated"};
  for(unsigned int i=0; i<2; i++) {
    sprintf(hname, "h_nTrk_%s", CollectionNames[i].c_str());
    sprintf(htit, "Number of %s tracks", CollectionNames[i].c_str());
    h_ntrk[i] = fs->make<TH1I>(hname, htit, 500, 0, 500);
  }
  std::string TrkNames[8]       = {"All", "Quality", "NoIso", "okEcal", "EcalCharIso", "HcalCharIso", "EcalNeutIso", "HcalNeutIso"};
  for(unsigned int i=0; i<8; i++) {
    sprintf(hname, "h_pt_%s", TrkNames[i].c_str());
    sprintf(htit, "p_{T} of %s tracks", TrkNames[i].c_str());
    h_pt[i]   = fs->make<TH1D>(hname, htit, 400, 0, 200.0);
    h_pt[i]->Sumw2();

    sprintf(hname, "h_p_%s", TrkNames[i].c_str());
    sprintf(htit, "Momentum of %s tracks", TrkNames[i].c_str());
    h_p[i]    = fs->make<TH1D>(hname, htit, 400, 0, 200.0);
    h_p[i]->Sumw2();

    sprintf(hname, "h_eta_%s", TrkNames[i].c_str());
    sprintf(htit, "Eta of %s tracks", TrkNames[i].c_str());
    h_eta[i]  = fs->make<TH1D>(hname, htit, 60, -3.0, 3.0);
    h_eta[i]->Sumw2();

    sprintf(hname, "h_phi_%s", TrkNames[i].c_str());
    sprintf(htit, "Phi of %s tracks", TrkNames[i].c_str());
    h_phi[i]  = fs->make<TH1D>(hname, htit, 100, -3.15, 3.15);
    h_phi[i]->Sumw2();
  }
  std::string IsolationNames[2] = {"Ecal", "Hcal"};
  for(unsigned int i=0; i<2; i++) {
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
}

void StudyHLT::endJob() {}

// ------------ method called when starting to processes a run  ------------
void StudyHLT::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  char hname[100], htit[100];
  std::cout  << "Run[" << nRun << "] " << iRun.run() << " hltconfig.init " 
	     << hltConfig_.init(iRun,iSetup,"HLT",changed) << std::endl;
  sprintf(hname, "h_HLTAccepts_%i", iRun.run());
  sprintf(htit, "HLT Accepts for Run No %i", iRun.run());
  TH1I *hnew = fs->make<TH1I>(hname, htit, 500, 0, 500);
  h_HLTAccepts.push_back(hnew);
  std::cout << "beginrun " << iRun.run() << std::endl;
  firstEvent = true;
}

// ------------ method called when ending the processing of a run  ------------
void StudyHLT::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun++;
  std::cout << "endrun[" << nRun << "] " << iRun.run() << std::endl;
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

DEFINE_FWK_MODULE(StudyHLT);


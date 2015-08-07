#include "IsoTrackCalib.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

IsoTrackCalib::IsoTrackCalib(const edm::ParameterSet& iConfig) : 
  changed(false), nRun(0), t_trackP(0), t_trackPx(0), t_trackPy(0),
  t_trackPz(0), t_trackEta(0), t_trackPhi(0), t_trackPt(0), t_neu_iso(0),
  t_charge_iso(0), t_emip(0), t_ehcal(0), t_trkL3mindr(0), t_ieta(0),
  t_disthotcell(0), t_ietahotcell(0), t_eventweight(0), t_l1pt(0), t_l1eta(0),
  t_l1phi(0), t_l3pt(0), t_l3eta(0), t_l3phi(0), t_leadingpt(0),
  t_leadingeta(0), t_leadingphi(0) {
   //now do whatever initialization is needed
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
  dr_L1                               = iConfig.getUntrackedParameter<double>("IsolationL1",1.0);
  a_coneR                             = iConfig.getUntrackedParameter<double>("ConeRadius",34.98);
  a_charIsoR                          = a_coneR + 28.9;
  a_neutIsoR                          = a_charIsoR*0.726;
  a_mipR                              = iConfig.getUntrackedParameter<double>("ConeRadiusMIP",14.0);
  a_neutR1                            = iConfig.getUntrackedParameter<double>("ConeRadiusNeut1",21.0);
  a_neutR2                            = iConfig.getUntrackedParameter<double>("ConeRadiusNeut2",29.0);
  cutMip                              = iConfig.getUntrackedParameter<double>("MIPCut", 1.0);
  cutCharge                           = iConfig.getUntrackedParameter<double>("ChargeIsolation",  2.0);
  cutNeutral                          = iConfig.getUntrackedParameter<double>("NeutralIsolation",  2.0);
  minRunNo                            = iConfig.getUntrackedParameter<int>("minRun");
  maxRunNo                            = iConfig.getUntrackedParameter<int>("maxRun");
  drCuts                              = iConfig.getUntrackedParameter<std::vector<double> >("DRCuts");
  bool isItAOD                        = iConfig.getUntrackedParameter<bool>("IsItAOD", true);
  triggerEvent_                       = edm::InputTag("hltTriggerSummaryAOD","","HLT");
  theTriggerResultsLabel              = edm::InputTag("TriggerResults","","HLT");

  // define tokens for access
  tok_lumi      = consumes<LumiDetails, edm::InLumi>(edm::InputTag("lumiProducer")); 
  tok_trigEvt   = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes   = consumes<edm::TriggerResults>(theTriggerResultsLabel);
  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_   = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_       = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  tok_ew_       = consumes<GenEventInfoProduct>(edm::InputTag("generator")); 
  tok_pf_       = consumes<reco::PFJetCollection>(edm::InputTag("ak5PFJets"));
 
  if (isItAOD) {
    tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));
    tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("reducedHcalRecHits", "hbhereco"));
  } else {
    tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
    tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEE"));
    tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  }

  if (verbosity>=0) {
    edm::LogInfo("IsoTrack") <<"Parameters read from config file \n" 
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
			     <<"\t a_neutR "         << a_neutR1 << ":" << a_neutR2
			     <<"\t cuts (MIP "       << cutMip << " : Charge " << cutCharge
			     <<" : Neutral "         << cutNeutral << ")";
    edm::LogInfo("IsoTrack") << trigNames.size() << " triggers to be studied";
    for (unsigned int k=0; k<trigNames.size(); ++k)
      edm::LogInfo("IsoTrack") << "Trigger[" << k << "] : " << trigNames[k];
    edm::LogInfo("IsoTrack") << drCuts.size() << " Delta R zones wrt trigger objects";
    for (unsigned int k=0; k<drCuts.size(); ++k)
      edm::LogInfo("IsoTrack") << "Cut[" << k << "]: " << drCuts[k];
  }
}

IsoTrackCalib::~IsoTrackCalib() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  if (t_trackP)     delete t_trackP;
  if (t_trackPx)    delete t_trackPx;
  if (t_trackPy)    delete t_trackPy;
  if (t_trackPz)    delete t_trackPz;
  if (t_trackEta)   delete t_trackEta;
  if (t_trackPhi)   delete t_trackPhi;
  if (t_trackPt)    delete t_trackPt;
  if (t_neu_iso)    delete t_neu_iso;
  if (t_charge_iso) delete t_charge_iso;
  if (t_emip)       delete t_emip;
  if (t_ehcal)      delete t_ehcal;
  if (t_trkL3mindr) delete t_trkL3mindr;
  if (t_ieta)       delete t_ieta;
  if (t_disthotcell)delete t_disthotcell;
  if (t_ietahotcell)delete t_ietahotcell;
  if (t_eventweight)delete t_eventweight;
  if (t_l1pt)       delete t_l1pt;
  if (t_l1eta)      delete t_l1eta;
  if (t_l1phi)      delete t_l1phi;
  if (t_l3pt)       delete t_l3pt;
  if (t_l3eta)      delete t_l3eta;
  if (t_l3phi)      delete t_l3phi;
  if (t_leadingpt)  delete t_leadingpt;
  if (t_leadingeta) delete t_leadingeta;
  if (t_leadingphi) delete t_leadingphi;
 }

void IsoTrackCalib::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  Run    = iEvent.id().run();
  Event  = iEvent.id().event();
  if (verbosity%10 > 0) 
    edm::LogInfo("IsoTrack") << "Run " << Run << " Event " << Event
			     << " Luminosity " << iEvent.luminosityBlock() 
			     << " Bunch " << iEvent.bunchCrossing() << " start";
  clearTreeVectors();

  //Get magnetic field and ECAL channel status
  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField *bField = bFieldH.product();
  
  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);

  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  
  //Get track collection
  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);
  reco::TrackCollection::const_iterator trkItr;
 
  double flatPtWeight = 0.0;
  //event weight for FLAT sample
  edm::Handle<GenEventInfoProduct> genEventInfo;
  iEvent.getByToken(tok_ew_, genEventInfo);
  flatPtWeight = genEventInfo->weight();  

  //jets info
  edm::Handle<reco::PFJetCollection> pfJetsHandle;
  iEvent.getByToken(tok_pf_, pfJetsHandle);   
  reco::PFJetCollection::const_iterator pfItr; 

  //Define the best vertex and the beamspot
  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);  
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);
  math::XYZPoint leadPV(0,0,0);
  if (recVtxs->size()>0 && !((*recVtxs)[0].isFake())) {
    leadPV = math::XYZPoint( (*recVtxs)[0].x(),(*recVtxs)[0].y(), (*recVtxs)[0].z() );
  } else if (beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
  if ((verbosity/100)%10>0) {
    edm::LogInfo("IsoTrack") << "Primary Vertex " << leadPV;
    if (beamSpotH.isValid()) edm::LogInfo("IsoTrack") << "Beam Spot " 
						      << beamSpotH->position();
  }
  
  // RecHits
  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);

  edm::Handle<LumiDetails> Lumid;
  iEvent.getLuminosityBlock().getByToken(tok_lumi, Lumid); 
  float mybxlumi=-1;
  if (Lumid.isValid()) 
  mybxlumi=Lumid->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing())*6.37;
  if (verbosity%10 > 0) edm::LogInfo("IsoTrack") << "Luminosity " << mybxlumi;

  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("IsoTrack") << "Error! Can't get the product "
				<< triggerEvent_.label();
  } else {
    triggerEvent = *(triggerEventHandle.product());
    
    const trigger::TriggerObjectCollection& TOC(triggerEvent.getObjects());
    /////////////////////////////TriggerResults
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByToken(tok_trigRes, triggerResults);
    if (triggerResults.isValid()) {
      std::vector<std::string> modules;
      h_nHLT->Fill(triggerResults->size());
      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
      
      const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
      for (unsigned int iHLT=0; iHLT<triggerResults->size(); iHLT++) {
      bool ok(false);
	unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[iHLT]);
	const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(triggerindx));
        edm::LogInfo("IsoTrack") << iHLT << "   " <<triggerNames_[iHLT];
	int ipos = -1;
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
	if (iHLT >= 645) edm::LogInfo("IsoTrack") << "Wrong trigger " << Run 
						  << " Event " << Event 
						  << " Hlt " << iHLT;
	for (unsigned int i=0; i<trigNames.size(); ++i) {
	  if (triggerNames_[iHLT].find(trigNames[i].c_str())!=std::string::npos) {
	    if (verbosity)
	      edm::LogInfo("IsoTrack") << "This is the trigger we are looking for " << triggerNames_[iHLT];
	    if (hlt > 0) ok = true;
	  }
	}
        if (verbosity%10 > 2) 
	  edm::LogInfo("IsoTrack") << "Trigger fired? : " << ok;
	if (ok) {
	  std::vector<math::XYZTLorentzVector> vec[3];
	  const std::pair<int,int> prescales(hltConfig_.prescaleValues(iEvent,iSetup,triggerNames_[iHLT]));
	  int preL1  = prescales.first;
	  int preHLT = prescales.second;
	  int prescale = preL1*preHLT;
	  if (verbosity%10 > 0) 
	    edm::LogInfo("IsoTrack") << triggerNames_[iHLT] << " accept " 
				     << hlt << " preL1 " << preL1 << " preHLT "
				     << preHLT << " preScale " << prescale;
	  std::pair<unsigned int, std::string> iRunTrig =
	    std::pair<unsigned int, std::string>(Run,triggerNames_[iHLT]);
	  if (TrigList.find(iRunTrig) != TrigList.end() ) {
	    TrigList[iRunTrig] += 1;
	  } else {
	    TrigList.insert(std::pair<std::pair<unsigned int, std::string>, unsigned int>(iRunTrig,1));
	    TrigPreList.insert(std::pair<std::pair<unsigned int, std::string>, std::pair<int, int>>(iRunTrig,prescales));
	  }
	  //loop over all trigger filters in event (i.e. filters passed)
	  for (unsigned int ifilter=0; ifilter<triggerEvent.sizeFilters(); ++ifilter) {  
	    std::vector<int> Keys;
	    std::string label = triggerEvent.filterTag(ifilter).label();
	    //loop over keys to objects passing this filter
	    for (unsigned int imodule=0; imodule<moduleLabels.size(); imodule++) {
	      if (label.find(moduleLabels[imodule]) != std::string::npos) {
		if (verbosity%10 > 0) 
		  edm::LogInfo("IsoTrack") << "FilterName " << label;
		for (unsigned int ifiltrKey=0; ifiltrKey<triggerEvent.filterKeys(ifilter).size(); ++ifiltrKey) {
		  Keys.push_back(triggerEvent.filterKeys(ifilter)[ifiltrKey]);
		  const trigger::TriggerObject& TO(TOC[Keys[ifiltrKey]]);
		  math::XYZTLorentzVector v4(TO.px(), TO.py(), TO.pz(), TO.energy());
                  if(label.find("hltSingleJet") != std::string::npos) {
		    vec[1].push_back(v4);
                  } else if (label.find("hlt1PFJet") != std::string::npos) {
		    vec[2].push_back(v4);
		  } else {
		    vec[0].push_back(v4);
		  }
		  if (verbosity%10 > 2)
		    edm::LogInfo("IsoTrack") << "key " << ifiltrKey << " : pt " 
					     << TO.pt() << " eta " << TO.eta()
					     << " phi " << TO.phi() << " mass "
					     << TO.mass() << " Id " << TO.id();
		}
	      }
	    }
	  }
	  //// Filling Pt, eta, phi of L1 and L2 objects
	  if (verbosity%10 > 0) {
	    for (int j=0; j<3; j++) {
	      for (unsigned int k=0; k<vec[j].size(); k++) {
		edm::LogInfo("IsoTrack") << "vec[" << j << "][" << k << "] pt " 
					 << vec[j][k].pt() << " eta " 
					 << vec[j][k].eta() << " phi "
					 << vec[j][k].phi();
	      }
	    }
	  }
	  double deta, dphi, dr;
	  //// deta, dphi and dR for leading L1 object with L2 objects
	  math::XYZTLorentzVector mindRvec1;
	  double mindR1(999);
	  for (int lvl=1; lvl<3; lvl++) {
	    for (unsigned int i=0; i<vec[lvl].size(); i++) {
	      deta = dEta(vec[0][0],vec[lvl][i]);
	      dphi = dPhi(vec[0][0],vec[lvl][i]);
	      dr   = dR(vec[0][0],vec[lvl][i]);
	      if (verbosity%10 > 2) 
		edm::LogInfo("IsoTrack") << "lvl " << lvl << " i " << i 
					 << " deta " << deta << " dphi " 
					 << dphi << " dR " << dr;
	      if (dr<mindR1) {
		mindR1    = dr;
		mindRvec1 = vec[lvl][i];
	      }
	    }
	  }
          //leading jet loop
          for(pfItr=pfJetsHandle->begin();pfItr!=pfJetsHandle->end(); pfItr++){
	    t_leadingpt->push_back(pfItr->pt());  
	    t_leadingeta->push_back(pfItr->eta());
	    t_leadingphi->push_back(pfItr->phi());
	    if (verbosity%10 > 0) 
	      edm::LogInfo("IsoTrack") << "Leading jet : pt/eta/phi " 
				       << pfItr->pt() << "/" << pfItr->eta()
				       << "/" << pfItr->phi();
	    break;
	  }
	  //tracks loop
	  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
	  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDirections, ((verbosity/100)%10>2));
	  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
	  unsigned int nTracks=0,nselTracks=0;
	  for (trkDetItr = trkCaloDirections.begin(),nTracks=0; 
	       trkDetItr != trkCaloDirections.end(); trkDetItr++,nTracks++) {
	    const reco::Track* pTrack = &(*(trkDetItr->trkItr));
            math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());
            TLorentzVector trackinfo;
            trackinfo.SetPxPyPzE(pTrack->px(), pTrack->py(), pTrack->pz(), pTrack->p());
	    if (verbosity%10 > 0) 
	      edm::LogInfo("IsoTrack") << "This track : " << nTracks 
				       << " (pt/eta/phi/p) :" << pTrack->pt() 
				       << "/" << pTrack->eta() << "/" 
				       << pTrack->phi() << "/" << pTrack->p();
	    math::XYZTLorentzVector mindRvec2;
	    double mindR2(999);
            if(pTrack->pt()>10){
	      for (unsigned int k=0; k<vec[2].size(); ++k) {
		dr   = dR(vec[2][k],v4); //changed 1 to 2
		if (dr<mindR2) {
		  mindR2    = dr;
		  mindRvec2 = vec[2][k];
		}
	      }
	      if (verbosity%10 > 2)
		edm::LogInfo("IsoTrack") << "Closest L3 object at mindr :" 
					 << mindR2 << " is " << mindRvec2;
	      double mindR = dR(mindRvec1,v4);
	    
	      unsigned int i1 = drCuts.size();
	      for (unsigned int ik=0; ik<drCuts.size(); ++ik) {
		if (mindR < drCuts[ik]) {
		  i1 = ik; break;
		}
	      }
	      unsigned int i2 = drCuts.size();
	      for (unsigned int ik=0; ik<drCuts.size(); ++ik) {
		if (mindR2 < drCuts[ik]) {
		  i2 = ik; break;
		}
	      }
	    
	      //Selection of good track
	      bool selectTk = spr::goodTrack(pTrack,leadPV,selectionParameters,((verbosity/100)%10>2));
	      int ieta(0);
	      if (trkDetItr->okHCAL) {
		HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
		ieta = detId.ieta();
	      }
	      if (verbosity%10 > 0) 
		edm::LogInfo("IsoTrack") << "seltlk/okECAL/okHCAL : "<< selectTk
					 << "/"  << trkDetItr->okECAL << "/" 
					 << trkDetItr->okHCAL  << " iEta " 
					 << ieta << " Classify " << i1 << ":" 
					 << i2;
	      if (selectTk && trkDetItr->okECAL && trkDetItr->okHCAL) {
		nselTracks++;
		int nRH_eMipDR=0, nNearTRKs=0;
		double e1 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
					     trkDetItr->pointHCAL, trkDetItr->pointECAL,
					     a_neutR1, trkDetItr->directionECAL, nRH_eMipDR);
		double e2 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
					     trkDetItr->pointHCAL, trkDetItr->pointECAL,
					     a_neutR2, trkDetItr->directionECAL, nRH_eMipDR);
		double eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
						trkDetItr->pointHCAL, trkDetItr->pointECAL,
						a_mipR, trkDetItr->directionECAL, nRH_eMipDR);
		int ietaHotCell(-99), iphiHotCell(-99), nRecHitsCone(-999);
		double distFromHotCell(-99.0);
		std::vector<DetId>    coneRecHitDetIds;
		GlobalPoint           gposHotCell(0.,0.,0.);
		double eHcal = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, trkDetItr->pointECAL,
					       a_coneR, trkDetItr->directionHCAL, nRecHitsCone, 
					       coneRecHitDetIds, distFromHotCell, 
					       ietaHotCell, iphiHotCell, gposHotCell);
	      
		double conehmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR, nNearTRKs, ((verbosity/100)%10>2));
		double e_inCone = e2 - e1;

		if (verbosity%10 > 0) {
		  edm::LogInfo("IsoTrack") << "This track : " << nTracks 
					   <<" (pt/eta/phi/p) :" << pTrack->pt()
					   << "/" << pTrack->eta() << "/" 
					   << pTrack->phi() << "/" 
					   << pTrack->p() << "\n"
					   << " (MIP/neu_isol/charge_iso/HCAL_energy/iEta/distfromHcell/iEtaHcell) = " 
					   << eMipDR << "/"<< e_inCone << "/" 
					   << conehmaxNearP << "/" << eHcal 
					   << "/" << ieta << "/" << distFromHotCell
					   << "/" <<ietaHotCell;
		}
		t_trackP->push_back(pTrack->p());
		t_trackPx->push_back(pTrack->px());
		t_trackPy->push_back(pTrack->py()); 
		t_trackPz->push_back(pTrack->pz());
		t_trackPt->push_back(pTrack->pt());;
		t_trackEta->push_back(pTrack->eta());
		t_trackPhi->push_back(pTrack->phi());
		t_emip->push_back(eMipDR);
		t_neu_iso->push_back(e_inCone);;
		t_charge_iso->push_back(conehmaxNearP);
		t_ehcal->push_back(eHcal);
		t_trkL3mindr->push_back(mindR2);
		t_ieta->push_back(ieta);
		t_disthotcell->push_back(distFromHotCell); 
		t_ietahotcell->push_back(ietaHotCell);
	      }
	    }
	  }
	  if (verbosity%10 > 0) {
	    edm::LogInfo("IsoTrack") << "selected tracks = " << nselTracks 
				     << "\nevent weight is = " << flatPtWeight 
				     << "\n L1 trigger object : pt/eta/phi " 
				     << vec[0][0].pt() << "/" << vec[0][0].eta()
				     << "/" << vec[0][0].phi()
				     << "\n L3 trigger object : pt/eta/phi " 
				     << vec[2][0].pt() << "/" << vec[2][0].eta()
				     << "/"<< vec[2][0].phi();
	  }
	  t_l1pt->push_back(vec[0][0].pt());
	  t_l1eta->push_back(vec[0][0].eta());
	  t_l1phi->push_back(vec[0][0].phi());
	  t_l3pt->push_back(vec[2][0].pt());
	  t_l3eta->push_back(vec[2][0].eta());
	  t_l3phi->push_back(vec[2][0].phi());
	  t_eventweight->push_back(flatPtWeight);
	  //	  break;
	}
      }
      // check if trigger names in (new) config                       
      if (changed) {
	changed = false;
	if ((verbosity/10)%10 > 1) {
	  edm::LogInfo("IsoTrack") <<"New trigger menu found !!!";
	  const unsigned int n(hltConfig_.size());
	  for (unsigned itrig=0; itrig<triggerNames_.size(); itrig++) {
	    unsigned int triggerindx = hltConfig_.triggerIndex(triggerNames_[itrig]);
	    if (triggerindx >= n)
	      edm::LogInfo("IsoTrack") << triggerNames_[itrig] << " " 
				       << triggerindx << " does not exist";
	    else
	      edm::LogInfo("IsoTrack") << triggerNames_[itrig] << " " 
				       << triggerindx << " exists";
	  }
	}
      }
    }
  }
  tree->Fill();
}

void IsoTrackCalib::beginJob() {
  h_nHLT        = fs->make<TH1I>("h_nHLT" , "size of trigger Names", 1000, 0, 1000);
  h_HLTAccept   = fs->make<TH1I>("h_HLTAccept", "HLT Accepts for all runs", 1000, 0, 1000);
  tree = fs->make<TTree>("tree", "tree");
   
  tree->Branch("Run",&Run,"Run/I");
  tree->Branch("Event",&Event,"Event/I");
  t_trackP      = new std::vector<double>();
  t_trackPx     = new std::vector<double>();
  t_trackPy     = new std::vector<double>();
  t_trackPz     = new std::vector<double>();
  t_trackEta    = new std::vector<double>();
  t_trackPhi    = new std::vector<double>();
  t_trackPt     = new std::vector<double>();
  t_neu_iso     = new std::vector<double>();
  t_charge_iso  = new std::vector<double>();
  t_emip        = new std::vector<double>();
  t_ehcal       = new std::vector<double>(); 
  t_trkL3mindr  = new std::vector<double>();
  t_ieta        = new std::vector<int>();
  t_disthotcell = new std::vector<double>();
  t_ietahotcell = new std::vector<double>(); 
  t_eventweight = new std::vector<double>();
  t_l1pt        = new std::vector<double>();
  t_l1eta       = new std::vector<double>();
  t_l1phi       = new std::vector<double>();
  t_l3pt        = new std::vector<double>();
  t_l3eta       = new std::vector<double>();
  t_l3phi       = new std::vector<double>();
  t_leadingpt   = new std::vector<double>();
  t_leadingeta  = new std::vector<double>();
  t_leadingphi  = new std::vector<double>();

  tree->Branch("t_trackP","std::vector<double>",&t_trackP);
  tree->Branch("t_trackPx","std::vector<double>",&t_trackPx);
  tree->Branch("t_trackPy","std::vector<double>",&t_trackPy);
  tree->Branch("t_trackPz","std::vector<double>",&t_trackPz);
  tree->Branch("t_trackEta","std::vector<double>",&t_trackEta);
  tree->Branch("t_trackPhi","vector<double>",&t_trackPhi);
  tree->Branch("t_trackPt","std::vector<double>",&t_trackPt);
  tree->Branch("t_neu_iso","std::vector<double>",&t_neu_iso);
  tree->Branch("t_charge_iso","std::vector<double>",&t_charge_iso);
  tree->Branch("t_emip","std::vector<double>",&t_emip);
  tree->Branch("t_ehcal","std::vector<double>",&t_ehcal);
  tree->Branch("t_trkL3mindr","std::vector<double>",&t_trkL3mindr);
  tree->Branch("t_ieta","std::vector<int>",&t_ieta);  
  tree->Branch("t_disthotcell","std::vector<double>",&t_disthotcell);
  tree->Branch("t_ietahotcell","std::vector<double>",&t_ietahotcell);
  tree->Branch("t_eventweight","std::vector<double>",&t_eventweight);
  tree->Branch("t_l1pt","std::vector<double>",&t_l1pt);
  tree->Branch("t_l1eta","std::vector<double>",&t_l1eta);
  tree->Branch("t_l1phi","std::vector<double>",&t_l1phi);
  tree->Branch("t_l3pt","std::vector<double>",&t_l3pt);
  tree->Branch("t_l3eta","std::vector<double>",&t_l3eta);
  tree->Branch("t_l3phi","std::vector<double>",&t_l3phi);
  tree->Branch("t_leadingpt","std::vector<double>",&t_leadingpt); 
  tree->Branch("t_leadingeta","std::vector<double>",&t_leadingeta);
  tree->Branch("t_leadingphi","std::vector<double>",&t_leadingphi);  
}

// ------------ method called once each job just after ending the event loop  ------------
void IsoTrackCalib::endJob() {
  unsigned int preL1, preHLT;
  std::map<std::pair<unsigned int, std::string>, unsigned int>::iterator itr;
  std::map<std::pair<unsigned int, std::string>, const std::pair<int, int>>::iterator itrPre;
  edm::LogInfo("IsoTrack") << "RunNo vs HLT accepts";
  unsigned int n = maxRunNo - minRunNo +1;
  g_Pre = fs->make<TH1I>("h_PrevsRN", "PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_PreL1 = fs->make<TH1I>("h_PreL1vsRN", "L1 PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_PreHLT = fs->make<TH1I>("h_PreHLTvsRN", "HLT PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_Accepts = fs->make<TH1I>("h_HLTAcceptsvsRN", "HLT Accepts Vs Run Number", n, minRunNo, maxRunNo); 

  for (itr=TrigList.begin(), itrPre=TrigPreList.begin(); itr!=TrigList.end(); itr++, itrPre++) {
    preL1 = (itrPre->second).first;
    preHLT = (itrPre->second).second;
    g_Accepts->Fill((itr->first).first, itr->second);
    g_PreL1->Fill((itr->first).first, preL1);
    g_PreHLT->Fill((itr->first).first, preHLT);
    g_Pre->Fill((itr->first).first, preL1*preHLT);
  }
}

// ------------ method called when starting to processes a run  ------------
void IsoTrackCalib::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogInfo("IsoTrack") << "Run[" << nRun <<"] " << iRun.run() 
			   << " hltconfig.init " << hltConfig_.init(iRun,iSetup,"HLT",changed);
  char  hname[100], htit[100];
  sprintf(hname, "h_HLTAccepts_%i", iRun.run());
  sprintf(htit, "HLT Accepts for Run No %i", iRun.run());
  TH1I *hnew = fs->make<TH1I>(hname, htit, 1000, 0, 1000);
  h_HLTAccepts.push_back(hnew);
  firstEvent = true;
}

// ------------ method called when ending the processing of a run  ------------
void IsoTrackCalib::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  nRun++;
  edm::LogInfo("IsoTrack") << "endRun[" << nRun << "] " << iRun.run();
}

// ------------ method called when starting to processes a luminosity block  ------------
void IsoTrackCalib::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method called when ending the processing of a luminosity block  ------------
void IsoTrackCalib::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void IsoTrackCalib::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

double IsoTrackCalib::dEta(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (vec1.eta()-vec2.eta());
}

double IsoTrackCalib::dPhi(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return reco::deltaPhi(vec1.phi(),vec2.phi());
}

double IsoTrackCalib::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return reco::deltaR(vec1.eta(),vec1.phi(),vec2.eta(),vec2.phi());
}

double IsoTrackCalib::dPt(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (vec1.pt()-vec2.pt());
}

double IsoTrackCalib::dP(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (std::abs(vec1.r()-vec2.r()));
}

double IsoTrackCalib::dinvPt(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return ((1/vec1.pt())-(1/vec2.pt()));
}

void IsoTrackCalib::clearTreeVectors(){
   t_trackP->clear();
   t_trackPx->clear();  
   t_trackPy->clear();
   t_trackPz->clear();
   t_trackPt->clear();
   t_trackEta->clear(); 
   t_trackPhi->clear();
   t_emip->clear();
   t_neu_iso->clear();;
   t_charge_iso->clear();
   t_ehcal->clear();
   t_trkL3mindr->clear();
   t_ieta->clear(); 
   t_disthotcell->clear();
   t_ietahotcell->clear();
   t_eventweight->clear();  
   t_l1pt->clear();
   t_l1eta->clear();
   t_l1phi->clear();
   t_l3pt->clear();
   t_l3eta->clear();
   t_l3phi->clear(); 
   t_leadingpt->clear();
   t_leadingeta->clear();
   t_leadingphi->clear();  
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsoTrackCalib);

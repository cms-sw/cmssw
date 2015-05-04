// -*- C++ -*-//
// Package:    IsoTrig
// Class:      IsoTrig
// 
/**\class IsoTrig IsoTrig.cc IsoTrig/IsoTrig/src/IsoTrig.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Ruchi Gupta
//         Created:  Fri May 25 12:02:48 CDT 2012
// $Id$
//
//
//#define DebugLog
#include "IsoTrig.h"


#include "FWCore/Common/interface/TriggerNames.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "HLTrigger/Timer/interface/FastTimerService.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

IsoTrig::IsoTrig(const edm::ParameterSet& iConfig) :
  hltPrescaleProvider_(iConfig, consumesCollector(), *this),
  changed(false), t_timeL2Prod(0), t_nPixCand(0), t_nPixSeed(0), t_nGoodTk(0),
  t_TrkhCone(0), t_TrkP(0), t_TrkselTkFlag(0), t_TrkqltyFlag(0),
  t_TrkMissFlag(0), t_TrkPVFlag(0), t_TrkNuIsolFlag(0),
  t_PixcandP(0), t_PixcandPt(0), t_PixcandEta(0),  t_PixcandPhi(0),
  t_PixcandMaxP(0), t_PixTrkcandP(0), t_PixTrkcandPt(0), t_PixTrkcandEta(0),
  t_PixTrkcandPhi(0), t_PixTrkcandMaxP(0), t_PixTrkcandselTk(0),
  t_NFcandP(0), t_NFcandPt(0), t_NFcandEta(0), t_NFcandPhi(0),
  t_NFcandEmip(0), t_NFTrkcandP(0), t_NFTrkcandPt(0), t_NFTrkcandEta(0),
  t_NFTrkcandPhi(0), t_NFTrkcandEmip(0), t_NFTrkMinDR(0), t_NFTrkMinDP1(0),
  t_NFTrkselTkFlag(0), t_NFTrkqltyFlag(0), t_NFTrkMissFlag(0), 
  t_NFTrkPVFlag(0), t_NFTrkPropFlag(0), t_NFTrkChgIsoFlag(0), 
  t_NFTrkNeuIsoFlag(0), t_NFTrkMipFlag(0), t_ECone(0) {
  //now do whatever initialization is neededOA
  trigNames                           = iConfig.getUntrackedParameter<std::vector<std::string> >("Triggers");
  PixcandTag_                          = iConfig.getParameter<edm::InputTag> ("PixcandTag");
  L1candTag_                          = iConfig.getParameter<edm::InputTag> ("L1candTag");
  L2candTag_                          = iConfig.getParameter<edm::InputTag> ("L2candTag");
  doL2L3                              = iConfig.getUntrackedParameter<bool>("DoL2L3",true);
  doTiming                            = iConfig.getUntrackedParameter<bool>("DoTimingTree",true);
  doMipCutTree                        = iConfig.getUntrackedParameter<bool>("DoMipCutTree",true);
  doTrkResTree                        = iConfig.getUntrackedParameter<bool>("DoTrkResTree",true);
  doChgIsolTree                        = iConfig.getUntrackedParameter<bool>("DoChgIsolTree",true);
  doStudyIsol                         = iConfig.getUntrackedParameter<bool>("DoStudyIsol",true);
  verbosity                           = iConfig.getUntrackedParameter<int>("Verbosity",0);
  theTrackQuality                     = iConfig.getUntrackedParameter<std::string>("TrackQuality","highPurity");
  processName                         = iConfig.getUntrackedParameter<std::string>("ProcessName","HLT");
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
  pixelTracksSources_                 = iConfig.getUntrackedParameter<std::vector<edm::InputTag> >("PixelTracksSources");
  pixelIsolationConeSizeAtEC_         = iConfig.getUntrackedParameter<std::vector<double> >("PixelIsolationConeSizeAtEC");
  minPTrackValue_                     = iConfig.getUntrackedParameter<double>("MinPTrackValue");
  vtxCutSeed_                         = iConfig.getUntrackedParameter<double>("VertexCutSeed");
  vtxCutIsol_                         = iConfig.getUntrackedParameter<double>("VertexCutIsol");
  tauUnbiasCone_                      = iConfig.getUntrackedParameter<double>("TauUnbiasCone");
  prelimCone_                         = iConfig.getUntrackedParameter<double>("PrelimCone");
  // define tokens for access
  tok_lumi      = consumes<LumiDetails, edm::InLumi>(edm::InputTag("lumiProducer"));
  edm::InputTag triggerEvent_ ("hltTriggerSummaryAOD","",processName); 
  tok_trigEvt   = consumes<trigger::TriggerEvent>(triggerEvent_);
  edm::InputTag theTriggerResultsLabel ("TriggerResults","",processName); 
  tok_trigRes   = consumes<edm::TriggerResults>(theTriggerResultsLabel);
  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_   = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_       = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  tok_EB_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  tok_EE_     = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEE"));
  tok_hbhe_   = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  tok_pixtk_    = consumes<reco::IsolatedPixelTrackCandidateCollection>(PixcandTag_);
  tok_l1cand_   = consumes<trigger::TriggerFilterObjectWithRefs>(L1candTag_);
  tok_l2cand_   = consumes<reco::IsolatedPixelTrackCandidateCollection>(L2candTag_);
  if (doTiming) {
    tok_verthb_ = consumes<reco::VertexCollection>(edm::InputTag("hltHITPixelVerticesHB"));
    tok_verthe_ = consumes<reco::VertexCollection>(edm::InputTag("hltHITPixelVerticesHB"));
    tok_hlt_    = consumes<trigger::TriggerFilterObjectWithRefs>(edm::InputTag("hltL1sL1SingleJet68"));
    tok_SeedingLayerhb = consumes<SeedingLayerSetsHits>(edm::InputTag("hltPixelLayerTripletsHITHB"));
    tok_SeedingLayerhe = consumes<SeedingLayerSetsHits>(edm::InputTag("hltPixelLayerTripletsHITHE"));
    tok_SiPixelRecHits = consumes<SiPixelRecHitCollection>(edm::InputTag("hltSiPixelRecHits"));
  }
  if(doChgIsolTree) {
    for (unsigned int k=0; k<pixelTracksSources_.size(); ++k) {
      //      edm::InputTag  pix (pixelTracksSources_[k],"",processName);
      //      tok_pixtks_.push_back(consumes<reco::TrackCollection>(pix));
      tok_pixtks_.push_back(consumes<reco::TrackCollection>(pixelTracksSources_[k]));
    }
  }
#ifdef DebugLog
  if (verbosity>=0) {
    std::cout <<"Parameters read from config file \n" 
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
	      <<" : Neutral "         << cutNeutral << ")"
	      << std::endl;
    std::cout <<"Charge Isolation parameters:"
	      <<"\t minPTrackValue " << minPTrackValue_
	      <<"\t vtxCutSeed "     << vtxCutSeed_
	      <<"\t vtxCutIsol "     << vtxCutIsol_
	      <<"\t tauUnbiasCone "  << tauUnbiasCone_
	      <<"\t prelimCone "     << prelimCone_
	      <<"\t pixelIsolationConeSizeAtEC";
    for (unsigned int k=0; k<pixelIsolationConeSizeAtEC_.size(); ++k)
      std::cout << " " << pixelIsolationConeSizeAtEC_[k];
    std::cout << std::endl;
  }
#endif
  double pl[] = {20,30,40,60,80,120};
  for (int i=0; i<6; ++i) pLimits[i] = pl[i];
  rEB_ = 123.8;
  zEE_ = 317.0;
}

IsoTrig::~IsoTrig() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  if (t_timeL2Prod)      delete t_timeL2Prod;
  if (t_nPixCand)        delete t_nPixCand;
  if (t_nPixSeed)        delete t_nPixSeed;
  if (t_nGoodTk)         delete t_nGoodTk;
  if (t_TrkhCone)        delete t_TrkhCone;
  if (t_TrkP)            delete t_TrkP;
  if (t_TrkselTkFlag)    delete t_TrkselTkFlag;
  if (t_TrkqltyFlag)     delete t_TrkqltyFlag;
  if (t_TrkMissFlag)     delete t_TrkMissFlag;
  if (t_TrkPVFlag)       delete t_TrkPVFlag;
  if (t_TrkNuIsolFlag)   delete t_TrkNuIsolFlag;
  if (t_PixcandP)        delete t_PixcandP;
  if (t_PixcandPt)       delete t_PixcandPt;
  if (t_PixcandEta)      delete t_PixcandEta;
  if (t_PixcandPhi)      delete t_PixcandPhi;
  if (t_PixcandMaxP)     delete t_PixcandMaxP;
  if (t_PixTrkcandP)     delete t_PixTrkcandP;
  if (t_PixTrkcandPt)    delete t_PixTrkcandPt;
  if (t_PixTrkcandEta)   delete t_PixTrkcandEta;
  if (t_PixTrkcandPhi)   delete t_PixTrkcandPhi;
  if (t_PixTrkcandMaxP)  delete t_PixTrkcandMaxP;
  if (t_PixTrkcandselTk) delete t_PixTrkcandselTk;
  if (t_NFcandP)         delete t_NFcandP;
  if (t_NFcandPt)        delete t_NFcandPt;
  if (t_NFcandEta)       delete t_NFcandEta;
  if (t_NFcandPhi)       delete t_NFcandPhi;
  if (t_NFcandEmip)      delete t_NFcandEmip;
  if (t_NFTrkcandP)      delete t_NFTrkcandP;
  if (t_NFTrkcandPt)     delete t_NFTrkcandPt;
  if (t_NFTrkcandEta)    delete t_NFTrkcandEta;
  if (t_NFTrkcandPhi)    delete t_NFTrkcandPhi;
  if (t_NFTrkcandEmip)   delete t_NFTrkcandEmip;
  if (t_NFTrkMinDR)      delete t_NFTrkMinDR;
  if (t_NFTrkMinDP1)     delete t_NFTrkMinDP1;
  if (t_NFTrkselTkFlag)  delete t_NFTrkselTkFlag;
  if (t_NFTrkqltyFlag)   delete t_NFTrkqltyFlag;
  if (t_NFTrkMissFlag)   delete t_NFTrkMissFlag;
  if (t_NFTrkPVFlag)     delete t_NFTrkPVFlag;
  if (t_NFTrkPropFlag)   delete t_NFTrkPropFlag;
  if (t_NFTrkChgIsoFlag) delete t_NFTrkChgIsoFlag;
  if (t_NFTrkNeuIsoFlag) delete t_NFTrkNeuIsoFlag;
  if (t_NFTrkMipFlag)    delete t_NFTrkMipFlag;
  if (t_ECone)           delete t_ECone;
}

void IsoTrig::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void IsoTrig::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
#ifdef DebugLog
  if (verbosity%10 > 0) std::cout << "Event starts====================================" << std::endl;
#endif
  int RunNo = iEvent.id().run();

  HLTConfigProvider const&  hltConfig = hltPrescaleProvider_.hltConfigProvider();

  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  iSetup.get<CaloGeometryRecord>().get(pG);
  const MagneticField *bField = bFieldH.product();   
  GlobalVector BField=bField->inTesla(GlobalPoint(0,0,0));
  bfVal = BField.mag();

  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("IsoTrack") << "Error! Can't get the product hltTriggerSummaryAOD";

  } else {
    triggerEvent = *(triggerEventHandle.product());
  }
  const trigger::TriggerObjectCollection& TOC(triggerEvent.getObjects());
  /////////////////////////////TriggerResults
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(tok_trigRes, triggerResults);  

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);

  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);

  iEvent.getByToken(tok_hbhe_, hbhe);

  iEvent.getByToken(tok_recVtx_, recVtxs);  
  iEvent.getByToken(tok_bs_, beamSpotH);
  if (recVtxs->size()>0 && !((*recVtxs)[0].isFake())) {
    leadPV = math::XYZPoint( (*recVtxs)[0].x(),(*recVtxs)[0].y(), (*recVtxs)[0].z() );
  } else if (beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }
#ifdef DebugLog
  if ((verbosity/100)%10>0) {
    std::cout << "Primary Vertex " << leadPV;
    if (beamSpotH.isValid()) std::cout << " Beam Spot " 
				       << beamSpotH->position();
    std::cout << std::endl;
  }
#endif
  pixelTrackRefsHE.clear(); pixelTrackRefsHB.clear();
  for (unsigned int iPix=0; iPix<pixelTracksSources_.size(); iPix++) {
    edm::Handle<reco::TrackCollection> iPixCol;
    iEvent.getByToken(tok_pixtks_[iPix],iPixCol); 
    if(iPixCol.isValid()){
      for (reco::TrackCollection::const_iterator pit=iPixCol->begin(); pit!=iPixCol->end(); pit++) {
	if(iPix==0) 
	  pixelTrackRefsHB.push_back(reco::TrackRef(iPixCol,pit-iPixCol->begin()));
	pixelTrackRefsHE.push_back(reco::TrackRef(iPixCol,pit-iPixCol->begin()));
      }
    }
  }
  if (doTiming) getGoodTracks(iEvent, trkCollection);

  for (unsigned int ifilter=0; ifilter<triggerEvent.sizeFilters(); 
       ++ifilter) {  
     std::string FilterNames[7] = {"hltL1sL1SingleJet68", "hltIsolPixelTrackL2FilterHE", "ecalIsolPixelTrackFilterHE", "hltIsolPixelTrackL3FilterHE",
				  "hltIsolPixelTrackL2FilterHB", "ecalIsolPixelTrackFilterHB", "hltIsolPixelTrackL3FilterHB"};
    std::string label = triggerEvent.filterTag(ifilter).label();
    for(int i=0; i<7; i++) {
      if(label==FilterNames[i]) h_Filters->Fill(i);
    }
  }
  edm::InputTag lumiProducer("LumiProducer", "", "RECO");
  edm::Handle<LumiDetails> Lumid;
  iEvent.getLuminosityBlock().getByToken(tok_lumi, Lumid);
  float mybxlumi=-1;
  if (Lumid.isValid()) 
    mybxlumi=Lumid->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing())*6.37;
#ifdef DebugLog
  if (verbosity%10 > 0)
    std::cout << "RunNo " << RunNo << " EvtNo " << iEvent.id().event() 
	      << " Lumi " << iEvent.luminosityBlock() << " Bunch " 
	      << iEvent.bunchCrossing() << " mybxlumi " << mybxlumi 
	      << std::endl;
#endif
  if (!triggerResults.isValid()) {
    edm::LogWarning("IsoTrack") << "Error! Can't get the product triggerResults";
    //      boost::shared_ptr<cms::Exception> const & error = triggerResults.whyFailed();
    //      edm::LogWarning(error->category()) << error->what();
  } else {
    std::vector<std::string> modules;
    h_nHLT->Fill(triggerResults->size());
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);

    const std::vector<std::string> & triggerNames_ = triggerNames.triggerNames();
#ifdef DebugLog
    if (verbosity%10 > 1) 
      std::cout << "number of HLTs " << triggerNames_.size() << std::endl;
#endif
    int hlt(-1), preL1(-1), preHLT(-1), prescale(-1);
    for (unsigned int i=0; i<triggerResults->size(); i++) {
      unsigned int triggerindx = hltConfig.triggerIndex(triggerNames_[i]);
      const std::vector<std::string>& moduleLabels(hltConfig.moduleLabels(triggerindx));
      
      for (unsigned int in=0; in<trigNames.size(); ++in) {
	//	  if (triggerNames_[i].find(trigNames[in].c_str())!=std::string::npos || triggerNames_[i]==" ") {
	if (triggerNames_[i].find(trigNames[in].c_str())!=std::string::npos) {
#ifdef DebugLog
	  if (verbosity%10 > 0) std::cout << "trigger that i want " << triggerNames_[i] << " accept " << triggerResults->accept(i) << std::endl;
#endif
	  hlt    = triggerResults->accept(i);
	  h_HLT      -> Fill(hlt);
	  //	    if (hlt>0 || triggerNames_[i]==" ") {
	  if (hlt>0) {
	    edm::Handle<reco::IsolatedPixelTrackCandidateCollection> Pixcands;
	    iEvent.getByToken(tok_pixtk_,Pixcands); 
	    edm::Handle<trigger::TriggerFilterObjectWithRefs> L1cands;
	    iEvent.getByToken(tok_l1cand_, L1cands); 
	    
	    const std::pair<int,int> prescales(hltPrescaleProvider_.prescaleValues(iEvent,iSetup,triggerNames_[i]));
	    preL1  = prescales.first;
	    preHLT = prescales.second;
	    prescale = preL1*preHLT;
#ifdef DebugLog
	    if (verbosity%10 > 0)
	      std::cout << triggerNames_[i] << " accept " << hlt << " preL1 " 
			<< preL1 << " preHLT " << preHLT << std::endl;
#endif	    
	    for (int iv=0; iv<3; ++iv) vec[iv].clear();
	    if (TrigList.find(RunNo) != TrigList.end() ) {
	      TrigList[RunNo] += 1;
	      } else {
	      TrigList.insert(std::pair<unsigned int, unsigned int>(RunNo,1));
	      TrigPreList.insert(std::pair<unsigned int, std::pair<int, int>>(RunNo,prescales));
	    }
	    //loop over all trigger filters in event (i.e. filters passed)
	    for (unsigned int ifilter=0; ifilter<triggerEvent.sizeFilters(); 
		 ++ifilter) {  
	      std::vector<int> Keys;
	      std::string label = triggerEvent.filterTag(ifilter).label();
	      //loop over keys to objects passing this filter
	      for (unsigned int imodule=0; imodule<moduleLabels.size(); 
		     imodule++) {
		if (label.find(moduleLabels[imodule]) != std::string::npos) {
#ifdef DebugLog
		  if (verbosity%10 > 0) std::cout << "FILTERNAME " << label << std::endl;
#endif
		  for (unsigned int ifiltrKey=0; ifiltrKey<triggerEvent.filterKeys(ifilter).size(); ++ifiltrKey) {
		    Keys.push_back(triggerEvent.filterKeys(ifilter)[ifiltrKey]);
		    const trigger::TriggerObject& TO(TOC[Keys[ifiltrKey]]);
		    math::XYZTLorentzVector v4(TO.px(), TO.py(), TO.pz(), TO.energy());
		    if (label.find("L2Filter") != std::string::npos) {
		      vec[1].push_back(v4);
		      } else if (label.find("L3Filter") != std::string::npos) {
		      vec[2].push_back(v4);
		    } else {
		      vec[0].push_back(v4);
		      h_L1ObjEnergy->Fill(TO.energy());
		    }
#ifdef DebugLog
		    if (verbosity%10 > 0)
		      std::cout << "key " << ifiltrKey << " : pt " << TO.pt() << " eta " << TO.eta() << " phi " << TO.phi() << " mass " << TO.mass() << " Id " << TO.id() << std::endl;
#endif
		  }
		}
	      }
	    }
	    std::vector<reco::TrackCollection::const_iterator> goodTks;
	    if (doL2L3) {
	      h_nL3Objs  -> Fill(vec[2].size());
	      studyTrigger(trkCollection, goodTks);
	    } else {
	      if (trkCollection.isValid()) {
		reco::TrackCollection::const_iterator trkItr;
		for (trkItr=trkCollection->begin(); 
		     trkItr!=trkCollection->end(); trkItr++) 
		    goodTks.push_back(trkItr);
	      }
	    }
	    // Now study isolation etc
	    if (doStudyIsol) studyIsolation(trkCollection, goodTks);
	    if (doTrkResTree) StudyTrkEbyP(trkCollection);
	    
	    std::pair<double,double> etaphi = etaPhiTrigger();
	    edm::Handle<reco::IsolatedPixelTrackCandidateCollection> L2cands;
	    iEvent.getByToken(tok_l2cand_,L2cands); 
	    if (!L2cands.isValid()) {
#ifdef DebugLog
	      if (verbosity%10 > 0) std::cout << " trigCand is not valid " << std::endl;
#endif
	    } else {
	      if(doMipCutTree) studyMipCut(trkCollection, L2cands);
	    }
	    if (pixelTracksSources_.size()>0)
	      if(doChgIsolTree && pixelTrackRefsHE.size()>0) chgIsolation(etaphi.first, etaphi.second, trkCollection, iEvent);
	  }
	  break;
	}
      }
    }
    h_PreL1    -> Fill(preL1);
    h_PreHLT   -> Fill(preHLT);
    h_Pre      -> Fill(prescale);
    h_PreL1wt  -> Fill(preL1, mybxlumi);
    h_PreHLTwt -> Fill(preHLT, mybxlumi);
    
    // check if trigger names in (new) config                       
    //      std::cout << "changed " <<changed << std::endl;
    if (changed) {
      changed = false;
#ifdef DebugLog
      if ((verbosity/10)%10 > 1) {
	std::cout << "New trigger menu found !!!" << std::endl;
	const unsigned int n(hltConfig.size());
	for (unsigned itrig=0; itrig<triggerNames_.size(); itrig++) {
	  unsigned int triggerindx = hltConfig.triggerIndex(triggerNames_[itrig]);
	  std::cout << triggerNames_[itrig] << " " << triggerindx << " ";
	  if (triggerindx >= n)
	    std::cout << "does not exist in the current menu" << std::endl;
	  else
	    std::cout << "exists" << std::endl;
	}
      }
#endif
    }
  }  
  if (doTiming) studyTiming(iEvent);
}

void IsoTrig::clearChgIsolnTreeVectors() { 
  t_PixcandP        ->clear();
  t_PixcandPt       ->clear();
  t_PixcandEta      ->clear();
  t_PixcandPhi      ->clear();
  for(unsigned int i=0; i< t_PixcandMaxP->size(); i++)
    t_PixcandMaxP[i].clear();
  t_PixcandMaxP     ->clear();
  t_PixTrkcandP     ->clear();
  t_PixTrkcandPt    ->clear();
  t_PixTrkcandEta   ->clear();
  t_PixTrkcandPhi   ->clear();
  t_PixTrkcandMaxP  ->clear();
  t_PixTrkcandselTk  ->clear();
}

void IsoTrig::clearMipCutTreeVectors() {
  t_NFcandP        ->clear();
  t_NFcandPt       ->clear();
  t_NFcandEta      ->clear();
  t_NFcandPhi      ->clear();
  t_NFcandEmip     ->clear();
  t_NFTrkcandP     ->clear();
  t_NFTrkcandPt    ->clear();
  t_NFTrkcandEta   ->clear();
  t_NFTrkcandPhi   ->clear();
  t_NFTrkcandEmip  ->clear();
  t_NFTrkMinDR     ->clear();
  t_NFTrkMinDP1    ->clear();
  t_NFTrkselTkFlag ->clear();
  t_NFTrkqltyFlag  ->clear();
  t_NFTrkMissFlag  ->clear();
  t_NFTrkPVFlag    ->clear();
  t_NFTrkPropFlag  ->clear();
  t_NFTrkChgIsoFlag->clear();
  t_NFTrkNeuIsoFlag->clear();
  t_NFTrkMipFlag   ->clear();
  t_ECone          ->clear();
}

void IsoTrig::pushChgIsolnTreeVecs(math::XYZTLorentzVector &Pixcand, math::XYZTLorentzVector &Trkcand, 
				   std::vector<double> &PixMaxP, double &TrkMaxP,
				   bool &selTk) {
  t_PixcandP         ->push_back(Pixcand.r());
  t_PixcandPt        ->push_back(Pixcand.pt());
  t_PixcandEta       ->push_back(Pixcand.eta());
  t_PixcandPhi       ->push_back(Pixcand.phi());
  t_PixcandMaxP      ->push_back(PixMaxP);
  t_PixTrkcandP      ->push_back(Trkcand.r());
  t_PixTrkcandPt     ->push_back(Trkcand.pt());
  t_PixTrkcandEta    ->push_back(Trkcand.eta());
  t_PixTrkcandPhi    ->push_back(Trkcand.phi());
  t_PixTrkcandMaxP   ->push_back(TrkMaxP); 
  t_PixTrkcandselTk  ->push_back(selTk); 
  
}

void IsoTrig::pushMipCutTreeVecs(math::XYZTLorentzVector &NFcand, 
				 math::XYZTLorentzVector &Trkcand, 
				 double &EmipNFcand, double &EmipTrkcand,
				 double &mindR, double &mindP1,
				 std::vector<bool> &Flags, double hCone) {
  t_NFcandP        ->push_back(NFcand.r());
  t_NFcandPt       ->push_back(NFcand.pt());
  t_NFcandEta      ->push_back(NFcand.eta());
  t_NFcandPhi      ->push_back(NFcand.phi());
  t_NFcandEmip     ->push_back(EmipNFcand);
  t_NFTrkcandP     ->push_back(Trkcand.r());
  t_NFTrkcandPt    ->push_back(Trkcand.pt());
  t_NFTrkcandEta   ->push_back(Trkcand.eta());
  t_NFTrkcandPhi   ->push_back(Trkcand.phi());
  t_NFTrkcandEmip  ->push_back(EmipTrkcand);
  t_NFTrkMinDR     ->push_back(mindR);
  t_NFTrkMinDP1    ->push_back(mindP1);
  t_NFTrkselTkFlag ->push_back(Flags[0]);
  t_NFTrkqltyFlag  ->push_back(Flags[1]);
  t_NFTrkMissFlag  ->push_back(Flags[2]);
  t_NFTrkPVFlag    ->push_back(Flags[3]);
  t_NFTrkPropFlag  ->push_back(Flags[4]);
  t_NFTrkChgIsoFlag->push_back(Flags[5]);
  t_NFTrkNeuIsoFlag->push_back(Flags[6]);
  t_NFTrkMipFlag   ->push_back(Flags[7]);
  t_ECone          ->push_back(hCone);
}

void IsoTrig::beginJob() {
  char hname[100], htit[100];
  std::string levels[20] = {"L1", "L2", "L3", 
			    "Reco", "RecoMatch", "RecoNoMatch", 
			    "L2Match", "L2NoMatch", "L3Match", "L3NoMatch", 
			    "HLTTrk", "HLTGoodTrk", "HLTIsoTrk", "HLTMip", "HLTSelect",
			    "nonHLTTrk", "nonHLTGoodTrk", "nonHLTIsoTrk", "nonHLTMip", "nonHLTSelect"};
  if (doTiming) {
    TimingTree = fs->make<TTree>("TimingTree", "TimingTree");
    t_timeL2Prod = new std::vector<double>();
    t_nPixCand   = new std::vector<int>();
    t_nPixSeed   = new std::vector<int>();
    t_nGoodTk    = new std::vector<int>();

    TimingTree->Branch("t_timeL2Prod", "std::vector<double>", &t_timeL2Prod);
    TimingTree->Branch("t_nPixCand",   "std::vector<int>", &t_nPixCand);
    TimingTree->Branch("t_nPixSeed",   "std::vector<int>", &t_nPixSeed);
    TimingTree->Branch("t_nGoodTk",    "std::vector<int>", &t_nGoodTk);
  }
  if (doTrkResTree) {
    TrkResTree = fs->make<TTree>("TrkRestree", "TrkRestree");
    t_TrkhCone     = new std::vector<double>();
    t_TrkP         = new std::vector<double>();
    t_TrkselTkFlag = new std::vector<bool>();
    t_TrkqltyFlag  = new std::vector<bool>();
    t_TrkMissFlag  = new std::vector<bool>();
    t_TrkPVFlag    = new std::vector<bool>();
    t_TrkNuIsolFlag= new std::vector<bool>();
    
    TrkResTree->Branch("t_TrkhCone",     "std::vector<double>", &t_TrkhCone);
    TrkResTree->Branch("t_TrkP",         "std::vector<double>", &t_TrkP);
    TrkResTree->Branch("t_TrkselTkFlag", "std::vector<bool>",   &t_TrkselTkFlag);
    TrkResTree->Branch("t_TrkqltyFlag",  "std::vector<bool>",   &t_TrkqltyFlag);
    TrkResTree->Branch("t_TrkMissFlag",  "std::vector<bool>",   &t_TrkMissFlag);
    TrkResTree->Branch("t_TrkPVFlag",    "std::vector<bool>",   &t_TrkPVFlag);
    TrkResTree->Branch("t_TrkNuIsolFlag","std::vector<bool>",   &t_TrkNuIsolFlag);
  }
  if (doChgIsolTree) {
    ChgIsolnTree = fs->make<TTree>("ChgIsolnTree", "ChgIsolntree");
    t_PixcandP        = new std::vector<double>();
    t_PixcandPt       = new std::vector<double>();
    t_PixcandEta      = new std::vector<double>();
    t_PixcandPhi      = new std::vector<double>();
    t_PixcandMaxP     = new std::vector<std::vector<double> >();
    t_PixTrkcandP     = new std::vector<double>();
    t_PixTrkcandPt    = new std::vector<double>();
    t_PixTrkcandEta   = new std::vector<double>();
    t_PixTrkcandPhi   = new std::vector<double>();
    t_PixTrkcandMaxP  = new std::vector<double>();
    t_PixTrkcandselTk = new std::vector<bool>();
    
    ChgIsolnTree->Branch("t_PixcandP",       "std::vector<double>", &t_PixcandP);
    ChgIsolnTree->Branch("t_PixcandPt",      "std::vector<double>", &t_PixcandPt);
    ChgIsolnTree->Branch("t_PixcandEta",     "std::vector<double>", &t_PixcandEta);
    ChgIsolnTree->Branch("t_PixcandPhi",     "std::vector<double>", &t_PixcandPhi);
    ChgIsolnTree->Branch("t_PixcandMaxP",    "std::vector<std::vector<double> >", &t_PixcandMaxP);
    ChgIsolnTree->Branch("t_PixTrkcandP",    "std::vector<double>", &t_PixTrkcandP);
    ChgIsolnTree->Branch("t_PixTrkcandPt",   "std::vector<double>", &t_PixTrkcandPt  );
    ChgIsolnTree->Branch("t_PixTrkcandEta",  "std::vector<double>", &t_PixTrkcandEta );
    ChgIsolnTree->Branch("t_PixTrkcandPhi",  "std::vector<double>", &t_PixTrkcandPhi );
    ChgIsolnTree->Branch("t_PixTrkcandMaxP", "std::vector<double>", &t_PixTrkcandMaxP);
    ChgIsolnTree->Branch("t_PixTrkcandselTk","std::vector<bool>", &t_PixTrkcandselTk);
  }
  if (doMipCutTree) {
    MipCutTree = fs->make<TTree>("MipCutTree", "MipCuttree");
    t_NFcandP        = new std::vector<double>();
    t_NFcandPt       = new std::vector<double>();
    t_NFcandEta      = new std::vector<double>();
    t_NFcandPhi      = new std::vector<double>();
    t_NFcandEmip     = new std::vector<double>();
    t_NFTrkcandP     = new std::vector<double>();
    t_NFTrkcandPt    = new std::vector<double>();
    t_NFTrkcandEta   = new std::vector<double>();
    t_NFTrkcandPhi   = new std::vector<double>();
    t_NFTrkcandEmip  = new std::vector<double>();
    t_NFTrkMinDR     = new std::vector<double>();
    t_NFTrkMinDP1    = new std::vector<double>();
    t_NFTrkselTkFlag = new std::vector<bool>();
    t_NFTrkqltyFlag  = new std::vector<bool>();
    t_NFTrkMissFlag  = new std::vector<bool>();
    t_NFTrkPVFlag    = new std::vector<bool>();
    t_NFTrkPropFlag  = new std::vector<bool>();
    t_NFTrkChgIsoFlag= new std::vector<bool>();
    t_NFTrkNeuIsoFlag= new std::vector<bool>();
    t_NFTrkMipFlag   = new std::vector<bool>();
    t_ECone          = new std::vector<double>();
    
    MipCutTree->Branch("t_NFcandP",        "std::vector<double>", &t_NFcandP);
    MipCutTree->Branch("t_NFcandPt",       "std::vector<double>", &t_NFcandPt);
    MipCutTree->Branch("t_NFcandEta",      "std::vector<double>", &t_NFcandEta);
    MipCutTree->Branch("t_NFcandPhi",      "std::vector<double>", &t_NFcandPhi);
    MipCutTree->Branch("t_NFcandEmip",     "std::vector<double>", &t_NFcandEmip);
    MipCutTree->Branch("t_NFTrkcandP",     "std::vector<double>", &t_NFTrkcandP);
    MipCutTree->Branch("t_NFTrkcandPt",    "std::vector<double>", &t_NFTrkcandPt  );
    MipCutTree->Branch("t_NFTrkcandEta",   "std::vector<double>", &t_NFTrkcandEta );
    MipCutTree->Branch("t_NFTrkcandPhi",   "std::vector<double>", &t_NFTrkcandPhi );
    MipCutTree->Branch("t_NFTrkcandEmip",  "std::vector<double>", &t_NFTrkcandEmip);
    MipCutTree->Branch("t_NFTrkMinDR",     "std::vector<double>", &t_NFTrkMinDR);
    MipCutTree->Branch("t_NFTrkMinDP1",    "std::vector<double>", &t_NFTrkMinDP1);
    MipCutTree->Branch("t_NFTrkselTkFlag", "std::vector<bool>",   &t_NFTrkselTkFlag);
    MipCutTree->Branch("t_NFTrkqltyFlag",  "std::vector<bool>",   &t_NFTrkqltyFlag);
    MipCutTree->Branch("t_NFTrkMissFlag",  "std::vector<bool>",   &t_NFTrkMissFlag);
    MipCutTree->Branch("t_NFTrkPVFlag",    "std::vector<bool>",   &t_NFTrkPVFlag);
    MipCutTree->Branch("t_NFTrkPropFlag",  "std::vector<bool>",   &t_NFTrkPropFlag);
    MipCutTree->Branch("t_NFTrkChgIsoFlag","std::vector<bool>",   &t_NFTrkChgIsoFlag);
    MipCutTree->Branch("t_NFTrkNeuIsoFlag","std::vector<bool>",   &t_NFTrkNeuIsoFlag);
    MipCutTree->Branch("t_NFTrkMipFlag",   "std::vector<bool>",   &t_NFTrkMipFlag);
    MipCutTree->Branch("t_ECone",          "std::vector<double>", &t_ECone);
  }
  h_Filters     = fs->make<TH1I>("h_Filters", "Filter Accepts", 10, 0, 10);
  std::string FilterNames[7] = {"hltL1sL1SingleJet68", "hltIsolPixelTrackL2FilterHE", "ecalIsolPixelTrackFilterHE", "hltIsolPixelTrackL3FilterHE",
				"hltIsolPixelTrackL2FilterHB", "ecalIsolPixelTrackFilterHB", "hltIsolPixelTrackL3FilterHB"};
  for(int i=0; i<7; i++) {
    h_Filters->GetXaxis()->SetBinLabel(i+1, FilterNames[i].c_str());
  }

  h_nHLT        = fs->make<TH1I>("h_nHLT" , "Size of trigger Names", 1000, 1, 1000);
  h_HLT         = fs->make<TH1I>("h_HLT"  ,  "HLT accept", 3, -1, 2);
  h_PreL1       = fs->make<TH1I>("h_PreL1",  "L1 Prescale", 500, 0, 500);
  h_PreHLT      = fs->make<TH1I>("h_PreHLT", "HLT Prescale", 50, 0, 50);
  h_Pre         = fs->make<TH1I>("h_Pre",    "Prescale", 3000, 0, 3000);

  h_PreL1wt     = fs->make<TH1D>("h_PreL1wt", "Weighted L1 Prescale", 500, 0, 500);
  h_PreHLTwt    = fs->make<TH1D>("h_PreHLTwt", "Weighted HLT Prescale", 500, 0, 100);
  h_L1ObjEnergy = fs->make<TH1D>("h_L1ObjEnergy", "Energy of L1Object", 500, 0.0, 500.0);

  h_EnIn = fs->make<TH1D>("h_EnInEcal", "EnergyIn Ecal", 200, 0.0, 20.0);
  h_EnOut = fs->make<TH1D>("h_EnOutEcal", "EnergyOut Ecal", 200, 0.0, 20.0);
  h_MipEnMatch = fs->make<TH2D>("h_MipEnMatch", "MipEn: HLT level vs Reco Level (Matched)", 200, 0.0, 20.0, 200, 0.0, 20.0);
  h_MipEnNoMatch = fs->make<TH2D>("h_MipEnNoMatch", "MipEn: HLT level vs Reco Level (No Match Found)", 200, 0.0, 20.0, 200, 0.0, 20.0);

  if (doL2L3) {
    h_nL3Objs     = fs->make<TH1I>("h_nL3Objs", "Number of L3 objects", 10, 0, 10);
    
    std::string pairs[9] = {"L2L3", "L2L3Match", "L2L3NoMatch", "L3Reco", "L3RecoMatch", "L3RecoNoMatch", "NewFilterReco", "NewFilterRecoMatch", "NewFilterRecoNoMatch"};
    for (int ipair=0; ipair<9; ipair++) {
      sprintf(hname, "h_dEta%s", pairs[ipair].c_str());
      sprintf(htit, "#Delta#eta for %s", pairs[ipair].c_str());
      h_dEta[ipair]        = fs->make<TH1D>(hname, htit, 200, -10.0, 10.0);
      h_dEta[ipair]->GetXaxis()->SetTitle("d#eta");
      
      sprintf(hname, "h_dPhi%s", pairs[ipair].c_str());
      sprintf(htit, "#Delta#phi for %s", pairs[ipair].c_str());
      h_dPhi[ipair]        = fs->make<TH1D>(hname, htit, 140, -7.0, 7.0);
      h_dPhi[ipair]->GetXaxis()->SetTitle("d#phi");

      sprintf(hname, "h_dPt%s", pairs[ipair].c_str());
      sprintf(htit, "#Delta dp_{T} for %s objects", pairs[ipair].c_str());
      h_dPt[ipair]         = fs->make<TH1D>(hname, htit, 400, -200.0, 200.0);
      h_dPt[ipair]->GetXaxis()->SetTitle("dp_{T} (GeV)");

      sprintf(hname, "h_dP%s", pairs[ipair].c_str());
      sprintf(htit, "#Delta p for %s objects", pairs[ipair].c_str());
      h_dP[ipair]         = fs->make<TH1D>(hname, htit, 400, -200.0, 200.0);
      h_dP[ipair]->GetXaxis()->SetTitle("dP (GeV)");

      sprintf(hname, "h_dinvPt%s", pairs[ipair].c_str());
      sprintf(htit, "#Delta (1/p_{T}) for %s objects", pairs[ipair].c_str());
      h_dinvPt[ipair]      = fs->make<TH1D>(hname, htit, 500, -0.4, 0.1);
      h_dinvPt[ipair]->GetXaxis()->SetTitle("d(1/p_{T})");
      sprintf(hname, "h_mindR%s", pairs[ipair].c_str());
      sprintf(htit, "min(#Delta R) for %s objects", pairs[ipair].c_str());
      h_mindR[ipair]       = fs->make<TH1D>(hname, htit, 500, 0.0, 1.0);
      h_mindR[ipair]->GetXaxis()->SetTitle("dR");
    }

    for (int lvl=0; lvl<2; lvl++) {
      sprintf(hname, "h_dEtaL1%s", levels[lvl+1].c_str());
      sprintf(htit, "#Delta#eta for L1 and %s objects", levels[lvl+1].c_str());
      h_dEtaL1[lvl] = fs->make<TH1D>(hname, htit, 400, -10.0, 10.0);

      sprintf(hname, "h_dPhiL1%s", levels[lvl+1].c_str());
      sprintf(htit, "#Delta#phi for L1 and %s objects", levels[lvl+1].c_str());
      h_dPhiL1[lvl] = fs->make<TH1D>(hname, htit, 280, -7.0, 7.0);

      sprintf(hname, "h_dRL1%s", levels[lvl+1].c_str());
      sprintf(htit, "#Delta R for L1 and %s objects", levels[lvl+1].c_str());
      h_dRL1[lvl] = fs->make<TH1D>(hname, htit, 100, 0.0, 10.0);
    }
  }

  int levmin = (doL2L3 ? 0 : 10);
  for (int ilevel=levmin; ilevel<20; ilevel++) {
    sprintf(hname, "h_p%s", levels[ilevel].c_str());
    sprintf(htit, "p for %s objects", levels[ilevel].c_str());
    h_p[ilevel] = fs->make<TH1D>(hname, htit, 100, 0.0, 500.0);
    h_p[ilevel]->GetXaxis()->SetTitle("p (GeV)");
    
    sprintf(hname, "h_pt%s", levels[ilevel].c_str());
    sprintf(htit, "p_{T} for %s objects", levels[ilevel].c_str());
    h_pt[ilevel] = fs->make<TH1D>(hname, htit, 100, 0.0, 500.0);
    h_pt[ilevel]->GetXaxis()->SetTitle("p_{T} (GeV)");
    
    sprintf(hname, "h_eta%s", levels[ilevel].c_str());
    sprintf(htit, "#eta for %s objects", levels[ilevel].c_str());
    h_eta[ilevel] = fs->make<TH1D>(hname, htit, 100, -5.0, 5.0);
    h_eta[ilevel]->GetXaxis()->SetTitle("#eta");
    
    sprintf(hname, "h_phi%s", levels[ilevel].c_str());
    sprintf(htit, "#phi for %s objects", levels[ilevel].c_str());
    h_phi[ilevel] = fs->make<TH1D>(hname, htit, 70, -3.5, 3.50);
    h_phi[ilevel]->GetXaxis()->SetTitle("#phi");
  }
  
  std::string cuts[2]  = {"HLTMatched", "HLTNotMatched"};
  std::string cuts2[2] = {"All", "Away from L1"};
  for (int icut=0; icut<2; icut++) {
    sprintf(hname, "h_eMip%s", cuts[icut].c_str());
    sprintf(htit, "eMip for %s tracks", cuts[icut].c_str());
    h_eMip[icut]     =fs->make<TH1D>(hname, htit, 200, 0.0, 10.0);
    h_eMip[icut]->GetXaxis()->SetTitle("E_{Mip} (GeV)");

    sprintf(hname, "h_eMaxNearP%s", cuts[icut].c_str());
    sprintf(htit, "eMaxNearP for %s tracks", cuts[icut].c_str());
    h_eMaxNearP[icut]=fs->make<TH1D>(hname, htit, 240, -2.0, 10.0);
    h_eMaxNearP[icut]->GetXaxis()->SetTitle("E_{MaxNearP} (GeV)");

    sprintf(hname, "h_eNeutIso%s", cuts[icut].c_str());
    sprintf(htit, "eNeutIso for %s ", cuts[icut].c_str());
    h_eNeutIso[icut] =fs->make<TH1D>(hname, htit, 200, 0.0, 10.0);
    h_eNeutIso[icut]->GetXaxis()->SetTitle("E_{NeutIso} (GeV)");

    for (int kcut=0; kcut<2; ++kcut) {
      for (int lim=0; lim<5; ++lim) {
	sprintf(hname, "h_etaCalibTracks%sCut%dLim%d", cuts[icut].c_str(), kcut, lim);
	sprintf(htit, "#eta for %s isolated MIP tracks (%4.1f < p < %5.1f Gev/c %s)", cuts[icut].c_str(), pLimits[lim], pLimits[lim+1], cuts2[kcut].c_str());
	h_etaCalibTracks[lim][icut][kcut]=fs->make<TH1D>(hname, htit, 60, -30.0, 30.0);
	h_etaCalibTracks[lim][icut][kcut]->GetXaxis()->SetTitle("i#eta");

	sprintf(hname, "h_etaMipTracks%sCut%dLim%d", cuts[icut].c_str(), kcut, lim);
	sprintf(htit, "#eta for %s charge isolated MIP tracks (%4.1f < p < %5.1f Gev/c %s)", cuts[icut].c_str(), pLimits[lim], pLimits[lim+1], cuts2[kcut].c_str());
	h_etaMipTracks[lim][icut][kcut]=fs->make<TH1D>(hname, htit, 60, -30.0, 30.0);
	h_etaMipTracks[lim][icut][kcut]->GetXaxis()->SetTitle("i#eta");
      }
    }
  }

  std::string ecut1[3] = {"all","HLTMatched","HLTNotMatched"};
  std::string ecut2[2] = {"without","with"};
  int etac[48] = {-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,
		  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  for (int icut=0; icut<6; icut++) {
    //    int i1 = (icut>3 ? 1 : 0);
    int i1 = (icut>2 ? 1 : 0);
    int i2 = icut - i1*3;
    for (int kcut=0; kcut<48; kcut++) {
      for (int lim=0; lim<5; ++lim) {
	sprintf(hname, "h_eta%dEnHcal%s%s%d", etac[kcut], ecut1[i2].c_str(), ecut2[i1].c_str(), lim);
	sprintf(htit, "HCAL energy for #eta=%d for %s tracks (p=%4.1f:%5.1f Gev) %s neutral isolation", etac[kcut], ecut1[i2].c_str(), pLimits[lim], pLimits[lim+1], ecut2[i1].c_str());
	h_eHcal[lim][icut][kcut]=fs->make<TH1D>(hname, htit, 750, 0.0, 150.0);
	h_eHcal[lim][icut][kcut]->GetXaxis()->SetTitle("Energy (GeV)");
	sprintf(hname, "h_eta%dEnCalo%s%s%d", etac[kcut], ecut1[i2].c_str(), ecut2[i1].c_str(), lim);
	sprintf(htit, "Calorimter energy for #eta=%d for %s tracks (p=%4.1f:%5.1f Gev) %s neutral isolation", etac[kcut], ecut1[i2].c_str(), pLimits[lim], pLimits[lim+1], ecut2[i1].c_str());
	h_eCalo[lim][icut][kcut]=fs->make<TH1D>(hname, htit, 750, 0.0, 150.0);
	h_eCalo[lim][icut][kcut]->GetXaxis()->SetTitle("Energy (GeV)");
      }
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void IsoTrig::endJob() {
  unsigned int preL1, preHLT;
  std::map<unsigned int, unsigned int>::iterator itr;
  std::map<unsigned int, const std::pair<int, int>>::iterator itrPre;
  edm::LogWarning ("IsoTrack") << trigNames.size() << "Triggers were run. RunNo vs HLT accepts for";
  for (unsigned int i=0; i<trigNames.size(); ++i) 
    edm::LogWarning("IsoTrack") << "[" << i << "]: " << trigNames[i];
  unsigned int n = maxRunNo - minRunNo +1;
  g_Pre = fs->make<TH1I>("h_PrevsRN", "PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_PreL1 = fs->make<TH1I>("h_PreL1vsRN", "L1 PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_PreHLT = fs->make<TH1I>("h_PreHLTvsRN", "HLT PreScale Vs Run Number", n, minRunNo, maxRunNo);
  g_Accepts = fs->make<TH1I>("h_HLTAcceptsvsRN", "HLT Accepts Vs Run Number", n, minRunNo, maxRunNo); 

  for (itr=TrigList.begin(), itrPre=TrigPreList.begin(); itr!=TrigList.end(); itr++, itrPre++) {
    preL1 = (itrPre->second).first;
    preHLT = (itrPre->second).second;
#ifdef DebugLog
    std::cout << itr->first << " " << itr->second << " " <<  itrPre->first << " " << preL1 << " " << preHLT << std::endl;
#endif
    g_Accepts->Fill(itr->first, itr->second);
    g_PreL1->Fill(itr->first, preL1);
    g_PreHLT->Fill(itr->first, preHLT);
    g_Pre->Fill(itr->first, preL1*preHLT);
  }
}

// ------------ method called when starting to processes a run  ------------
void IsoTrig::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogWarning ("IsoTrack") << "Run " << iRun.run() << " hltconfig.init " 
			       << hltPrescaleProvider_.init(iRun,iSetup,processName,changed);
}

// ------------ method called when ending the processing of a run  ------------
void IsoTrig::endRun(edm::Run const&, edm::EventSetup const&) {}

// ------------ method called when starting to processes a luminosity block  ------------
void IsoTrig::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method called when ending the processing of a luminosity block  ------------
void IsoTrig::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void IsoTrig::StudyTrkEbyP(edm::Handle<reco::TrackCollection>& trkCollection) {

  t_TrkselTkFlag->clear();
  t_TrkqltyFlag->clear();
  t_TrkMissFlag->clear();
  t_TrkPVFlag->clear();
  t_TrkNuIsolFlag->clear();
  t_TrkhCone->clear();
  t_TrkP->clear();
  
  if (!trkCollection.isValid()) {
#ifdef DebugLog
    std::cout << "trkCollection.isValid is false" << std::endl;
#endif
  } else {
    std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr; 
    const MagneticField *bField = bFieldH.product();
    const CaloGeometry* geo = pG.product();
    std::vector<spr::propagatedTrackDirection> trkCaloDirections1;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDirections1, ((verbosity/100)%10>2));
    unsigned int nTracks=0;
    int nRH_eMipDR=0, nNearTRKs=0;
    std::vector<bool> selFlags;
    for (trkDetItr = trkCaloDirections1.begin(); trkDetItr != trkCaloDirections1.end(); trkDetItr++,nTracks++) {
      double conehmaxNearP = 0, hCone=0, eMipDR=0.0;
      const reco::Track* pTrack = &(*(trkDetItr->trkItr));
#ifdef DebugLog
      if (verbosity%10>0) std::cout << "track no. " << nTracks << " p(): " << pTrack->p() << std::endl;
#endif
      if (pTrack->p() > 20) {
	math::XYZTLorentzVector v2(pTrack->px(), pTrack->py(),  
				   pTrack->pz(), pTrack->p());
	eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
				 trkDetItr->pointHCAL, trkDetItr->pointECAL,
				 a_mipR, trkDetItr->directionECAL, nRH_eMipDR);
	bool selectTk = spr::goodTrack(pTrack,leadPV,selectionParameters,((verbosity/100)%10>1));
	spr::trackSelectionParameters oneCutParameters = selectionParameters;
	oneCutParameters.maxDxyPV  = 10;
	oneCutParameters.maxDzPV   = 100;
	oneCutParameters.maxInMiss = 2;
	oneCutParameters.maxOutMiss= 2;
	bool qltyFlag     =  spr::goodTrack(pTrack,leadPV,oneCutParameters,((verbosity/100)%10>1));
	oneCutParameters           = selectionParameters;
	oneCutParameters.maxDxyPV  = 10;
	oneCutParameters.maxDzPV   = 100;
	bool qltyMissFlag =  spr::goodTrack(pTrack,leadPV,oneCutParameters,((verbosity/100)%10>1));
	oneCutParameters           = selectionParameters;
	oneCutParameters.maxInMiss = 2;
	oneCutParameters.maxOutMiss= 2;
	bool qltyPVFlag   =  spr::goodTrack(pTrack,leadPV,oneCutParameters,((verbosity/100)%10>1));
#ifdef DebugLog
	/*
	std::cout << "sel " << selectTk << std::endl;
	std::cout << "ntracks " << nTracks;
	std::cout << " a_charIsoR " << a_charIsoR;
	std::cout << " nNearTRKs " << nNearTRKs << std::endl; 
	*/
#endif
	conehmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections1, a_charIsoR, nNearTRKs, ((verbosity/100)%10>1));
#ifdef DebugLog
	/*
	std::cout << "coneh " << conehmaxNearP << std::endl;
	std::cout << "ok " << trkDetItr->okECAL << " " << trkDetItr->okHCAL << std::endl;
	*/
#endif
	double e1 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
				     trkDetItr->pointHCAL, trkDetItr->pointECAL,
				     a_neutR1, trkDetItr->directionECAL, nRH_eMipDR);
	double e2 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
				     trkDetItr->pointHCAL, trkDetItr->pointECAL,
				     a_neutR2, trkDetItr->directionECAL, nRH_eMipDR);
	double e_inCone  = e2 - e1;
	bool chgIsolFlag = (conehmaxNearP < cutCharge);
	bool mipFlag     = (eMipDR < cutMip);
	bool neuIsolFlag = (e_inCone < cutNeutral);
	bool trkpropFlag = ((trkDetItr->okECAL) && (trkDetItr->okHCAL));
	selFlags.clear();
	selFlags.push_back(selectTk);        selFlags.push_back(qltyFlag);
	selFlags.push_back(qltyMissFlag);    selFlags.push_back(qltyPVFlag);
#ifdef DebugLog
	if (verbosity%10>0) std::cout << "emip: " << eMipDR << "<" << cutMip << "(" << mipFlag << ")" 
				      << " ; ok: " << trkDetItr->okECAL << "/" << trkDetItr->okHCAL 
				      << " ; chgiso: " << conehmaxNearP << "<" << cutCharge << "(" << chgIsolFlag << ")" << std::endl;
#endif
	
	if(chgIsolFlag && mipFlag && trkpropFlag) {
	  double             distFromHotCell=-99.0;
	  int                nRecHitsCone=-99, ietaHotCell=-99, iphiHotCell=-99;
	  GlobalPoint        gposHotCell(0.,0.,0.);
	  std::vector<DetId> coneRecHitDetIds;
	  hCone    = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, 
				     trkDetItr->pointECAL,
				     a_coneR, trkDetItr->directionHCAL, 
				     nRecHitsCone, coneRecHitDetIds,
				     distFromHotCell, ietaHotCell, iphiHotCell,
				     gposHotCell, -1);
	  // push vectors into the Tree
	  t_TrkselTkFlag ->push_back(selFlags[0]);
	  t_TrkqltyFlag  ->push_back(selFlags[1]);
	  t_TrkMissFlag  ->push_back(selFlags[2]);
	  t_TrkPVFlag    ->push_back(selFlags[3]);
	  t_TrkNuIsolFlag->push_back(neuIsolFlag);
	  t_TrkhCone     ->push_back(hCone);
	  t_TrkP         ->push_back(pTrack->p());
	}
      }
    }
#ifdef DebugLog
    if (verbosity%10>0) std::cout << "Filling " << t_TrkP->size() << " tracks in TrkRestree out of " << nTracks << std::endl;
#endif
  }
  TrkResTree->Fill();
}

void IsoTrig::studyTiming(const edm::Event& theEvent) {
  t_timeL2Prod->clear(); t_nPixCand->clear(); t_nPixSeed->clear();

#ifdef DebugLog
  edm::Handle<SeedingLayerSetsHits> hblayers, helayers;
  theEvent.getByToken(tok_SeedingLayerhb, hblayers);
  theEvent.getByToken(tok_SeedingLayerhe, helayers);
  const SeedingLayerSetsHits* layershb = hblayers.product();
  const SeedingLayerSetsHits* layershe = helayers.product();
  std::cout << "size of Seeding TripletLayers hb/he " << layershb->size() << "/" << layershe->size() << std::endl;
  edm::Handle<SiPixelRecHitCollection> rchts;
  theEvent.getByToken(tok_SiPixelRecHits, rchts);
  const SiPixelRecHitCollection* rechits = rchts.product();
  std::cout << "size of SiPixelRechits " << rechits->size() << std::endl;;
#endif
  double tHB=0.0, tHE=0.0;
  int nCandHB=pixelTrackRefsHB.size(), nCandHE=pixelTrackRefsHE.size();
  int nSeedHB=0, nSeedHE=0;

  if(nCandHE>0) {
    edm::Handle<reco::VertexCollection> pVertHB, pVertHE;
    theEvent.getByToken(tok_verthb_,pVertHB);
    theEvent.getByToken(tok_verthe_,pVertHE);
    edm::Handle<trigger::TriggerFilterObjectWithRefs> l1trigobj;
    theEvent.getByToken(tok_l1cand_, l1trigobj);
    
    std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1tauobjref;
    std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1jetobjref;
    std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1forjetobjref;
    
    l1trigobj->getObjects(trigger::TriggerL1TauJet, l1tauobjref);
    l1trigobj->getObjects(trigger::TriggerL1CenJet, l1jetobjref);
    l1trigobj->getObjects(trigger::TriggerL1ForJet, l1forjetobjref);
    
    double ptTriggered  = -10;
    double etaTriggered = -100;
    double phiTriggered = -100;
    for (unsigned int p=0; p<l1tauobjref.size(); p++) {
      if (l1tauobjref[p]->pt()>ptTriggered) {
	ptTriggered  = l1tauobjref[p]->pt(); 
	phiTriggered = l1tauobjref[p]->phi();
	etaTriggered = l1tauobjref[p]->eta();
      }
    }
    for (unsigned int p=0; p<l1jetobjref.size(); p++) {
      if (l1jetobjref[p]->pt()>ptTriggered) {
	ptTriggered  = l1jetobjref[p]->pt();
	phiTriggered = l1jetobjref[p]->phi();
	etaTriggered = l1jetobjref[p]->eta();
      }
    }
    for (unsigned int p=0; p<l1forjetobjref.size(); p++) {
      if (l1forjetobjref[p]->pt()>ptTriggered) {
	ptTriggered=l1forjetobjref[p]->pt();
	phiTriggered=l1forjetobjref[p]->phi();
	etaTriggered=l1forjetobjref[p]->eta();
      }
    }
    for(unsigned iS=0; iS<pixelTrackRefsHE.size(); iS++) {
      reco::VertexCollection::const_iterator vitSel;
      double minDZ = 100;
      bool vtxMatch;
      for (reco::VertexCollection::const_iterator vit=pVertHE->begin(); vit!=pVertHE->end(); vit++) {
	if (fabs(pixelTrackRefsHE[iS]->dz(vit->position()))<minDZ) {
	  minDZ  = fabs(pixelTrackRefsHE[iS]->dz(vit->position()));
	  vitSel = vit;
	}
      }
      //cut on dYX:
      if (minDZ!=100&&fabs(pixelTrackRefsHE[iS]->dxy(vitSel->position()))<vtxCutSeed_) vtxMatch=true;
      if (minDZ==100) vtxMatch=true;
      
      //select tracks not matched to triggered L1 jet
      double R=deltaR(etaTriggered, phiTriggered, pixelTrackRefsHE[iS]->eta(), pixelTrackRefsHE[iS]->phi());
      if (R>tauUnbiasCone_ && vtxMatch) nSeedHE++;
    }
    for(unsigned iS=0; iS<pixelTrackRefsHB.size(); iS++) {
      reco::VertexCollection::const_iterator vitSel;
      double minDZ = 100;
      bool vtxMatch;
      for (reco::VertexCollection::const_iterator vit=pVertHB->begin(); vit!=pVertHB->end(); vit++) {
	if (fabs(pixelTrackRefsHB[iS]->dz(vit->position()))<minDZ) {
	  minDZ  = fabs(pixelTrackRefsHB[iS]->dz(vit->position()));
	  vitSel = vit;
	}
      }
      //cut on dYX:
      if (minDZ!=100&&fabs(pixelTrackRefsHB[iS]->dxy(vitSel->position()))<101.0) vtxMatch=true;
      if (minDZ==100) vtxMatch=true;
      
      //select tracks not matched to triggered L1 jet
      double R=deltaR(etaTriggered, phiTriggered, pixelTrackRefsHB[iS]->eta(), pixelTrackRefsHB[iS]->phi());
      if (R>1.2 && vtxMatch) nSeedHB++;
    }
    
    edm::Service<FastTimerService> ft;
    tHE = ft->queryModuleTimeByLabel(theEvent.streamID(),"hltIsolPixelTrackProdHE") ;
    tHB = ft->queryModuleTimeByLabel(theEvent.streamID(),"hltIsolPixelTrackProdHB");
#ifdef DebugLog    
    if (verbosity%10>0) std::cout << "(HB/HE) time: " << tHB <<"/" << tHE 
				  << " nCand: " << nCandHB << "/" << nCandHE 
				  << "nSeed: " << nSeedHB << "/" << nSeedHE 
				  << std::endl;
#endif
  }
  t_timeL2Prod->push_back(tHB);   t_timeL2Prod->push_back(tHE);
  t_nPixSeed->push_back(nSeedHB); t_nPixSeed->push_back(nSeedHE);
  t_nPixCand->push_back(nCandHB); t_nPixCand->push_back(nCandHE);

  TimingTree->Fill();
}
void IsoTrig::studyMipCut(edm::Handle<reco::TrackCollection>& trkCollection,
			  edm::Handle<reco::IsolatedPixelTrackCandidateCollection>& L2cands) {

  clearMipCutTreeVectors();
#ifdef DebugLog
  if (verbosity%10>0) std::cout << "inside studymipcut" << std::endl;
#endif
  if (!trkCollection.isValid()) {
    edm::LogWarning("IsoTrack") << "trkCollection.isValid is false";
  } else {
    std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr; 
    const MagneticField *bField = bFieldH.product();
    const CaloGeometry* geo = pG.product();
    std::vector<spr::propagatedTrackDirection> trkCaloDirections1;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDirections1, ((verbosity/100)%10>2));
#ifdef DebugLog
    if (verbosity%10>0) std::cout << "Number of L2cands:" << L2cands->size() 
				     << " to be matched to something out of " 
				     << trkCaloDirections1.size() << " reco tracks" << std::endl;
#endif
    for (unsigned int i=0; i<L2cands->size(); i++) {    
      edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref =
	edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(L2cands, i);  
      double enIn = candref->energyIn();
      h_EnIn->Fill(candref->energyIn());
      h_EnOut->Fill(candref->energyOut());
      math::XYZTLorentzVector v1(candref->track()->px(),candref->track()->py(),
				 candref->track()->pz(),candref->track()->p());
#ifdef DebugLog
      if (verbosity%10>1) 
	std::cout << "HLT Level Candidate eta/phi/pt/enIn:" 
		  << candref->track()->eta() << "/" << candref->track()->phi() 
		  << "/" << candref->track()->pt() << "/" << candref->energyIn()
		  << std::endl;   
#endif
      math::XYZTLorentzVector mindRvec; 
      double mindR=999.9, mindP1=999.9, eMipDR=0.0;
      std::vector<bool> selFlags;
      unsigned int nTracks=0;
      double conehmaxNearP = 0, hCone=0;
      for (trkDetItr = trkCaloDirections1.begin(); trkDetItr != trkCaloDirections1.end(); trkDetItr++,nTracks++){
	const reco::Track* pTrack = &(*(trkDetItr->trkItr));
	math::XYZTLorentzVector v2(pTrack->px(), pTrack->py(),  
				   pTrack->pz(), pTrack->p());
	double dr   = dR(v1,v2);
	double dp1  = std::abs(1./v1.r() - 1./v2.r());
	//	std::cout <<"This recotrack(eta/phi/pt) " << pTrack->eta() << "/" 
	//		  << pTrack->phi() << "/" << pTrack->pt() << " has dr/dp= " 
	//		  << dr << "/" << dp << "/" << dp1 << std::endl;
	if (dr<mindR) {
	  mindR = dr;
	  mindP1= dp1;
	  mindRvec=v2;
	  int nRH_eMipDR=0, nNearTRKs=0;
	  eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
				   trkDetItr->pointHCAL, trkDetItr->pointECAL,
				   a_mipR, trkDetItr->directionECAL, nRH_eMipDR);
	  bool selectTk = spr::goodTrack(pTrack,leadPV,selectionParameters,((verbosity/100)%10>1));
	  spr::trackSelectionParameters oneCutParameters = selectionParameters;
	  oneCutParameters.maxDxyPV  = 10;
	  oneCutParameters.maxDzPV     = 100;
	  oneCutParameters.maxInMiss = 2;
	  oneCutParameters.maxOutMiss= 2;
	  bool qltyFlag     =  spr::goodTrack(pTrack,leadPV,oneCutParameters,((verbosity/100)%10>1));
	  oneCutParameters           = selectionParameters;
	  oneCutParameters.maxDxyPV  = 10;
	  oneCutParameters.maxDzPV     = 100;
	  bool qltyMissFlag =  spr::goodTrack(pTrack,leadPV,oneCutParameters,((verbosity/100)%10>1));
	  oneCutParameters           = selectionParameters;
	  oneCutParameters.maxInMiss = 2;
	  oneCutParameters.maxOutMiss= 2;
	  bool qltyPVFlag   =  spr::goodTrack(pTrack,leadPV,oneCutParameters,((verbosity/100)%10>1));
	  conehmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections1, a_charIsoR, nNearTRKs, ((verbosity/100)%10>1));
	  double e1 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
				       trkDetItr->pointHCAL, trkDetItr->pointECAL,
				       a_neutR1, trkDetItr->directionECAL, nRH_eMipDR);
	  double e2 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
				       trkDetItr->pointHCAL, trkDetItr->pointECAL,
				       a_neutR2, trkDetItr->directionECAL, nRH_eMipDR);
	  double e_inCone  = e2 - e1;
	  bool chgIsolFlag = (conehmaxNearP < cutCharge);
	  bool mipFlag     = (eMipDR < cutMip);
	  bool neuIsolFlag = (e_inCone < cutNeutral);
	  bool trkpropFlag = ((trkDetItr->okECAL) && (trkDetItr->okHCAL));
	  selFlags.clear();
	  selFlags.push_back(selectTk);        selFlags.push_back(qltyFlag);
	  selFlags.push_back(qltyMissFlag);    selFlags.push_back(qltyPVFlag);
	  selFlags.push_back(trkpropFlag);     selFlags.push_back(chgIsolFlag);
	  selFlags.push_back(neuIsolFlag);     selFlags.push_back(mipFlag);
	  double             distFromHotCell=-99.0;
	  int                nRecHitsCone=-99, ietaHotCell=-99, iphiHotCell=-99;
	  GlobalPoint        gposHotCell(0.,0.,0.);
	  std::vector<DetId> coneRecHitDetIds;
	  hCone    = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, 
				     trkDetItr->pointECAL,
				     a_coneR, trkDetItr->directionHCAL, 
				     nRecHitsCone, coneRecHitDetIds,
				     distFromHotCell, ietaHotCell, iphiHotCell,
				     gposHotCell, -1);
	}
      }
      pushMipCutTreeVecs(v1, mindRvec, enIn, eMipDR, mindR, mindP1, selFlags, hCone);
      fillDifferences(6, v1, mindRvec, (verbosity%10 >0));
      if (mindR>=0.05) {
	fillDifferences(8, v1, mindRvec, (verbosity%10 >0));
	h_MipEnNoMatch->Fill(candref->energyIn(), eMipDR);
      } else {
	fillDifferences(7, v1, mindRvec, (verbosity%10 >0));
	h_MipEnMatch->Fill(candref->energyIn(), eMipDR);
      }
    }
  }
  MipCutTree->Fill();  
}

void IsoTrig::studyTrigger(edm::Handle<reco::TrackCollection>& trkCollection,
			   std::vector<reco::TrackCollection::const_iterator>& goodTks) {

#ifdef DebugLog
  if (verbosity%10 > 0) std::cout << "Inside StudyTrigger" << std::endl;
#endif
  //// Filling Pt, eta, phi of L1, L2 and L3 objects
  for (int j=0; j<3; j++) {
    for (unsigned int k=0; k<vec[j].size(); k++) {
#ifdef DebugLog
      if (verbosity%10 > 0) std::cout << "vec[" << j << "][" << k << "] pt " << vec[j][k].pt() << " eta " << vec[j][k].eta() << " phi " << vec[j][k].phi() << std::endl;
#endif
      fillHist(j, vec[j][k]);
    }
  }
	  
  double deta, dphi, dr;
  //// deta, dphi and dR for leading L1 object with L2 and L3 objects
  for (int lvl=1; lvl<3; lvl++) {
    for (unsigned int i=0; i<vec[lvl].size(); i++) {
      deta = dEta(vec[0][0],vec[lvl][i]);
      dphi = dPhi(vec[0][0],vec[lvl][i]);
      dr   = dR(vec[0][0],vec[lvl][i]);
#ifdef DebugLog
      if (verbosity%10 > 1) std::cout << "lvl " <<lvl << " i " << i << " deta " << deta << " dphi " << dphi << " dR " << dr << std::endl;
#endif
      h_dEtaL1[lvl-1] -> Fill(deta);
      h_dPhiL1[lvl-1] -> Fill(dphi);
      h_dRL1[lvl-1]   -> Fill(std::sqrt(dr));
    }
  }

  math::XYZTLorentzVector mindRvec;
  double mindR;
  for (unsigned int k=0; k<vec[2].size(); ++k) {
    //// Find min of deta/dphi/dR for each of L3 objects with L2 objects
    mindR=999.9;
#ifdef DebugLog
    if (verbosity%10 > 1) std::cout << "L3obj: pt " << vec[2][k].pt() << " eta " << vec[2][k].eta() << " phi " << vec[2][k].phi() << std::endl;
#endif
    for (unsigned int j=0; j<vec[1].size(); j++) {
      dr   = dR(vec[2][k],vec[1][j]);
      if (dr<mindR) {
	mindR=dr;
	mindRvec=vec[1][j];
      }
    }
    fillDifferences(0, vec[2][k], mindRvec, (verbosity%10 >0));
    if (mindR < 0.03) {
      fillDifferences(1, vec[2][k], mindRvec, (verbosity%10 >0));
      fillHist(6, mindRvec);
      fillHist(8, vec[2][k]);
    } else {
      fillDifferences(2, vec[2][k], mindRvec, (verbosity%10 >0));
      fillHist(7, mindRvec);
      fillHist(9, vec[2][k]);
    }
	      	      
    ////// Minimum deltaR for each of L3 objs with Reco::tracks
    mindR=999.9;
#ifdef DebugLog
    if (verbosity%10 > 0) 
      std::cout << "Now Matching L3 track with reco: L3 Track (eta, phi) " 
		<< vec[2][k].eta() << ":" << vec[2][k].phi() << " L2 Track "
		<< mindRvec.eta() << ":" << mindRvec.phi() << " dR "
		<< mindR << std::endl;
#endif
    reco::TrackCollection::const_iterator goodTk = trkCollection->end();
    if (trkCollection.isValid()) {
      double mindP(9999.9);
      reco::TrackCollection::const_iterator trkItr;
      for (trkItr=trkCollection->begin(); 
	   trkItr!=trkCollection->end(); trkItr++) {
	math::XYZTLorentzVector v4(trkItr->px(), trkItr->py(), 
				   trkItr->pz(), trkItr->p());
	double deltaR = dR(v4, vec[2][k]);
	double dp     = std::abs(v4.r()/vec[2][k].r()-1.0);
	if (deltaR<mindR) {
	  mindR    = deltaR;
	  mindP    = dp;
	  mindRvec = v4;
	  goodTk   = trkItr;
	}
#ifdef DebugLog
	if ((verbosity/10)%10>1 && deltaR<1.0) {
	  std::cout << "reco track: pt " << v4.pt() << " eta " << v4.eta()
		    << " phi " << v4.phi() << " DR " << deltaR 
		    << std::endl;
	}
#endif
      }
#ifdef DebugLog
      if (verbosity%10 > 0) 
	std::cout << "Now Matching at Reco level in step 1 DR: "  << mindR
		  << ":" << mindP << " eta:phi " << mindRvec.eta() << ":" 
		  << mindRvec.phi() << std::endl;
#endif
      if (mindR < 0.03 && mindP > 0.1) {
	for (trkItr=trkCollection->begin(); 
	     trkItr!=trkCollection->end(); trkItr++) {
	  math::XYZTLorentzVector v4(trkItr->px(), trkItr->py(), 
				     trkItr->pz(), trkItr->p());
	  double deltaR = dR(v4, vec[2][k]);
	  double dp     = std::abs(v4.r()/vec[2][k].r()-1.0);
	  if (dp<mindP && deltaR<0.03) {
	    mindR    = deltaR;
	    mindP    = dp;
	    mindRvec = v4;
	    goodTk   = trkItr;
	  }
	}
#ifdef DebugLog
	if (verbosity%10 > 0) 
	  std::cout << "Now Matching at Reco level in step 2 DR: "  << mindR
		    << ":" << mindP << " eta:phi " << mindRvec.eta() << ":" 
		    << mindRvec.phi() << std::endl;
#endif
      }
      fillDifferences(3, vec[2][k], mindRvec, (verbosity%10 >0));
      fillHist(3, mindRvec);
      if(mindR < 0.03) {
	fillDifferences(4, vec[2][k], mindRvec, (verbosity%10 >0));
	fillHist(4, mindRvec);
      } else {
	fillDifferences(5, vec[2][k], mindRvec, (verbosity%10 >0));
	fillHist(5, mindRvec);
      }
      if (goodTk != trkCollection->end()) goodTks.push_back(goodTk);
    }
  }
}

void IsoTrig::studyIsolation(edm::Handle<reco::TrackCollection>& trkCollection,
			     std::vector<reco::TrackCollection::const_iterator>& goodTks) {

  if (trkCollection.isValid()) {
    // get handles to calogeometry and calotopology
    const CaloGeometry* geo = pG.product();
    const MagneticField *bField = bFieldH.product();
    std::vector<spr::propagatedTrackDirection> trkCaloDirections;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDirections, ((verbosity/100)%10>2));

    std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
#ifdef DebugLog
    if ((verbosity/1000)%10 > 1) {
      std::cout << "n of barrelRecHitsHandle " << barrelRecHitsHandle->size() << std::endl;
      for (EcalRecHitCollection::const_iterator hit = barrelRecHitsHandle->begin(); hit != barrelRecHitsHandle->end(); ++hit) {
	std::cout << "hit : detid(ieta,iphi) " << (EBDetId)(hit->id()) << " time " << hit->time() << " energy " <<  hit->energy() << std::endl;
      }
      std::cout << "n of endcapRecHitsHandle " << endcapRecHitsHandle->size() << std::endl;
      for (EcalRecHitCollection::const_iterator hit = endcapRecHitsHandle->begin(); hit != endcapRecHitsHandle->end(); ++hit) {
	std::cout << "hit : detid(ieta,iphi) " << (EEDetId)(hit->id()) << " time " << hit->time() << " energy " <<  hit->energy() << std::endl;
      }
      std::cout << "n of hbhe " << hbhe->size() << std::endl;
      for (HBHERecHitCollection::const_iterator hit = hbhe->begin(); hit != hbhe->end(); ++hit) {
	std::cout << "hit : detid(ieta,iphi) " << hit->id() << " time " << hit->time() << " energy " <<  hit->energy() << std::endl;
      }
    }
#endif
    unsigned int nTracks=0, ngoodTk=0, nselTk=0;
    int          ieta=999;
    for (trkDetItr = trkCaloDirections.begin(); trkDetItr != trkCaloDirections.end(); trkDetItr++,nTracks++){
      bool l3Track  = (std::find(goodTks.begin(),goodTks.end(),trkDetItr->trkItr) != goodTks.end());
      const reco::Track* pTrack = &(*(trkDetItr->trkItr));
      math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), 
				 pTrack->pz(), pTrack->p());
      bool selectTk = spr::goodTrack(pTrack,leadPV,selectionParameters,((verbosity/100)%10>1));
      double eMipDR=9999., e_inCone=0, conehmaxNearP=0, mindR=999.9, hCone=0;
      if (trkDetItr->okHCAL) {
	HcalDetId detId = (HcalDetId)(trkDetItr->detIdHCAL);
	ieta = detId.ieta();
      }
      for (unsigned k=0; k<vec[0].size(); ++k) {
	double deltaR = dR(v4, vec[0][k]);
	if (deltaR<mindR) mindR = deltaR;
      }
#ifdef DebugLog
      if ((verbosity/100)%10 > 1) std::cout << "Track ECAL " << trkDetItr->okECAL << " HCAL " << trkDetItr->okHCAL << " Flag " << selectTk << std::endl;
#endif
      if (selectTk && trkDetItr->okECAL && trkDetItr->okHCAL) {
	ngoodTk++;
	int nRH_eMipDR=0, nNearTRKs=0;
	double e1 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
				     trkDetItr->pointHCAL, trkDetItr->pointECAL,
				     a_neutR1, trkDetItr->directionECAL, nRH_eMipDR);
	double e2 =  spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
				     trkDetItr->pointHCAL, trkDetItr->pointECAL,
				     a_neutR2, trkDetItr->directionECAL, nRH_eMipDR);
	eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, endcapRecHitsHandle,
				 trkDetItr->pointHCAL, trkDetItr->pointECAL,
				 a_mipR, trkDetItr->directionECAL, nRH_eMipDR);
	conehmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR, nNearTRKs, ((verbosity/100)%10>1));
	e_inCone = e2 - e1;
	double             distFromHotCell=-99.0;
	int                nRecHitsCone=-99, ietaHotCell=-99, iphiHotCell=-99;
	GlobalPoint        gposHotCell(0.,0.,0.);
	std::vector<DetId> coneRecHitDetIds;
  	hCone    = spr::eCone_hcal(geo, hbhe, trkDetItr->pointHCAL, 
				   trkDetItr->pointECAL,
				   a_coneR, trkDetItr->directionHCAL, 
				   nRecHitsCone, coneRecHitDetIds,
				   distFromHotCell, ietaHotCell, iphiHotCell,
				   gposHotCell, -1);
	if (eMipDR<1.0) nselTk++;
      }
      if (l3Track) {
	fillHist(10,v4);
	if (selectTk) {
	  fillHist(11,v4);
	  fillCuts(0, eMipDR, conehmaxNearP, e_inCone, v4, ieta, (mindR>dr_L1));
	  if (conehmaxNearP < cutCharge) {
	    fillHist(12,v4);
	    if (eMipDR < cutMip) {
	      fillHist(13,v4);
	      fillEnergy(1, ieta, hCone, eMipDR, v4);
	      fillEnergy(0, ieta, hCone, eMipDR, v4);
	      if (e_inCone < cutNeutral) {
		fillHist(14, v4);
		fillEnergy(4, ieta, hCone, eMipDR, v4);
		fillEnergy(3, ieta, hCone, eMipDR, v4);
	      }
	    }
	  }
	}
      } else if (doL2L3) {
	fillHist(15,v4);
	if (selectTk) {
	  fillHist(16,v4);
	  fillCuts(1, eMipDR, conehmaxNearP, e_inCone, v4, ieta, (mindR>dr_L1));
	  if (conehmaxNearP < cutCharge) {
	    fillHist(17,v4);
	    if (eMipDR < cutMip) {
	      fillHist(18,v4);
	      fillEnergy(2, ieta, hCone, eMipDR, v4);
	      fillEnergy(0, ieta, hCone, eMipDR, v4);
	      if (e_inCone < cutNeutral) {
		fillHist(19, v4);
		fillEnergy(5, ieta, hCone, eMipDR, v4);
		fillEnergy(3, ieta, hCone, eMipDR, v4);
	      }
	    }
	  }
	}
      }
    }
    //    std::cout << "Number of tracks selected offline " << nselTk << std::endl;      
  }
}

void IsoTrig::chgIsolation(double& etaTriggered, double& phiTriggered, 
			   edm::Handle<reco::TrackCollection>& trkCollection, 
			   const edm::Event& theEvent) {
  clearChgIsolnTreeVectors();
#ifdef DebugLog
  if (verbosity%10>0)  std::cout << "Inside chgIsolation() with eta/phi Triggered: " << etaTriggered << "/" << phiTriggered << std::endl;
#endif
  std::vector<double> maxP;
  
  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr; 
  const MagneticField *bField = bFieldH.product();
  const CaloGeometry* geo = pG.product();
  std::vector<spr::propagatedTrackDirection> trkCaloDirections1;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDirections1, ((verbosity/100)%10>2));
#ifdef DebugLog
  if (verbosity%10>0) std::cout << "Propagated TrkCollection" << std::endl;
#endif
  for (unsigned int k=0; k<pixelIsolationConeSizeAtEC_.size(); ++k)
    maxP.push_back(0);
  unsigned i = pixelTrackRefsHE.size();
  std::vector<std::pair<unsigned int, std::pair<double, double>>>  VecSeedsatEC;
  //loop to select isolated tracks
  for (unsigned iS=0; iS<pixelTrackRefsHE.size(); iS++) {
    if (pixelTrackRefsHE[iS]->p() > minPTrackValue_) {
      bool vtxMatch = false;
      //associate to vertex (in Z) 
      unsigned int ivSel = recVtxs->size();
      double minDZ = 100;
      for (unsigned int iv = 0; iv < recVtxs->size(); ++iv) {
	if (fabs(pixelTrackRefsHE[iS]->dz((*recVtxs)[iv].position()))<minDZ) {
	  minDZ  = fabs(pixelTrackRefsHE[iS]->dz((*recVtxs)[iv].position()));
	  ivSel  = iv;
	}
      }
      //cut on dYX:
      if (ivSel == recVtxs->size()) {
	vtxMatch = true;
      } else if (fabs(pixelTrackRefsHE[iS]->dxy((*recVtxs)[ivSel].position()))<vtxCutSeed_){
	vtxMatch = true;
      }
      //select tracks not matched to triggered L1 jet
      double R = deltaR(etaTriggered, phiTriggered, pixelTrackRefsHE[iS]->eta(), 
			pixelTrackRefsHE[iS]->phi());
      if (R > tauUnbiasCone_ && vtxMatch) {
	//propagate seed track to ECAL surface:
	std::pair<double,double> seedCooAtEC;
	// in case vertex is found:
	if (minDZ!=100) seedCooAtEC=GetEtaPhiAtEcal(pixelTrackRefsHE[iS]->eta(), pixelTrackRefsHE[iS]->phi(), pixelTrackRefsHE[iS]->pt(), pixelTrackRefsHE[iS]->charge(), (*recVtxs)[ivSel].z());
	//in case vertex is not found:
	else            seedCooAtEC=GetEtaPhiAtEcal(pixelTrackRefsHE[iS]->eta(), pixelTrackRefsHE[iS]->phi(), pixelTrackRefsHE[iS]->pt(), pixelTrackRefsHE[iS]->charge(), 0);
	VecSeedsatEC.push_back(std::make_pair(iS, seedCooAtEC));
      }
    }
  }
  for (unsigned int l=0; l<VecSeedsatEC.size(); l++) {
    unsigned int iSeed =  VecSeedsatEC[l].first;
    math::XYZTLorentzVector v1(pixelTrackRefsHE[iSeed]->px(),pixelTrackRefsHE[iSeed]->py(),
			       pixelTrackRefsHE[iSeed]->pz(),pixelTrackRefsHE[iSeed]->p());

    for (unsigned int j=0; j<VecSeedsatEC.size(); j++) {
      unsigned int iSurr = VecSeedsatEC[j].first;
      if (iSeed != iSurr) {
	//define preliminary cone around seed track impact point from which tracks will be extrapolated:
	//	edm::Ref<reco::IsolatedPixelTrackCandidateCollection> cand2ref =
	//	  edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(L2cands, iSurr);
	if (deltaR(pixelTrackRefsHE[iSeed]->eta(), pixelTrackRefsHE[iSeed]->phi(), pixelTrackRefsHE[iSurr]->eta(), 
		   pixelTrackRefsHE[iSurr]->phi()) < prelimCone_) {
	  unsigned int ivSel = recVtxs->size();
	  double minDZ2=100;
	  for (unsigned int iv = 0; iv < recVtxs->size(); ++iv) {
	    if (fabs(pixelTrackRefsHE[iSurr]->dz((*recVtxs)[iv].position()))<minDZ2) {
	      minDZ2  = fabs(pixelTrackRefsHE[iSurr]->dz((*recVtxs)[iv].position()));
	      ivSel   = iv;
	    }
	  }
	  //cut ot dXY:
	  if (minDZ2==100 || fabs(pixelTrackRefsHE[iSurr]->dxy((*recVtxs)[ivSel].position()))<vtxCutIsol_) {
	    //calculate distance at ECAL surface and update isolation: 
	    double dist = getDistInCM(VecSeedsatEC[i].second.first, VecSeedsatEC[i].second.second, VecSeedsatEC[j].second.first, VecSeedsatEC[j].second.second);
	    for (unsigned int k=0; k<pixelIsolationConeSizeAtEC_.size(); ++k) {
	      if (dist<pixelIsolationConeSizeAtEC_[k]) {
		if (pixelTrackRefsHE[iSurr]->p() > maxP[k]) 
		  maxP[k] = pixelTrackRefsHE[iSurr]->p();
	      }
	    }
	  }
	}
      }
    }
  
    double conehmaxNearP = -1; bool selectTk=false;
    double mindR=999.9; int nTracks=0;
    math::XYZTLorentzVector mindRvec;
    for (trkDetItr = trkCaloDirections1.begin(); trkDetItr != trkCaloDirections1.end(); trkDetItr++, nTracks++){
      int nNearTRKs=0;
      const reco::Track* pTrack = &(*(trkDetItr->trkItr));
      math::XYZTLorentzVector v2(pTrack->px(), pTrack->py(),  
				 pTrack->pz(), pTrack->p());
      double dr   = dR(v1,v2);
      if (dr<mindR) {
	selectTk = spr::goodTrack(pTrack,leadPV,selectionParameters,((verbosity/100)%10>1));
	conehmaxNearP = spr::chargeIsolationCone(nTracks, trkCaloDirections1, a_charIsoR, nNearTRKs, ((verbosity/100)%10>1));
	mindR = dr;
	mindRvec = v2;
      }
    }
    pushChgIsolnTreeVecs(v1, mindRvec, maxP, conehmaxNearP, selectTk);
  }
  ChgIsolnTree->Fill();
}

void IsoTrig::getGoodTracks(const edm::Event& iEvent,
			    edm::Handle<reco::TrackCollection>& trkCollection){

  t_nGoodTk->clear();
  std::vector<int> nGood(4,0);
  if (trkCollection.isValid()) {
    // get handles to calogeometry and calotopology
    const CaloGeometry* geo = pG.product();
    const MagneticField *bField = bFieldH.product();
    std::vector<spr::propagatedTrackDirection> trkCaloDirections;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDirections, ((verbosity/100)%10>2));

    // get the trigger jet
    edm::Handle<trigger::TriggerFilterObjectWithRefs> l1trigobj;
    iEvent.getByToken(tok_l1cand_, l1trigobj);
  
    std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1tauobjref;
    l1trigobj->getObjects(trigger::TriggerL1TauJet, l1tauobjref);
    std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1jetobjref;
    l1trigobj->getObjects(trigger::TriggerL1CenJet, l1jetobjref);
    std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1forjetobjref;
    l1trigobj->getObjects(trigger::TriggerL1ForJet, l1forjetobjref);

    double ptTriggered(-10), etaTriggered(-100), phiTriggered(-100); 
    for (unsigned int p=0; p<l1tauobjref.size(); p++) {
      if (l1tauobjref[p]->pt()>ptTriggered) {
	ptTriggered  = l1tauobjref[p]->pt(); 
	phiTriggered = l1tauobjref[p]->phi();
	etaTriggered = l1tauobjref[p]->eta();
      }
    }
    for (unsigned int p=0; p<l1jetobjref.size(); p++) {
      if (l1jetobjref[p]->pt()>ptTriggered) {
	ptTriggered  = l1jetobjref[p]->pt();
	phiTriggered = l1jetobjref[p]->phi();
	etaTriggered = l1jetobjref[p]->eta();
      }
    }
    for (unsigned int p=0; p<l1forjetobjref.size(); p++) {
      if (l1forjetobjref[p]->pt()>ptTriggered) {
	ptTriggered=l1forjetobjref[p]->pt();
	phiTriggered=l1forjetobjref[p]->phi();
	etaTriggered=l1forjetobjref[p]->eta();
      }
    }
    double pTriggered = ptTriggered*cosh(etaTriggered);
    math::XYZTLorentzVector pTrigger(ptTriggered*cos(phiTriggered),
				     ptTriggered*sin(phiTriggered),
				     pTriggered*tanh(etaTriggered), pTriggered);

    std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;
    unsigned int nTracks(0);
    for (trkDetItr = trkCaloDirections.begin(); trkDetItr != trkCaloDirections.end(); trkDetItr++,nTracks++){
      const reco::Track* pTrack = &(*(trkDetItr->trkItr));
      math::XYZTLorentzVector v4(pTrack->px(), pTrack->py(), 
				 pTrack->pz(), pTrack->p());
      bool selectTk = spr::goodTrack(pTrack,leadPV,selectionParameters,((verbosity/100)%10>1));
      double mindR = dR(v4, pTrigger);
#ifdef DebugLog
      if ((verbosity/100)%10 > 1) std::cout << "Track ECAL " << trkDetItr->okECAL << " HCAL " << trkDetItr->okHCAL << " Flag " << selectTk << std::endl;
#endif
      if (selectTk && trkDetItr->okECAL && trkDetItr->okHCAL && mindR > 1.0) {
	int nRH_eMipDR(0), nNearTRKs(0);
	double eMipDR = spr::eCone_ecal(geo, barrelRecHitsHandle, 
					endcapRecHitsHandle,
					trkDetItr->pointHCAL, 
					trkDetItr->pointECAL, a_mipR, 
					trkDetItr->directionECAL, nRH_eMipDR);
	double conehmaxNearP = spr::chargeIsolationCone(nTracks, 
							trkCaloDirections, 
							a_charIsoR, nNearTRKs, 
							((verbosity/100)%10>1));
	if (conehmaxNearP < 2.0 && eMipDR<1.0) {
	  if (pTrack->p() >= 20 && pTrack->p() < 30) {
	    ++nGood[0];
	  } else if (pTrack->p() >= 30 && pTrack->p() < 40) {
	    ++nGood[1];
	  } else if (pTrack->p() >= 40 && pTrack->p() < 60) {
	    ++nGood[2];
	  } else if (pTrack->p() >= 60 && pTrack->p() < 100) {
	    ++nGood[3];
	  }
	}
      }
    }
  }

  for (unsigned int ii=0; ii<nGood.size(); ++ii)
    t_nGoodTk->push_back(nGood[ii]);
}

void IsoTrig::fillHist(int indx, math::XYZTLorentzVector& vec) {
  h_p[indx]->Fill(vec.r());
  h_pt[indx]->Fill(vec.pt());
  h_eta[indx]->Fill(vec.eta());
  h_phi[indx]->Fill(vec.phi());
}

void IsoTrig::fillDifferences(int indx, math::XYZTLorentzVector& vec1, 
			      math::XYZTLorentzVector& vec2, bool debug) {
  double dr     = dR(vec1,vec2);
  double deta   = dEta(vec1, vec2);
  double dphi   = dPhi(vec1, vec2);
  double dpt    = dPt(vec1, vec2);
  double dp     = dP(vec1, vec2);
  double dinvpt = dinvPt(vec1, vec2);
  h_dEta[indx]  ->Fill(deta);
  h_dPhi[indx]  ->Fill(dphi);
  h_dPt[indx]   ->Fill(dpt);
  h_dP[indx]    ->Fill(dp);
  h_dinvPt[indx]->Fill(dinvpt);
  h_mindR[indx] ->Fill(dr);
#ifdef DebugLog
  if (debug) std::cout << "mindR for index " << indx << " is " << dr << " deta " << deta << " dphi " << dphi << " dpt " << dpt <<  " dinvpt " << dinvpt <<std::endl;
#endif
}

void IsoTrig::fillCuts(int indx, double eMipDR, double conehmaxNearP, double e_inCone, math::XYZTLorentzVector& vec, int ieta, bool cut) {
  h_eMip[indx]     ->Fill(eMipDR);
  h_eMaxNearP[indx]->Fill(conehmaxNearP);
  h_eNeutIso[indx] ->Fill(e_inCone);
  if ((conehmaxNearP < cutCharge) && (eMipDR < cutMip)) {
    for (int lim=0; lim<5; ++lim) {
      if ((vec.r()>pLimits[lim]) && (vec.r()<=pLimits[lim+1])) {
	h_etaMipTracks[lim][indx][0]->Fill((double)(ieta));
	if (cut) h_etaMipTracks[lim][indx][1]->Fill((double)(ieta));
	if (e_inCone < cutNeutral) {
	  h_etaCalibTracks[lim][indx][0]->Fill((double)(ieta));
	  if (cut) h_etaCalibTracks[lim][indx][1]->Fill((double)(ieta));
	}
      }
    }
  }
}

void IsoTrig::fillEnergy(int indx, int ieta, double hCone, double eMipDR, math::XYZTLorentzVector& vec) {
  int kk(-1);
  if      (ieta > 0 && ieta < 25)  kk = 23 + ieta;
  else if (ieta > -25 && ieta < 0) kk = -(ieta + 1);
  if (kk >= 0 && eMipDR > 0.01 && hCone > 1.0) {
    for (int lim=0; lim<5; ++lim) {
      if ((vec.r()>pLimits[lim]) && (vec.r()<=pLimits[lim+1])) {
	h_eHcal[lim][indx][kk]     ->Fill(hCone);
	h_eCalo[lim][indx][kk]     ->Fill(hCone+eMipDR);
      }
    }
  }
}

double IsoTrig::dEta(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (vec1.eta()-vec2.eta());
}

double IsoTrig::dPhi(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {

  double phi1 = vec1.phi();
  if (phi1 < 0) phi1 += 2.0*M_PI;
  double phi2 = vec2.phi();
  if (phi2 < 0) phi2 += 2.0*M_PI;
  double dphi = phi1-phi2;
  if (dphi > M_PI)       dphi -= 2.*M_PI;
  else if (dphi < -M_PI) dphi += 2.*M_PI;
  return dphi;
}

double IsoTrig::dR(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  double deta = dEta(vec1,vec2);
  double dphi = dPhi(vec1,vec2);
  return std::sqrt(deta*deta + dphi*dphi);
}

double IsoTrig::dPt(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (vec1.pt()-vec2.pt());
}

double IsoTrig::dP(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return (std::abs(vec1.r()-vec2.r()));
}

double IsoTrig::dinvPt(math::XYZTLorentzVector& vec1, math::XYZTLorentzVector& vec2) {
  return ((1/vec1.pt())-(1/vec2.pt()));
}

std::pair<double,double> IsoTrig::etaPhiTrigger() {
  double eta(0), phi(0), ptmax(0);
  for (unsigned int k=0; k<vec[0].size(); ++k) {
    if (k == 0) {
      eta   = vec[0][k].eta();
      phi   = vec[0][k].phi();
      ptmax = vec[0][k].pt();
    } else if (vec[0][k].pt() > ptmax) {
      eta   = vec[0][k].eta();
      phi   = vec[0][k].phi();
      ptmax = vec[0][k].pt();
    }
  }
  return std::pair<double,double>(eta,phi);
}

std::pair<double,double> IsoTrig::GetEtaPhiAtEcal(double etaIP, double phiIP, 
						  double pT, int charge, 
						  double vtxZ) {

  double deltaPhi=0;
  double etaEC = 100;
  double phiEC = 100;

  double Rcurv = 9999999;
  if (bfVal!=0) Rcurv=pT*33.3*100/(bfVal*10); //r(m)=pT(GeV)*33.3/B(kG)

  double ecDist = zEE_;
  double ecRad  = rEB_;  //radius of ECAL barrel (cm)
  double theta  = 2*atan(exp(-etaIP));
  double zNew   = 0;
  if (theta>0.5*M_PI) theta=M_PI-theta;
  if (fabs(etaIP)<1.479) {
    if ((0.5*ecRad/Rcurv)>1) {
      etaEC         = 10000;
      deltaPhi      = 0;
    } else {
      deltaPhi      =-charge*asin(0.5*ecRad/Rcurv);
      double alpha1 = 2*asin(0.5*ecRad/Rcurv);
      double z      = ecRad/tan(theta);
      if (etaIP>0) zNew = z*(Rcurv*alpha1)/ecRad+vtxZ; //new z-coordinate of track
      else         zNew =-z*(Rcurv*alpha1)/ecRad+vtxZ; //new z-coordinate of track
      double zAbs=fabs(zNew);
      if (zAbs<ecDist) {
        etaEC    = -log(tan(0.5*atan(ecRad/zAbs)));
        deltaPhi = -charge*asin(0.5*ecRad/Rcurv);
      }
      if (zAbs>ecDist) {
        zAbs           = (fabs(etaIP)/etaIP)*ecDist;
        double Zflight = fabs(zAbs-vtxZ);
        double alpha   = (Zflight*ecRad)/(z*Rcurv);
        double Rec     = 2*Rcurv*sin(alpha/2);
        deltaPhi       =-charge*alpha/2;
        etaEC          =-log(tan(0.5*atan(Rec/ecDist)));
      }
    }
  } else {
    zNew           = (fabs(etaIP)/etaIP)*ecDist;
    double Zflight = fabs(zNew-vtxZ);
    double Rvirt   = fabs(Zflight*tan(theta));
    double Rec     = 2*Rcurv*sin(Rvirt/(2*Rcurv));
    deltaPhi       =-(charge)*(Rvirt/(2*Rcurv));
    etaEC          =-log(tan(0.5*atan(Rec/ecDist)));
  }

  if (zNew<0) etaEC=-etaEC;
  phiEC            = phiIP+deltaPhi;

  if (phiEC<-M_PI) phiEC += 2*M_PI;
  if (phiEC>M_PI)  phiEC -= 2*M_PI;

  std::pair<double,double> retVal(etaEC,phiEC);
  return retVal;
}

double IsoTrig::getDistInCM(double eta1,double phi1, double eta2,double phi2) {
  double Rec;
  double theta1=2*atan(exp(-eta1));
  double theta2=2*atan(exp(-eta2));
  if (fabs(eta1)<1.479) Rec=rEB_; //radius of ECAL barrel
  else if (fabs(eta1)>1.479&&fabs(eta1)<7.0) Rec=tan(theta1)*zEE_; //distance from IP to ECAL endcap
  else return 1000;

  //|vect| times tg of acos(scalar product)
  double angle=acos((sin(theta1)*sin(theta2)*(sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2))+cos(theta1)*cos(theta2)));
  if (angle<0.5*M_PI) return fabs((Rec/sin(theta1))*tan(angle));
  else return 1000;
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsoTrig);

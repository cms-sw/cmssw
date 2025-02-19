// $Id: FourVectorHLTriggerOffline.cc,v 1.40 2011/10/28 16:34:34 olzem Exp $
// See header file for information. 
#include "TMath.h"
#include "HLTriggerOffline/Common/interface/FourVectorHLTriggerOffline.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include <map>
#include <utility>


using namespace edm;
using namespace trigger;

FourVectorHLTriggerOffline::FourVectorHLTriggerOffline(const edm::ParameterSet& iConfig):
  resetMe_(true),  currentRun_(-99)
{
  LogDebug("FourVectorHLTriggerOffline") << "constructor...." ;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("FourVectorHLTriggerOffline") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  dirname_ = iConfig.getUntrackedParameter("dirname",
					   std::string("HLT/FourVector/"));
  //dirname_ +=  iConfig.getParameter<std::string>("@module_label");
  
  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirname_);
  }
  
  processname_ = iConfig.getParameter<std::string>("processname");

  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",20);
  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);
     // this is the list of paths to look at.
     std::vector<edm::ParameterSet> paths = 
     iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");
     for(std::vector<edm::ParameterSet>::iterator 
	pathconf = paths.begin() ; pathconf != paths.end(); 
      pathconf++) {
       std::pair<std::string, std::string> custompathnamepair;
       custompathnamepair.first =pathconf->getParameter<std::string>("pathname"); 
       custompathnamepair.second = pathconf->getParameter<std::string>("denompathname");   
       custompathnamepairs_.push_back(custompathnamepair);
       //    customdenompathnames_.push_back(pathconf->getParameter<std::string>("denompathname"));  
       // custompathnames_.push_back(pathconf->getParameter<std::string>("pathname"));  
    }

  if (hltPaths_.size() > 0)
    {
      // book a histogram of scalers
     scalersSelect = dbe_->book1D("selectedScalers","Selected Scalers", hltPaths_.size(), 0.0, (double)hltPaths_.size());
    }

 
  triggerSummaryLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerResultsLabel");

  electronEtaMax_ = iConfig.getUntrackedParameter<double>("electronEtaMax",2.5);
  electronEtMin_ = iConfig.getUntrackedParameter<double>("electronEtMin",3.0);
  electronDRMatch_  =iConfig.getUntrackedParameter<double>("electronDRMatch",0.3); 

  muonEtaMax_ = iConfig.getUntrackedParameter<double>("muonEtaMax",2.1);
  muonEtMin_ = iConfig.getUntrackedParameter<double>("muonEtMin",3.0);
  muonDRMatch_  =iConfig.getUntrackedParameter<double>("muonDRMatch",0.3); 

  tauEtaMax_ = iConfig.getUntrackedParameter<double>("tauEtaMax",2.5);
  tauEtMin_ = iConfig.getUntrackedParameter<double>("tauEtMin",3.0);
  tauDRMatch_  =iConfig.getUntrackedParameter<double>("tauDRMatch",0.3); 

  jetEtaMax_ = iConfig.getUntrackedParameter<double>("jetEtaMax",5.0);
  jetEtMin_ = iConfig.getUntrackedParameter<double>("jetEtMin",10.0);
  jetDRMatch_  =iConfig.getUntrackedParameter<double>("jetDRMatch",0.3); 

  bjetEtaMax_ = iConfig.getUntrackedParameter<double>("bjetEtaMax",2.5);
  bjetEtMin_ = iConfig.getUntrackedParameter<double>("bjetEtMin",10.0);
  bjetDRMatch_  =iConfig.getUntrackedParameter<double>("bjetDRMatch",0.3); 

  photonEtaMax_ = iConfig.getUntrackedParameter<double>("photonEtaMax",2.5);
  photonEtMin_ = iConfig.getUntrackedParameter<double>("photonEtMin",3.0);
  photonDRMatch_  =iConfig.getUntrackedParameter<double>("photonDRMatch",0.3); 

  trackEtaMax_ = iConfig.getUntrackedParameter<double>("trackEtaMax",2.5);
  trackEtMin_ = iConfig.getUntrackedParameter<double>("trackEtMin",3.0);
  trackDRMatch_  =iConfig.getUntrackedParameter<double>("trackDRMatch",0.3); 

  metMin_ = iConfig.getUntrackedParameter<double>("metMin",10.0);
  htMin_ = iConfig.getUntrackedParameter<double>("htMin",10.0);
  sumEtMin_ = iConfig.getUntrackedParameter<double>("sumEtMin",10.0);
  
}


FourVectorHLTriggerOffline::~FourVectorHLTriggerOffline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
FourVectorHLTriggerOffline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace trigger;
  ++nev_;
  LogDebug("FourVectorHLTriggerOffline")<< " analyze...." ;
  
  Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByLabel("genParticles", genParticles);
  if(!genParticles.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "genParticles not found, "
      "skipping event"; 
    return;
  }

  Handle<reco::GenJetCollection> genJets;
  iEvent.getByLabel("iterativeCone5GenJets",genJets);
  if(!genJets.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "genJets not found, "
      "skipping event"; 
    return;
  }

  Handle<reco::GenMETCollection> genMets;
  iEvent.getByLabel("genMetTrue",genMets);
  if(!genMets.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "genMets not found, "
      "skipping event"; 
    return;
  }

  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(triggerResultsLabel_,triggerResults);
  if(!triggerResults.isValid()) {
    edm::InputTag triggerResultsLabelFU(triggerResultsLabel_.label(),triggerResultsLabel_.instance(), "FU");
   iEvent.getByLabel(triggerResultsLabelFU,triggerResults);
  if(!triggerResults.isValid()) {
    edm::LogInfo("FourVectorHLTriggerOffline") << "TriggerResults not found, "
      "skipping event"; 
    return;
   }
  }
  const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
  int npath = triggerResults->size();

  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) {
    edm::InputTag triggerSummaryLabelFU(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(), "FU");
   iEvent.getByLabel(triggerSummaryLabelFU,triggerObj);
  if(!triggerObj.isValid()) {
    edm::LogInfo("FourVectorHLTriggerOffline") << "TriggerEvent not found, "
      "skipping event"; 
    return;
   }
  }

  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByLabel("muons",muonHandle);
  if(!muonHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "muonHandle not found, ";
    //  "skipping event"; 
    //  return;
   }

  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  iEvent.getByLabel("gsfElectrons",gsfElectrons); 
  if(!gsfElectrons.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "gsfElectrons not found, ";
      //"skipping event"; 
      //return;
  }

  edm::Handle<reco::CaloTauCollection> tauHandle;
  iEvent.getByLabel("caloRecoTauProducer",tauHandle);
  if(!tauHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "tauHandle not found, ";
      //"skipping event"; 
      //return;
  }

  edm::Handle<reco::CaloJetCollection> jetHandle;
  iEvent.getByLabel("iterativeCone5CaloJets",jetHandle);
  if(!jetHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "jetHandle not found, ";
      //"skipping event"; 
      //return;
  }
 
   // Get b tag information
 edm::Handle<reco::JetTagCollection> bTagIPHandle;
 iEvent.getByLabel("jetProbabilityBJetTags", bTagIPHandle);
 if (!bTagIPHandle.isValid()) {
    edm::LogInfo("FourVectorHLTriggerOffline") << "mTagIPHandle trackCountingHighEffJetTags not found, ";
      //"skipping event"; 
      //return;
  }

   // Get b tag information
 edm::Handle<reco::JetTagCollection> bTagMuHandle;
 iEvent.getByLabel("softMuonBJetTags", bTagMuHandle);
 if (!bTagMuHandle.isValid()) {
    edm::LogInfo("FourVectorHLTriggerOffline") << "bTagMuHandle  not found, ";
      //"skipping event"; 
      //return;
  }

  edm::Handle<reco::CaloMETCollection> metHandle;
  iEvent.getByLabel("met",metHandle);
  if(!metHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "metHandle not found, ";
      //"skipping event"; 
      //return;
  }

  edm::Handle<reco::PhotonCollection> photonHandle;
  iEvent.getByLabel("photons",photonHandle);
  if(!photonHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "photonHandle not found, ";
      //"skipping event"; 
      //return;
  }

  edm::Handle<reco::TrackCollection> trackHandle;
  iEvent.getByLabel("pixelTracks",trackHandle);
  if(!trackHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTriggerOffline") << "trackHandle not found, ";
      //"skipping event"; 
      //return;
  }

  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());

  // electron Monitor
	// ------------
  objMon<reco::GsfElectronCollection> eleMon;
  eleMon.setReco(gsfElectrons);
  eleMon.setMC(genParticles, 11, 1);
  eleMon.setLimits(electronEtaMax_, electronEtMin_, electronDRMatch_);
  
  eleMon.pushTriggerType(TriggerElectron);
  eleMon.pushTriggerType(TriggerL1NoIsoEG);
  eleMon.pushTriggerType(TriggerL1IsoEG);

  eleMon.pushL1TriggerType(TriggerL1NoIsoEG);
  eleMon.pushL1TriggerType(TriggerL1IsoEG);

  // muon Monitor
	// ------------
  objMon<reco::MuonCollection>  muoMon;
  muoMon.setReco(muonHandle);
  muoMon.setMC(genParticles, 13, 1);
  muoMon.setLimits(muonEtaMax_, muonEtMin_, muonDRMatch_);
  
  muoMon.pushTriggerType(TriggerMuon);
  muoMon.pushTriggerType(TriggerL1Mu);

  muoMon.pushL1TriggerType(TriggerL1Mu);
	
  // tau Monitor
	// ------------
  objMon<reco::CaloTauCollection>  tauMon;
  tauMon.setReco(tauHandle);
  tauMon.setMC(genParticles, 15, 3);
  tauMon.setLimits(tauEtaMax_, tauEtMin_, tauDRMatch_);
  
  tauMon.pushTriggerType(TriggerTau);
  tauMon.pushTriggerType(TriggerL1TauJet);

  tauMon.pushL1TriggerType(TriggerL1TauJet);
  tauMon.pushL1TriggerType(TriggerL1CenJet);
  tauMon.pushL1TriggerType(TriggerL1ForJet);
	
  // photon Monitor
	// ------------
  objMon<reco::PhotonCollection> phoMon;
  phoMon.setReco(photonHandle);
  phoMon.setMC(genParticles, 22, 1);
  phoMon.setLimits(photonEtaMax_, photonEtMin_, photonDRMatch_);
  
  phoMon.pushTriggerType(TriggerPhoton);

  phoMon.pushL1TriggerType(TriggerL1NoIsoEG);
  phoMon.pushL1TriggerType(TriggerL1IsoEG);

  // jet Monitor - NOTICE: we use genJets for MC
	// -------------------------------------------
  objMon<reco::CaloJetCollection> jetMon;
  jetMon.setReco(jetHandle);
  jetMon.setGenJets(true, genJets);
  jetMon.setLimits(jetEtaMax_, jetEtMin_, jetDRMatch_);

  jetMon.pushTriggerType(TriggerJet);
  jetMon.pushTriggerType(TriggerL1CenJet);
  jetMon.pushTriggerType(TriggerL1ForJet);
  
  jetMon.pushL1TriggerType(TriggerL1CenJet);
  jetMon.pushL1TriggerType(TriggerL1ForJet);
  jetMon.pushL1TriggerType(TriggerL1TauJet);

  // btag Monitor - NOTICE: we use genJets for MC
	// -------------------------------------------
  //objMon<reco::JetTagCollection> btagIPMon;
  objMon<reco::CaloJetCollection> btagIPMon; // CaloJet will not be used, this is only place holder
  //btagIPMon.setReco(jetHandle);
  btagIPMon.setRecoB(bTagIPHandle);
  btagIPMon.setMC(genParticles, 5, 3);
  btagIPMon.setGenJets(true, genJets);
  btagIPMon.setBJetsFlag(true);
  btagIPMon.setLimits(bjetEtaMax_, bjetEtMin_, bjetDRMatch_);

  btagIPMon.pushTriggerType(TriggerBJet);
  btagIPMon.pushTriggerType(TriggerJet);

  btagIPMon.pushL1TriggerType(TriggerL1CenJet);
  btagIPMon.pushL1TriggerType(TriggerL1ForJet);
  btagIPMon.pushL1TriggerType(TriggerL1TauJet);

  //objMon<reco::JetTagCollection> btagMuMon;
  objMon<reco::CaloJetCollection> btagMuMon; // CaloJet will not be used, this is only place holder
  //btagMuMon.setReco(jetHandle);
  btagMuMon.setRecoB(bTagMuHandle);
  btagMuMon.setMC(genParticles, 5, 3);
  btagMuMon.setGenJets(true, genJets);
  btagMuMon.setBJetsFlag(true);
  btagMuMon.setLimits(bjetEtaMax_, bjetEtMin_, bjetDRMatch_);

  btagMuMon.pushTriggerType(TriggerBJet);
  btagMuMon.pushTriggerType(TriggerJet);

  btagMuMon.pushL1TriggerType(TriggerL1CenJet);
  btagMuMon.pushL1TriggerType(TriggerL1ForJet);
  btagMuMon.pushL1TriggerType(TriggerL1TauJet);

 
  objMon<reco::CaloJetCollection> btagMon; // Generic btagMon

    for(PathInfoCollection::iterator v = hltPaths_.begin();
	v!= hltPaths_.end(); ++v ) 
{ 
    //LogTrace("FourVectorHLTriggerOffline") << " path " << v->getPath() << std::endl;
	      if (v->getPath().find("BTagIP") != std::string::npos ) btagMon = btagIPMon;
				else btagMon = btagMuMon;


  // did we pass the denomPath?
  bool denompassed = false;  
  for(int i = 0; i < npath; ++i) {
     if (triggerNames.triggerName(i).find(v->getDenomPath()) != std::string::npos && triggerResults->accept(i))
       { 
        denompassed = true;
        break;
       }
  }

  if (denompassed)
  {  

      eleMon.clearSets();
      muoMon.clearSets();
      tauMon.clearSets();
      phoMon.clearSets();
      jetMon.clearSets();
      btagMon.clearSets();

      // UNUSED
      //       int triggertype = 0;     
      //       triggertype = v->getObjectType();

      bool l1accept = false;
      edm::InputTag l1testTag(v->getl1Path(),"",processname_);
      const int l1index = triggerObj->filterIndex(l1testTag);
      if ( l1index >= triggerObj->sizeFilters() ) {
        edm::LogInfo("FourVectorHLTriggerOffline") << "no index "<< l1index << " of that name " << v->getl1Path() << "\t" << "\t" << l1testTag;
	continue; // not in this event
      }

      const trigger::Vids & idtype = triggerObj->filterIds(l1index);
      const trigger::Keys & l1k = triggerObj->filterKeys(l1index);
      l1accept = l1k.size() > 0;


      eleMon.monitorDenominator(v, l1accept, idtype, l1k, toc);
  		muoMon.monitorDenominator(v, l1accept, idtype, l1k, toc);
  		tauMon.monitorDenominator(v, l1accept, idtype, l1k, toc);
  		phoMon.monitorDenominator(v, l1accept, idtype, l1k, toc);
  		jetMon.monitorDenominator(v, l1accept, idtype, l1k, toc);
  		btagMon.monitorDenominator(v, l1accept, idtype, l1k, toc);
  
  		eleMon.fillL1Match(this);
  		muoMon.fillL1Match(this);
  		tauMon.fillL1Match(this);
  		phoMon.fillL1Match(this);
  		jetMon.fillL1Match(this);
  		btagMon.fillL1Match(this);
  
     // did we pass the numerator path?
     bool numpassed = false;
     for(int i = 0; i < npath; ++i) {

        if (triggerNames.triggerName(i) == v->getPath() && triggerResults->accept(i)) numpassed = true;

     }

     if (numpassed)
     { 

        if (!l1accept) {
           edm::LogInfo("FourVectorHLTriggerOffline") << "l1 seed path not accepted for hlt path "<< v->getPath() << "\t" << v->getl1Path();
        }

        // fill scaler histograms
        edm::InputTag filterTag = v->getTag();

	      // loop through indices and see if the filter is on the list of filters used by this path
      
        if (v->getLabel() == "dummy"){
        const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
	      //loop over labels
        for (std::vector<std::string>::const_iterator labelIter= filterLabels.begin(); labelIter!=filterLabels.end(); labelIter++)          
	      {

	        //std::cout << v->getPath() << "\t" << *labelIter << std::endl;
          // last match wins...
	        edm::InputTag testTag(*labelIter,"",processname_);
          int testindex = triggerObj->filterIndex(testTag);
          if ( !(testindex >= triggerObj->sizeFilters()) ) {

	          //std::cout << "found one! " << v->getPath() << "\t" << testTag.label() << std::endl; 
            filterTag = testTag; v->setLabel(*labelIter);}
	        }
        }
	
        const int index = triggerObj->filterIndex(filterTag);
        if ( index >= triggerObj->sizeFilters() ) {

	        //        std::cout << "WTF no index "<< index << " of that name "
	        //	     << filterTag << std::endl;
	        continue; // not in this event

        }

        //LogDebug("FourVectorHLTriggerOffline") << "filling ... " ;
        const trigger::Keys & k = triggerObj->filterKeys(index);
        //      const trigger::Vids & idtype = triggerObj->filterIds(index);
        // assume for now the first object type is the same as all objects in the collection
        //    std::cout << filterTag << "\t" << idtype.size() << "\t" << k.size() << std::endl;
        //     std::cout << "path " << v->getPath() << " trigger type "<<triggertype << std::endl;
        //if (k.size() > 0) v->getNOnHisto()->Fill(k.size());


        unsigned int NOn=0;

        for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {

          eleMon.monitorOnline(idtype, l1k, ki, toc, NOn);
          muoMon.monitorOnline(idtype, l1k, ki, toc, NOn);
          tauMon.monitorOnline(idtype, l1k, ki, toc, NOn);
          phoMon.monitorOnline(idtype, l1k, ki, toc, NOn);
          jetMon.monitorOnline(idtype, l1k, ki, toc, NOn);
          btagMon.monitorOnline(idtype, l1k, ki, toc, NOn);

        } //online object loop

        eleMon.fillOnlineMatch(this, l1k, toc);
        muoMon.fillOnlineMatch(this, l1k, toc);
        tauMon.fillOnlineMatch(this, l1k, toc);
        phoMon.fillOnlineMatch(this, l1k, toc);
        jetMon.fillOnlineMatch(this, l1k, toc);
        btagMon.fillOnlineMatch(this, l1k, toc);

      	eleMon.monitorOffline(this);
      	muoMon.monitorOffline(this);
      	tauMon.monitorOffline(this);
      	phoMon.monitorOffline(this);
      	jetMon.monitorOffline(this);
      	btagMon.monitorOffline(this);

      	eleMon.fillOffMatch(this);
      	muoMon.fillOffMatch(this);
      	tauMon.fillOffMatch(this);
      	phoMon.fillOffMatch(this);
      	jetMon.fillOffMatch(this);
      	btagMon.fillOffMatch(this);


     } //numpassed

    

   } //denompassed

  } //pathinfo loop

}



// -- method called once each job just before starting event loop  --------
void 
FourVectorHLTriggerOffline::beginJob()
{
  nev_ = 0;
  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    dbe->rmdir(dirname_);
  }
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    }  
}

// - method called once each job just after ending the event loop  ------------
void 
FourVectorHLTriggerOffline::endJob() 
{
   LogInfo("FourVectorHLTriggerOffline") << "analyzed " << nev_ << " events";
   return;
}


// BeginRun
void FourVectorHLTriggerOffline::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTriggerOffline") << "beginRun, run " << run.id();
  // HLT config does not change within runs!

  bool changed(false);  
  if (!hltConfig_.init(run, c, processname_, changed)) {
    processname_ = "FU";
    if (!hltConfig_.init(run, c, processname_, changed)){
  LogDebug("FourVectorHLTriggerOffline") << "HLTConfigProvider failed to initialize.";
    }
    // check if trigger name in (new) config
    //	std::cout << "Available TriggerNames are: " << std::endl;
    //	hltConfig_.dump("Triggers");
      }


  if (1)
 {
  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
  }


    const unsigned int n(hltConfig_.size());
    if (plotAll_){
    for (unsigned int j=0; j!=n; ++j) {
    std::string pathname = hltConfig_.triggerName(j);  
    std::string l1pathname = "dummy";
    for (unsigned int i=0; i!=n; ++i) {
      // std::cout << hltConfig_.triggerName(i) << std::endl;
    
    std::string denompathname = hltConfig_.triggerName(i);  
    int objectType = 0;
    int denomobjectType = 0;
    //parse pathname to guess object type
    if (pathname.find("MET") != std::string::npos) 
      objectType = trigger::TriggerMET;    
    if (pathname.find("SumET") != std::string::npos) 
      objectType = trigger::TriggerTET;    
    if (pathname.find("HT") != std::string::npos) 
      objectType = trigger::TriggerTET;    
    if (pathname.find("Jet") != std::string::npos) 
      objectType = trigger::TriggerJet;    
    if (pathname.find("Mu") != std::string::npos)
      objectType = trigger::TriggerMuon;    
    if (pathname.find("Ele") != std::string::npos) 
      objectType = trigger::TriggerElectron;    
    if (pathname.find("Photon") != std::string::npos) 
      objectType = trigger::TriggerPhoton;    
    if (pathname.find("Tau") != std::string::npos) 
      objectType = trigger::TriggerTau;    
    if (pathname.find("IsoTrack") != std::string::npos) 
      objectType = trigger::TriggerTrack;    
    if (pathname.find("BTag") != std::string::npos)
      objectType = trigger::TriggerBJet;    

    //parse denompathname to guess denomobject type
    if (denompathname.find("MET") != std::string::npos) 
      denomobjectType = trigger::TriggerMET;    
    if (denompathname.find("SumET") != std::string::npos) 
      denomobjectType = trigger::TriggerTET;    
    if (denompathname.find("HT") != std::string::npos) 
      denomobjectType = trigger::TriggerTET;    
    if (denompathname.find("Jet") != std::string::npos) 
      denomobjectType = trigger::TriggerJet;    
    if (denompathname.find("Mu") != std::string::npos)
      denomobjectType = trigger::TriggerMuon;    
    if (denompathname.find("Ele") != std::string::npos) 
      denomobjectType = trigger::TriggerElectron;    
    if (denompathname.find("Photon") != std::string::npos) 
      denomobjectType = trigger::TriggerPhoton;    
    if (denompathname.find("Tau") != std::string::npos) 
      denomobjectType = trigger::TriggerTau;    
    if (denompathname.find("IsoTrack") != std::string::npos) 
      denomobjectType = trigger::TriggerTrack;    
    if (denompathname.find("BTag") != std::string::npos) 
      denomobjectType = trigger::TriggerBJet;    

    // find L1 condition for numpath with numpath objecttype 

    // find PSet for L1 global seed for numpath, 
    // list module labels for numpath
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);

            for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
    	  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      //  std::cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << std::endl;
	      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
		{
		  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
		  //                  std::cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << std::endl;
		  //  l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
                  l1pathname = *numpathmodule; 
                  break; 
		}
  } 
   
    

    std::string filtername("dummy");
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (plotAll_ && denomobjectType == objectType && objectType != 0)
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));

    }
    }

    }
    else
    {
     // plot all diagonal combinations plus any other specified pairs
    for (unsigned int i=0; i!=n; ++i) {
      std::string denompathname = "";  
      std::string pathname = hltConfig_.triggerName(i);  
      std::string l1pathname = "dummy";
      int objectType = 0;
      //       int denomobjectType = 0; // UNUSED
    //parse pathname to guess object type
    if (pathname.find("MET") != std::string::npos) 
      objectType = trigger::TriggerMET;    
    if (pathname.find("SumET") != std::string::npos) 
      objectType = trigger::TriggerTET;    
    if (pathname.find("HT") != std::string::npos) 
      objectType = trigger::TriggerTET;    
    if (pathname.find("Jet") != std::string::npos) 
      objectType = trigger::TriggerJet;    
    if (pathname.find("Mu") != std::string::npos)
      objectType = trigger::TriggerMuon;    
    if (pathname.find("Ele") != std::string::npos) 
      objectType = trigger::TriggerElectron;    
    if (pathname.find("Photon") != std::string::npos) 
      objectType = trigger::TriggerPhoton;    
    if (pathname.find("Tau") != std::string::npos) 
      objectType = trigger::TriggerTau;    
    if (pathname.find("IsoTrack") != std::string::npos) 
      objectType = trigger::TriggerTrack;    
    if (pathname.find("BTag") != std::string::npos) 
      objectType = trigger::TriggerBJet;    

    // UNUSED
    //     //parse denompathname to guess denomobject type
    //     if (denompathname.find("MET") != std::string::npos) 
    //       denomobjectType = trigger::TriggerMET;    
    //     if (denompathname.find("SumET") != std::string::npos) 
    //       denomobjectType = trigger::TriggerTET;    
    //     if (denompathname.find("HT") != std::string::npos) 
    //       denomobjectType = trigger::TriggerTET;    
    //     if (denompathname.find("Jet") != std::string::npos) 
    //       denomobjectType = trigger::TriggerJet;    
    //     if (denompathname.find("Mu") != std::string::npos)
    //       denomobjectType = trigger::TriggerMuon;    
    //     if (denompathname.find("Ele") != std::string::npos) 
    //       denomobjectType = trigger::TriggerElectron;    
    //     if (denompathname.find("Photon") != std::string::npos) 
    //       denomobjectType = trigger::TriggerPhoton;    
    //     if (denompathname.find("Tau") != std::string::npos) 
    //       denomobjectType = trigger::TriggerTau;    
    //     if (denompathname.find("IsoTrack") != std::string::npos) 
    //       denomobjectType = trigger::TriggerTrack;    
    //     if (denompathname.find("BTag") != std::string::npos) 
    //       denomobjectType = trigger::TriggerBJet;    
    // find L1 condition for numpath with numpath objecttype 
    
    // find PSet for L1 global seed for numpath, 
    // list module labels for numpath
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);

    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
    	  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      //  std::cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << std::endl;
	      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
		{
		  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
		  //                  std::cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << std::endl;
                  //l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression"); 
                  l1pathname = *numpathmodule;
                  break; 
		}
    } 
   
    



    std::string filtername("dummy");
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (objectType == trigger::TriggerPhoton) ptMax = 100.0;
    if (objectType == trigger::TriggerElectron) ptMax = 100.0;
    if (objectType == trigger::TriggerMuon) ptMax = 100.0;
    if (objectType == trigger::TriggerTau) ptMax = 100.0;
    if (objectType == trigger::TriggerJet) ptMax = 300.0;
    if (objectType == trigger::TriggerBJet) ptMax = 300.0;
    if (objectType == trigger::TriggerMET) ptMax = 300.0;
    if (objectType == trigger::TriggerTET) ptMax = 300.0;
    if (objectType == trigger::TriggerTrack) ptMax = 100.0;

    if (objectType != 0){
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));
      //create folder for pathname
     }
    }
    // now loop over denom/num path pairs specified in cfg, 
    // recording the off-diagonal ones
    for (std::vector<std::pair<std::string, std::string> >::iterator custompathnamepair = custompathnamepairs_.begin(); custompathnamepair != custompathnamepairs_.end(); ++custompathnamepair)
    {
      if (custompathnamepair->first != custompathnamepair->second)
	{

      std::string denompathname = custompathnamepair->second;  
      std::string pathname = custompathnamepair->first;  
     
      // check that these exist
      bool foundfirst = false;
      bool foundsecond = false;
      for (unsigned int i=0; i!=n; ++i) {
	if (hltConfig_.triggerName(i) == denompathname) foundsecond = true;
	if (hltConfig_.triggerName(i) == pathname) foundfirst = true;
      } 
      if (!foundfirst)
	{
	  edm::LogInfo("FourVectorHLTriggerOffline") << "pathname not found, ignoring " << pathname;
          continue;
	}
      if (!foundsecond)
	{
	  edm::LogInfo("FourVectorHLTriggerOffline") << "denompathname not found, ignoring " << pathname;
          continue;
	}

     //std::cout << pathname << "\t" << denompathname << std::endl;
      std::string l1pathname = "dummy";
      int objectType = 0;
      //int denomobjectType = 0;
    //parse pathname to guess object type
    if (pathname.find("MET") != std::string::npos) 
      objectType = trigger::TriggerMET;    
    if (pathname.find("SumET") != std::string::npos) 
      objectType = trigger::TriggerTET;    
    if (pathname.find("HT") != std::string::npos) 
      objectType = trigger::TriggerTET;    
    if (pathname.find("Jet") != std::string::npos) 
      objectType = trigger::TriggerJet;    
    if (pathname.find("Mu") != std::string::npos)
      objectType = trigger::TriggerMuon;    
    if (pathname.find("Ele") != std::string::npos) 
      objectType = trigger::TriggerElectron;    
    if (pathname.find("Photon") != std::string::npos) 
      objectType = trigger::TriggerPhoton;    
    if (pathname.find("Tau") != std::string::npos) 
      objectType = trigger::TriggerTau;    
    if (pathname.find("IsoTrack") != std::string::npos) 
      objectType = trigger::TriggerTrack;    
    if (pathname.find("BTag") != std::string::npos)
      objectType = trigger::TriggerBJet;    
    // find L1 condition for numpath with numpath objecttype 

    // find PSet for L1 global seed for numpath, 
    // list module labels for numpath
  
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
    
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
    	  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      //  std::cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << std::endl;
	      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
		{
		  edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
		  //                  std::cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << std::endl;
		  // l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
                  l1pathname = *numpathmodule;
                  //std::cout << *numpathmodule << std::endl; 
                  break; 
		}
    }
    
    



    std::string filtername("dummy");
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (objectType == trigger::TriggerPhoton) ptMax = 100.0;
    if (objectType == trigger::TriggerElectron) ptMax = 100.0;
    if (objectType == trigger::TriggerMuon) ptMax = 100.0;
    if (objectType == trigger::TriggerTau) ptMax = 100.0;
    if (objectType == trigger::TriggerJet) ptMax = 300.0;
    if (objectType == trigger::TriggerBJet) ptMax = 300.0;
    if (objectType == trigger::TriggerMET) ptMax = 300.0;
    if (objectType == trigger::TriggerTET) ptMax = 300.0;
    if (objectType == trigger::TriggerTrack) ptMax = 100.0;

    if (objectType != 0)
    hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));
    
	}
    }

    }



    // now set up all of the histos for each path
    for(PathInfoCollection::iterator v = hltPaths_.begin();
	  v!= hltPaths_.end(); ++v ) {
    	MonitorElement *NOn, *onEtOn, *onEtavsonPhiOn=0;
    	MonitorElement *NMc, *mcEtMc, *mcEtavsmcPhiMc=0;
	MonitorElement *NOff, *offEtOff, *offEtavsoffPhiOff=0;
	MonitorElement *NL1, *l1EtL1, *l1Etavsl1PhiL1=0;
    	MonitorElement *NL1On, *l1EtL1On, *l1Etavsl1PhiL1On=0;
	MonitorElement *NL1Off, *offEtL1Off, *offEtavsoffPhiL1Off=0;
	MonitorElement *NOnOff, *offEtOnOff, *offEtavsoffPhiOnOff=0;
	MonitorElement *NL1Mc, *mcEtL1Mc, *mcEtavsmcPhiL1Mc=0;
	MonitorElement *NOffMc, *mcEtOffMc, *mcEtavsmcPhiOffMc=0;
	MonitorElement *NOnMc, *mcEtOnMc, *mcEtavsmcPhiOnMc=0;
    	MonitorElement *NL1OnUM, *l1EtL1OnUM, *l1Etavsl1PhiL1OnUM=0;
	MonitorElement *NL1OffUM, *offEtL1OffUM, *offEtavsoffPhiL1OffUM=0;
	MonitorElement *NOnOffUM, *offEtOnOffUM, *offEtavsoffPhiOnOffUM=0;
	MonitorElement *NL1McUM, *mcEtL1McUM, *mcEtavsmcPhiL1McUM=0;
	MonitorElement *NOffMcUM, *mcEtOffMcUM, *mcEtavsmcPhiOffMcUM=0;
	MonitorElement *NOnMcUM, *mcEtOnMcUM, *mcEtavsmcPhiOnMcUM=0;
  MonitorElement *mcDRL1Mc, *mcDROnMc, *mcDROffMc,
                   *offDRL1Off, *offDROnOff, *l1DRL1On=0;
	std::string labelname("dummy");
        labelname = v->getPath() + "_wrt_" + v->getDenomPath();
	std::string histoname(labelname+"_NOn");
	std::string title(labelname+" N online");



        double histEtaMax = 2.5;
        if (v->getObjectType() == trigger::TriggerMuon || v->getObjectType() == trigger::TriggerL1Mu) 
	  {
	    histEtaMax = muonEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerElectron || v->getObjectType() == trigger::TriggerL1NoIsoEG || v->getObjectType() == trigger::TriggerL1IsoEG )
	  {
	    histEtaMax = electronEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerTau || v->getObjectType() == trigger::TriggerL1TauJet )
	  {
	    histEtaMax = tauEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerJet || v->getObjectType() == trigger::TriggerL1CenJet || v->getObjectType() == trigger::TriggerL1ForJet )
	  {
	    histEtaMax = jetEtaMax_; 
	  }
        else if (v->getObjectType() == trigger::TriggerBJet)
	  {
	    histEtaMax = bjetEtaMax_;
	  }
        else if (v->getObjectType() == trigger::TriggerMET || v->getObjectType() == trigger::TriggerL1ETM )
	  {
	    histEtaMax = 5.0; 
	  }
        else if (v->getObjectType() == trigger::TriggerPhoton)
	  {
	    histEtaMax = photonEtaMax_; 
	  }
        else if (v->getObjectType() == trigger::TriggerTrack)
	  {
	    histEtaMax = trackEtaMax_; 
	  }

        TString pathfolder = dirname_ + TString("/") + v->getPath();
        dbe_->setCurrentFolder(pathfolder.Data());

	NOn =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NMc";
	title = labelname+" N Mc";
	NMc =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NOff";
	title = labelname+" N Off";
	NOff =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);
      
	histoname = labelname+"_NL1";
	title = labelname+" N L1";
	NL1 =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1On";
	title = labelname+" N L1On";
	NL1On =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1Off";
	title = labelname+" N L1Off";
	NL1Off =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NOnOff";
	title = labelname+" N OnOff";
	NOnOff =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1Mc";
	title = labelname+" N L1Mc";
	NL1Mc =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NOffMc";
	title = labelname+" N OffMc";
	NOffMc =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NOnMc";
	title = labelname+" N OnMc";
	NOnMc =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1OnUM";
	title = labelname+" N L1OnUM";
	NL1OnUM =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1OffUM";
	title = labelname+" N L1OffUM";
	NL1OffUM =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NOnOffUM";
	title = labelname+" N OnOffUM";
	NOnOffUM =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NL1McUM";
	title = labelname+" N L1McUM";
	NL1McUM =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NOffMcUM";
	title = labelname+" N OffMcUM";
	NOffMcUM =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_NOnMcUM";
	title = labelname+" N OnMcUM";
	NOnMcUM =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);

	histoname = labelname+"_mcEtMc";
	title = labelname+" mcE_t Mc";
	mcEtMc =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

        histoname = labelname+"_onEtOn";
	title = labelname+" onE_t online";
	onEtOn =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_offEtOff";
	title = labelname+" offE_t offline";
	offEtOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_l1EtL1";
	title = labelname+" l1E_t L1";
	l1EtL1 =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

        int nBins2D = 10;

	histoname = labelname+"_mcEtamcPhiMc";
	title = labelname+" mc#eta vs mc#phi Mc";
	mcEtavsmcPhiMc =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_onEtaonPhiOn";
	title = labelname+" on#eta vs on#phi online";
	onEtavsonPhiOn =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_offEtaoffPhiOff";
	title = labelname+" off#eta vs off#phi offline";
	offEtavsoffPhiOff =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_l1Etal1PhiL1";
	title = labelname+" l1#eta vs l1#phi L1";
	l1Etavsl1PhiL1 =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_l1EtL1On";
	title = labelname+" l1E_t L1+online";
	l1EtL1On =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_offEtL1Off";
	title = labelname+" offE_t L1+offline";
	offEtL1Off =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_offEtOnOff";
	title = labelname+" offE_t online+offline";
	offEtOnOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_mcEtL1Mc";
	title = labelname+" mcE_t L1+MC truth";
	mcEtL1Mc =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_mcEtOffMc";
	title = labelname+" mcE_t Off+MC truth";
	mcEtOffMc =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_mcEtOnMc";
	title = labelname+" mcE_t online+MC truth";
	mcEtOnMc =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_l1Etal1PhiL1On";
	title = labelname+" l1#eta vs l1#phi L1+online";
	l1Etavsl1PhiL1On =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_offEtaoffPhiL1Off";
	title = labelname+" off#eta vs off#phi L1+offline";
	offEtavsoffPhiL1Off =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_offEtaoffPhiOnOff";
	title = labelname+" off#eta vs off#phi online+offline";
	offEtavsoffPhiOnOff =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_mcEtamcPhiL1Mc";
	title = labelname+" mc#eta vs mc#phi L1+MC truth";
	mcEtavsmcPhiL1Mc =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_mcEtamcPhiOffMc";
	title = labelname+" mc#eta vs mc#phi Off+MC truth";
	mcEtavsmcPhiOffMc =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());


	histoname = labelname+"_mcEtamcPhiOnMc";
	title = labelname+" mc#eta vs mc#phi online+MC truth";
	mcEtavsmcPhiOnMc =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_l1EtL1OnUM";
	title = labelname+" l1E_t L1+onlineUM";
	l1EtL1OnUM =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_offEtL1OffUM";
	title = labelname+" offE_t L1+offlineUM";
	offEtL1OffUM =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_offEtOnOffUM";
	title = labelname+" offE_t online+offlineUM";
	offEtOnOffUM =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_mcEtL1McUM";
	title = labelname+" mcE_t L1+MC truthUM";
	mcEtL1McUM =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_mcEtOffMcUM";
	title = labelname+" mcE_t Off+MC truthUM";
	mcEtOffMcUM =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_mcEtOnMcUM";
	title = labelname+" mcE_t online+MC truthUM";
	mcEtOnMcUM =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_l1Etal1PhiL1OnUM";
	title = labelname+" l1#eta vs l1#phi L1+onlineUM";
	l1Etavsl1PhiL1OnUM =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_offEtaoffPhiL1OffUM";
	title = labelname+" off#eta vs off#phi L1+offlineUM";
	offEtavsoffPhiL1OffUM =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_offEtaoffPhiOnOffUM";
	title = labelname+" off#eta vs off#phi online+offlineUM";
	offEtavsoffPhiOnOffUM =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_mcEtamcPhiL1McUM";
	title = labelname+" mc#eta vs mc#phi L1+MC truthUM";
	mcEtavsmcPhiL1McUM =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_mcEtamcPhiOffMcUM";
	title = labelname+" mc#eta vs mc#phi Off+MC truthUM";
	mcEtavsmcPhiOffMcUM =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_mcEtamcPhiOnMcUM";
	title = labelname+" mc#eta vs mc#phi online+MC truthUM";
	mcEtavsmcPhiOnMcUM =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins2D,-histEtaMax,histEtaMax,
				nBins2D,-TMath::Pi(), TMath::Pi());

	histoname = labelname+"_l1DRL1On";
	title = labelname+" l1DR L1+online";
	l1DRL1On =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 0, 1.);

	histoname = labelname+"_offDRL1Off";
	title = labelname+" offDR L1+offline";
	offDRL1Off =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 0, 1.);

	histoname = labelname+"_offDROnOff";
	title = labelname+" offDR online+offline";
	offDROnOff =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 0, 1.);

	histoname = labelname+"_mcDRL1Mc";
	title = labelname+" mcDR L1+MC truth";
	mcDRL1Mc =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 0, 1.);

	histoname = labelname+"_mcDROffMc";
	title = labelname+" mcDR Off+MC truth";
	mcDROffMc =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 0, 1.);

	histoname = labelname+"_mcDROnMc";
	title = labelname+" mcDR online+MC truth";
	mcDROnMc =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 0, 1.);


	v->setHistos( NMc, mcEtMc, mcEtavsmcPhiMc, NOn, onEtOn, onEtavsonPhiOn, NOff, offEtOff, offEtavsoffPhiOff, NL1, l1EtL1, l1Etavsl1PhiL1, NL1On, l1EtL1On, l1Etavsl1PhiL1On, NL1Off, offEtL1Off, offEtavsoffPhiL1Off, NOnOff, offEtOnOff, offEtavsoffPhiOnOff, NL1Mc, mcEtL1Mc, mcEtavsmcPhiL1Mc, NOffMc, mcEtOffMc, mcEtavsmcPhiOffMc, NOnMc, mcEtOnMc, mcEtavsmcPhiOnMc, NL1OnUM, l1EtL1OnUM, l1Etavsl1PhiL1OnUM, NL1OffUM, offEtL1OffUM, offEtavsoffPhiL1OffUM, NOnOffUM, offEtOnOffUM, offEtavsoffPhiOnOffUM, NL1McUM, mcEtL1McUM, mcEtavsmcPhiL1McUM, NOffMcUM, mcEtOffMcUM, mcEtavsmcPhiOffMcUM, NOnMcUM, mcEtOnMcUM, mcEtavsmcPhiOnMcUM, mcDRL1Mc, mcDROnMc, mcDROffMc, offDRL1Off, offDROnOff, l1DRL1On
);


    }
 }
 return;



}

/// EndRun
void FourVectorHLTriggerOffline::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTriggerOffline") << "endRun, run " << run.id();
}

/// Clean DR Match Set
void FourVectorHLTriggerOffline::cleanDRMatchSet(mmset& tempSet)
{

 LogDebug("FourVectorHLTriggerOffline") << "cleanDRMatchSet(mmset& tempSet) " << std::endl;
 LogDebug("FourVectorHLTriggerOffline") << "size of the set (before CLEANED)= " << tempSet.size() << " maps." << std::endl;

 if(tempSet.size() < 2) return;
 
 bool cleanedOneMap = false;
 
 // cleaning needed if the set has at least two maps

 while(! cleanedOneMap && tempSet.size() > 1) {

 cleanedOneMap=false;

 //LogTrace("FourVectorHLTriggerOffline") << "cleaning: size of the set  = " << tempSet.size() << " maps." << std::endl;

 int imap = 0;
 for ( mmset::iterator setIter_i = tempSet.begin( ); setIter_i != tempSet.end( ); setIter_i++ ) 
 {

      fimmap tempMap_j = *setIter_i;

      //LogTrace("FourVectorHLTriggerOffline") << " map " << imap << std::endl;
      //LogTrace("FourVectorHLTriggerOffline") << " --------" << std::endl;
      for (fimmap::iterator it = tempMap_j.begin(); it != tempMap_j.end(); ++it)
      {

        //LogTrace("FourVectorHLTriggerOffline") << " " <<   (*it).first << " :  " << (*it).second << std::endl;

      }

      imap++;

 }

 // loop i
 for ( mmset::iterator setIter_i = tempSet.begin( ); setIter_i != tempSet.end( ); setIter_i++ ) 
 {
     
    fimmap tempMap_i = *setIter_i;
    fimmap::iterator it = tempMap_i.begin();
    int topValue = (*it).second;
    //LogTrace("FourVectorHLTriggerOffline") << " topValue = " << topValue << std::endl;

    
    mmset::iterator tempIter_i = setIter_i;

    // from all the other maps, clean entries that have mapped value "topValue"
    // loop j
    for ( mmset::iterator setIter_j = ++tempIter_i; setIter_j != tempSet.end( ); setIter_j++ ) 
    {

      fimmap tempMap_j = *setIter_j;
      //LogTrace("FourVectorHLTriggerOffline") << "  size of the map  = " << tempMap_j.size() << std::endl;

      for (fimmap::iterator it = tempMap_j.begin(); it != tempMap_j.end(); ++it)
      {

        if(topValue == (*it).second) 
	{
				  
          //LogTrace("FourVectorHLTriggerOffline") << "   Ridding map of a doubly-matched object." << std::endl;
	  tempMap_j.erase(it);

	  cleanedOneMap = true;

	}

     } //end for

     if(cleanedOneMap) 
     {
        
	//remove the old map from the set
	tempSet.erase(setIter_j);

	// insert in the set the new map if it is not an empty map
	if(! tempMap_j.empty()) tempSet.insert(tempMap_j);

	break; // break from loop j

     } // end if

    }// end loop j 

    if(cleanedOneMap) break; // break from loop i

 } // end loop i

 if(cleanedOneMap) 
 { 

  // continue cleaning (in while loop)
  // but reset flag first
  cleanedOneMap=false;
  continue; 

 }
 else {

  // finished cleaing (break from while loop)
  break; 

 }

} // end while

 //LogTrace("FourVectorHLTriggerOffline") << "cleaned: size of the set  = " << tempSet.size() << " maps." << std::endl;
 int jmap = 0;

 for ( mmset::iterator setIter_i = tempSet.begin( ); setIter_i != tempSet.end( ); setIter_i++ ) 
 {

      fimmap tempMap_j = *setIter_i;

      //LogTrace("FourVectorHLTriggerOffline") << " map " << jmap << std::endl;
      //LogTrace("FourVectorHLTriggerOffline") << " --------" << std::endl;

      for (fimmap::iterator it = tempMap_j.begin(); it != tempMap_j.end(); ++it)
      {

        //LogTrace("FourVectorHLTriggerOffline") << " " <<   (*it).first << " :  " << (*it).second << std::endl;
      
      }

      jmap++;

 } // end for


 return;
}


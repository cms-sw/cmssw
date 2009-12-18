// $Id: FourVectorHLTOffline.cc,v 1.58 2009/12/11 19:06:50 rekovic Exp $
// See header file for information. 
#include "TMath.h"
#include "DQMOffline/Trigger/interface/FourVectorHLTOffline.h"


#include <map>
#include <utility>


using namespace edm;
using namespace trigger;

FourVectorHLTOffline::FourVectorHLTOffline(const edm::ParameterSet& iConfig):
  resetMe_(true),  currentRun_(-99)
{

  LogDebug("FourVectorHLTOffline") << "constructor...." ;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("FourVectorHLTOffline") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  dirname_ = iConfig.getUntrackedParameter("dirname", std::string("HLT/FourVector/"));
  //dirname_ +=  iConfig.getParameter<std::string>("@module_label");
  
  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirname_);
  }
  
  processname_ = iConfig.getParameter<std::string>("processname");

  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",20);
  nLS_   = iConfig.getUntrackedParameter<unsigned int>("NLumSegs",500);

  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);
     // this is the list of paths to look at.
  std::vector<edm::ParameterSet> paths = 
  iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");

  for(std::vector<edm::ParameterSet>::iterator pathconf = paths.begin() ; pathconf != paths.end(); pathconf++) {

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

  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  muonRecoCollectionName_ = iConfig.getUntrackedParameter("muonRecoCollectionName", std::string("muons"));

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

  metEtaMax_ = iConfig.getUntrackedParameter<double>("metEtaMax",5);
  metMin_ = iConfig.getUntrackedParameter<double>("metMin",10.0);
  metDRMatch_  =iConfig.getUntrackedParameter<double>("metDRMatch",0.5); 

  htEtaMax_ = iConfig.getUntrackedParameter<double>("htEtaMax",5);
  htMin_ = iConfig.getUntrackedParameter<double>("htMin",10.0);
  htDRMatch_  =iConfig.getUntrackedParameter<double>("htDRMatch",0.5); 

  sumEtMin_ = iConfig.getUntrackedParameter<double>("sumEtMin",10.0);

  specialPaths_ = iConfig.getParameter<std::vector<std::string > >("SpecialPaths");

  pathsSummaryFolder_ = iConfig.getUntrackedParameter ("pathsSummaryFolder",std::string("HLT/FourVector/PathsSummary/"));
  pathsSummaryHLTCorrelationsFolder_ = iConfig.getUntrackedParameter ("hltCorrelationsFolder",std::string("HLT/FourVector/PathsSummary/HLT Correlations/"));
  pathsSummaryFilterCountsFolder_ = iConfig.getUntrackedParameter ("filterCountsFolder",std::string("HLT/FourVector/PathsSummary/Filters Counts/"));

  pathsSummaryHLTPathsPerLSFolder_ = iConfig.getUntrackedParameter ("individualPathsPerLSFolder",std::string("HLT/FourVector/PathsSummary/HLT LS/"));
  pathsIndividualHLTPathsPerLSFolder_ = iConfig.getUntrackedParameter ("individualPathsPerLSFolder",std::string("HLT/FourVector/PathsSummary/HLT LS/Paths/"));

  fLumiFlag = true;
  ME_HLTAll_LS_ = NULL;
  ME_HLT_bx_ = NULL;
  
}


FourVectorHLTOffline::~FourVectorHLTOffline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
FourVectorHLTOffline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //if(! fLumiFlag ) return;

  using namespace edm;
  using namespace trigger;
  ++nev_;
  LogDebug("FourVectorHLTOffline")<< " analyze...." ;
  
  /*
  Handle<GenParticleCollection> genParticles;
  iEvent.getByLabel("genParticles", genParticles);
  if(!genParticles.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "genParticles not found, "
      "skipping event"; 
    return;
  }

  Handle<GenJetCollection> genJets;
  iEvent.getByLabel("iterativeCone5GenJets",genJets);
  if(!genJets.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "genJets not found, "
      "skipping event"; 
    return;
  }

  Handle<GenMETCollection> genMets;
  iEvent.getByLabel("genMetTrue",genMets);
  if(!genMets.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "genMets not found, "
      "skipping event"; 
    return;
  }
  */

  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(triggerResultsLabel_,triggerResults);
  if(!triggerResults.isValid()) {
    edm::InputTag triggerResultsLabelFU(triggerResultsLabel_.label(),triggerResultsLabel_.instance(), "FU");
   iEvent.getByLabel(triggerResultsLabelFU,triggerResults);
  if(!triggerResults.isValid()) {
    edm::LogInfo("FourVectorHLTOffline") << "TriggerResults not found, "
      "skipping event"; 
    return;
   }
  }
  triggerResults_ = triggerResults;
  TriggerNames triggerNames(*triggerResults);  
  int npath = triggerResults->size();

  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) {

    edm::InputTag triggerSummaryLabelFU(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(), "FU");
    iEvent.getByLabel(triggerSummaryLabelFU,triggerObj);

    if(!triggerObj.isValid()) {

      edm::LogInfo("FourVectorHLTOffline") << "TriggerEvent not found, " "skipping event"; 
      return;

    }

  }

  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByLabel(muonRecoCollectionName_,muonHandle);
  if(!muonHandle.isValid()) { 

    edm::LogInfo("FourVectorHLTOffline") << "muonHandle not found, ";
    //  "skipping event"; 
    //  return;

  }

  if(muonHandle.isValid()) { 

    for( reco::MuonCollection::const_iterator iter = muonHandle->begin(), iend = muonHandle->end(); iter != iend; ++iter )
    {


       LogTrace("FourVectorHLTOffline")<< "Found a reco muon" << endl;

       if (iter->isStandAloneMuon()) {
        LogTrace("FourVectorHLTOffline") << "This muon is STA" <<endl;
       }
       else if (iter->isGlobalMuon()){ 
        LogTrace("FourVectorHLTOffline") << "This muon is Global" <<endl;
       }
       else if (iter->isTrackerMuon()){ 
        LogTrace("FourVectorHLTOffline") << "This muon is Tracker" <<endl;
       }

   } // end for
  } // end if

  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  iEvent.getByLabel("gsfElectrons",gsfElectrons); 
  if(!gsfElectrons.isValid()) { 

    edm::LogInfo("FourVectorHLTOffline") << "gsfElectrons not found, ";
      //"skipping event"; 
      //return;
  }

  edm::Handle<reco::CaloTauCollection> tauHandle;
  iEvent.getByLabel("caloRecoTauProducer",tauHandle);
  if(!tauHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "tauHandle not found, ";
      //"skipping event"; 
      //return;
  }

  edm::Handle<reco::CaloJetCollection> jetHandle;
  iEvent.getByLabel("iterativeCone5CaloJets",jetHandle);
  if(!jetHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "jetHandle not found, ";
      //"skipping event"; 
      //return;
  }
 
   // Get b tag information
 edm::Handle<reco::JetTagCollection> bTagIPHandle;
 iEvent.getByLabel("jetProbabilityBJetTags", bTagIPHandle);
 if (!bTagIPHandle.isValid()) {
    edm::LogInfo("FourVectorHLTOffline") << "mTagIPHandle trackCountingHighEffJetTags not found, ";
      //"skipping event"; 
      //return;
  }

   // Get b tag information
 edm::Handle<reco::JetTagCollection> bTagMuHandle;
 iEvent.getByLabel("softMuonBJetTags", bTagMuHandle);
 if (!bTagMuHandle.isValid()) {
    edm::LogInfo("FourVectorHLTOffline") << "bTagMuHandle  not found, ";
      //"skipping event"; 
      //return;
  }

  edm::Handle<reco::CaloMETCollection> metHandle;
  iEvent.getByLabel("met",metHandle);
  if(!metHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "metHandle not found, ";
      //"skipping event"; 
      //return;
  }

  edm::Handle<reco::PhotonCollection> photonHandle;
  iEvent.getByLabel("photons",photonHandle);
  if(!photonHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "photonHandle not found, ";
      //"skipping event"; 
      //return;
  }

  edm::Handle<reco::TrackCollection> trackHandle;
  iEvent.getByLabel("pixelTracks",trackHandle);
  if(!trackHandle.isValid()) { 
    edm::LogInfo("FourVectorHLTOffline") << "trackHandle not found, ";
      //"skipping event"; 
      //return;
  }

  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());

  // Monitors
  // ---------------

  // electron Monitor
  objMonData<reco::GsfElectronCollection> eleMon;
  eleMon.setReco(gsfElectrons);
  eleMon.setLimits(electronEtaMax_, electronEtMin_, electronDRMatch_);
  
  eleMon.pushTriggerType(TriggerElectron);
  eleMon.pushTriggerType(TriggerL1NoIsoEG);
  eleMon.pushTriggerType(TriggerL1IsoEG);

  eleMon.pushL1TriggerType(TriggerL1NoIsoEG);
  eleMon.pushL1TriggerType(TriggerL1IsoEG);

  // muon Monitor
  objMonData<reco::MuonCollection>  muoMon;
  muoMon.setReco(muonHandle);
  muoMon.setLimits(muonEtaMax_, muonEtMin_, muonDRMatch_);
  
  muoMon.pushTriggerType(TriggerMuon);
  muoMon.pushTriggerType(TriggerL1Mu);

  muoMon.pushL1TriggerType(TriggerL1Mu);
  
  // tau Monitor
  objMonData<reco::CaloTauCollection>  tauMon;
  tauMon.setReco(tauHandle);
  tauMon.setLimits(tauEtaMax_, tauEtMin_, tauDRMatch_);
  
  tauMon.pushTriggerType(TriggerTau);
  tauMon.pushTriggerType(TriggerL1TauJet);

  tauMon.pushL1TriggerType(TriggerL1TauJet);
  tauMon.pushL1TriggerType(TriggerL1ForJet);
  
  // photon Monitor
  objMonData<reco::PhotonCollection> phoMon;
  phoMon.setReco(photonHandle);
  phoMon.setLimits(photonEtaMax_, photonEtMin_, photonDRMatch_);
  
  phoMon.pushTriggerType(TriggerPhoton);

  phoMon.pushL1TriggerType(TriggerL1NoIsoEG);
  phoMon.pushL1TriggerType(TriggerL1IsoEG);

  // jet Monitor - NOTICE: we use genJets for MC
  objMonData<reco::CaloJetCollection> jetMon;
  jetMon.setReco(jetHandle);
  jetMon.setLimits(jetEtaMax_, jetEtMin_, jetDRMatch_);

  jetMon.pushTriggerType(TriggerJet);
  jetMon.pushTriggerType(TriggerL1CenJet);
  jetMon.pushTriggerType(TriggerL1ForJet);
  
  jetMon.pushL1TriggerType(TriggerL1CenJet);
  jetMon.pushL1TriggerType(TriggerL1ForJet);
  jetMon.pushL1TriggerType(TriggerL1TauJet);

  // bjet Monitor - NOTICE: we use genJets for MC
  objMonData<reco::CaloJetCollection> btagIPMon; // CaloJet will not be used, this is only place holder
  //btagIPMon.setReco(jetHandle);
  btagIPMon.setRecoB(bTagIPHandle);
  btagIPMon.setBJetsFlag(true);
  btagIPMon.setLimits(bjetEtaMax_, bjetEtMin_, bjetDRMatch_);

  btagIPMon.pushTriggerType(TriggerBJet);
  btagIPMon.pushTriggerType(TriggerJet);

  btagIPMon.pushL1TriggerType(TriggerL1CenJet);
  btagIPMon.pushL1TriggerType(TriggerL1ForJet);
  btagIPMon.pushL1TriggerType(TriggerL1TauJet);

  objMonData<reco::CaloJetCollection> btagMuMon; // CaloJet will not be used, this is only place holder
  //btagMuMon.setReco(jetHandle);
  btagMuMon.setRecoB(bTagMuHandle);
  btagMuMon.setBJetsFlag(true);
  btagMuMon.setLimits(bjetEtaMax_, bjetEtMin_, bjetDRMatch_);

  btagMuMon.pushTriggerType(TriggerBJet);
  btagMuMon.pushTriggerType(TriggerJet);

  btagMuMon.pushL1TriggerType(TriggerL1CenJet);
  btagMuMon.pushL1TriggerType(TriggerL1ForJet);
  btagMuMon.pushL1TriggerType(TriggerL1TauJet);


  objMonData<reco::CaloJetCollection> btagMon; // Generic btagMon
 
  // met Monitor
  objMonData<reco::CaloMETCollection> metMon;
  metMon.setReco(metHandle);
  metMon.setLimits(metEtaMax_, metMin_, metDRMatch_);
  
  metMon.pushTriggerType(TriggerMET);

  metMon.pushL1TriggerType(TriggerL1ETM);


  // vector to hold monitors 
  // interface is through virtual class BaseMonitor
  std::vector<BaseMonitor*> monitors;

  //monitors.push_back(&jetMon);

  monitors.push_back(&muoMon);
  monitors.push_back(&eleMon);
  monitors.push_back(&tauMon);
  monitors.push_back(&phoMon);
  monitors.push_back(&jetMon);
  monitors.push_back(&btagMon);
  monitors.push_back(&metMon);

  int bx = iEvent.bunchCrossing();
  /*
  // Fill HLTPassed_Correlation Matrix bin (i,j) = (Any,Any)
  // --------------------------------------------------------
  int anyBinNumber = ME_HLTPassPass_->getTH2F()->GetXaxis()->FindBin("Any HLT");      
  // any triger accepted
  if(triggerResults->accept()){

    ME_HLTPassPass_->Fill(anyBinNumber-1,anyBinNumber-1);//binNumber1 = 0 = first filter

  }
  */

  vector<string> name;
  name.push_back("All");
  name.push_back("Muon");
  name.push_back("Egamma");
  name.push_back("JetMET");
  name.push_back("Rest");
  name.push_back("Special");
  
  fillHltMatrix(name);



  // Main loop over paths
  // --------------------
  for(PathInfoCollection::iterator v = hltPaths_.begin(); v!= hltPaths_.end(); ++v ) { 

    LogTrace("FourVectorHLTOffline") << " path " << v->getPath() << endl;

    if (v->getPath().find("BTagIP") != std::string::npos ) btagMon = btagIPMon;
    else btagMon = btagMuMon;

    //if (v->getPath().find("HLT_L1Jet6U") == std::string::npos ) continue;

    unsigned int pathByIndex = triggerNames.triggerIndex(v->getPath());

    if(pathByIndex >= triggerResults_->size() ) continue;
  
    // Fill HLTPassed Matrix and HLTPassFail Matrix
    // --------------------------------------------------------

    if(triggerResults->accept(pathByIndex)){
  
      int pathBinNumber = ME_HLT_bx_->getTH2F()->GetYaxis()->FindBin(v->getPath().c_str());      
      ME_HLT_bx_->Fill(bx,pathBinNumber-1);

    }


  
    // Fill histogram of filter ocupancy for each HLT path
    // ---------------------------------
    unsigned int lastModule = triggerResults->index(pathByIndex);
  
    //go through the list of filters
    for(unsigned int filt = 0; filt < v->filtersAndIndices.size(); filt++){
      
      int binNumber = v->getFiltersHisto()->getTH1()->GetXaxis()->FindBin(v->filtersAndIndices[filt].first.c_str());      
      
      //check if filter passed
      if(triggerResults->accept(pathByIndex)){
        v->getFiltersHisto()->Fill(binNumber-1);//binNumber1 = 0 = first filter
      }
      //otherwise the module that issued the decision is the first fail
      //so that all the ones before it passed
      else if(v->filtersAndIndices[filt].second < lastModule){
        v->getFiltersHisto()->Fill(binNumber-1);//binNumber1 = 0 = first filter
      }
  
    } // end for filt

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

     //LogTrace("FourVectorHLTOffline") << " denominator path " << v->getPath() << endl;


     // Get the righe monitor for this path
     // -----------------------------------
     BaseMonitor* mon = NULL;

     for(std::vector<BaseMonitor*>::iterator mit = monitors.begin(); mit!= monitors.end(); ++mit ) {
       
       if((*mit)->isTriggerType(v->getObjectType())) {

         mon = *mit;
         break;

       }

     }

     // if cannot find moniotor for the path, go to next path
     if(!mon) continue;

     // clear sets of matched objects
     mon->clearSets();

     int triggertype = 0;     
     triggertype = v->getObjectType();

     bool l1accept = false;
     edm::InputTag l1testTag(v->getl1Path(),"",processname_);
     const int l1index = triggerObj->filterIndex(l1testTag);
     

     if ( l1index >= triggerObj->sizeFilters() ) {

       edm::LogInfo("FourVectorHLTOffline") << "no index "<< l1index << " of that name " << v->getl1Path() << "\t" << "\t" << l1testTag;
       continue; // not in this event

     }

     const trigger::Vids & idtype = triggerObj->filterIds(l1index);
     const trigger::Keys & l1k = triggerObj->filterKeys(l1index);
     l1accept = l1k.size() > 0;
     //LogTrace("FourVectorHLTOffline") << " triggertype = " << triggertype << " TriggerMuon  " <<  TriggerMuon << "   l1accept = " << l1accept << endl;
     //if (l1k.size() == 0) cout << v->getl1Path() << endl;
     //l1accept = true;

     mon->monitorDenominator(v, l1accept, idtype, l1k, toc);
     mon->fillL1Match(this);

    // did we pass the numerator path?
    bool numpassed = false;

    for(int i = 0; i < npath; ++i) {

      if (triggerNames.triggerName(i) == v->getPath() && triggerResults->accept(i)) numpassed = true;

    }

    if (numpassed)
    { 

      if (!l1accept) {

            edm::LogInfo("FourVectorHLTOffline") << "l1 seed path not accepted for hlt path "<< v->getPath() << "\t" << v->getl1Path();

      }

      // fill scaler histograms
      edm::InputTag filterTag = v->getTag();

      // loop through indices and see if the filter is on the list of filters used by this path
      
      if (v->getLabel() == "dummy"){
      const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());

      //loop over labels
      for (std::vector<std::string>::const_iterator labelIter= filterLabels.begin(); labelIter!=filterLabels.end(); labelIter++) {

        //cout << v->getPath() << "\t" << *labelIter << endl;
        // last match wins...
        edm::InputTag testTag(*labelIter,"",processname_);
        //           cout << v->getPath() << "\t" << testTag.label() << "\t" << testTag.process() << endl;
        int testindex = triggerObj->filterIndex(testTag);
        if ( !(testindex >= triggerObj->sizeFilters()) ) {

          //cout << "found one! " << v->getPath() << "\t" << testTag.label() << endl; 
          filterTag = testTag; v->setLabel(*labelIter);}
        }

      } // end for
  
      const int index = triggerObj->filterIndex(filterTag);
      if ( index >= triggerObj->sizeFilters() ) {

      //cout << "WTF no index "<< index << " of that name "
      //<< filterTag << endl;
        continue; // not in this event

      }

      const trigger::Keys & k = triggerObj->filterKeys(index);
      //      const trigger::Vids & idtype = triggerObj->filterIds(index);
      // assume for now the first object type is the same as all objects in the collection
      //    cout << filterTag << "\t" << idtype.size() << "\t" << k.size() << endl;
      //     cout << "path " << v->getPath() << " trigger type "<<triggertype << endl;
      //if (k.size() > 0) v->getNOnHisto()->Fill(k.size());


      unsigned int NOnCount=0;

      // Loop over HLT objects
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {


        mon->monitorOnline(idtype, l1k, ki, toc, NOnCount);

      } // online object loop

      if(NOnCount>0) v->getNOnHisto()->Fill(NOnCount);


        
      mon->fillOnlineMatch(this, l1k, toc);

      //mon->monitorOffline(this);
      //mon->fillOffMatch(this);

    } //numpassed


   } //denompassed


 } //pathinfo loop

}



// -- method called once each job just before starting event loop  --------
void 
FourVectorHLTOffline::beginJob()
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
FourVectorHLTOffline::endJob() 
{
   LogInfo("FourVectorHLTOffline") << "analyzed " << nev_ << " events";
   return;
}


// BeginRun
void FourVectorHLTOffline::beginRun(const edm::Run& run, const edm::EventSetup& c)
{

  LogDebug("FourVectorHLTOffline") << "beginRun, run " << run.id();
// HLT config does not change within runs!
 
  if (!hltConfig_.init(processname_)) {

    processname_ = "FU";

    if (!hltConfig_.init(processname_)){

      LogDebug("FourVectorHLTOffline") << "HLTConfigProvider failed to initialize.";

    }

    // check if trigger name in (new) config
    //  cout << "Available TriggerNames are: " << endl;
    //  hltConfig_.dump("Triggers");
  }

  if (1) {

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
      
            //  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
            if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed") {
      
              edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
              // cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
              // l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
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

    } // end if plotAll
    else {

      // plot all diagonal combinations plus any other specified pairs
      for (unsigned int i=0; i!=n; ++i) {

         std::string denompathname = "";  
         std::string pathname = hltConfig_.triggerName(i);  
         std::string l1pathname = "dummy";
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
  
          //  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
          if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed") {
  
            edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
            // cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
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
        if (objectType == trigger::TriggerMuon) ptMax = 150.0;
        if (objectType == trigger::TriggerTau) ptMax = 100.0;
        if (objectType == trigger::TriggerJet) ptMax = 300.0;
        if (objectType == trigger::TriggerBJet) ptMax = 300.0;
        if (objectType == trigger::TriggerMET) ptMax = 300.0;
        if (objectType == trigger::TriggerTET) ptMax = 300.0;
        if (objectType == trigger::TriggerTrack) ptMax = 100.0;
    
        // keep track of all paths, except for FinalPath
        if (objectType != -1 && pathname.find("FinalPath") == std::string::npos){
  
          hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));
          //create folder for pathname
  
        }

      } // end for i

        
      // now loop over denom/num path pairs specified in cfg, 
      // recording the off-diagonal ones
      for (std::vector<std::pair<std::string, std::string> >::iterator custompathnamepair = custompathnamepairs_.begin(); custompathnamepair != custompathnamepairs_.end(); ++custompathnamepair) {
            
        std::string numpathname = custompathnamepair->first;  
        std::string denompathname = custompathnamepair->second;  
  
        if (numpathname != denompathname) {
  
          // check that denominator exists
          bool founddenominator = false;
          for (unsigned int k=0; k!=n; ++k) {

            string n_pathname = hltConfig_.triggerName(k);

            if (n_pathname.find(denompathname) != std::string::npos) {
              
              LogDebug("FourVectorHLTOffline") << "denompathname is selected to be = " << n_pathname << endl;;
              founddenominator = true;

              break;

            }
          }

          if (!founddenominator) {
  
            edm::LogInfo("FourVectorHLTOffline") << "denompathname not found, go to the next pair numearator-denominator" << endl;
            
            // go to the next pair
            continue;
  
          }

          // check that numerator exists
          bool foundnumerator = false;
          for (unsigned int j=0; j!=n; ++j) {

            string pathname = hltConfig_.triggerName(j);

            LogDebug("FourVectorHLTOffline") << "check if path " << pathname << " is numpathname = " << numpathname << endl;
            if (hltConfig_.triggerName(j).find(numpathname)!= std::string::npos) {
              
              LogDebug("FourVectorHLTOffline") << "pathname is selected to be = " << denompathname << endl;;
              foundnumerator = true;

            }
  
  
            if (!foundnumerator) {
    
              edm::LogInfo("FourVectorHLTOffline") << "pathname not found, ignoring " << pathname;
              continue;
  
            }
  
  
            //cout << pathname << "\t" << denompathname << endl;
            std::string l1pathname = "dummy";
            int objectType = 0;
            //int denomobjectType = 0;
            //parse pathname to guess object type
            if (pathname.find("MET") != std::string::npos) objectType = trigger::TriggerMET;    
            if (pathname.find("SumET") != std::string::npos) objectType = trigger::TriggerTET;    
            if (pathname.find("HT") != std::string::npos) objectType = trigger::TriggerTET;    
            if (pathname.find("Jet") != std::string::npos) objectType = trigger::TriggerJet;    
            if (pathname.find("Mu") != std::string::npos) objectType = trigger::TriggerMuon;    
            if (pathname.find("Ele") != std::string::npos) objectType = trigger::TriggerElectron;    
            if (pathname.find("Photon") != std::string::npos) objectType = trigger::TriggerPhoton;    
            if (pathname.find("Tau") != std::string::npos) objectType = trigger::TriggerTau;    
            if (pathname.find("IsoTrack") != std::string::npos) objectType = trigger::TriggerTrack;    
            if (pathname.find("BTag") != std::string::npos) objectType = trigger::TriggerBJet;    
            // find L1 condition for numpath with numpath objecttype 
  
            // find PSet for L1 global seed for numpath, 
            // list module labels for numpath
      
            std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
      
            for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
            numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
  
            //  cout << pathname << "\t" << *numpathmodule << "\t" << hltConfig_.moduleType(*numpathmodule) << endl;
            if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed") {
  
              edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
              //                  cout << l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression") << endl;
              // l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
              l1pathname = *numpathmodule;
              //cout << *numpathmodule << endl; 
              break; 
  
            }
  
          } // end for
      
  
          std::string filtername("dummy");
          float ptMin = 0.0;
          float ptMax = 100.0;
          if (objectType == trigger::TriggerPhoton) ptMax = 100.0;
          if (objectType == trigger::TriggerElectron) ptMax = 100.0;
          if (objectType == trigger::TriggerMuon) ptMax = 150.0;
          if (objectType == trigger::TriggerTau) ptMax = 100.0;
          if (objectType == trigger::TriggerJet) ptMax = 300.0;
          if (objectType == trigger::TriggerBJet) ptMax = 300.0;
          if (objectType == trigger::TriggerMET) ptMax = 300.0;
          if (objectType == trigger::TriggerTET) ptMax = 300.0;
          if (objectType == trigger::TriggerTrack) ptMax = 100.0;
  
          // monitor regardless of the objectType of the path
          if (objectType != 0)
            hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));
      
        } // end for j, loop over paths

       }  // end if not same num and denominator 
  
      } // end for pair

    } // end else


    vector<string> muonPaths;
    vector<string> egammaPaths;
    vector<string> jetmetPaths;
    vector<string> restPaths;
    vector<string> allPaths;
    // fill vectors of Muon, Egamma, JetMET, Rest, and Special paths
    for(PathInfoCollection::iterator v = hltPaths_.begin();
	  v!= hltPaths_.end(); ++v ) {

      std::string pathName = v->getPath();
      int objectType = v->getObjectType();

      // initialize pair <pathname,count>
      // --------------------------------
      std::pair<std::string, int> tPair;
      tPair.first = pathName;
      tPair.second = 0;
      fPathTempCountPair.push_back(tPair);
      allPaths.push_back(pathName);

      switch (objectType) {
        case trigger::TriggerMuon :
          muonPaths.push_back(pathName);
          break;

        case trigger::TriggerElectron :
        case trigger::TriggerPhoton :
          egammaPaths.push_back(pathName);
          break;

        case trigger::TriggerJet :
        case trigger::TriggerMET :
          jetmetPaths.push_back(pathName);
          break;

        default:
          restPaths.push_back(pathName);
      }

    } // end for

    std::pair<std::string, int> tPair;
    tPair.first = "Any HLT";
    tPair.second = 0;
    fPathTempCountPair.push_back(tPair);


    dbe_->setCurrentFolder(pathsSummaryFolder_.c_str());

    setupHltMatrix("All", allPaths);

    setupHltMatrix("Muon", muonPaths);

    setupHltMatrix("Egamma", egammaPaths);

    setupHltMatrix("JetMET", jetmetPaths);

    setupHltMatrix("Rest", restPaths);

    setupHltMatrix("Special", specialPaths_);

    setupHltLsPlots();

    setupHltBxPlots();


    // now set up all of the histos for each path
    for(PathInfoCollection::iterator v = hltPaths_.begin(); v!= hltPaths_.end(); ++v ) {

      MonitorElement *NOn, *onEtOn, *onEtavsonPhiOn=0;
      MonitorElement *NOff, *offEtOff, *offEtavsoffPhiOff=0;
      MonitorElement *NL1, *l1EtL1, *l1Etavsl1PhiL1=0;
      MonitorElement *NL1On, *l1EtL1On, *l1Etavsl1PhiL1On=0;
      MonitorElement *NL1Off, *offEtL1Off, *offEtavsoffPhiL1Off=0;
      MonitorElement *NOnOff, *offEtOnOff, *offEtavsoffPhiOnOff=0;
      MonitorElement *NL1OnUM, *l1EtL1OnUM, *l1Etavsl1PhiL1OnUM=0;
      MonitorElement *NL1OffUM, *offEtL1OffUM, *offEtavsoffPhiL1OffUM=0;
      MonitorElement *NOnOffUM, *offEtOnOffUM, *offEtavsoffPhiOnOffUM=0;
      MonitorElement *offDRL1Off, *offDROnOff, *l1DRL1On=0;
      MonitorElement *filters=0;
      

      std::string labelname("dummy");
      labelname = v->getPath() + "_wrt_" + v->getDenomPath();
      std::string histoname(labelname+"_NOn");
      std::string title(labelname+" N online");
      double histEtaMax = 2.5;

      if (v->getObjectType() == trigger::TriggerMuon || v->getObjectType() == trigger::TriggerL1Mu) {

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
        histEtaMax = metEtaMax_; 
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

      NOn =  dbe->book1D(histoname.c_str(), title.c_str(),10, 0.5, 10.5);


       histoname = labelname+"_NOff";
       title = labelname+" N Off";
       NOff =  dbe->book1D(histoname.c_str(), title.c_str(),10, 0.5, 10.5);
       
       histoname = labelname+"_NL1";
       title = labelname+" N L1";
       NL1 =  dbe->book1D(histoname.c_str(), title.c_str(),10, 0.5, 10.5);
       
       histoname = labelname+"_NL1On";
       title = labelname+" N L1On";
       NL1On =  dbe->book1D(histoname.c_str(), title.c_str(),10, 0.5, 10.5);
       
       histoname = labelname+"_NL1Off";
       title = labelname+" N L1Off";
       NL1Off =  dbe->book1D(histoname.c_str(), title.c_str(),10, 0.5, 10.5);
       
       histoname = labelname+"_NOnOff";
       title = labelname+" N OnOff";
       NOnOff =  dbe->book1D(histoname.c_str(), title.c_str(),10, 0.5, 10.5);
       
       
       histoname = labelname+"_NL1OnUM";
       title = labelname+" N L1OnUM";
       NL1OnUM =  dbe->book1D(histoname.c_str(), title.c_str(),10, 0.5, 10.5);
       
       histoname = labelname+"_NL1OffUM";
       title = labelname+" N L1OffUM";
       NL1OffUM =  dbe->book1D(histoname.c_str(), title.c_str(),10, 0.5, 10.5);
       
       histoname = labelname+"_NOnOffUM";
       title = labelname+" N OnOffUM";
       NOnOffUM =  dbe->book1D(histoname.c_str(), title.c_str(),10, 0.5, 10.5);
       
       
       histoname = labelname+"_onEtOn";
       title = labelname+" onE_t online";
       onEtOn =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, v->getPtMin(), v->getPtMax());
       
       histoname = labelname+"_offEtOff";
       title = labelname+" offE_t offline";
       offEtOff =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, v->getPtMin(), v->getPtMax());
       
       histoname = labelname+"_l1EtL1";
       title = labelname+" l1E_t L1";
       l1EtL1 =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, v->getPtMin(), v->getPtMax());
       
       int nBins2D = 10;
       
       
       histoname = labelname+"_onEtaonPhiOn";
       title = labelname+" on#eta vs on#phi online";
       onEtavsonPhiOn =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D,-histEtaMax,histEtaMax, nBins2D,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_offEtaoffPhiOff";
       title = labelname+" off#eta vs off#phi offline";
       offEtavsoffPhiOff =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D,-histEtaMax,histEtaMax, nBins2D,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_l1Etal1PhiL1";
       title = labelname+" l1#eta vs l1#phi L1";
       l1Etavsl1PhiL1 =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D,-histEtaMax,histEtaMax, nBins2D,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_l1EtL1On";
       title = labelname+" l1E_t L1+online";
       l1EtL1On =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, v->getPtMin(), v->getPtMax());
       
       histoname = labelname+"_offEtL1Off";
       title = labelname+" offE_t L1+offline";
       offEtL1Off =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, v->getPtMin(), v->getPtMax());
       
       histoname = labelname+"_offEtOnOff";
       title = labelname+" offE_t online+offline";
       offEtOnOff =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, v->getPtMin(), v->getPtMax());
       
       histoname = labelname+"_l1Etal1PhiL1On";
       title = labelname+" l1#eta vs l1#phi L1+online";
       l1Etavsl1PhiL1On =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D,-histEtaMax,histEtaMax, nBins2D,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_offEtaoffPhiL1Off";
       title = labelname+" off#eta vs off#phi L1+offline";
       offEtavsoffPhiL1Off =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D,-histEtaMax,histEtaMax, nBins2D,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_offEtaoffPhiOnOff";
       title = labelname+" off#eta vs off#phi online+offline";
       offEtavsoffPhiOnOff =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D,-histEtaMax,histEtaMax, nBins2D,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_l1EtL1OnUM";
       title = labelname+" l1E_t L1+onlineUM";
       l1EtL1OnUM =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, v->getPtMin(), v->getPtMax());
       
       histoname = labelname+"_offEtL1OffUM";
       title = labelname+" offE_t L1+offlineUM";
       offEtL1OffUM =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, v->getPtMin(), v->getPtMax());
       
       histoname = labelname+"_offEtOnOffUM";
       title = labelname+" offE_t online+offlineUM";
       offEtOnOffUM =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, v->getPtMin(), v->getPtMax());
       
       histoname = labelname+"_l1Etal1PhiL1OnUM";
       title = labelname+" l1#eta vs l1#phi L1+onlineUM";
       l1Etavsl1PhiL1OnUM =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D,-histEtaMax,histEtaMax, nBins2D,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_offEtaoffPhiL1OffUM";
       title = labelname+" off#eta vs off#phi L1+offlineUM";
       offEtavsoffPhiL1OffUM =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D,-histEtaMax,histEtaMax, nBins2D,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_offEtaoffPhiOnOffUM";
       title = labelname+" off#eta vs off#phi online+offlineUM";
       offEtavsoffPhiOnOffUM =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D,-histEtaMax,histEtaMax, nBins2D,-TMath::Pi(), TMath::Pi());
       
       
       
       
       histoname = labelname+"_l1DRL1On";
       title = labelname+" l1DR L1+online";
       l1DRL1On =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, 0, 1.); 
       
       histoname = labelname+"_offDRL1Off";
       title = labelname+" offDR L1+offline";
       offDRL1Off =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, 0, 1.);
       
       histoname = labelname+"_offDROnOff";
       title = labelname+" offDR online+offline";
       offDROnOff =  dbe->book1D(histoname.c_str(), title.c_str(),nBins_, 0, 1.); 

       // -------------------------
       //
       //  Filters for each path
       //
       // -------------------------
       
       // get all modules in this HLT path
       vector<string> moduleNames = hltConfig_.moduleLabels( v->getPath() ); 
       
       int numModule = 0;
       string moduleName, moduleType;
       unsigned int moduleIndex;
       
       //print module name
       vector<string>::const_iterator iDumpModName;
       for (iDumpModName = moduleNames.begin();iDumpModName != moduleNames.end();iDumpModName++) {

         moduleName = *iDumpModName;
         moduleType = hltConfig_.moduleType(moduleName);
         moduleIndex = hltConfig_.moduleIndex(v->getPath(), moduleName);

         LogTrace ("FourVectorHLTOffline") << "Module "      << numModule
             << " is called " << moduleName
             << " , type = "  << moduleType
             << " , index = " << moduleIndex
             << endl;

         numModule++;

         if((moduleType.find("Filter") != string::npos && moduleType.find("HLTTriggerTypeFilter") == string::npos ) || 
            (moduleType.find("Associator") != string::npos) || 
            (moduleType.find("HLTLevel1GTSeed") != string::npos) || 
            (moduleType.find("HLTGlobalSumsCaloMET") != string::npos) ||
            (moduleType.find("HLTPrescaler") != string::npos) ) {

           std::pair<std::string, int> filterIndexPair;
           filterIndexPair.first   = moduleName;
           filterIndexPair.second  = moduleIndex;
           v->filtersAndIndices.push_back(filterIndexPair);

         }


       }//end for modulesName

       dbe_->setCurrentFolder(pathsSummaryFilterCountsFolder_.c_str()); 

       //int nbin_sub = 5;
       int nbin_sub = v->filtersAndIndices.size()+2;
    
       // count plots for subfilter
       filters = dbe_->book1D("Filters_" + v->getPath(), 
                              "Filters_" + v->getPath(),
                              nbin_sub+1, -0.5, 0.5+(double)nbin_sub);
       
       for(unsigned int filt = 0; filt < v->filtersAndIndices.size(); filt++){

         filters->setBinLabel(filt+1, (v->filtersAndIndices[filt]).first);

       }

       // book Count vs LS
       dbe_->setCurrentFolder(pathsIndividualHLTPathsPerLSFolder_.c_str());
       MonitorElement* tempME = dbe_->book1D(v->getPath() + "_count_per_LS", 
                              v->getPath() + " count per LS",
                              nLS_, 0,nLS_);
       tempME->setAxisTitle("Luminosity Section");


       v->setHistos( NOn, onEtOn, onEtavsonPhiOn, NOff, offEtOff, offEtavsoffPhiOff, NL1, l1EtL1, l1Etavsl1PhiL1, NL1On, l1EtL1On, l1Etavsl1PhiL1On, NL1Off, offEtL1Off, offEtavsoffPhiL1Off, NOnOff, offEtOnOff, offEtavsoffPhiOnOff, NL1OnUM, l1EtL1OnUM, l1Etavsl1PhiL1OnUM, NL1OffUM, offEtL1OffUM, offEtavsoffPhiL1OffUM, NOnOffUM, offEtOnOffUM, offEtavsoffPhiOnOffUM, offDRL1Off, offDROnOff, l1DRL1On, filters
);


    }  // end for hltPath

  } // end if(1) dummy

 return;

}

/// EndRun
void FourVectorHLTOffline::endRun(const edm::Run& run, const edm::EventSetup& c)
{

  LogDebug("FourVectorHLTOffline") << "endRun, run " << run.id();

}

/// Clean DR Match Set
void FourVectorHLTOffline::cleanDRMatchSet(mmset& tempSet)
{

  LogDebug("FourVectorHLTOffline") << "cleanDRMatchSet(mmset& tempSet) " << "size of the set (before CLEANING) = " << tempSet.size() << " maps." << endl;

  if(tempSet.size() < 2) return;

  if(tempSet.size() > 10) {

    LogDebug("FourVectorHLTOffline") << "size of the set is too large.  It will be truncated to 10." << endl;
    mmset::iterator it = tempSet.begin();
    for (int i=0;i<10;i++) {    it++; }
    tempSet.erase( it, tempSet.end());
    LogDebug("FourVectorHLTOffline") << "size of the set is now = " << tempSet.size() << " maps." << endl;

  }
 
  bool cleanedOneMap = false;
 
  // cleaning needed if the set has at least two maps

  while(! cleanedOneMap && tempSet.size() > 1) {

    cleanedOneMap=false;

    //LogTrace("FourVectorHLTOffline") << "cleaning: size of the set  = " << tempSet.size() << " maps." << endl;

    int imap = 0;
    for ( mmset::iterator setIter_i = tempSet.begin( ); setIter_i != tempSet.end( ); setIter_i++ ) {

      fimmap tempMap_j = *setIter_i;

      //LogTrace("FourVectorHLTOffline") << " map " << imap << endl;
      //LogTrace("FourVectorHLTOffline") << " --------" << endl;

      for (fimmap::iterator it = tempMap_j.begin(); it != tempMap_j.end(); ++it) {

        //LogTrace("FourVectorHLTOffline") << " " <<   (*it).first << " :  " << (*it).second << endl;

      }

      imap++;

    }

    // loop i
    for ( mmset::iterator setIter_i = tempSet.begin( ); setIter_i != tempSet.end( ); setIter_i++ ) {
     
      fimmap tempMap_i = *setIter_i;
      fimmap::iterator it = tempMap_i.begin();
      int topValue = (*it).second;
      //LogTrace("FourVectorHLTOffline") << " topValue = " << topValue << endl;
  
      
      mmset::iterator tempIter_i = setIter_i;
  
      // from all the other maps, clean entries that have mapped value "topValue"
      // loop j
      for ( mmset::iterator setIter_j = ++tempIter_i; setIter_j != tempSet.end( ); setIter_j++ ) {
  
        fimmap tempMap_j = *setIter_j;
        //LogTrace("FourVectorHLTOffline") << "  size of the map  = " << tempMap_j.size() << endl;
  
        for (fimmap::iterator it = tempMap_j.begin(); it != tempMap_j.end(); ++it)
        {
  
          if(topValue == (*it).second) {
            
            //LogTrace("FourVectorHLTOffline") << "   Ridding map of a doubly-matched object." << endl;
            tempMap_j.erase(it);
            cleanedOneMap = true;
  
          }
  
        } //end for
  
        if(cleanedOneMap) {
          
          //remove the old map from the set
          tempSet.erase(setIter_j);
  
          // insert in the set the new map if it is not an empty map
          if(! tempMap_j.empty()) tempSet.insert(tempMap_j);
  
          break; // break from loop j
  
       } // end if
  
      }// end loop j 
  
      if(cleanedOneMap) break; // break from loop i

    } // end loop i

    if(cleanedOneMap) { 

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

  //LogTrace("FourVectorHLTOffline") << "cleaned: size of the set  = " << tempSet.size() << " maps." << endl;
  int jmap = 0;

  for ( mmset::iterator setIter_i = tempSet.begin( ); setIter_i != tempSet.end( ); setIter_i++ ) 
  {

    fimmap tempMap_j = *setIter_i;

    //LogTrace("FourVectorHLTOffline") << " map " << jmap << endl;
    //LogTrace("FourVectorHLTOffline") << " --------" << endl;

    for (fimmap::iterator it = tempMap_j.begin(); it != tempMap_j.end(); ++it) {

        //LogTrace("FourVectorHLTOffline") << " " <<   (*it).first << " :  " << (*it).second << endl;
      
    }

    jmap++;

  } // end for

  return;

}

void FourVectorHLTOffline::setupHltMatrix(std::string label, vector<std::string> paths) {

    paths.push_back("Any HLT");

    string h_name; 
    string h_title; 

    dbe_->setCurrentFolder(pathsSummaryFolder_.c_str());
    h_name= "HLT_"+label+"_PassPass";
    h_title = "HLT_"+label+"_PassPass (x=Pass, y=Pass)";
    MonitorElement* ME = dbe_->book2D(h_name.c_str(), h_title.c_str(),
                           paths.size(), -0.5, paths.size()-0.5, paths.size(), -0.5, paths.size()-0.5);
    h_name= "HLT_"+label+"_Pass_Any";
    h_title = "HLT_"+label+"_Pass (x=Pass, Any=Pass) normalized to Any HLT Pass";
    MonitorElement* ME_Any = dbe_->book1D(h_name.c_str(), h_title.c_str(),
                           paths.size(), -0.5, paths.size()-0.5);

    dbe_->setCurrentFolder(pathsSummaryHLTCorrelationsFolder_.c_str());
    h_name= "HLT_"+label+"_PassPass_Normalized";
    h_title = "HLT_"+label+"_PassPass (x=Pass, y=Pass) normalized to xBin=Pass";
    MonitorElement* ME_Normalized = dbe_->book2D(h_name.c_str(), h_title.c_str(),
                           paths.size(), -0.5, paths.size()-0.5, paths.size(), -0.5, paths.size()-0.5);
    h_name= "HLT_"+label+"_Pass_Normalized_Any";
    h_title = "HLT_"+label+"_Pass (x=Pass, Any=Pass) normalized to Any HLT Pass";
    MonitorElement* ME_Normalized_Any = dbe_->book1D(h_name.c_str(), h_title.c_str(),
                           paths.size(), -0.5, paths.size()-0.5);

    for(unsigned int i = 0; i < paths.size(); i++){

      ME->getTH2F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME->getTH2F()->GetYaxis()->SetBinLabel(i+1, (paths[i]).c_str());

      ME_Normalized->getTH2F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Normalized->getTH2F()->GetYaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Normalized_Any->getTH1F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Any->getTH1F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());

    }

}

void FourVectorHLTOffline::fillHltMatrix(vector<std::string> name) {

TriggerNames triggerNames(*triggerResults_);

string fullPathToME; 

for (unsigned int mi=0;mi<name.size();mi++) {

  fullPathToME = "HLT/FourVector/PathsSummary/HLT_"+name[mi]+"_PassPass";
  MonitorElement* ME_2d = dbe_->get(fullPathToME);
  fullPathToME = "HLT/FourVector/PathsSummary/HLT_"+name[mi]+"_Pass_Any";
  MonitorElement* ME_1d = dbe_->get(fullPathToME);
  if(!ME_2d || !ME_1d) {  

    LogTrace("FourVectorHLTOffline") << " ME not valid although I gave full path" << endl;
    continue;

  }

  TH2F * hist_2d = ME_2d->getTH2F();
  TH1F * hist_1d = ME_1d->getTH1F();

  // Fill HLTPassed Matrix bin (i,j) = (Any,Any)
  // --------------------------------------------------------
  int anyBinNumber = hist_2d->GetXaxis()->FindBin("Any HLT");      
  // any triger accepted
  if(triggerResults_->accept()){

    hist_2d->Fill(anyBinNumber-1,anyBinNumber-1);//binNumber1 = 0 = first filter
    hist_1d->Fill(anyBinNumber-1);//binNumber1 = 0 = first filter

  }


  // Main loop over paths
  // --------------------
  for (int i=1; i< hist_2d->GetNbinsX();i++) { 


    unsigned int pathByIndex = triggerNames.triggerIndex(hist_2d->GetXaxis()->GetBinLabel(i));
    if(pathByIndex >= triggerResults_->size() ) continue;

  
    // Fill HLTPassed Matrix and HLTPassFail Matrix
    // --------------------------------------------------------

    if(triggerResults_->accept(pathByIndex)){
  
      hist_2d->Fill(i-1,anyBinNumber-1);//binNumber1 = 0 = first filter
      hist_2d->Fill(anyBinNumber-1,i-1);//binNumber1 = 0 = first filter

      hist_1d->Fill(i-1);//binNumber1 = 0 = first filter

      for (int j=1; j< hist_2d->GetNbinsY();j++) {
  
        unsigned int crosspathByIndex = triggerNames.triggerIndex(hist_2d->GetXaxis()->GetBinLabel(j));
        if(crosspathByIndex >= triggerResults_->size() ) continue;
  
        if(triggerResults_->accept(crosspathByIndex)){
  
          hist_2d->Fill(i-1,j-1);//binNumber1 = 0 = first filter
  
        } // end if j path passed
  
      } // end for j 
  
    } // end if i passed

  } // end for i

 } // end for mi

}

void FourVectorHLTOffline::setupHltBxPlots()
{

  //pathsSummaryFolder_ = TString("HLT/FourVector/PathsSummary/");
  //dbe_->setCurrentFolder(pathsSummaryFolder_.c_str());
  dbe_->setCurrentFolder(pathsSummaryFolder_);

  // setup HLT bx plot
  int Nbx = 3600;
  unsigned int npaths = hltPaths_.size();

  ME_HLT_bx_ = dbe_->book2D("HLT_bx",
                         "HLT counts vs Event bx",
                         Nbx, -0.5, Nbx-0.5, npaths, -0.5, npaths-0.5);
  ME_HLT_bx_->setAxisTitle("Bunch Crossing");


  // Set up bin labels on Y axis continuing to cover all npaths
  for(unsigned int i = 0; i < npaths; i++){

    ME_HLT_bx_->getTH2F()->GetYaxis()->SetBinLabel(i+1, (hltPaths_[i]).getPath().c_str());

  }


}

void FourVectorHLTOffline::setupHltLsPlots()
{
 
  unsigned int npaths = hltPaths_.size();

  //pathsSummaryHLTPathsPerLSFolder_ = TString("HLT/FourVector/PathsSummary/HLT LS/");
  //dbe_->setCurrentFolder(pathsSummaryHLTPathsPerLSFolder_.c_str());
  dbe_->setCurrentFolder(pathsSummaryHLTPathsPerLSFolder_);

  ME_HLTAll_LS_  = dbe_->book2D("All_count_LS",
                    "All paths per LS ",
                         nLS_, 0, nLS_, npaths+1, -0.5, npaths+1-0.5);
  ME_HLTAll_LS_->setAxisTitle("Luminosity Section");

  // Set up bin labels on Y axis continuing to cover all npaths
  for(unsigned int i = 0; i < npaths; i++){

    ME_HLTAll_LS_->getTH2F()->GetYaxis()->SetBinLabel(i+1, (hltPaths_[i]).getPath().c_str());

  }

  unsigned int i = npaths;
  ME_HLTAll_LS_->getTH2F()->GetYaxis()->SetBinLabel(i+1, "Any HLT");

  int nBinsPerLSHisto = 20;
  int nLSHistos = npaths/nBinsPerLSHisto;
  for (int nh=0;nh<nLSHistos+1;nh++) {

    char name[200];
    char title[200];

    sprintf(name, "Group_%d_paths_count_LS",nLSHistos-1-nh);
    sprintf(title, "Group %d,  paths count per LS",nLSHistos-1-nh);

    MonitorElement* tempME  = dbe_->book2D(name,title,
                    nLS_, 0, nLS_, nBinsPerLSHisto+1, -0.5, nBinsPerLSHisto+1-0.5);

    tempME->setAxisTitle("LS");

    // Set up bin labels on Y axis continuing to cover all npaths
    for(int i = nh*nBinsPerLSHisto; i < (nh+1)*nBinsPerLSHisto; i++){

      if (i == int(npaths)) break;

      int bin;
      if(nh == 0){

       bin = i;

      }
      else {

       bin = i % nBinsPerLSHisto;

      }

      tempME->setBinLabel(bin+1, hltPaths_[i].getPath().c_str(), 2);

    }

    tempME->setBinLabel(nBinsPerLSHisto+1, "Any HLT", 2);

    v_ME_HLTAll_LS_.push_back(tempME);

  }


}



void FourVectorHLTOffline::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){   

   //int lumi = int(lumiSeg.id().luminosityBlock());
   //if(lumi < 74 || lumi > 77) fLumiFlag = false;
   //else fLumiFlag = true;

}

void FourVectorHLTOffline::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c)
{

   int lumi = int(lumiSeg.id().luminosityBlock());
   LogTrace("FourVectorHLTOffline") << " end lumiSection number " << lumi << endl;

    // get the count of path up to now
   string fullPathToME = "HLT/FourVector/PathsSummary/HLT_All_PassPass";
   MonitorElement* ME_2d = dbe_->get(fullPathToME);

   if(! ME_2d) {

     LogTrace("FourVectorHLTOffline") << " could not fine 2d matrix " << fullPathToME << endl;

     return;

   }

   TH2F * hist_2d = ME_2d->getTH2F();

   for (std::vector<std::pair<std::string, int> >::iterator ip = fPathTempCountPair.begin(); ip != fPathTempCountPair.end(); ++ip) {
  
    // get the path and its previous count
    std::string pathname = ip->first;  
    int prevCount = ip->second;  
    
    // get the current count of path up to now
    int pathBin = hist_2d->GetXaxis()->FindBin(pathname.c_str());      

    if(pathBin > hist_2d->GetNbinsX()) {
      
      cout << " Cannot find the bin for path " << pathname << endl;
      continue;

    }

    int currCount = int(hist_2d->GetBinContent(pathBin, pathBin));

    // count due to prev lumi sec is a difference bw current and previous
    int diffCount = currCount - prevCount;

    LogTrace("FourVectorHLTOffline") << " lumi = " << lumi << "  path " << pathname << "  count " << diffCount <<  endl;

    // set the counter in the pair to current count
    ip->second = currCount;  

    //////////////////////////////////////
    // fill the 2D All paths' count per LS
    //////////////////////////////////////
    if ( ME_HLTAll_LS_) {

      TH2F* hist_All = ME_HLTAll_LS_->getTH2F();

      // find the bin
      int pathBinNumber = hist_All->GetYaxis()->FindBin(pathname.c_str());
      
      // update  the bin content  (must do that since events don't ncessarily come in the order
      int currentLumiCount = int(hist_All->GetBinContent(lumi+1,pathBinNumber));
      int updatedLumiCount = currentLumiCount + diffCount;
      hist_All->SetBinContent(lumi+1,pathBinNumber,updatedLumiCount);
    
    }
    else {

      LogDebug("FourVectorHLTOffline") << " cannot find ME_HLTAll_LS_" <<  endl;

    }
    
    for (unsigned int i=0 ; i< v_ME_HLTAll_LS_.size(); i++) {  
      
      MonitorElement* tempME = v_ME_HLTAll_LS_[i];

      if ( tempME ) {
  
        TH2F* hist_All = tempME->getTH2F();
  
        // find the bin
        int pathBinNumber = hist_All->GetYaxis()->FindBin(pathname.c_str());
        // update  the bin content  (must do that since events don't ncessarily come in the order
        int currentLumiCount = int(hist_All->GetBinContent(lumi+1,pathBinNumber));
        int updatedLumiCount = currentLumiCount + diffCount;
        hist_All->SetBinContent(lumi+1,pathBinNumber,updatedLumiCount);
      
      }
      else {
  
        LogDebug("FourVectorHLTOffline") << " cannot find tempME " <<  endl;
  
      }

    }


    ///////////////////////////////////////////
    // fill the 1D individual path count per LS
    ///////////////////////////////////////////
    string fullPathToME_count = pathsIndividualHLTPathsPerLSFolder_.c_str() + pathname + "_count_per_LS";
    MonitorElement* ME_1d = dbe_->get(fullPathToME_count);
    if ( ME_1d) { 

      // update  the bin content  (must do that since events don't ncessarily come in the order
      int currentLumiCount = int(ME_1d->getTH1()->GetBinContent(lumi+1));
      int updatedLumiCount = currentLumiCount + diffCount;
      ME_1d->getTH1()->SetBinContent(lumi+1,updatedLumiCount);

    }
    else {

      LogDebug("FourVectorHLTOffline") << " cannot find ME " << fullPathToME_count  <<  endl;

    }

  } // end for ip

}


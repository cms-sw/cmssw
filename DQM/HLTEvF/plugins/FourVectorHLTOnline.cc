// $Id: FourVectorHLTOnline.cc,v 1.42 2012/02/28 10:54:57 fwyzard Exp $
// See header file for information. 
#include "TMath.h"
#include "DQM/HLTEvF/interface/FourVectorHLTOnline.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include <map>
#include <utility>


using namespace edm;
using namespace trigger;
using namespace std;

FourVectorHLTOnline::FourVectorHLTOnline(const edm::ParameterSet& iConfig):
  resetMe_(true),  currentRun_(-99)
{

  LogDebug("FourVectorHLTOnline") << "constructor...." ;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("FourVectorHLTOnline") << "unabel to get DQMStore service?";
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
  fCustomBXPath = iConfig.getUntrackedParameter<std::string>("customBXPath", std::string("HLT_MinBiasBSC"));

  referenceBX_ = iConfig.getUntrackedParameter<unsigned int>("referenceBX",51);
  Nbx_ = iConfig.getUntrackedParameter<unsigned int>("Nbx",3564);

  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",20);
  nBinsOneOverEt_ = iConfig.getUntrackedParameter<unsigned int>("NbinsOneOverEt",10000);
  nLS_   = iConfig.getUntrackedParameter<unsigned int>("NLuminositySegments",10);

  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);
     // this is the list of paths to look at.
  std::vector<edm::ParameterSet> paths = 
  iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");

  for(std::vector<edm::ParameterSet>::iterator pathconf = paths.begin() ; pathconf != paths.end(); pathconf++) {

    //std::pair<std::string, std::string> custompathnamepair;
    //custompathnamepair.first =pathconf->getParameter<std::string>("pathname"); 
    //custompathnamepair.second = pathconf->getParameter<std::string>("denompathname");   
    //custompathnamepairs_.push_back(custompathnamepair);
    custompathnamepairs_.push_back(
        make_pair(
          pathconf->getParameter<std::string>("pathname"),
          pathconf->getParameter<std::string>("denompathname")
        )
    );

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
  pathsSummaryHLTPathsPerBXFolder_ = iConfig.getUntrackedParameter ("individualPathsPerBXFolder",std::string("HLT/FourVector/PathsSummary/HLT BX/"));

  fLumiFlag = true;
  ME_HLTAll_LS_ = NULL;
  ME_HLT_BX_ = NULL;
  ME_HLT_CUSTOM_BX_ = NULL;
  
}


FourVectorHLTOnline::~FourVectorHLTOnline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
FourVectorHLTOnline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //if(! fLumiFlag ) return;

  using namespace edm;
  using namespace trigger;
  ++nev_;
  LogDebug("FourVectorHLTOnline")<< " analyze...." ;


  
  /*
  Handle<GenParticleCollection> genParticles;
  iEvent.getByLabel("genParticles", genParticles);
  if(!genParticles.isValid()) { 
    edm::LogInfo("FourVectorHLTOnline") << "genParticles not found, "
      "skipping event"; 
    return;
  }

  Handle<GenJetCollection> genJets;
  iEvent.getByLabel("iterativeCone5GenJets",genJets);
  if(!genJets.isValid()) { 
    edm::LogInfo("FourVectorHLTOnline") << "genJets not found, "
      "skipping event"; 
    return;
  }

  Handle<GenMETCollection> genMets;
  iEvent.getByLabel("genMetTrue",genMets);
  if(!genMets.isValid()) { 
    edm::LogInfo("FourVectorHLTOnline") << "genMets not found, "
      "skipping event"; 
    return;
  }
  */

  // Get trigger results
  
  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(triggerResultsLabel_,triggerResults);
  if(!triggerResults.isValid()) {
    edm::InputTag triggerResultsLabelFU(triggerResultsLabel_.label(),triggerResultsLabel_.instance(), "FU");
   iEvent.getByLabel(triggerResultsLabelFU,triggerResults);
  if(!triggerResults.isValid()) {
    edm::LogInfo("FourVectorHLTOnline") << "TriggerResults not found, "
      "skipping event"; 
    return;
   }
  }
  triggerResults_ = triggerResults;
  const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
  int npath = triggerResults->size();

  iEvent.getByLabel(triggerSummaryLabel_,fTriggerObj); 
  if(!fTriggerObj.isValid()) {

    edm::InputTag triggerSummaryLabelFU(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(), "FU");
    iEvent.getByLabel(triggerSummaryLabelFU,fTriggerObj);

    if(!fTriggerObj.isValid()) {

      edm::LogInfo("FourVectorHLTOnline") << "TriggerEvent not found, " "skipping event"; 
      return;

    }

  }

  edm::Handle<TriggerResults> muonHandle;
  iEvent.getByLabel(muonRecoCollectionName_,muonHandle);
  if(!muonHandle.isValid()) { 

    edm::LogInfo("FourVectorHLTOnline") << "muonHandle not found, ";
    //  "skipping event"; 
    //  return;

  }


  edm::Handle<TriggerResults> gsfElectrons;
  iEvent.getByLabel("gsfElectrons",gsfElectrons); 
  if(!gsfElectrons.isValid()) 
    edm::LogInfo("FourVectorHLTOnline") << "gsfElectrons not found, ";

  edm::Handle<TriggerResults> tauHandle;
  iEvent.getByLabel("caloRecoTauProducer",tauHandle);
  if(!tauHandle.isValid()) 
    edm::LogInfo("FourVectorHLTOnline") << "tauHandle not found, ";

  edm::Handle<TriggerResults> jetHandle;
  iEvent.getByLabel("iterativeCone5CaloJets",jetHandle);
  if(!jetHandle.isValid()) 
    edm::LogInfo("FourVectorHLTOnline") << "jetHandle not found, ";
 
   // Get b tag information
 edm::Handle<TriggerResults> bTagIPHandle;
 iEvent.getByLabel("jetProbabilityBJetTags", bTagIPHandle);
 if (!bTagIPHandle.isValid()) 
    edm::LogInfo("FourVectorHLTOnline") << "mTagIPHandle trackCountingHighEffJetTags not found, ";

   // Get b tag information
 edm::Handle<TriggerResults> bTagMuHandle;
 iEvent.getByLabel("softMuonBJetTags", bTagMuHandle);
 if (!bTagMuHandle.isValid()) 
    edm::LogInfo("FourVectorHLTOnline") << "bTagMuHandle  not found, ";

  edm::Handle<TriggerResults> metHandle;
  iEvent.getByLabel("met",metHandle);
  if(!metHandle.isValid()) 
    edm::LogInfo("FourVectorHLTOnline") << "metHandle not found, ";

  edm::Handle<TriggerResults> photonHandle;
  iEvent.getByLabel("photons",photonHandle);
  if(!photonHandle.isValid()) 
    edm::LogInfo("FourVectorHLTOnline") << "photonHandle not found, ";

  edm::Handle<TriggerResults> trackHandle;
  iEvent.getByLabel("pixelTracks",trackHandle);
  if(!trackHandle.isValid()) 
    edm::LogInfo("FourVectorHLTOnline") << "trackHandle not found, ";

  // ---------------------
  // Monitors
  // ---------------------

  // electron Monitor
  objMonData<TriggerResults> eleMon;
  eleMon.setReco(gsfElectrons);
  eleMon.setLimits(electronEtaMax_, electronEtMin_, electronDRMatch_);
  
  eleMon.pushTriggerType(TriggerElectron);
  eleMon.pushTriggerType(TriggerL1NoIsoEG);
  eleMon.pushTriggerType(TriggerL1IsoEG);

  eleMon.pushL1TriggerType(TriggerL1NoIsoEG);
  eleMon.pushL1TriggerType(TriggerL1IsoEG);

  // muon Monitor
  objMonData<TriggerResults>  muoMon;
  muoMon.setReco(muonHandle);
  muoMon.setLimits(muonEtaMax_, muonEtMin_, muonDRMatch_);
  
  muoMon.pushTriggerType(TriggerMuon);
  muoMon.pushTriggerType(TriggerL1Mu);

  muoMon.pushL1TriggerType(TriggerL1Mu);
  
  // tau Monitor
  objMonData<TriggerResults>  tauMon;
  tauMon.setReco(tauHandle);
  tauMon.setLimits(tauEtaMax_, tauEtMin_, tauDRMatch_);
  
  tauMon.pushTriggerType(TriggerTau);
  tauMon.pushTriggerType(TriggerL1TauJet);

  tauMon.pushL1TriggerType(TriggerL1TauJet);
  tauMon.pushL1TriggerType(TriggerL1ForJet);
  
  // photon Monitor
  objMonData<TriggerResults> phoMon;
  phoMon.setReco(photonHandle);
  phoMon.setLimits(photonEtaMax_, photonEtMin_, photonDRMatch_);
  
  phoMon.pushTriggerType(TriggerPhoton);

  phoMon.pushL1TriggerType(TriggerL1NoIsoEG);
  phoMon.pushL1TriggerType(TriggerL1IsoEG);

  // jet Monitor - NOTICE: we use genJets for MC
  objMonData<TriggerResults> jetMon;
  jetMon.setReco(jetHandle);
  jetMon.setLimits(jetEtaMax_, jetEtMin_, jetDRMatch_);

  jetMon.pushTriggerType(TriggerJet);
  jetMon.pushTriggerType(TriggerL1CenJet);
  jetMon.pushTriggerType(TriggerL1ForJet);
  
  jetMon.pushL1TriggerType(TriggerL1CenJet);
  jetMon.pushL1TriggerType(TriggerL1ForJet);
  jetMon.pushL1TriggerType(TriggerL1TauJet);

  // bjet Monitor - NOTICE: we use genJets for MC
  objMonData<TriggerResults> btagIPMon; // CaloJet will not be used, this is only place holder
  //btagIPMon.setReco(jetHandle);
  btagIPMon.setBJetsFlag(true);
  btagIPMon.setLimits(bjetEtaMax_, bjetEtMin_, bjetDRMatch_);

  btagIPMon.pushTriggerType(TriggerBJet);
  btagIPMon.pushTriggerType(TriggerJet);

  btagIPMon.pushL1TriggerType(TriggerL1CenJet);
  btagIPMon.pushL1TriggerType(TriggerL1ForJet);
  btagIPMon.pushL1TriggerType(TriggerL1TauJet);

  objMonData<TriggerResults> btagMuMon; // CaloJet will not be used, this is only place holder
  //btagMuMon.setReco(jetHandle);
  btagMuMon.setBJetsFlag(true);
  btagMuMon.setLimits(bjetEtaMax_, bjetEtMin_, bjetDRMatch_);

  btagMuMon.pushTriggerType(TriggerBJet);
  btagMuMon.pushTriggerType(TriggerJet);

  btagMuMon.pushL1TriggerType(TriggerL1CenJet);
  btagMuMon.pushL1TriggerType(TriggerL1ForJet);
  btagMuMon.pushL1TriggerType(TriggerL1TauJet);


  objMonData<TriggerResults> btagMon; // Generic btagMon
 
  // met Monitor
  objMonData<TriggerResults> metMon;
  metMon.setReco(metHandle);
  metMon.setLimits(metEtaMax_, metMin_, metDRMatch_);
  
  metMon.pushTriggerType(TriggerMET);

  metMon.pushL1TriggerType(TriggerL1ETM);

  // tet Monitor
  objMonData<TriggerResults> tetMon;
  tetMon.setReco(metHandle);
  //tetMon.setLimits(tetEtaMax_=999., tetEtMin_=10, tetDRMatch_=999);
  tetMon.setLimits(999., 10., 999.);
  
  tetMon.pushTriggerType(TriggerTET);

  tetMon.pushL1TriggerType(TriggerL1ETT);

  // default Monitor
  //objMonData<trigger::TriggerEvent> defMon;
  objMonData<TriggerResults> defMon;
  defMon.setLimits(999., 3., 999.);

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
  monitors.push_back(&tetMon);

  int bx = iEvent.bunchCrossing();
  /*
  // Fill HLTPassed_Correlation Matrix bin (i,j) = (Any,Any)
  // --------------------------------------------------------
  int anyBinNumber = ME_HLTPassPass_->getTH2F()->GetXaxis()->FindBin("HLT_Any");      
  // any triger accepted
  if(triggerResults->accept()){

    ME_HLTPassPass_->Fill(anyBinNumber-1,anyBinNumber-1);//binNumber1 = 0 = first filter

  }
  */

  //fillHltMatrix(triggerNames);


  // Loop over paths
  // --------------------
  for(PathInfoCollection::iterator v = hltPathsDiagonal_.begin(); v!= hltPathsDiagonal_.end(); ++v ) { 

    LogTrace("FourVectorHLTOnline") << " unique path " << v->getPath() << std::endl;

    unsigned int pathByIndex = triggerNames.triggerIndex(v->getPath());

    // path must be in the menu
    if(pathByIndex >= triggerResults_->size() ) continue;

  
    // Fill HLTPassed Matrix and HLTPassFail Matrix
    // --------------------------------------------------------

    if(triggerResults->accept(pathByIndex)){
  
      int pathBinNumber = ME_HLT_BX_->getTH2F()->GetYaxis()->FindBin(v->getPath().c_str());      
      ME_HLT_BX_->Fill(bx,pathBinNumber-1);

      if(hasHLTPassed(fCustomBXPath,triggerNames)) {

        ME_HLT_CUSTOM_BX_->Fill(bx,pathBinNumber-1);

      }

    } // end if accept
  
    // Fill histogram of filter ocupancy for each HLT path
    // ---------------------------------
    unsigned int lastModule = triggerResults->index(pathByIndex);

    //check if hlt passed
    bool hltPathPassed = triggerResults->accept(pathByIndex);
  
    //go through the list of filters
    for(unsigned int filt = 0; filt < v->filtersAndIndices.size(); filt++){
      
      int binNumber = v->getFiltersHisto()->getTH1()->GetXaxis()->FindBin(v->filtersAndIndices[filt].first.c_str());      
      
      // if hlt path passed, then all the modules passed
      // otherwise the module that issued the decision is the first fail
      // this means that all modules before it passed
      if(hltPathPassed) {

        v->getFiltersHisto()->Fill(binNumber-1);//binNumber1 = 0 = first filter
      }
      else if(v->filtersAndIndices[filt].second < lastModule){
        v->getFiltersHisto()->Fill(binNumber-1);//binNumber1 = 0 = first filter
      }
  
    } // end for filt

  } // end for diag paths


  // Main loop over paths
  // --------------------
  for(PathInfoCollection::iterator v = hltPaths_.begin(); v!= hltPaths_.end(); ++v ) { 

    LogTrace("FourVectorHLTOnline") << " path " << v->getPath() << std::endl;

    if (v->getPath().find("BTagIP") != std::string::npos ) btagMon = btagIPMon;
    else btagMon = btagMuMon;

    //if(*v != "HLT_L1Jet6U") continue;

    unsigned int pathByIndex = triggerNames.triggerIndex(v->getPath());

    if(pathByIndex >= triggerResults_->size() ) continue;
  
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

     //LogTrace("FourVectorHLTOnline") << " denominator path " << v->getPath() << std::endl;


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
     //if(!mon) continue;

     // if cannot find moniotor for the path, assign the default
     if(!mon) mon = &defMon;

     // attach this path to mon
     mon->setPath(v);

     // clear sets of matched objects
     mon->clearSets();

     //int triggertype = 0;     
     //triggertype = v->getObjectType();

     // monitor L1 (only if L1 passed and can find GTSeed)
     ////////////////////////////////////////////////////
     bool l1accept = false;
     edm::InputTag l1testTag(v->getl1Path(),"",processname_);
     const int l1Index = fTriggerObj->filterIndex(l1testTag);
     
     if ( l1Index >= fTriggerObj->sizeFilters() ) {

       LogTrace("FourVectorHLTOnline") << "Cannot find L1GTSeed of the path " << v->getPath() << std::endl;
       LogTrace("FourVectorHLTOnline") << "\t L1GTSeed name = " << v->getl1Path() << std::endl;
       LogTrace("FourVectorHLTOnline") <<  "\t tried with tag " << l1testTag << std::endl;
       LogTrace("FourVectorHLTOnline") <<"\t module index = "<< l1Index << std::endl;

     } 

	   // monitor L1 object
	   mon->monitorL1(l1Index, this);
	
    // monitor Online (only if HLT passed)
    //////////////////////////////////////
	  bool numpassed = false;

    // did we pass the numerator path, i.e. HLT path?
    for(int i = 0; i < npath; ++i) {

      if (triggerNames.triggerName(i) == v->getPath() && triggerResults->accept(i)) numpassed = true;

    }

    if (!numpassed) continue;

    if (!l1accept) {

          LogDebug("FourVectorHLTOnline") 
            << "This should not happen.  HLT passed, but L1 Seed not passed for hlt path "<< std::endl 
            << "HLT path: " << v->getPath() << "\t HLTLevel1GTSeed: " << v->getl1Path();

    }

    // fill scaler histograms
    edm::InputTag filterTag = v->getTag();

    // loop through indices and see if the filter is 
    // on the list of filters used by this path
    //----------------------------------------------
    if (v->getLabel() == "dummy"){
    const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());

    //loop over labels
    for (std::vector<std::string>::const_iterator labelIter= filterLabels.begin(); labelIter!=filterLabels.end(); labelIter++) {

      edm::InputTag testTag(*labelIter,"",processname_);

      int testindex = fTriggerObj->filterIndex(testTag);

      // last match wins...
      if ( !(testindex >= fTriggerObj->sizeFilters()) ) {

        filterTag = testTag; 
        v->setLabel(*labelIter);}

      }

    } // end for
  
    const int hltIndex = fTriggerObj->filterIndex(filterTag);
    if ( hltIndex >= fTriggerObj->sizeFilters() ) {

      LogTrace("FourVectorHLTOnline") << "WTF no index of that name " << filterTag << std::endl;
      continue; // not in this event

    }

    mon->monitorOnline(hltIndex, l1Index, this);

    mon->fillOnlineMatch(l1Index, this);

   } //denompassed


 } //pathinfo loop

}



// -- method called once each job just before starting event loop  --------
void 
FourVectorHLTOnline::beginJob()
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
FourVectorHLTOnline::endJob() 
{
   LogInfo("FourVectorHLTOnline") << "analyzed " << nev_ << " events";
   return;
}


// BeginRun
void FourVectorHLTOnline::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTOnline") << "beginRun, run " << run.id();

  // HLT config does not change within runs!
  bool changed=false;
 
  if (!hltConfig_.init(run, c, processname_, changed)) {
    processname_ = "FU";

    if (!hltConfig_.init(run, c, processname_, changed)){

      LogDebug("FourVectorHLTOnline") << "HLTConfigProvider failed to initialize.";

    }

    // check if trigger name in (new) config
    //  std::cout << "Available TriggerNames are: " << std::endl;
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

        std::string l1pathname = getL1ConditionModuleName(pathname);
      
        int objectType =  getTriggerTypeParsePathName(pathname);
  
        for (unsigned int i=0; i!=n; ++i) {
  
          std::string denompathname = hltConfig_.triggerName(i);  
          int denomobjectType =  getTriggerTypeParsePathName(denompathname);

      
         
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
         //parse pathname to guess object type
         int objectType =  getTriggerTypeParsePathName(pathname);

        std::string l1pathname = getL1ConditionModuleName(pathname);
  
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

          hltPathsDiagonal_.push_back(PathInfo(denompathname, pathname, l1pathname, filtername, processname_, objectType, ptMin, ptMax));
  
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

            std::string n_pathname = hltConfig_.triggerName(k);

            if (n_pathname.find(denompathname) != std::string::npos) {
              
              LogDebug("FourVectorHLTOnline") << "denompathname is selected to be = " << n_pathname << std::endl;;
              founddenominator = true;

              break;

            }
          }

          if (!founddenominator) {
  
            edm::LogInfo("FourVectorHLTOnline") << "denompathname not found, go to the next pair numearator-denominator" << std::endl;
            
            // go to the next pair
            continue;
  
          }

          // check that numerator exists
          bool foundnumerator = false;
          for (unsigned int j=0; j!=n; ++j) {

            std::string pathname = hltConfig_.triggerName(j);

            LogDebug("FourVectorHLTOnline") << "check if path " << pathname << " is numpathname = " << numpathname << std::endl;
            if (hltConfig_.triggerName(j).find(numpathname)!= std::string::npos) {
              
              LogDebug("FourVectorHLTOnline") << "pathname is selected to be = " << denompathname << std::endl;;
              foundnumerator = true;

            }
  
  
            if (!foundnumerator) {
    
              edm::LogInfo("FourVectorHLTOnline") << "pathname not found, ignoring " << pathname;
              continue;
  
            }
  
  
            std::string l1pathname = getL1ConditionModuleName(pathname);
            int objectType =  getTriggerTypeParsePathName(pathname);
  
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


    std::vector<std::string> muonPaths;
    std::vector<std::string> egammaPaths;
    std::vector<std::string> tauPaths;
    std::vector<std::string> jetmetPaths;
    std::vector<std::string> restPaths;
    std::vector<std::string> allPaths;
    // fill vectors of Muon, Egamma, JetMet, Rest, and Special paths
    for(PathInfoCollection::iterator v = hltPathsDiagonal_.begin(); v!= hltPathsDiagonal_.end(); ++v ) {

      std::string pathName = v->getPath();
      int objectType = v->getObjectType();

      std::vector<int> tempCount(5,0);

      fPathTempCountPair.push_back(make_pair(pathName,0));
      fPathBxTempCountPair.push_back(make_pair(pathName,tempCount));

      allPaths.push_back(pathName);

      switch (objectType) {
        case trigger::TriggerMuon :
          muonPaths.push_back(pathName);
          break;

        case trigger::TriggerElectron :
        case trigger::TriggerPhoton :
          egammaPaths.push_back(pathName);
          break;

        case trigger::TriggerTau :
          tauPaths.push_back(pathName);
          break;

        case trigger::TriggerJet :
        case trigger::TriggerMET :
          jetmetPaths.push_back(pathName);
          break;

        default:
          restPaths.push_back(pathName);
      }

    }

    fPathTempCountPair.push_back(make_pair("HLT_Any",0));

    fGroupName.push_back("All");
    fGroupName.push_back("Muon");
    fGroupName.push_back("Egamma");
    fGroupName.push_back("Tau");
    fGroupName.push_back("JetMet");
    fGroupName.push_back("Rest");
    fGroupName.push_back("Special");

    for(unsigned int g=0; g<fGroupName.size(); g++) {

      //fGroupTempCountPair.push_back(make_pair(fGroupName[g],0));
      //fGroupL1TempCountPair.push_back(make_pair(fGroupName[g],0));

    }
  
    dbe_->setCurrentFolder(pathsSummaryFolder_.c_str());



    fGroupNamePathsPair.push_back(make_pair("All",allPaths));

    fGroupNamePathsPair.push_back(make_pair("Muon",muonPaths));

    fGroupNamePathsPair.push_back(make_pair("Egamma",egammaPaths));

    fGroupNamePathsPair.push_back(make_pair("Tau",tauPaths));

    fGroupNamePathsPair.push_back(make_pair("JetMet",jetmetPaths));

    fGroupNamePathsPair.push_back(make_pair("Rest",restPaths));

    fGroupNamePathsPair.push_back(make_pair("Special",specialPaths_));

    /// add dataset name and thier triggers to the list 
    std::vector<std::string> datasetNames =  hltConfig_.datasetNames() ;
    for (unsigned int i=0;i<datasetNames.size();i++) {

      std::vector<std::string> datasetPaths = hltConfig_.datasetContent(datasetNames[i]);
 
      fGroupNamePathsPair.push_back(make_pair(datasetNames[i],datasetPaths));
      //setupHltMatrix(datasetNames[i],datasetPaths);

    }

    /*
    setupHltMatrix("All", allPaths);
    setupHltMatrix("Muon", muonPaths);
    setupHltMatrix("Egamma", egammaPaths);
    setupHltMatrix("Tau", tauPaths);
    setupHltMatrix("JetMET", jetmetPaths);
    setupHltMatrix("Rest", restPaths);
    setupHltMatrix("Special", specialPaths_);
    */

    for (unsigned int g=0;g<fGroupNamePathsPair.size();g++) {

      fGroupTempCountPair.push_back(make_pair(fGroupNamePathsPair[g].first,0));
      fGroupL1TempCountPair.push_back(make_pair(fGroupNamePathsPair[g].first,0));
      //setupHltMatrix(fGroupNamePathsPair[g].first,fGroupNamePathsPair[g].second);

    }
    /*
    */

    setupHltLsPlots();

    setupHltBxPlots();


    for(PathInfoCollection::iterator v = hltPathsDiagonal_.begin(); v!= hltPathsDiagonal_.end(); ++v ) {

       // -------------------------
       //
       //  Filters for each path
       //
       // -------------------------
       
       // get all modules in this HLT path
       std::vector<std::string> moduleNames = hltConfig_.moduleLabels( v->getPath() ); 
       
       int numModule = 0;
       std::string moduleName, moduleType;
       unsigned int moduleIndex;
       
       //print module name
       std::vector<std::string>::const_iterator iDumpModName;
       for (iDumpModName = moduleNames.begin();iDumpModName != moduleNames.end();iDumpModName++) {

         moduleName = *iDumpModName;
         moduleType = hltConfig_.moduleType(moduleName);
         moduleIndex = hltConfig_.moduleIndex(v->getPath(), moduleName);

         LogTrace ("FourVectorHLTOnline") << "Module "      << numModule
             << " is called " << moduleName
             << " , type = "  << moduleType
             << " , index = " << moduleIndex
             << std::endl;

         numModule++;

         if((moduleType.find("Filter") != std::string::npos && moduleType.find("HLTTriggerTypeFilter") == std::string::npos ) || 
            (moduleType.find("Associator") != std::string::npos) || 
            (moduleType.find("HLTLevel1GTSeed") != std::string::npos) || 
            (moduleType.find("HLTGlobalSumsCaloMET") != std::string::npos) ||
            (moduleType.find("HLTPrescaler") != string::npos) ||
            (hltConfig_.moduleEDMType(moduleName).find("EDFilter") != string::npos))
          {

           //std::pair<std::string, int> filterIndexPair;
           //filterIndexPair.first   = moduleName;
           //filterIndexPair.second  = moduleIndex;
           //v->filtersAndIndices.push_back(filterIndexPair);
           v->filtersAndIndices.push_back(make_pair(moduleName,moduleIndex));

         }


       }//end for modulesName


       dbe_->setCurrentFolder(pathsSummaryFilterCountsFolder_.c_str()); 
      
       //int nbin_sub = 5;
       int nbin_sub = v->filtersAndIndices.size()+2;
    
       std::string pathName = v->getPath();
          
       vector<string> datasetNames =  hltConfig_.datasetNames();
      
       //Creates subfolders in FourVector->PathsSummary->FiltersCounts for each dataset,
       //and moves each path's filter histogram to the proper dataset folder
       for (unsigned int k=0;k<datasetNames.size();k++) { //Loop to cycle through datasets
	 // std::cout << "Dataset " << datasetNames[k] << " has trigger paths " << std::endl;
	
	 std::string datasetFolder = pathsSummaryFilterCountsFolder_ + datasetNames[k];
	 vector<string> datasetPaths = hltConfig_.datasetContent(datasetNames[k]);
	 
	 for (unsigned int m=0;m<datasetPaths.size();m++){ //Loop to cycle through trigger paths
	   // std::cout << "              " <<  datasetPaths[m] << std::endl;
	 
	   if(datasetPaths[m]==pathName){ //Moves path to proper dataset directory
	     dbe_->setCurrentFolder(datasetFolder.c_str());
	   }
	 }
       }
	     

       // count plots for subfilter
       MonitorElement* filters = dbe_->book1D("Filters_" + v->getPath(), 
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

       v->setFilterHistos(filters);

    } // end for paths

    // now set up all of the histos for each path-denom
    for(PathInfoCollection::iterator v = hltPaths_.begin(); v!= hltPaths_.end(); ++v ) {

      MonitorElement *NOn, *onEtOn, *onOneOverEtOn, *onEtavsonPhiOn=0;
      MonitorElement *NOff, *offEtOff, *offEtavsoffPhiOff=0;
      MonitorElement *NL1, *l1EtL1, *l1Etavsl1PhiL1=0;
      MonitorElement *NL1On, *l1EtL1On, *l1Etavsl1PhiL1On=0;
      MonitorElement *NL1Off, *offEtL1Off, *offEtavsoffPhiL1Off=0;
      MonitorElement *NOnOff, *offEtOnOff, *offEtavsoffPhiOnOff=0;
      MonitorElement *NL1OnUM, *l1EtL1OnUM, *l1Etavsl1PhiL1OnUM=0;
      MonitorElement *NL1OffUM, *offEtL1OffUM, *offEtavsoffPhiL1OffUM=0;
      MonitorElement *NOnOffUM, *offEtOnOffUM, *offEtavsoffPhiOnOffUM=0;
      MonitorElement *offDRL1Off, *offDROnOff, *l1DRL1On=0;
      

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
       
       histoname = labelname+"_onOneOverEtOn";
       title = labelname+" 1 / onE_t online";
       onOneOverEtOn =  dbe->book1D(histoname.c_str(), title.c_str(),nBinsOneOverEt_, 0, 1);
       onOneOverEtOn->setAxisTitle("HLT 1/Et [1/GeV]");
       
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


       v->setHistos( NOn, onEtOn, onOneOverEtOn, onEtavsonPhiOn, NOff, offEtOff, offEtavsoffPhiOff, NL1, l1EtL1, l1Etavsl1PhiL1, NL1On, l1EtL1On, l1Etavsl1PhiL1On, NL1Off, offEtL1Off, offEtavsoffPhiL1Off, NOnOff, offEtOnOff, offEtavsoffPhiOnOff, NL1OnUM, l1EtL1OnUM, l1Etavsl1PhiL1OnUM, NL1OffUM, offEtL1OffUM, offEtavsoffPhiL1OffUM, NOnOffUM, offEtOnOffUM, offEtavsoffPhiOnOffUM, offDRL1Off, offDROnOff, l1DRL1On 
);


    }  // end for hltPath

    // HLT_Any
    // book Count vs LS
    dbe_->setCurrentFolder(pathsIndividualHLTPathsPerLSFolder_.c_str());
    MonitorElement* tempME = dbe_->book1D("HLT_Any_count_per_LS", 
                           "HLT_Any count per LS",
                           nLS_, 0,nLS_);
    tempME->setAxisTitle("Luminosity Section");

  } // end if(1) dummy

 return;

}

/// EndRun
void FourVectorHLTOnline::endRun(const edm::Run& run, const edm::EventSetup& c)
{

  LogDebug("FourVectorHLTOnline") << "endRun, run " << run.id();

}

/// Clean DR Match Set
void FourVectorHLTOnline::cleanDRMatchSet(mmset& tempSet)
{

  LogDebug("FourVectorHLTOnline") << "cleanDRMatchSet(mmset& tempSet) " << "size of the set (before CLEANING) = " << tempSet.size() << " maps." << std::endl;

  if(tempSet.size() < 2) return;

  if(tempSet.size() > 10) {

    LogDebug("FourVectorHLTOnline") << "size of the set is too large.  It will be truncated to 10." << std::endl;
    mmset::iterator it = tempSet.begin();
    for (int i=0;i<10;i++) {    it++; }
    tempSet.erase( it, tempSet.end());
    LogDebug("FourVectorHLTOnline") << "size of the set is now = " << tempSet.size() << " maps." << std::endl;

  }
 
  bool cleanedOneMap = false;
 
  // cleaning needed if the set has at least two maps

  while(! cleanedOneMap && tempSet.size() > 1) {

    cleanedOneMap=false;

    //LogTrace("FourVectorHLTOnline") << "cleaning: size of the set  = " << tempSet.size() << " maps." << std::endl;

    int imap = 0;
    for ( mmset::iterator setIter_i = tempSet.begin( ); setIter_i != tempSet.end( ); setIter_i++ ) {

      fimmap tempMap_j = *setIter_i;

      //LogTrace("FourVectorHLTOnline") << " map " << imap << std::endl;
      //LogTrace("FourVectorHLTOnline") << " --------" << std::endl;

      for (fimmap::iterator it = tempMap_j.begin(); it != tempMap_j.end(); ++it) {

        //LogTrace("FourVectorHLTOnline") << " " <<   (*it).first << " :  " << (*it).second << std::endl;

      }

      imap++;

    }

    // loop i
    for ( mmset::iterator setIter_i = tempSet.begin( ); setIter_i != tempSet.end( ); setIter_i++ ) {
     
      fimmap tempMap_i = *setIter_i;
      fimmap::iterator it = tempMap_i.begin();
      int topValue = (*it).second;
      //LogTrace("FourVectorHLTOnline") << " topValue = " << topValue << std::endl;
  
      
      mmset::iterator tempIter_i = setIter_i;
  
      // from all the other maps, clean entries that have mapped value "topValue"
      // loop j
      for ( mmset::iterator setIter_j = ++tempIter_i; setIter_j != tempSet.end( ); setIter_j++ ) {
  
        fimmap tempMap_j = *setIter_j;
        //LogTrace("FourVectorHLTOnline") << "  size of the map  = " << tempMap_j.size() << std::endl;
  
        for (fimmap::iterator it = tempMap_j.begin(); it != tempMap_j.end(); ++it)
        {
  
          if(topValue == (*it).second) {
            
            //LogTrace("FourVectorHLTOnline") << "   Ridding map of a doubly-matched object." << std::endl;
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

  //LogTrace("FourVectorHLTOnline") << "cleaned: size of the set  = " << tempSet.size() << " maps." << std::endl;
  int jmap = 0;

  for ( mmset::iterator setIter_i = tempSet.begin( ); setIter_i != tempSet.end( ); setIter_i++ ) 
  {

    fimmap tempMap_j = *setIter_i;

    //LogTrace("FourVectorHLTOnline") << " map " << jmap << std::endl;
    //LogTrace("FourVectorHLTOnline") << " --------" << std::endl;

    for (fimmap::iterator it = tempMap_j.begin(); it != tempMap_j.end(); ++it) {

        //LogTrace("FourVectorHLTOnline") << " " <<   (*it).first << " :  " << (*it).second << std::endl;
      
    }

    jmap++;

  } // end for

  return;

}

void FourVectorHLTOnline::setupHltMatrix(std::string label, std::vector<std::string> paths) {

    //string groupLabelAny = "HLT_"+label+"_Any";
    //paths.push_back(groupLabelAny.c_str());
    paths.push_back("HLT_"+label+"_L1_Any");
    paths.push_back("HLT_"+label+"_Any");
    paths.push_back("HLT_Any");

    std::string h_name; 
    std::string h_title; 

    dbe_->setCurrentFolder(pathsSummaryFolder_.c_str());

    h_name= "HLT_"+label+"_PassPass";
    h_title = "HLT_"+label+"_PassPass (x=Pass, y=Pass)";
    MonitorElement* ME = dbe_->book2D(h_name.c_str(), h_title.c_str(),
                           paths.size(), -0.5, paths.size()-0.5, paths.size(), -0.5, paths.size()-0.5);

    h_name= "HLT_"+label+"_Pass_Any";
    h_title = "HLT_"+label+"_Pass (x=Pass, Any=Pass) normalized to HLT_Any Pass";
    MonitorElement* ME_Any = dbe_->book1D(h_name.c_str(), h_title.c_str(),
                           paths.size(), -0.5, paths.size()-0.5);

    dbe_->setCurrentFolder(pathsSummaryHLTCorrelationsFolder_.c_str());
    h_name= "HLT_"+label+"_PassPass_Normalized";
    h_title = "HLT_"+label+"_PassPass (x=Pass, y=Pass) normalized to xBin=Pass";
    MonitorElement* ME_Normalized = dbe_->book2D(h_name.c_str(), h_title.c_str(),
                           paths.size(), -0.5, paths.size()-0.5, paths.size(), -0.5, paths.size()-0.5);
    h_name= "HLT_"+label+"_Pass_Normalized_Any";
    h_title = "HLT_"+label+"_Pass (x=Pass, Any=Pass) normalized to HLT_Any Pass";
    MonitorElement* ME_Normalized_Any = dbe_->book1D(h_name.c_str(), h_title.c_str(),
                           paths.size(), -0.5, paths.size()-0.5);

    dbe_->setCurrentFolder(pathsSummaryHLTPathsPerLSFolder_.c_str());
    h_name= "HLT_"+label+"_Total_LS";
    h_title = label+" HLT paths total count combined per LS ";
    MonitorElement* ME_Total_LS = dbe_->book1D(h_name.c_str(), h_title.c_str(), nLS_, 0, nLS_);
    ME_Total_LS->setAxisTitle("LS");

    h_name= "HLT_"+label+"_LS";
    h_title = label+" HLT paths count per LS ";
    MonitorElement* ME_Group_LS = dbe_->book2D(h_name.c_str(), h_title.c_str(), nLS_, 0, nLS_, paths.size(), -0.5, paths.size()-0.5);
    ME_Group_LS->setAxisTitle("LS");
    /// add this path to the vector of 2D LS paths
    v_ME_HLTAll_LS_.push_back(ME_Group_LS);

    h_name= "HLT_"+label+"_L1_Total_LS";
    h_title = label+" HLT paths total count combined per LS ";
    MonitorElement* ME_Total_L1_LS = dbe_->book1D(h_name.c_str(), h_title.c_str(), nLS_, 0, nLS_);
    ME_Total_L1_LS->setAxisTitle("LS");

    h_name= "HLT_"+label+"_L1_LS";
    h_title = label+" HLT L1s count per LS ";
    MonitorElement* ME_Group_L1_LS = dbe_->book2D(h_name.c_str(), h_title.c_str(), nLS_, 0, nLS_, paths.size(), -0.5, paths.size()-0.5);
    ME_Group_L1_LS->setAxisTitle("LS");

    dbe_->setCurrentFolder(pathsSummaryHLTPathsPerBXFolder_.c_str());
    h_name= "HLT_"+label+"_BX_LS";
    h_title = label+" HLT paths total count combined per BX ";
    MonitorElement* ME_Total_BX = dbe_->book2D(h_name.c_str(), h_title.c_str(),  nLS_, 0, nLS_, 5, -2.5, 2.5);
    ME_Total_BX->setAxisTitle("LS",1);

    for(unsigned int i = 0; i < paths.size(); i++){

      ME->getTH2F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME->getTH2F()->GetYaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Group_LS->getTH2F()->GetYaxis()->SetBinLabel(i+1, (paths[i]).c_str());

      ME_Normalized->getTH2F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Normalized->getTH2F()->GetYaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Normalized_Any->getTH1F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Any->getTH1F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());

    }
    
}

void FourVectorHLTOnline::fillHltMatrix(const edm::TriggerNames & triggerNames) {

 std::string fullPathToME; 

 //for (unsigned int mi=0;mi<fGroupName.size();mi++) {
  for (unsigned int mi=0;mi<fGroupNamePathsPair.size();mi++) {


  fullPathToME = "HLT/FourVector/PathsSummary/HLT_"+fGroupNamePathsPair[mi].first+"_PassPass";
  MonitorElement* ME_2d = dbe_->get(fullPathToME);
  fullPathToME = "HLT/FourVector/PathsSummary/HLT_"+fGroupNamePathsPair[mi].first+"_Pass_Any";
  MonitorElement* ME_1d = dbe_->get(fullPathToME);
  if(!ME_2d || !ME_1d) {  

    LogTrace("FourVectorHLTOnline") << " ME not valid although I gave full path" << std::endl;
    continue;

  }

  TH2F * hist_2d = ME_2d->getTH2F();
  TH1F * hist_1d = ME_1d->getTH1F();

  // Fill HLTPassed Matrix bin (i,j) = (Any,Any)
  // --------------------------------------------------------
  int anyBinNumber = hist_2d->GetXaxis()->FindBin("HLT_Any");      
  // any triger accepted
  if(triggerResults_->accept()){

    hist_2d->Fill(anyBinNumber-1,anyBinNumber-1);//binNumber1 = 0 = first filter
    hist_1d->Fill(anyBinNumber-1);//binNumber1 = 0 = first filter

  }

  bool groupPassed = false;
  bool groupL1Passed = false;

  // Main loop over paths
  // --------------------

  for (int i=1; i< hist_2d->GetNbinsX();i++) { 

  std::string hltpathname =  hist_2d->GetXaxis()->GetBinLabel(i);


    // check if this is hlt path name
    unsigned int pathByIndex = triggerNames.triggerIndex(hltpathname);
    if(pathByIndex >= triggerResults_->size() ) continue;

    // check if its L1 passed
    if(hasL1Passed(hltpathname,triggerNames)) groupL1Passed = true;

    // Fill HLTPassed Matrix and HLTPassFail Matrix
    // --------------------------------------------------------

    if(triggerResults_->accept(pathByIndex)){

      groupPassed = true;
      //groupL1Passed = true;

      if(groupPassed && !groupL1Passed) 
  
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

  std::string groupBinLabel = "HLT_"+fGroupNamePathsPair[mi].first+"_Any";
  int groupBinNumber = hist_2d->GetXaxis()->FindBin(groupBinLabel.c_str());      
  if(groupPassed) hist_1d->Fill(groupBinNumber-1);//binNumber1 = 0 = first filter

  std::string groupL1BinLabel = "HLT_"+fGroupNamePathsPair[mi].first+"_L1_Any";
  int groupL1BinNumber = hist_2d->GetXaxis()->FindBin(groupL1BinLabel.c_str());      

  if(groupL1Passed) hist_1d->Fill(groupL1BinNumber-1);//binNumber1 = 0 = first filter
 } // end for mi

}

void FourVectorHLTOnline::setupHltBxPlots()
{

  //pathsSummaryFolder_ = TString("HLT/FourVector/PathsSummary/");
  //dbe_->setCurrentFolder(pathsSummaryFolder_.c_str());
  dbe_->setCurrentFolder(pathsSummaryFolder_);

  // setup HLT bx plot
  unsigned int npaths = hltPathsDiagonal_.size();

  ME_HLT_BX_ = dbe_->book2D("HLT_bx",
                         "HLT counts vs Event bx",
                         Nbx_+1, -0.5, Nbx_+1-0.5, npaths, -0.5, npaths-0.5);
  ME_HLT_CUSTOM_BX_ = dbe_->book2D("HLT_Custom_bx",
                         "HLT counts vs Event bx",
                         Nbx_+1, -0.5, Nbx_+1-0.5, npaths, -0.5, npaths-0.5);
  ME_HLT_BX_->setAxisTitle("Bunch Crossing");
  ME_HLT_CUSTOM_BX_->setAxisTitle("Bunch Crossing");


  // Set up bin labels on Y axis continuing to cover all npaths
  for(unsigned int i = 0; i < npaths; i++){

    ME_HLT_BX_->getTH2F()->GetYaxis()->SetBinLabel(i+1, (hltPathsDiagonal_[i]).getPath().c_str());
    ME_HLT_CUSTOM_BX_->getTH2F()->GetYaxis()->SetBinLabel(i+1, (hltPathsDiagonal_[i]).getPath().c_str());

  }


}

void FourVectorHLTOnline::setupHltLsPlots()
{
 
  unsigned int npaths = hltPathsDiagonal_.size();

  //pathsSummaryHLTPathsPerLSFolder_ = TString("HLT/FourVector/PathsSummary/HLT LS/");
  //dbe_->setCurrentFolder(pathsSummaryHLTPathsPerLSFolder_.c_str());
  dbe_->setCurrentFolder(pathsSummaryHLTPathsPerLSFolder_);

  ME_HLTAll_LS_  = dbe_->book2D("All_count_LS",
                    "All paths per LS ",
                         nLS_, 0, nLS_, npaths+1, -0.5, npaths+1-0.5);
  ME_HLTAll_LS_->setAxisTitle("Luminosity Section");

  // Set up bin labels on Y axis continuing to cover all npaths
  for(unsigned int i = 0; i < npaths; i++){

    ME_HLTAll_LS_->getTH2F()->GetYaxis()->SetBinLabel(i+1, (hltPathsDiagonal_[i]).getPath().c_str());

  }

  unsigned int i = npaths;
  ME_HLTAll_LS_->getTH2F()->GetYaxis()->SetBinLabel(i+1, "HLT_Any");

  int nBinsPerLSHisto = 20;
  int nLSHistos = npaths/nBinsPerLSHisto;
  for (int nh=0;nh<nLSHistos+1;nh++) {

    char name[200];
    char title[200];

    sprintf(name, "Group_%d_paths_count_LS",nLSHistos-nh);
    sprintf(title, "Group %d,  paths count per LS",nLSHistos-nh);

    MonitorElement* tempME  = dbe_->book2D(name,title,
                    nLS_, 0, nLS_, nBinsPerLSHisto+3, -0.5, nBinsPerLSHisto+3-0.5);

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

      tempME->setBinLabel(bin+1, hltPathsDiagonal_[i].getPath().c_str(), 2);

    }

    tempME->setBinLabel(nBinsPerLSHisto+3, "HLT_Any", 2);
    tempME->setBinLabel(nBinsPerLSHisto+2, "HLT_PhysicsDeclared", 2);

    v_ME_HLTAll_LS_.push_back(tempME);

  }


}


void FourVectorHLTOnline::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){   

   //int lumi = int(lumiSeg.id().luminosityBlock());
   //if(lumi < 74 || lumi > 77) fLumiFlag = false;
   //else fLumiFlag = true;

}

void FourVectorHLTOnline::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c)
{

   int lumi = int(lumiSeg.id().luminosityBlock());
   LogTrace("FourVectorHLTOnline") << " end lumiSection number " << lumi << std::endl;

   //countHLTPathHitsEndLumiBlock(lumi);
   //countHLTGroupHitsEndLumiBlock(lumi);
   //countHLTGroupL1HitsEndLumiBlock(lumi);

   //countHLTGroupBXHitsEndLumiBlock(lumi);

}

void FourVectorHLTOnline::countHLTGroupBXHitsEndLumiBlock(const int& lumi)
{

 LogTrace("FourVectorHLTOnline") << " countHLTGroupBXHitsEndLumiBlock() lumiSection number " << lumi << std::endl;

   TH2F * hist_2d_bx = ME_HLT_BX_->getTH2F();

   for (std::vector<std::pair<std::string, std::vector<int> > >::iterator ip = fPathBxTempCountPair.begin(); ip != fPathBxTempCountPair.end(); ++ip) {
  
    // get the path and its previous count
    std::string pathname = ip->first;  
    std::vector<int> prevCount = ip->second;  

    // vector of 5 zeros
    std::vector<int> currCount (5,0);
    std::vector<int> diffCount (5,0);
    
    // get the current count of path up to now
    int pathBin = hist_2d_bx->GetYaxis()->FindBin(pathname.c_str());      

    if(pathBin > hist_2d_bx->GetNbinsY()) {
      
      LogTrace("FourVectorHLTOnline") << " Cannot find the bin for path " << pathname << std::endl;
      continue;

    }

    for (unsigned int j =0;j<currCount.size();j++) { 

      int bxOffset = j-2;
      int bunch = referenceBX_+bxOffset;
      if(bunch < 1) bunch += Nbx_ ;
      int bxBin = bunch +1; // add one to get the right bin

      
      currCount[j] = int(hist_2d_bx->GetBinContent(bxBin, pathBin));  // add one to get the right bin

      LogTrace("FourVectorHLTOnline") << "currCount = " << currCount[j] << std::endl;

      // count due to prev lumi sec is a difference bw current and previous
      diffCount[j] = currCount[j] - prevCount[j];

      LogTrace("FourVectorHLTOnline") << " lumi = " << lumi << "  path " << pathname << "bxOffset = " << bxOffset << "  count = " << diffCount[j] <<  std::endl;

      // set the counter in the pair to current count
      ip->second = currCount;  

    } // end for bx j

   //////////////////////////////////////
   // fill the 2D Group paths'  BX count per LS
   //////////////////////////////////////
   LogTrace("FourVectorHLTOnline")  << "Find " << pathname << std::endl;
   for(unsigned int i=0; i<fGroupName.size(); i++){

    bool isMember = false;
    //check if the path is in this group
    for (unsigned int j=0;j<fGroupNamePathsPair.size();j++) { 

      LogTrace("FourVectorHLTOnline")  << " ---- Group " << fGroupNamePathsPair[j].first << std::endl;
      if(fGroupNamePathsPair[j].first != fGroupName[i]) continue;

      for (unsigned int k = 0; k<(fGroupNamePathsPair[j].second).size();k++) {

        LogTrace("FourVectorHLTOnline")  << " comparing to " <<  fGroupNamePathsPair[j].second[k] << std::endl; 

        if(fGroupNamePathsPair[j].second[k] == pathname) {

          isMember = true;
          break;

        }

      } // end for k
    } // end for j


    if(!isMember) {
      
      LogTrace("FourVectorHLTOnline")  << "Could not find a group to which the path belongs, path = " << pathname << "    group = " << fGroupName[i] << std::endl;
      continue;

    }

    std::string fullPathToME = pathsSummaryHLTPathsPerBXFolder_ + "HLT_"+fGroupName[i] + "_BX_LS";
    MonitorElement* ME_2d = dbe_->get(fullPathToME);

    if ( ME_2d) {

      TH2F* hist_All = ME_2d->getTH2F();

      for (unsigned int j = 0; j<diffCount.size();j++) {

        // find the bin
        int binNumber = j+1; // add one to get right bin

        // update  the bin content  (must do that since events don't ncessarily come in the order
        int currentLumiCount = int(hist_All->GetBinContent(lumi+1,binNumber));
        int updatedLumiCount = currentLumiCount + diffCount[j];
        hist_All->SetBinContent(lumi+1,binNumber,updatedLumiCount);
      } // end for bx j
    
    }
    else {

      LogDebug("FourVectorHLTOnline") << " cannot find ME_2d" << ME_2d->getName() <<  std::endl;

    }
    

   } // end for group i

  } // end for ip

}

void FourVectorHLTOnline::countHLTGroupL1HitsEndLumiBlock(const int& lumi)
{

 LogTrace("FourVectorHLTOnline") << " countHLTGroupL1HitsEndLumiBlock() lumiSection number " << lumi << std::endl;

 //for(unsigned int i=0; i<fGroupName.size(); i++){
 for(unsigned int i=0; i<fGroupNamePathsPair.size(); i++){

   // get the count of path up to now
   std::string fullPathToME = "HLT/FourVector/PathsSummary/HLT_" + fGroupNamePathsPair[i].first+ "_Pass_Any";
   MonitorElement* ME_1d = dbe_->get(fullPathToME);

   if(! ME_1d) {

     LogTrace("FourVectorHLTOnline") << " could not find 1d matrix " << fullPathToME << std::endl;

     continue;

   }

   LogTrace("FourVectorHLTOnline") << " Looking in histogram "  << fullPathToME << std::endl;

   TH1F * hist_1d = ME_1d->getTH1F();

   for (std::vector<std::pair<std::string, int> >::iterator ip = fGroupL1TempCountPair.begin(); ip != fGroupL1TempCountPair.end(); ++ip) {
  
    // get the path and its previous count
    std::string pathname = ip->first;  
    int prevCount = ip->second;  

    std::string binLabel = "HLT_"+pathname+"_L1_Any";
    
    LogTrace("FourVectorHLTOnline") << " Looking for binLabel = " << binLabel <<  std::endl;
    // get the current count of path up to now
    int pathBin = hist_1d->GetXaxis()->FindBin(binLabel.c_str());      

    LogTrace("FourVectorHLTOnline") << " pathBin = " << pathBin <<  "  out of histogram total number of bins " << hist_1d->GetNbinsX() <<  std::endl;
    if(pathBin == -1) {
      
      LogTrace("FourVectorHLTOnline") << " Cannot find the bin for path " << pathname << std::endl;
      continue;

    }

    int currCount = int(hist_1d->GetBinContent(pathBin));

    // count due to prev lumi sec is a difference bw current and previous
    int diffCount = currCount - prevCount;

    LogTrace("FourVectorHLTOnline") << " lumi = " << lumi << "  path " << pathname << "  count " << diffCount <<  std::endl;

    // set the counter in the pair to current count
    ip->second = currCount;  


    ///////////////////////////////////////////
    // fill the 1D individual path count per LS
    ///////////////////////////////////////////
    std::string fullPathToME_count = pathsSummaryHLTPathsPerLSFolder_ +"HLT_" + pathname + "_L1_Total_LS";
    MonitorElement* ME_1d = dbe_->get(fullPathToME_count);
    if ( ME_1d) { 

      // update  the bin content  (must do that since events don't ncessarily come in the order
      int currentLumiCount = int(ME_1d->getTH1()->GetBinContent(lumi+1));
      int updatedLumiCount = currentLumiCount + diffCount;
      ME_1d->getTH1()->SetBinContent(lumi+1,updatedLumiCount);

    }
    else {

      LogDebug("FourVectorHLTOnline") << " cannot find ME " << fullPathToME_count  <<  std::endl;

    }

   } // end for ip

 } // end for i

}


void FourVectorHLTOnline::countHLTGroupHitsEndLumiBlock(const int& lumi)
{

 LogTrace("FourVectorHLTOnline") << " countHLTGroupHitsEndLumiBlock() lumiSection number " << lumi << std::endl;
 //for(unsigned int i=0; i<fGroupName.size(); i++){
 for(unsigned int i=0; i<fGroupNamePathsPair.size(); i++){

    // get the count of path up to now
   //string fullPathToME = "HLT/FourVector/PathsSummary/HLT_" + fGroupName[i] + "_Pass_Any";
   std::string fullPathToME = "HLT/FourVector/PathsSummary/HLT_" + fGroupNamePathsPair[i].first + "_Pass_Any";
   MonitorElement* ME_1d = dbe_->get(fullPathToME);

   if(! ME_1d) {

     LogTrace("FourVectorHLTOnline") << " could not find 1d matrix " << fullPathToME << std::endl;

     continue;

   }

   LogTrace("FourVectorHLTOnline") << " Looking in histogram "  << fullPathToME << std::endl;

   TH1F * hist_1d = ME_1d->getTH1F();

   for (std::vector<std::pair<std::string, int> >::iterator ip = fGroupTempCountPair.begin(); ip != fGroupTempCountPair.end(); ++ip) {
  
    // get the path and its previous count
    std::string pathname = ip->first;  
    int prevCount = ip->second;  

    std::string binLabel = "HLT_"+pathname+"_Any";
    
    LogTrace("FourVectorHLTOnline") << " Looking for binLabel = " << binLabel <<  std::endl;
    // get the current count of path up to now
    int pathBin = hist_1d->GetXaxis()->FindBin(binLabel.c_str());      

    LogTrace("FourVectorHLTOnline") << " pathBin = " << pathBin <<  "  out of histogram total number of bins " << hist_1d->GetNbinsX() <<  std::endl;
    if(pathBin == -1) {
      
      LogTrace("FourVectorHLTOnline") << " Cannot find the bin for path " << pathname << std::endl;
      continue;

    }

    int currCount = int(hist_1d->GetBinContent(pathBin));

    // count due to prev lumi sec is a difference bw current and previous
    int diffCount = currCount - prevCount;

    LogTrace("FourVectorHLTOnline") << " lumi = " << lumi << "  path " << pathname << "  count " << diffCount <<  std::endl;

    // set the counter in the pair to current count
    ip->second = currCount;  


    ///////////////////////////////////////////
    // fill the 1D individual path count per LS
    ///////////////////////////////////////////
    std::string fullPathToME_count = pathsSummaryHLTPathsPerLSFolder_ +"HLT_" + pathname + "_Total_LS";
    MonitorElement* ME_1d = dbe_->get(fullPathToME_count);
    if ( ME_1d) { 

      // update  the bin content  (must do that since events don't ncessarily come in the order
      int currentLumiCount = int(ME_1d->getTH1()->GetBinContent(lumi+1));
      int updatedLumiCount = currentLumiCount + diffCount;
      ME_1d->getTH1()->SetBinContent(lumi+1,updatedLumiCount);

    }
    else {

      LogDebug("FourVectorHLTOnline") << " cannot find ME " << fullPathToME_count  <<  std::endl;

    }

   } // end for ip

 } // end for i

}


void FourVectorHLTOnline::countHLTPathHitsEndLumiBlock(const int& lumi)
{

   LogTrace("FourVectorHLTOnline") << " countHLTPathHitsEndLumiBlock() lumiSection number " << lumi << std::endl;
    // get the count of path up to now
   std::string fullPathToME = "HLT/FourVector/PathsSummary/HLT_All_PassPass";
   MonitorElement* ME_2d = dbe_->get(fullPathToME);

   if(! ME_2d) {

     LogTrace("FourVectorHLTOnline") << " could not fine 2d matrix " << fullPathToME << std::endl;

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
      
      LogTrace("FourVectorHLTOnline") << " Cannot find the bin for path " << pathname << std::endl;
      continue;

    }

    int currCount = int(hist_2d->GetBinContent(pathBin, pathBin));

    // count due to prev lumi sec is a difference bw current and previous
    int diffCount = currCount - prevCount;

    LogTrace("FourVectorHLTOnline") << " lumi = " << lumi << "  path " << pathname << "  count " << diffCount <<  std::endl;

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

      LogDebug("FourVectorHLTOnline") << " cannot find ME_HLTAll_LS_" <<  std::endl;

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
  
        LogDebug("FourVectorHLTOnline") << " cannot find tempME " <<  std::endl;
  
      }

    }


    ///////////////////////////////////////////
    // fill the 1D individual path count per LS
    ///////////////////////////////////////////
    std::string fullPathToME_count = pathsIndividualHLTPathsPerLSFolder_ + pathname + "_count_per_LS";
    MonitorElement* ME_1d = dbe_->get(fullPathToME_count);
    if ( ME_1d) { 

      // update  the bin content  (must do that since events don't ncessarily come in the order
      int currentLumiCount = int(ME_1d->getTH1()->GetBinContent(lumi+1));
      int updatedLumiCount = currentLumiCount + diffCount;
      ME_1d->getTH1()->SetBinContent(lumi+1,updatedLumiCount);

    }
    else {

      LogDebug("FourVectorHLTOnline") << " cannot find ME " << fullPathToME_count  <<  std::endl;

    }

  } // end for ip

}

int FourVectorHLTOnline::getTriggerTypeParsePathName(const std::string& pathname)
{

   int objectType = 0;

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

   return objectType;
}

const std::string FourVectorHLTOnline::getL1ConditionModuleName(const std::string& pathname)
{

  // find L1 condition for numpath with numpath objecttype 
  // find PSet for L1 global seed for numpath, 
  // list module labels for numpath
  std::string l1pathname = "dummy";

  std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);

  for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {

    if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed") {

     l1pathname = *numpathmodule;
     break; 

    }

  } // end for

  return l1pathname;

}


bool FourVectorHLTOnline::hasL1Passed(const std::string& pathname, const edm::TriggerNames & triggerNames)
{
  
  bool rc = false;

  std::string l1ModuleName = getL1ConditionModuleName(pathname);
  int l1ModuleIndex = hltConfig_.moduleIndex(pathname, l1ModuleName);

  unsigned int pathByIndex = triggerNames.triggerIndex(pathname);
  if(pathByIndex >= triggerResults_->size() ) return rc; // path is not in the menu

  // get index of the last module that issued the decision
  int lastModule = triggerResults_->index(pathByIndex);

  // if L1 passed, then it must not be the module that 
  // issued the last decision
  rc = (l1ModuleIndex < lastModule);

  return rc;

}

bool FourVectorHLTOnline::hasHLTPassed(const std::string& pathname, const edm::TriggerNames & triggerNames)
{
  
  bool rc = false;

  unsigned int pathByIndex = triggerNames.triggerIndex(pathname);
  if(pathByIndex >= triggerResults_->size() ) return rc; // path is not in the menu

  rc  = triggerResults_->accept(pathByIndex);

  return rc;

}

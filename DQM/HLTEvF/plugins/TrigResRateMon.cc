// $Id: TrigResRateMon.cc,v 1.7 2010/07/23 19:04:56 rekovic Exp $
// See header file for information. 
#include "TMath.h"
#include "DQM/HLTEvF/interface/TrigResRateMon.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include <map>
#include <utility>


using namespace edm;
using namespace trigger;
using namespace std;

TrigResRateMon::TrigResRateMon(const edm::ParameterSet& iConfig): currentRun_(-99)
{

  LogDebug("TrigResRateMon") << "constructor...." ;

  fIsSetup = false;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("TrigResRateMon") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  dirname_ = iConfig.getUntrackedParameter("dirname", std::string("HLT/TrigResults/"));
  //dirname_ +=  iConfig.getParameter<std::string>("@module_label");
  
  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirname_);
  }

  doCombineRuns_ = iConfig.getUntrackedParameter<bool>("doCombineRuns", false);
  doVBTFMuon_ = iConfig.getUntrackedParameter<bool>("doVBTFMuon", true);
  
  processname_ = iConfig.getParameter<std::string>("processname");
  fCustomBXPath = iConfig.getUntrackedParameter<std::string>("customBXPath", std::string("HLT_MinBiasBSC"));

  referenceBX_ = iConfig.getUntrackedParameter<unsigned int>("referenceBX",51);
  Nbx_ = iConfig.getUntrackedParameter<unsigned int>("Nbx",3564);

  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",20);
  nBins2D_ = iConfig.getUntrackedParameter<unsigned int>("Nbins2D",40);
  dRMax_ = iConfig.getUntrackedParameter<double>("dRMax",1.0);
  dRMaxElectronMuon_ = iConfig.getUntrackedParameter<double>("dRMaxElectronMuon",0.3);
  nBinsDR_ = iConfig.getUntrackedParameter<unsigned int>("NbinsDR",10);
  nBinsOneOverEt_ = iConfig.getUntrackedParameter<unsigned int>("NbinsOneOverEt",10000);
  nLS_   = iConfig.getUntrackedParameter<unsigned int>("NLuminositySegments",10);
  LSsize_   = iConfig.getUntrackedParameter<double>("LuminositySegmentSize",23);
  thresholdFactor_ = iConfig.getUntrackedParameter<double>("thresholdFactor",1.0);

  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);
     // this is the list of paths to look at.
  std::vector<edm::ParameterSet> paths = 
  iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");

  for(std::vector<edm::ParameterSet>::iterator pathconf = paths.begin() ; pathconf != paths.end(); pathconf++) {

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


  /*
  muonRecoCollectionName_ = iConfig.getUntrackedParameter("muonRecoCollectionName", std::string("muons"));

  electronEtaMax_ = iConfig.getUntrackedParameter<double>("electronEtaMax",2.5);
  electronEtMin_ = iConfig.getUntrackedParameter<double>("electronEtMin",3.0);
  electronDRMatch_  =iConfig.getUntrackedParameter<double>("electronDRMatch",0.3); 
  electronL1DRMatch_  =iConfig.getUntrackedParameter<double>("electronL1DRMatch",0.3); 

  muonEtaMax_ = iConfig.getUntrackedParameter<double>("muonEtaMax",2.1);
  muonEtMin_ = iConfig.getUntrackedParameter<double>("muonEtMin",0.0);
  muonDRMatch_  =iConfig.getUntrackedParameter<double>("muonDRMatch",0.3); 
  muonL1DRMatch_  =iConfig.getUntrackedParameter<double>("muonL1DRMatch",0.3); 

  tauEtaMax_ = iConfig.getUntrackedParameter<double>("tauEtaMax",2.5);
  tauEtMin_ = iConfig.getUntrackedParameter<double>("tauEtMin",3.0);
  tauDRMatch_  =iConfig.getUntrackedParameter<double>("tauDRMatch",0.3); 
  tauL1DRMatch_  =iConfig.getUntrackedParameter<double>("tauL1DRMatch",0.5); 

  jetEtaMax_ = iConfig.getUntrackedParameter<double>("jetEtaMax",5.0);
  jetEtMin_ = iConfig.getUntrackedParameter<double>("jetEtMin",10.0);
  jetDRMatch_  =iConfig.getUntrackedParameter<double>("jetDRMatch",0.3); 
  jetL1DRMatch_  =iConfig.getUntrackedParameter<double>("jetL1DRMatch",0.5); 

  bjetEtaMax_ = iConfig.getUntrackedParameter<double>("bjetEtaMax",2.5);
  bjetEtMin_ = iConfig.getUntrackedParameter<double>("bjetEtMin",10.0);
  bjetDRMatch_  =iConfig.getUntrackedParameter<double>("bjetDRMatch",0.3); 
  bjetL1DRMatch_  =iConfig.getUntrackedParameter<double>("bjetL1DRMatch",0.3); 

  photonEtaMax_ = iConfig.getUntrackedParameter<double>("photonEtaMax",2.5);
  photonEtMin_ = iConfig.getUntrackedParameter<double>("photonEtMin",3.0);
  photonDRMatch_  =iConfig.getUntrackedParameter<double>("photonDRMatch",0.3); 
  photonL1DRMatch_  =iConfig.getUntrackedParameter<double>("photonL1DRMatch",0.3); 

  trackEtaMax_ = iConfig.getUntrackedParameter<double>("trackEtaMax",2.5);
  trackEtMin_ = iConfig.getUntrackedParameter<double>("trackEtMin",3.0);
  trackDRMatch_  =iConfig.getUntrackedParameter<double>("trackDRMatch",0.3); 
  trackL1DRMatch_  =iConfig.getUntrackedParameter<double>("trackL1DRMatch",0.3); 

  metEtaMax_ = iConfig.getUntrackedParameter<double>("metEtaMax",5);
  metMin_ = iConfig.getUntrackedParameter<double>("metMin",10.0);
  metDRMatch_  =iConfig.getUntrackedParameter<double>("metDRMatch",0.5); 
  metL1DRMatch_  =iConfig.getUntrackedParameter<double>("metL1DRMatch",0.5); 

  htEtaMax_ = iConfig.getUntrackedParameter<double>("htEtaMax",5);
  htMin_ = iConfig.getUntrackedParameter<double>("htMin",10.0);
  htDRMatch_  =iConfig.getUntrackedParameter<double>("htDRMatch",0.5); 
  htL1DRMatch_  =iConfig.getUntrackedParameter<double>("htL1DRMatch",0.5); 
  */

  sumEtMin_ = iConfig.getUntrackedParameter<double>("sumEtMin",10.0);

      // Muon quality cuts
      dxyCut_ = iConfig.getUntrackedParameter<double>("DxyCut", 0.2);   // dxy < 0.2 cm 
      normalizedChi2Cut_ = iConfig.getUntrackedParameter<double>("NormalizedChi2Cut", 10.); // chi2/ndof (of global fit) <10.0
      trackerHitsCut_ = iConfig.getUntrackedParameter<int>("TrackerHitsCut", 11);  // Tracker Hits >10 
      pixelHitsCut_ = iConfig.getUntrackedParameter<int>("PixelHitsCut", 1); // Pixel Hits >0
      muonHitsCut_ = iConfig.getUntrackedParameter<int>("MuonHitsCut", 1);  // Valid Muon Hits >0 
      isAlsoTrackerMuon_ = iConfig.getUntrackedParameter<bool>("IsAlsoTrackerMuon", true);
      nMatchesCut_ = iConfig.getUntrackedParameter<int>("NMatchesCut", 2); // At least 2 Chambers with matches 

  specialPaths_ = iConfig.getParameter<std::vector<std::string > >("SpecialPaths");

  pathsSummaryFolder_ = iConfig.getUntrackedParameter ("pathsSummaryFolder",std::string("HLT/TrigResults/PathsSummary/"));
  pathsSummaryHLTCorrelationsFolder_ = iConfig.getUntrackedParameter ("hltCorrelationsFolder",std::string("HLT/TrigResults/PathsSummary/HLT Correlations/"));
  pathsSummaryFilterCountsFolder_ = iConfig.getUntrackedParameter ("filterCountsFolder",std::string("HLT/TrigResults/PathsSummary/Filters Counts/"));

  pathsSummaryHLTPathsPerLSFolder_ = iConfig.getUntrackedParameter ("individualPathsPerLSFolder",std::string("HLT/TrigResults/PathsSummary/HLT LS/"));
  pathsIndividualHLTPathsPerLSFolder_ = iConfig.getUntrackedParameter ("individualPathsPerLSFolder",std::string("HLT/TrigResults/PathsSummary/HLT LS/Paths/"));
  pathsSummaryHLTPathsPerBXFolder_ = iConfig.getUntrackedParameter ("individualPathsPerBXFolder",std::string("HLT/TrigResults/PathsSummary/HLT BX/"));

  fLumiFlag = true;
  ME_HLTAll_LS = NULL;
  ME_HLT_BX = NULL;
  ME_HLT_CUSTOM_BX = NULL;

  //jetID = new reco::helper::JetIDHelper(iConfig.getParameter<ParameterSet>("JetIDParams"));

    recHitsEBTag_ = iConfig.getUntrackedParameter<edm::InputTag>("RecHitsEBTag",edm::InputTag("reducedEcalRecHitsEB"));
      recHitsEETag_ = iConfig.getUntrackedParameter<edm::InputTag>("RecHitsEETag",edm::InputTag("reducedEcalRecHitsEE"));

  
}


TrigResRateMon::~TrigResRateMon()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TrigResRateMon::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //if(! fLumiFlag ) return;

  using namespace edm;
  using namespace trigger;
  ++nev_;
  LogDebug("TrigResRateMon")<< " analyze...." ;


  

  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(triggerResultsLabel_,triggerResults);
  if(!triggerResults.isValid()) {
    edm::InputTag triggerResultsLabelFU(triggerResultsLabel_.label(),triggerResultsLabel_.instance(), "FU");
   iEvent.getByLabel(triggerResultsLabelFU,triggerResults);
  if(!triggerResults.isValid()) {
    edm::LogInfo("TrigResRateMon") << "TriggerResults not found, "
      "skipping event"; 
    return;
   }
  }
  triggerResults_ = triggerResults;
  const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);
  //int npath = triggerResults->size();


  // int bx = iEvent.bunchCrossing();
  /*
  // Fill HLTPassed_Correlation Matrix bin (i,j) = (Any,Any)
  // --------------------------------------------------------
  int anyBinNumber = ME_HLTPassPass_->getTH2F()->GetXaxis()->FindBin("HLT_Any");      
  // any triger accepted
  if(triggerResults->accept()){

    ME_HLTPassPass_->Fill(anyBinNumber-1,anyBinNumber-1);//binNumber1 = 0 = first filter

  }
  */

  fillHltMatrix(triggerNames);


  // Loop over paths
  // --------------------
  for(PathInfoCollection::iterator v = hltPathsDiagonal_.begin(); v!= hltPathsDiagonal_.end(); ++v ) { 

    /*
    LogTrace("TrigResRateMon") << " unique path " << v->getPath() << endl;

    unsigned int pathByIndex = triggerNames.triggerIndex(v->getPath());

    // path must be in the menu
    if(pathByIndex >= triggerResults_->size() ) continue;

  
    // Fill HLTPassed Matrix and HLTPassFail Matrix
    // --------------------------------------------------------

    if(triggerResults->accept(pathByIndex)){
  
      int pathBinNumber = ME_HLT_BX->getTH2F()->GetYaxis()->FindBin(v->getPath().c_str());      
      ME_HLT_BX->Fill(bx,pathBinNumber-1);

      //if(hasHLTPassed(fCustomBXPath,triggerNames)) {

        ME_HLT_CUSTOM_BX->Fill(bx,pathBinNumber-1);

      //}

    } // end if accept

    */
  
    /*
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
    */

  } // end for diag paths

}



// -- method called once each job just before starting event loop  --------
void 
TrigResRateMon::beginJob()
{
  nev_ = 0;
  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
  }
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
  }  

  MonitorElement* reportSummaryFloat = dbe->bookFloat("reportSummaryMap");
  if(reportSummaryFloat) reportSummaryFloat->Fill(1);


}

// - method called once each job just after ending the event loop  ------------
void 
TrigResRateMon::endJob() 
{
   LogInfo("TrigResRateMon") << "analyzed " << nev_ << " events";
   return;
}


// BeginRun
void TrigResRateMon::beginRun(const edm::Run& run, const edm::EventSetup& c)
{

  LogDebug("TrigResRateMon") << "beginRun, run " << run.id();

  if(fIsSetup) return;
  
  // HLT config does not change within runs!
  bool changed=false;
 
  if (!hltConfig_.init(run, c, processname_, changed)) {

    processname_ = "FU";

    if (!hltConfig_.init(run, c, processname_, changed)){

      LogDebug("TrigResRateMon") << "HLTConfigProvider failed to initialize.";

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

        string l1pathname = getL1ConditionModuleName(pathname);

        int l1ModuleIndex = hltConfig_.moduleIndex(pathname, l1pathname);
      
        int objectType =  getTriggerTypeParsePathName(pathname);
  
        for (unsigned int i=0; i!=n; ++i) {
  
          std::string denompathname = hltConfig_.triggerName(i);  
          int denomobjectType =  getTriggerTypeParsePathName(denompathname);

      
         
          std::string filtername("dummy");
          float ptMin = 0.0;
          float ptMax = 100.0;
          if (plotAll_ && denomobjectType == objectType && objectType != 0) {
          
            int hltThreshold = getThresholdFromName(pathname);
            int l1Threshold = getThresholdFromName(l1pathname);
            hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, l1ModuleIndex, filtername, processname_, objectType, ptMin, ptMax, hltThreshold, l1Threshold));

          }

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

        string l1pathname = getL1ConditionModuleName(pathname);
        int l1ModuleIndex = hltConfig_.moduleIndex(pathname, l1pathname);
  
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

          int hltThreshold = getThresholdFromName(pathname);
          int l1Threshold = getThresholdFromName(l1pathname);
  
          hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, l1ModuleIndex, filtername, processname_, objectType, ptMin, ptMax, hltThreshold, l1Threshold));

          hltPathsDiagonal_.push_back(PathInfo(denompathname, pathname, l1pathname, l1ModuleIndex, filtername, processname_, objectType, ptMin, ptMax, hltThreshold, l1Threshold));
  
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
              
              LogDebug("TrigResRateMon") << "denompathname is selected to be = " << n_pathname << endl;;
              founddenominator = true;

              break;

            }
          }

          if (!founddenominator) {
  
            edm::LogInfo("TrigResRateMon") << "denompathname not found, go to the next pair numearator-denominator" << endl;
            
            // go to the next pair
            continue;
  
          }

          // check that numerator exists
          bool foundnumerator = false;
          for (unsigned int j=0; j!=n; ++j) {

            string pathname = hltConfig_.triggerName(j);

            LogDebug("TrigResRateMon") << "check if path " << pathname << " is numpathname = " << numpathname << endl;
            if (hltConfig_.triggerName(j).find(numpathname)!= std::string::npos) {
              
              LogDebug("TrigResRateMon") << "pathname is selected to be = " << denompathname << endl;;
              foundnumerator = true;

            }
  
  
            if (!foundnumerator) {
    
              edm::LogInfo("TrigResRateMon") << "pathname not found, ignoring " << pathname;
              continue;
  
            }
  
  
            string l1pathname = getL1ConditionModuleName(pathname);
            int l1ModuleIndex = hltConfig_.moduleIndex(pathname, l1pathname);
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
          if (objectType != 0) {
            int hltThreshold = getThresholdFromName(pathname);
            int l1Threshold = getThresholdFromName(l1pathname);
            hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, l1ModuleIndex, filtername, processname_, objectType, ptMin, ptMax, hltThreshold, l1Threshold));

          }
      
        } // end for j, loop over paths

       }  // end if not same num and denominator 
  
      } // end for pair

    } // end else


    vector<string> muonPaths;
    vector<string> egammaPaths;
    vector<string> tauPaths;
    vector<string> jetmetPaths;
    vector<string> restPaths;
    vector<string> allPaths;
    // fill vectors of Muon, Egamma, JetMET, Rest, and Special paths
    for(PathInfoCollection::iterator v = hltPathsDiagonal_.begin(); v!= hltPathsDiagonal_.end(); ++v ) {

      std::string pathName = v->getPath();
      int objectType = v->getObjectType();

      vector<int> tempCount(5,0);

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
    fGroupName.push_back("JetMET");
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

    fGroupNamePathsPair.push_back(make_pair("JetMET",jetmetPaths));

    fGroupNamePathsPair.push_back(make_pair("Rest",restPaths));

    fGroupNamePathsPair.push_back(make_pair("Special",specialPaths_));

    /// add dataset name and thier triggers to the list 
    vector<string> datasetNames =  hltConfig_.datasetNames() ;
    for (unsigned int i=0;i<datasetNames.size();i++) {

      vector<string> datasetPaths = hltConfig_.datasetContent(datasetNames[i]);
      fGroupNamePathsPair.push_back(make_pair(datasetNames[i],datasetPaths));

    }


    for (unsigned int g=0;g<fGroupNamePathsPair.size();g++) {

      fGroupTempCountPair.push_back(make_pair(fGroupNamePathsPair[g].first,0));
      fGroupL1TempCountPair.push_back(make_pair(fGroupNamePathsPair[g].first,0));
      setupHltMatrix(fGroupNamePathsPair[g].first,fGroupNamePathsPair[g].second);

    }

    setupHltLsPlots();

    setupHltBxPlots();


    for(PathInfoCollection::iterator v = hltPathsDiagonal_.begin(); v!= hltPathsDiagonal_.end(); ++v ) {

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

         LogTrace ("TrigResRateMon") << "Module "      << numModule
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

      /*
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
      */
      

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

      TString pathfolder = dirname_ + TString("/FourVector/") + v->getPath();
      /*
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
       
       
       histoname = labelname+"_onEtaonPhiOn";
       title = labelname+" on#eta vs on#phi online";
       onEtavsonPhiOn =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D_,-histEtaMax,histEtaMax, nBins2D_,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_offEtaoffPhiOff";
       title = labelname+" off#eta vs off#phi offline";
       offEtavsoffPhiOff =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D_,-histEtaMax,histEtaMax, nBins2D_,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_l1Etal1PhiL1";
       title = labelname+" l1#eta vs l1#phi L1";
       l1Etavsl1PhiL1 =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D_,-histEtaMax,histEtaMax, nBins2D_,-TMath::Pi(), TMath::Pi());
       
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
       l1Etavsl1PhiL1On =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D_,-histEtaMax,histEtaMax, nBins2D_,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_offEtaoffPhiL1Off";
       title = labelname+" off#eta vs off#phi L1+offline";
       offEtavsoffPhiL1Off =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D_,-histEtaMax,histEtaMax, nBins2D_,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_offEtaoffPhiOnOff";
       title = labelname+" off#eta vs off#phi online+offline";
       offEtavsoffPhiOnOff =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D_,-histEtaMax,histEtaMax, nBins2D_,-TMath::Pi(), TMath::Pi());
       
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
       l1Etavsl1PhiL1OnUM =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D_,-histEtaMax,histEtaMax, nBins2D_,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_offEtaoffPhiL1OffUM";
       title = labelname+" off#eta vs off#phi L1+offlineUM";
       offEtavsoffPhiL1OffUM =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D_,-histEtaMax,histEtaMax, nBins2D_,-TMath::Pi(), TMath::Pi());
       
       histoname = labelname+"_offEtaoffPhiOnOffUM";
       title = labelname+" off#eta vs off#phi online+offlineUM";
       offEtavsoffPhiOnOffUM =  dbe->book2D(histoname.c_str(), title.c_str(), nBins2D_,-histEtaMax,histEtaMax, nBins2D_,-TMath::Pi(), TMath::Pi());
       
       
       
       
       histoname = labelname+"_l1DRL1On";
       title = labelname+" l1DR L1+online";
       l1DRL1On =  dbe->book1D(histoname.c_str(), title.c_str(),nBinsDR_, 0, dRMax_); 
       
       histoname = labelname+"_offDRL1Off";
       title = labelname+" offDR L1+offline";
       offDRL1Off =  dbe->book1D(histoname.c_str(), title.c_str(),nBinsDR_, 0, dRMax_);
       
       histoname = labelname+"_offDROnOff";
       title = labelname+" offDR online+offline";
       offDROnOff =  dbe->book1D(histoname.c_str(), title.c_str(),nBinsDR_, 0, dRMax_); 
       */


       //v->setHistos( NOn, onEtOn, onOneOverEtOn, onEtavsonPhiOn, NOff, offEtOff, offEtavsoffPhiOff, NL1, l1EtL1, l1Etavsl1PhiL1, NL1On, l1EtL1On, l1Etavsl1PhiL1On, NL1Off, offEtL1Off, offEtavsoffPhiL1Off, NOnOff, offEtOnOff, offEtavsoffPhiOnOff, NL1OnUM, l1EtL1OnUM, l1Etavsl1PhiL1OnUM, NL1OffUM, offEtL1OffUM, offEtavsoffPhiL1OffUM, NOnOffUM, offEtOnOffUM, offEtavsoffPhiOnOffUM, offDRL1Off, offDROnOff, l1DRL1On );


    }  // end for hltPath

    // HLT_Any
    // book Count vs LS
    dbe_->setCurrentFolder(pathsIndividualHLTPathsPerLSFolder_.c_str());
    MonitorElement* tempME = dbe_->book1D("HLT_Any_count_per_LS", 
                           "HLT_Any count per LS",
                           nLS_, 0,nLS_);
    tempME->setAxisTitle("Luminosity Section");

  } // end if(1) dummy

 if(doCombineRuns_) fIsSetup = true;

 return;

}

/// EndRun
void TrigResRateMon::endRun(const edm::Run& run, const edm::EventSetup& c)
{

  LogDebug("TrigResRateMon") << "endRun, run " << run.id();

}


void TrigResRateMon::setupHltMatrix(const std::string& label, vector<std::string>& paths) {

    //string groupLabelAny = "HLT_"+label+"_Any";
    //paths.push_back(groupLabelAny.c_str());
    paths.push_back("HLT_"+label+"_L1_Any");
    paths.push_back("HLT_"+label+"_Any");
    paths.push_back("HLT_Any");

    string h_name; 
    string h_title; 

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
    v_ME_HLTAll_LS.push_back(ME_Group_LS);

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
    v_ME_Total_BX.push_back(ME_Total_BX);

    h_name= "HLT_"+label+"_BX_LS_Norm";
    h_title = label+" HLT paths total count combined per BX Normalized to LS";
    MonitorElement* ME_Total_BX_Norm = dbe_->book2D(h_name.c_str(), h_title.c_str(),  nLS_, 0, nLS_, 5, -2.5, 2.5);
    ME_Total_BX_Norm->setAxisTitle("LS",1);
    v_ME_Total_BX_Norm.push_back(ME_Total_BX_Norm);

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

void TrigResRateMon::fillHltMatrix(const edm::TriggerNames & triggerNames) {

 string fullPathToME; 

 //for (unsigned int mi=0;mi<fGroupName.size();mi++) {
  for (unsigned int mi=0;mi<fGroupNamePathsPair.size();mi++) {


  fullPathToME = "HLT/TrigResults/PathsSummary/HLT_"+fGroupNamePathsPair[mi].first+"_PassPass";
  MonitorElement* ME_2d = dbe_->get(fullPathToME);
  fullPathToME = "HLT/TrigResults/PathsSummary/HLT_"+fGroupNamePathsPair[mi].first+"_Pass_Any";
  MonitorElement* ME_1d = dbe_->get(fullPathToME);
  if(!ME_2d || !ME_1d) {  

    LogTrace("TrigResRateMon") << " ME not valid although I gave full path" << endl;
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

  string hltpathname =  hist_2d->GetXaxis()->GetBinLabel(i);


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

  string groupBinLabel = "HLT_"+fGroupNamePathsPair[mi].first+"_Any";
  int groupBinNumber = hist_2d->GetXaxis()->FindBin(groupBinLabel.c_str());      
  if(groupPassed) hist_1d->Fill(groupBinNumber-1);//binNumber1 = 0 = first filter

  string groupL1BinLabel = "HLT_"+fGroupNamePathsPair[mi].first+"_L1_Any";
  int groupL1BinNumber = hist_2d->GetXaxis()->FindBin(groupL1BinLabel.c_str());      

  if(groupL1Passed) hist_1d->Fill(groupL1BinNumber-1);//binNumber1 = 0 = first filter
 } // end for mi

}

void TrigResRateMon::setupHltBxPlots()
{

  //pathsSummaryFolder_ = TString("HLT/TrigResults/PathsSummary/");
  //dbe_->setCurrentFolder(pathsSummaryFolder_.c_str());
  dbe_->setCurrentFolder(pathsSummaryFolder_);

  // setup HLT bx plot
  unsigned int npaths = hltPathsDiagonal_.size();

  ME_HLT_BX = dbe_->book2D("HLT_bx",
                         "HLT counts vs Event bx",
                         Nbx_+1, -0.5, Nbx_+1-0.5, npaths, -0.5, npaths-0.5);
  ME_HLT_CUSTOM_BX = dbe_->book2D("HLT_Custom_bx",
                         "HLT counts vs Event bx",
                         Nbx_+1, -0.5, Nbx_+1-0.5, npaths, -0.5, npaths-0.5);
  ME_HLT_BX->setAxisTitle("Bunch Crossing");
  ME_HLT_CUSTOM_BX->setAxisTitle("Bunch Crossing");


  // Set up bin labels on Y axis continuing to cover all npaths
  for(unsigned int i = 0; i < npaths; i++){

    ME_HLT_BX->getTH2F()->GetYaxis()->SetBinLabel(i+1, (hltPathsDiagonal_[i]).getPath().c_str());
    ME_HLT_CUSTOM_BX->getTH2F()->GetYaxis()->SetBinLabel(i+1, (hltPathsDiagonal_[i]).getPath().c_str());

  }


}

void TrigResRateMon::setupHltLsPlots()
{
 
  unsigned int npaths = hltPathsDiagonal_.size();

  //pathsSummaryHLTPathsPerLSFolder_ = TString("HLT/TrigResults/PathsSummary/HLT LS/");
  //dbe_->setCurrentFolder(pathsSummaryHLTPathsPerLSFolder_.c_str());
  dbe_->setCurrentFolder(pathsSummaryHLTPathsPerLSFolder_);

  ME_HLTAll_LS  = dbe_->book2D("All_count_LS",
                    "All paths per LS ",
                         nLS_, 0, nLS_, npaths+1, -0.5, npaths+1-0.5);
  ME_HLTAll_LS->setAxisTitle("Luminosity Section");

  // Set up bin labels on Y axis continuing to cover all npaths
  for(unsigned int i = 0; i < npaths; i++){

    ME_HLTAll_LS->getTH2F()->GetYaxis()->SetBinLabel(i+1, (hltPathsDiagonal_[i]).getPath().c_str());

  }

  unsigned int i = npaths;
  ME_HLTAll_LS->getTH2F()->GetYaxis()->SetBinLabel(i+1, "HLT_Any");

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

    v_ME_HLTAll_LS.push_back(tempME);

  }


}


void TrigResRateMon::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){   

   //int lumi = int(lumiSeg.id().luminosityBlock());
   //if(lumi < 74 || lumi > 77) fLumiFlag = false;
   //else fLumiFlag = true;

}

void TrigResRateMon::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c)
{

   int lumi = int(lumiSeg.id().luminosityBlock());
   LogTrace("TrigResRateMon") << " end lumiSection number " << lumi << endl;

  countHLTPathHitsEndLumiBlock(lumi);
  countHLTGroupHitsEndLumiBlock(lumi);
  countHLTGroupL1HitsEndLumiBlock(lumi);

  countHLTGroupBXHitsEndLumiBlock(lumi);

}

void TrigResRateMon::countHLTGroupBXHitsEndLumiBlock(const int& lumi)
{

 LogTrace("TrigResRateMon") << " countHLTGroupBXHitsEndLumiBlock() lumiSection number " << lumi << endl;

   TH2F * hist_2d_bx = ME_HLT_BX->getTH2F();

   for (std::vector<std::pair<std::string, vector<int> > >::iterator ip = fPathBxTempCountPair.begin(); ip != fPathBxTempCountPair.end(); ++ip) {
  
    // get the path and its previous count
    std::string pathname = ip->first;  
    vector<int> prevCount = ip->second;  

    // vector of 5 zeros
    vector<int> currCount (5,0);
    vector<int> diffCount (5,0);
    
    // get the current count of path up to now
    int pathBin = hist_2d_bx->GetYaxis()->FindBin(pathname.c_str());      

    if(pathBin > hist_2d_bx->GetNbinsY()) {
      
      LogTrace("TrigResRateMon") << " Cannot find the bin for path " << pathname << endl;
      continue;

    }

    for (unsigned int b =0;b<currCount.size();b++) { 

      int bxOffset = b-2;
      int bunch = referenceBX_+bxOffset;
      if(bunch < 1) bunch += Nbx_ ;
      int bxBin = bunch +1; // add one to get the right bin

      
      currCount[b] = int(hist_2d_bx->GetBinContent(bxBin, pathBin));  // add one to get the right bin

      LogTrace("TrigResRateMon") << "currCount = " << currCount[b] << endl;

      // count due to prev lumi sec is a difference bw current and previous
      diffCount[b] = currCount[b] - prevCount[b];

      LogTrace("TrigResRateMon") << " lumi = " << lumi << "  path " << pathname << "bxOffset = " << bxOffset << "  count = " << diffCount[b] <<  endl;

    } // end for bx b

    // set the counter in the pair to current count
    ip->second = currCount;  

   ////////////////////////////////////////////////////////////
   // fill the 2D Group paths' BX count per LS, using currCount
   ////////////////////////////////////////////////////////////
   LogTrace("TrigResRateMon")  << "Find " << pathname << endl;

   //check if the path is in this group
   for (unsigned int j=0;j<fGroupNamePathsPair.size();j++) { 

      bool isMember = false;

      LogTrace("TrigResRateMon")  << " ---- Group " << fGroupNamePathsPair[j].first << endl;

      // decide if pathname is member of this group
      for (unsigned int k = 0; k<(fGroupNamePathsPair[j].second).size();k++) {

        LogTrace("TrigResRateMon")  << " comparing to " <<  fGroupNamePathsPair[j].second[k] << endl; 

        if(fGroupNamePathsPair[j].second[k] == pathname) {

          isMember = true;
          break;

        }

      } // end for k

      if(!isMember) {
      
      LogTrace("TrigResRateMon")  << "Could not find a group to which the path belongs, path = " << pathname << "    group = " << fGroupNamePathsPair[j].first << endl;
      continue;

      }

      MonitorElement* ME_2d = v_ME_Total_BX[j];

      if (! ME_2d) {

        LogDebug("TrigResRateMon") << " cannot find ME_2d for group " << fGroupNamePathsPair[j].first  <<  endl;
        continue;

      }

      vector<int> updatedLumiCount(5,0);

      float entireBXWindowUpdatedLumiCount = 0;
      
      TH2F* hist_All = ME_2d->getTH2F();

      for (unsigned int b = 0; b<diffCount.size();b++) {

        // find the bin
        int binNumber = b+1; // add one to get right bin

        // update  the bin content  (must do that since events don't ncessarily come in the order
        int currentLumiCount = int(hist_All->GetBinContent(lumi+1,binNumber));
        updatedLumiCount[b] = currentLumiCount + diffCount[b];
        hist_All->SetBinContent(lumi+1,binNumber,updatedLumiCount[b]);

        entireBXWindowUpdatedLumiCount += updatedLumiCount[b];
        
      } // end for bx b

      MonitorElement* ME_2d_Norm = v_ME_Total_BX_Norm[j];

      if (! ME_2d_Norm) {

        LogDebug("TrigResRateMon") << " cannot find ME_2d_Norm for group " << fGroupNamePathsPair[j].first  <<  endl;
        continue;

      }

      TH2F* hist_All_Norm = ME_2d_Norm->getTH2F();

      for (unsigned int b = 0; b<diffCount.size();b++) {

        // find the bin
        int binNumber = b+1; // add one to get right bin

        // update  the bin content  but normalized to the whole columb (BX windw +/- 2)
        hist_All_Norm->SetBinContent(lumi+1,binNumber,float(updatedLumiCount[b])/entireBXWindowUpdatedLumiCount);

      } // end for bx b
    
    } // end for group j
    
  } // end for ip

}

void TrigResRateMon::countHLTGroupL1HitsEndLumiBlock(const int& lumi)
{

 LogTrace("TrigResRateMon") << " countHLTGroupL1HitsEndLumiBlock() lumiSection number " << lumi << endl;

 for(unsigned int i=0; i<fGroupNamePathsPair.size(); i++){

   // get the count of path up to now
   string fullPathToME = "HLT/TrigResults/PathsSummary/HLT_" + fGroupNamePathsPair[i].first+ "_Pass_Any";
   MonitorElement* ME_1d = dbe_->get(fullPathToME);

   if(! ME_1d) {

     LogTrace("TrigResRateMon") << " could not find 1d matrix " << fullPathToME << endl;

     continue;

   }

   LogTrace("TrigResRateMon") << " Looking in histogram "  << fullPathToME << endl;

   TH1F * hist_1d = ME_1d->getTH1F();

   for (std::vector<std::pair<std::string, float> >::iterator ip = fGroupL1TempCountPair.begin(); ip != fGroupL1TempCountPair.end(); ++ip) {
  
    // get the path and its previous count
    string pathname = ip->first;  
    float prevCount = ip->second;  

    string binLabel = "HLT_"+pathname+"_L1_Any";
    
    LogTrace("TrigResRateMon") << " Looking for binLabel = " << binLabel <<  endl;
    // get the current count of path up to now
    int pathBin = hist_1d->GetXaxis()->FindBin(binLabel.c_str());      

    LogTrace("TrigResRateMon") << " pathBin = " << pathBin <<  "  out of histogram total number of bins " << hist_1d->GetNbinsX() <<  endl;
    if(pathBin == -1) {
      
      LogTrace("TrigResRateMon") << " Cannot find the bin for path " << pathname << endl;
      continue;

    }

    float currCount = hist_1d->GetBinContent(pathBin)/LSsize_;

    // count due to prev lumi sec is a difference bw current and previous
    float diffCount = currCount - prevCount;

    LogTrace("TrigResRateMon") << " lumi = " << lumi << "  path " << pathname << "  count " << diffCount <<  endl;

    // set the counter in the pair to current count
    ip->second = currCount;  


    ///////////////////////////////////////////
    // fill the 1D individual path count per LS
    ///////////////////////////////////////////
    string fullPathToME_count = pathsSummaryHLTPathsPerLSFolder_ +"HLT_" + pathname + "_L1_Total_LS";
    MonitorElement* ME_1d = dbe_->get(fullPathToME_count);
    if ( ME_1d) { 

      // update  the bin content  (must do that since events don't ncessarily come in the order
      float currentLumiCount = ME_1d->getTH1()->GetBinContent(lumi+1);
      float updatedLumiCount = currentLumiCount + diffCount;
      ME_1d->getTH1()->SetBinContent(lumi+1,updatedLumiCount);

    }
    else {

      LogDebug("TrigResRateMon") << " cannot find ME " << fullPathToME_count  <<  endl;

    }

   } // end for ip

 } // end for i

}


void TrigResRateMon::countHLTGroupHitsEndLumiBlock(const int& lumi)
{

 LogTrace("TrigResRateMon") << " countHLTGroupHitsEndLumiBlock() lumiSection number " << lumi << endl;
 for(unsigned int i=0; i<fGroupNamePathsPair.size(); i++){

   // get the count of path up to now
   string fullPathToME = "HLT/TrigResults/PathsSummary/HLT_" + fGroupNamePathsPair[i].first + "_Pass_Any";
   MonitorElement* ME_1d = dbe_->get(fullPathToME);

   if(! ME_1d) {

     LogTrace("TrigResRateMon") << " could not find 1d matrix " << fullPathToME << endl;

     continue;

   }

   LogTrace("TrigResRateMon") << " Looking in histogram "  << fullPathToME << endl;

   TH1F * hist_1d = ME_1d->getTH1F();

   for (std::vector<std::pair<std::string, float> >::iterator ip = fGroupTempCountPair.begin(); ip != fGroupTempCountPair.end(); ++ip) {
  
    // get the path and its previous count
    string pathname = ip->first;  
    float prevCount = ip->second;  

    string binLabel = "HLT_"+pathname+"_Any";
    
    LogTrace("TrigResRateMon") << " Looking for binLabel = " << binLabel <<  endl;
    // get the current count of path up to now
    int pathBin = hist_1d->GetXaxis()->FindBin(binLabel.c_str());      

    LogTrace("TrigResRateMon") << " pathBin = " << pathBin <<  "  out of histogram total number of bins " << hist_1d->GetNbinsX() <<  endl;
    if(pathBin == -1) {
      
      LogTrace("TrigResRateMon") << " Cannot find the bin for path " << pathname << endl;
      continue;

    }

    float currCount = hist_1d->GetBinContent(pathBin)/LSsize_;

    // count due to prev lumi sec is a difference bw current and previous
    float diffCount = currCount - prevCount;

    LogTrace("TrigResRateMon") << " lumi = " << lumi << "  path " << pathname << "  count " << diffCount <<  endl;

    // set the counter in the pair to current count
    ip->second = currCount;  


    ///////////////////////////////////////////
    // fill the 1D individual path count per LS
    ///////////////////////////////////////////
    string fullPathToME_count = pathsSummaryHLTPathsPerLSFolder_ +"HLT_" + pathname + "_Total_LS";
    MonitorElement* ME_1d = dbe_->get(fullPathToME_count);
    if ( ME_1d) { 

      // update  the bin content  (must do that since events don't ncessarily come in the order
      float currentLumiCount = ME_1d->getTH1()->GetBinContent(lumi+1);
      float updatedLumiCount = currentLumiCount + diffCount;
      ME_1d->getTH1()->SetBinContent(lumi+1,updatedLumiCount);

    }
    else {

      LogDebug("TrigResRateMon") << " cannot find ME " << fullPathToME_count  <<  endl;

    }

   } // end for ip

 } // end for i

}


void TrigResRateMon::countHLTPathHitsEndLumiBlock(const int& lumi)
{

   LogTrace("TrigResRateMon") << " countHLTPathHitsEndLumiBlock() lumiSection number " << lumi << endl;
    // get the count of path up to now
   string fullPathToME = "HLT/TrigResults/PathsSummary/HLT_All_PassPass";
   MonitorElement* ME_2d = dbe_->get(fullPathToME);

   if(! ME_2d) {

     LogTrace("TrigResRateMon") << " could not fine 2d matrix " << fullPathToME << endl;

     return;

   }

   TH2F * hist_2d = ME_2d->getTH2F();

   for (std::vector<std::pair<std::string, float> >::iterator ip = fPathTempCountPair.begin(); ip != fPathTempCountPair.end(); ++ip) {
  
    // get the path and its previous count
    std::string pathname = ip->first;  
    float prevCount = ip->second;  
    
    // get the current count of path up to now
    float pathBin = hist_2d->GetXaxis()->FindBin(pathname.c_str());      

    if(pathBin > hist_2d->GetNbinsX()) {
      
      LogTrace("TrigResRateMon") << " Cannot find the bin for path " << pathname << endl;
      continue;

    }

    float currCount = hist_2d->GetBinContent(pathBin, pathBin)/LSsize_;

    // count due to prev lumi sec is a difference bw current and previous
    float diffCount = currCount - prevCount;

    LogTrace("TrigResRateMon") << " lumi = " << lumi << "  path " << pathname << "  count " << diffCount <<  endl;

    // set the counter in the pair to current count
    ip->second = currCount;  

    //////////////////////////////////////
    // fill the 2D All paths' count per LS
    //////////////////////////////////////
    if ( ME_HLTAll_LS) {

      TH2F* hist_All = ME_HLTAll_LS->getTH2F();

      // find the bin
      int pathBinNumber = hist_All->GetYaxis()->FindBin(pathname.c_str());
      
      // update  the bin content  (must do that since events don't ncessarily come in the order
      float currentLumiCount = hist_All->GetBinContent(lumi+1,pathBinNumber);
      float updatedLumiCount = currentLumiCount + diffCount;
      hist_All->SetBinContent(lumi+1,pathBinNumber,updatedLumiCount);
    
    }
    else {

      LogDebug("TrigResRateMon") << " cannot find ME_HLTAll_LS" <<  endl;

    }
    
    for (unsigned int i=0 ; i< v_ME_HLTAll_LS.size(); i++) {  
      
      MonitorElement* tempME = v_ME_HLTAll_LS[i];

      if ( tempME ) {
  
        TH2F* hist_All = tempME->getTH2F();
  
        // find the bin
        int pathBinNumber = hist_All->GetYaxis()->FindBin(pathname.c_str());
        // update  the bin content  (must do that since events don't ncessarily come in the order
        float currentLumiCount = hist_All->GetBinContent(lumi+1,pathBinNumber);
        float updatedLumiCount = currentLumiCount + diffCount;
        hist_All->SetBinContent(lumi+1,pathBinNumber,updatedLumiCount);
      
      }
      else {
  
        LogDebug("TrigResRateMon") << " cannot find tempME " <<  endl;
  
      }

    }


    ///////////////////////////////////////////
    // fill the 1D individual path count per LS
    ///////////////////////////////////////////
    string fullPathToME_count = pathsIndividualHLTPathsPerLSFolder_ + pathname + "_count_per_LS";
    MonitorElement* ME_1d = dbe_->get(fullPathToME_count);
    if ( ME_1d) { 

      // update  the bin content  (must do that since events don't ncessarily come in the order
      float currentLumiCount = ME_1d->getTH1()->GetBinContent(lumi+1);
      float updatedLumiCount = currentLumiCount + diffCount;
      ME_1d->getTH1()->SetBinContent(lumi+1,updatedLumiCount);

    }
    else {

      LogDebug("TrigResRateMon") << " cannot find ME " << fullPathToME_count  <<  endl;

    }

  } // end for ip

}

int TrigResRateMon::getTriggerTypeParsePathName(const string& pathname)
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
	 if (pathname.find("EG") != std::string::npos) 
	   objectType = trigger::TriggerPhoton;    
	 if (pathname.find("Tau") != std::string::npos) 
	   objectType = trigger::TriggerTau;    
	 if (pathname.find("IsoTrack") != std::string::npos) 
	   objectType = trigger::TriggerTrack;    
	 if (pathname.find("BTag") != std::string::npos) 
	   objectType = trigger::TriggerBJet;    

   return objectType;
}

const string TrigResRateMon::getL1ConditionModuleName(const string& pathname)
{

  // find L1 condition for numpath with numpath objecttype 
  // find PSet for L1 global seed for numpath, 
  // list module labels for numpath
  string l1pathname = "dummy";

  vector<string> numpathmodules = hltConfig_.moduleLabels(pathname);

  for(vector<string>::iterator numpathmodule = numpathmodules.begin();
  numpathmodule!= numpathmodules.end(); ++numpathmodule ) {

    if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed") {

     l1pathname = *numpathmodule;
     break; 

    }

  } // end for

  return l1pathname;

}


bool TrigResRateMon::hasL1Passed(const string& pathname, const edm::TriggerNames & triggerNames)
{
  
  bool rc = false;
  int l1ModuleIndex = 999;
  // --------------------
  for(PathInfoCollection::iterator v = hltPathsDiagonal_.begin(); v!= hltPathsDiagonal_.end(); ++v ) { 

    if(v->getPath() == pathname ) l1ModuleIndex = v->getL1ModuleIndex();

  }

  unsigned int pathByIndex = triggerNames.triggerIndex(pathname);
  if(pathByIndex >= triggerResults_->size() ) return rc; // path is not in the menu

  // get index of the last module that issued the decision
  int lastModule = triggerResults_->index(pathByIndex);

  // if L1 passed, then it must not be the module that 
  // issued the last decision
  rc = (l1ModuleIndex < lastModule);

  return rc;

}

bool TrigResRateMon::hasHLTPassed(const string& pathname, const edm::TriggerNames & triggerNames)
{
  
  bool rc = false;

  unsigned int pathByIndex = triggerNames.triggerIndex(pathname);
  if(pathByIndex >= triggerResults_->size() ) return rc; // path is not in the menu

  rc  = triggerResults_->accept(pathByIndex);

  return rc;

}


int TrigResRateMon::getThresholdFromName(const string & name)
{
  
  std::string pathname = name;
  //cout << "----------------------------------------------" << endl;
  //cout << pathname << endl;

  //remove "L1" substr
  if(pathname.find("L1") != std::string::npos) pathname.replace(pathname.find("L1"),2,"");
  //remove "L2" substr
  if(pathname.find("L2") != std::string::npos) pathname.replace(pathname.find("L2"),2,"");
  //remove "8E29" substr
  if(pathname.find("8E29") != std::string::npos) pathname.replace(pathname.find("8E29"),4,"");

  int digitLocation=0;
  for (unsigned int i=0; i < pathname.length(); i++)
  {
     if (isdigit(pathname.at(i))) {

       digitLocation = i;
       break;

     }
  }

  // get the string from the location of the first digit to the end
  string hltThresholdString = pathname.substr(digitLocation);

  int hltThreshold = 0;

  // get intiger at the begining of the string
  sscanf (hltThresholdString.c_str(),"%d%*s",&hltThreshold);
  //printf ("%s -> %s -> %d\n",pathname.c_str(), hltThresholdString.c_str(), hltThreshold);

  return hltThreshold;

}


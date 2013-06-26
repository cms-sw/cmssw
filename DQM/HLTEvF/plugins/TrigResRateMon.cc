// $Id: TrigResRateMon.cc,v 1.26 2012/02/21 10:32:34 slaunwhj Exp $
// See header file for information. 
#include "TMath.h"
#include "TString.h"
#include "DQM/HLTEvF/interface/TrigResRateMon.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

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

  testPathsFolder_ = iConfig.getUntrackedParameter ("testPathsFolder",std::string("HLT/TrigResults/testPaths/"));
  pathsSummaryFolder_ = iConfig.getUntrackedParameter ("pathsSummaryFolder",std::string("HLT/TrigResults/PathsSummary/HLT Counts/"));
  pathsSummaryStreamsFolder_ = iConfig.getUntrackedParameter ("pathsSummaryFolder",std::string("HLT/TrigResults/PathsSummary/"));
  //pathsSummaryStreamsFolder_ = iConfig.getUntrackedParameter ("pathsSummaryFolder",std::string("HLT/TrigResults/PathsSummary/Streams/"));
  pathsSummaryHLTCorrelationsFolder_ = iConfig.getUntrackedParameter ("hltCorrelationsFolder",std::string("HLT/TrigResults/PathsSummary/HLT Correlations/"));
  pathsSummaryFilterCountsFolder_ = iConfig.getUntrackedParameter ("filterCountsFolder",std::string("HLT/TrigResults/PathsSummary/Filters Counts/"));

  pathsSummaryHLTPathsPerLSFolder_ = iConfig.getUntrackedParameter ("individualPathsPerLSFolder",std::string("HLT/TrigResults/PathsSummary/HLT LS/"));
  pathsIndividualHLTPathsPerLSFolder_ = iConfig.getUntrackedParameter ("individualPathsPerLSFolder",std::string("HLT/TrigResults/PathsSummary/HLT LS/Paths/"));
  pathsSummaryHLTPathsPerBXFolder_ = iConfig.getUntrackedParameter ("individualPathsPerBXFolder",std::string("HLT/TrigResults/PathsSummary/HLT BX/"));


  // mask off some of the paths so that they don't appear in the plots

  maskedPaths_ = iConfig.getParameter<std::vector<std::string > >("MaskedPaths");

  referenceTrigInput_ = iConfig.getParameter<std::string> ("ReferenceTrigger");
  foundReferenceTrigger_ = false;

  //Robin
  testPaths_ = iConfig.getParameter<std::vector<std::string > >("testPaths");
  
  fLumiFlag = true;
  ME_HLTAll_LS = NULL;
  ME_HLT_BX = NULL;
  ME_HLT_CUSTOM_BX = NULL;

  //jetID = new reco::helper::JetIDHelper(iConfig.getParameter<ParameterSet>("JetIDParams"));

    recHitsEBTag_ = iConfig.getUntrackedParameter<edm::InputTag>("RecHitsEBTag",edm::InputTag("reducedEcalRecHitsEB"));
      recHitsEETag_ = iConfig.getUntrackedParameter<edm::InputTag>("RecHitsEETag",edm::InputTag("reducedEcalRecHitsEE"));


      jmsDebug = false;
      jmsFakeZBCounts = false;
      found_zbIndex = false;
      if (jmsDebug ) std::cout << "Printing extra info " << std::endl;
      
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

  //Robin---
  nStream_++;
  passAny = false ;

  // Find which index is zero bias for this run
  
// //   if(!found_zbIndex){
// //     // set default to something that is out-of-bounds
// //     zbIndex = triggerNames.size() +2;
// //     for (unsigned int i = 0; i < triggerNames.size(); i ++){
// //       std::string thisName = triggerNames.triggerName(i);
// //       TString checkName(thisName.c_str());
// //       if (checkName.Contains("HLT_ZeroBias_v")){
// //         zbIndex = i;
// //         found_zbIndex = true;
// //         if(jmsDebug) std::cout << "Found the ZeroBias index!!!!!!!! It is   " << zbIndex <<std::endl;
// //       }
// //     }
// //   }

  // int bx = iEvent.bunchCrossing();
  /*
  // Fill HLTPassed_Correlation Matrix bin (i,j) = (Any,Any)
  // --------------------------------------------------------
  int anyBinNumber = ME_HLTPassPass->getTH2F()->GetXaxis()->FindBin("HLT_Any");      
  // any triger accepted
  if(triggerResults->accept()){

    ME_HLTPassPass->Fill(anyBinNumber-1,anyBinNumber-1);//binNumber1 = 0 = first filter

  }
  */


  
  fillHltMatrix(triggerNames, iEvent, iSetup);




  /////////////////////////////////////////////////////////
  //
  //  Experimental Testing Area
  //  Try to get the lumi information for these events 
  //
  ////////////////////////////////////////////////////////////

  edm::Handle<LumiScalersCollection> lumiScalers;
  bool lumiHandleOK = iEvent.getByLabel(InputTag("hltScalersRawToDigi","",""), lumiScalers);

  if (jmsDebug) std::cout << "Tried to get lumi handle result = " << lumiHandleOK << std::endl;
  
  if (lumiHandleOK) {
    if (jmsDebug) std::cout << "LumiScalers size is:  " << lumiScalers->size()  << std::endl;

    if (lumiScalers->size()) {
      LumiScalersCollection::const_iterator it3 = lumiScalers->begin();
      unsigned int lumisection = it3->sectionNumber();
      if(lumisection){

        if (jmsDebug) std::cout << "Instanteous Lumi is " << it3->instantLumi() << std::endl;
        if (jmsDebug) std::cout << "Instanteous Lumi Error is " <<it3->instantLumiErr() << std::endl;
        if (jmsDebug) std::cout << "Lumi Fill is " <<it3->lumiFill() << std::endl;
        if (jmsDebug) std::cout << "Lumi Fill is " <<it3->lumiRun() << std::endl;
        if (jmsDebug) std::cout << "Live Lumi Fill is " <<it3->liveLumiFill() << std::endl;
        if (jmsDebug) std::cout << "Live Lumi Run is " <<it3->liveLumiRun() << std::endl;

        addLumiToAverage(it3->instantLumi());
        
        
      } // end
    }// end if lumi scalers exist
    
  }// end if lumi handle ok

  
  fillCountsPerPath(iEvent, iSetup);

  if (passAny) nPass_ ++ ;
  
  return;

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
  
  
  MonitorElement* reportSummaryME = dbe->book1D("reportSummaryMap","report Summary Map",2,0,2);
  if(reportSummaryME) reportSummaryME->Fill(1);


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

      if (jmsDebug) std::cout << "HLTConfigProvider failed to initialize.";

    } else {
      if (jmsDebug) std::cout << "Initialized HLTConfigProvider with name FU " << std::endl;
    }

    // check if trigger name in (new) config
    //  cout << "Available TriggerNames are: " << endl;
    //  hltConfig_.dump("Triggers");
  } else {
    if (jmsDebug) std::cout << "Initialized HLTConfigProvider with name HLT " << std::endl;
  }

  if (1) {

    DQMStore *dbe = 0;
    dbe = Service<DQMStore>().operator->();
  
    if (dbe) {
      dbe->setCurrentFolder(dirname_);
    }


    //    const unsigned int n(hltConfig_.size());

    TotalDroppedCounts = 0;
    //Robin-------Diagnostic plots--------
    meDiagnostic = dbe->book1D("DroppedCounts Diagnose", "LSs vs Status;Status;LSs", 3, -0.5,2.5 );
    meCountsDroppedPerLS = dbe->book1D("CountsDroppedVsLS", "Counts vs LumiSec;LS;Dropped Stream Counts", nLS_, 0.5, nLS_+0.5);
    meCountsPassPerLS = dbe->book1D("CountsPassVsLS", "Counts vs LumiSec;LS;Passed Stream Counts", nLS_, 0.5, nLS_+0.5);
    meCountsStreamPerLS = dbe->book1D("CountsStreamVsLS", "Counts vs LumiSec;LS;Stream Counts", nLS_, 0.5, nLS_+0.5);
    meXsecStreamPerLS = dbe->book1D("XsecStreamVsLS", "Xsec vs LumiSec;LS;Stream Xsec", nLS_, 0.5, nLS_+0.5);

//     meXsecPerLS = dbe->book1D("XsecVsLS", "Xsec vs LumiSec;LS;Xsec", nLS_, 0.5, nLS_+0.5);
//     meXsec = dbe->book1D("Xsec", "histo for Xsec ", 20, 0.01, 0.06);
//     //    meXsecPerIL = dbe->book2D("XsecVsIL", "Xsec vs Inst Lumi;#mub^{-1}*s^{-1}; Xsec", 200, 700, 900, 100, 0.01, 0.1);
//     TProfile tempProfile("XsecVsIL", "Xsec vs Inst Lumi;#mub^{-1}*s^{-1}; Xsec", 40, 600, 3000);
//     meXsecPerIL = dbe_->bookProfile("XsecVsIL", &tempProfile);

    bookTestHisto(); //Robin
    dbe->setCurrentFolder(dirname_); //Robin

    // JMS fill the counts per path
    bookCountsPerPath();
    clearLumiAverage();
    averageInstLumi3LS = 0 ;
    findReferenceTriggerIndex();
    meAverageLumiPerLS = dbe->book1D("InstLumiVsLS", "Instantaneous Luminosity vs LumiSec;LS;#mub^{-1}*s^{-1}", nLS_, 0.5, nLS_+0.5);
// //     if (plotAll_){

// //       for (unsigned int j=0; j!=n; ++j) {
  
// //         std::string pathname = hltConfig_.triggerName(j);  

// //         string l1pathname = getL1ConditionModuleName(pathname);

// //         int l1ModuleIndex = hltConfig_.moduleIndex(pathname, l1pathname);
      
// //         int objectType =  getTriggerTypeParsePathName(pathname);
  
// //         for (unsigned int i=0; i!=n; ++i) {
  
// //           std::string denompathname = hltConfig_.triggerName(i);  
// //           int denomobjectType =  getTriggerTypeParsePathName(denompathname);

      
         
// //           std::string filtername("dummy");
// //           float ptMin = 0.0;
// //           float ptMax = 100.0;
// //           if (plotAll_ && denomobjectType == objectType && objectType != 0) {
          
// //             int hltThreshold = getThresholdFromName(pathname);
// //             int l1Threshold = getThresholdFromName(l1pathname);
// //             hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, l1ModuleIndex, filtername, processname_, objectType, ptMin, ptMax, hltThreshold, l1Threshold));

// //           }

// //         }
// //       }

// //     } // end if plotAll
// //     else {

      // plot all diagonal combinations plus any other specified pairs
// //       for (unsigned int i=0; i!=n; ++i) {

// //          std::string denompathname = "";  
// //          std::string pathname = hltConfig_.triggerName(i);  
// //          //parse pathname to guess object type
// //          int objectType =  getTriggerTypeParsePathName(pathname);

// //         string l1pathname = getL1ConditionModuleName(pathname);
// //         int l1ModuleIndex = hltConfig_.moduleIndex(pathname, l1pathname);
  
// //         std::string filtername("dummy");
// //         float ptMin = 0.0;
// //         float ptMax = 100.0;

// //         if (objectType == trigger::TriggerPhoton) ptMax = 100.0;
// //         if (objectType == trigger::TriggerElectron) ptMax = 100.0;
// //         if (objectType == trigger::TriggerMuon) ptMax = 150.0;
// //         if (objectType == trigger::TriggerTau) ptMax = 100.0;
// //         if (objectType == trigger::TriggerJet) ptMax = 300.0;
// //         if (objectType == trigger::TriggerBJet) ptMax = 300.0;
// //         if (objectType == trigger::TriggerMET) ptMax = 300.0;
// //         if (objectType == trigger::TriggerTET) ptMax = 300.0;
// //         if (objectType == trigger::TriggerTrack) ptMax = 100.0;
    
// //         // keep track of all paths, except for FinalPath
// //         if (objectType != -1 && pathname.find("FinalPath") == std::string::npos){

// //           int hltThreshold = getThresholdFromName(pathname);
// //           int l1Threshold = getThresholdFromName(l1pathname);
  
// //           hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, l1ModuleIndex, filtername, processname_, objectType, ptMin, ptMax, hltThreshold, l1Threshold));

// //           hltPathsDiagonal_.push_back(PathInfo(denompathname, pathname, l1pathname, l1ModuleIndex, filtername, processname_, objectType, ptMin, ptMax, hltThreshold, l1Threshold));
  
// //         }

// //       } // end for i

        
      // now loop over denom/num path pairs specified in cfg, 
      // recording the off-diagonal ones
// //       for (std::vector<std::pair<std::string, std::string> >::iterator custompathnamepair = custompathnamepairs_.begin(); custompathnamepair != custompathnamepairs_.end(); ++custompathnamepair) {
            
// //         std::string numpathname = custompathnamepair->first;  
// //         std::string denompathname = custompathnamepair->second;  
  
// //         if (numpathname != denompathname) {
  
// //           // check that denominator exists
// //           bool founddenominator = false;
// //           for (unsigned int k=0; k!=n; ++k) {

// //             string n_pathname = hltConfig_.triggerName(k);

// //             if (n_pathname.find(denompathname) != std::string::npos) {
              
// //               LogDebug("TrigResRateMon") << "denompathname is selected to be = " << n_pathname << endl;;
// //               founddenominator = true;

// //               break;

// //             }
// //           }

// //           if (!founddenominator) {
  
// //             edm::LogInfo("TrigResRateMon") << "denompathname not found, go to the next pair numearator-denominator" << endl;
            
// //             // go to the next pair
// //             continue;
  
// //           }

// //           // check that numerator exists
// //           bool foundnumerator = false;
// //           for (unsigned int j=0; j!=n; ++j) {

// //             string pathname = hltConfig_.triggerName(j);

// //             LogDebug("TrigResRateMon") << "check if path " << pathname << " is numpathname = " << numpathname << endl;
// //             if (hltConfig_.triggerName(j).find(numpathname)!= std::string::npos) {
              
// //               LogDebug("TrigResRateMon") << "pathname is selected to be = " << denompathname << endl;;
// //               foundnumerator = true;

// //             }
  
  
// //             if (!foundnumerator) {
    
// //               edm::LogInfo("TrigResRateMon") << "pathname not found, ignoring " << pathname;
// //               continue;
  
// //             }
  
  
// //             string l1pathname = getL1ConditionModuleName(pathname);
// //             int l1ModuleIndex = hltConfig_.moduleIndex(pathname, l1pathname);
// //             int objectType =  getTriggerTypeParsePathName(pathname);
  
// //           std::string filtername("dummy");
// //           float ptMin = 0.0;
// //           float ptMax = 100.0;
// //           if (objectType == trigger::TriggerPhoton) ptMax = 100.0;
// //           if (objectType == trigger::TriggerElectron) ptMax = 100.0;
// //           if (objectType == trigger::TriggerMuon) ptMax = 150.0;
// //           if (objectType == trigger::TriggerTau) ptMax = 100.0;
// //           if (objectType == trigger::TriggerJet) ptMax = 300.0;
// //           if (objectType == trigger::TriggerBJet) ptMax = 300.0;
// //           if (objectType == trigger::TriggerMET) ptMax = 300.0;
// //           if (objectType == trigger::TriggerTET) ptMax = 300.0;
// //           if (objectType == trigger::TriggerTrack) ptMax = 100.0;
  
// //           // monitor regardless of the objectType of the path
// //           if (objectType != 0) {
// //             int hltThreshold = getThresholdFromName(pathname);
// //             int l1Threshold = getThresholdFromName(l1pathname);
// //             hltPaths_.push_back(PathInfo(denompathname, pathname, l1pathname, l1ModuleIndex, filtername, processname_, objectType, ptMin, ptMax, hltThreshold, l1Threshold));

// //           }
      
// //         } // end for j, loop over paths

// //        }  // end if not same num and denominator 
  
// //       }
//       // end for pair

//     }
  // end else


    /*
    vector<string> muonPaths;
    vector<string> egammaPaths;
    vector<string> tauPaths;
    vector<string> jetmetPaths;
    vector<string> restPaths;
    */
    vector<string> allPaths;

    // fill vectors of Muon, Egamma, JetMET, Rest, and Special paths
    for(PathInfoCollection::iterator v = hltPathsDiagonal_.begin(); v!= hltPathsDiagonal_.end(); ++v ) {

      std::string pathName = v->getPath();
      //int objectType = v->getObjectType();

      vector<int> tempCount(5,0);

      fPathTempCountPair.push_back(make_pair(pathName,0));
      fPathBxTempCountPair.push_back(make_pair(pathName,tempCount));

      allPaths.push_back(pathName);

      /*
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
      */

    }

    fPathTempCountPair.push_back(make_pair("HLT_Any",0));

    fGroupName.push_back("AllSelectedPaths");
    /*
    fGroupName.push_back("Muon");
    fGroupName.push_back("Egamma");
    fGroupName.push_back("Tau");
    fGroupName.push_back("JetMET");
    fGroupName.push_back("Rest");
    fGroupName.push_back("Special");
    */

    for(unsigned int g=0; g<fGroupName.size(); g++) {

      //fGroupTempCountPair.push_back(make_pair(fGroupName[g],0));
      //fGroupL1TempCountPair.push_back(make_pair(fGroupName[g],0));

    }
  
    dbe_->setCurrentFolder(pathsSummaryFolder_.c_str());



//     fGroupNamePathsPair.push_back(make_pair("AllSelectedPaths",allPaths));

    /*
    fGroupNamePathsPair.push_back(make_pair("Muon",muonPaths));

    fGroupNamePathsPair.push_back(make_pair("Egamma",egammaPaths));

    fGroupNamePathsPair.push_back(make_pair("Tau",tauPaths));

    fGroupNamePathsPair.push_back(make_pair("JetMET",jetmetPaths));

    fGroupNamePathsPair.push_back(make_pair("Rest",restPaths));

    fGroupNamePathsPair.push_back(make_pair("Special",specialPaths_));
    */

    /// add dataset name and thier triggers to the list 
    //vector<string> datasetNames =  hltConfig_.datasetNames() ;
    vector<string> datasetNames =  hltConfig_.streamContent("A") ;
    for (unsigned int i=0;i<datasetNames.size();i++) {

      vector<string> datasetPaths = hltConfig_.datasetContent(datasetNames[i]);
      fGroupNamePathsPair.push_back(make_pair(datasetNames[i],datasetPaths));

      DatasetInfo tempDS;
      tempDS.datasetName = datasetNames[i];
      tempDS.setPaths(datasetPaths);
      tempDS.countsPerPathME_Name = pathsSummaryFolder_ + "HLT_" + datasetNames[i] + "_Pass_Any";
      tempDS.xsecPerPathME_Name = pathsSummaryFolder_ + "HLT_" + datasetNames[i] + "_Xsec";
      tempDS.rawCountsPerPathME_Name = pathsSummaryFolder_ + "HLT_" + datasetNames[i] + "_RawCounts";
      tempDS.scaledXsecPerPathME_Name = pathsSummaryFolder_ + "HLT_" + datasetNames[i] + "_XsecScaled";
      tempDS.ratePerLSME_Name = pathsSummaryFolder_ + "HLT_" + datasetNames[i] + "_Rate"; //Robin
      tempDS.setMaskedPaths(maskedPaths_);
      if (jmsDebug) tempDS.printMaskedPaths();
      primaryDataSetInformation.push_back(tempDS);

      rawCountsPerPD.push_back(0);  //Robin
    }

    // push stream A and its PDs
    fGroupNamePathsPair.push_back(make_pair("A",datasetNames));


    for (unsigned int g=0;g<fGroupNamePathsPair.size();g++) {

      fGroupTempCountPair.push_back(make_pair(fGroupNamePathsPair[g].first,0));
      fGroupL1TempCountPair.push_back(make_pair(fGroupNamePathsPair[g].first,0));
      setupHltMatrix(fGroupNamePathsPair[g].first,fGroupNamePathsPair[g].second);

    }

/*
    // HLT matrices from Streams
    const std::vector<std::string> streamNames = hltConfig_.streamNames();

    for (unsigned int s=0;s<streamNames.size();s++) {

      /// add dataset name and thier triggers to the list 
      vector<string> hltConfig =  streamDatasetNames_.streamContent(streamNames[s]) ;
      if(streamNames[s] == "A") setupStreamMatrix(streamNames[s],streamDatasetNames);

    }
*/

// //     setupHltLsPlots();

// //     setupHltBxPlots();


//     for(PathInfoCollection::iterator v = hltPathsDiagonal_.begin(); v!= hltPathsDiagonal_.end(); ++v ) {

//        // -------------------------
//        //
//        //  Filters for each path
//        //
//        // -------------------------
       
//        // get all modules in this HLT path
//        vector<string> moduleNames = hltConfig_.moduleLabels( v->getPath() ); 
       
//        int numModule = 0;
//        string moduleName, moduleType;
//        unsigned int moduleIndex;
       
//        //print module name
//        vector<string>::const_iterator iDumpModName;
//        for (iDumpModName = moduleNames.begin();iDumpModName != moduleNames.end();iDumpModName++) {

//          moduleName = *iDumpModName;
//          moduleType = hltConfig_.moduleType(moduleName);
//          moduleIndex = hltConfig_.moduleIndex(v->getPath(), moduleName);

//          LogTrace ("TrigResRateMon") << "Module "      << numModule
//              << " is called " << moduleName
//              << " , type = "  << moduleType
//              << " , index = " << moduleIndex
//              << endl;

//          numModule++;

//          if((moduleType.find("Filter") != string::npos && moduleType.find("HLTTriggerTypeFilter") == string::npos ) || 
//             (moduleType.find("Associator") != string::npos) || 
//             (moduleType.find("HLTLevel1GTSeed") != string::npos) || 
//             (moduleType.find("HLTGlobalSumsCaloMET") != string::npos) ||
//             (moduleType.find("HLTPrescaler") != string::npos) ) {

//            //std::pair<std::string, int> filterIndexPair;
//            //filterIndexPair.first   = moduleName;
//            //filterIndexPair.second  = moduleIndex;
//            //v->filtersAndIndices.push_back(filterIndexPair);
//            v->filtersAndIndices.push_back(make_pair(moduleName,moduleIndex));

//          }


//        }//end for modulesName

//        // dbe_->setCurrentFolder(pathsSummaryFilterCountsFolder_.c_str()); 

// //        //int nbin_sub = 5;
// //        int nbin_sub = v->filtersAndIndices.size()+2;
    
// //        // count plots for subfilter
// //        MonitorElement* filters = dbe_->book1D("Filters_" + v->getPath(), 
// //                               "Filters_" + v->getPath(),
// //                               nbin_sub+1, -0.5, 0.5+(double)nbin_sub);
       
// //        for(unsigned int filt = 0; filt < v->filtersAndIndices.size(); filt++){

// //          filters->setBinLabel(filt+1, (v->filtersAndIndices[filt]).first);

// //        }

//        // book Count vs LS
//        dbe_->setCurrentFolder(pathsIndividualHLTPathsPerLSFolder_.c_str());
//        MonitorElement* tempME = dbe_->book1D(v->getPath() + "_count_per_LS", 
//                               v->getPath() + " rate [Hz]",
//                               nLS_, 0,nLS_);
//        tempME->setAxisTitle("Luminosity Section");

//        //v->setFilterHistos(filters);

//     }
    // end for paths

    // now set up all of the histos for each path-denom
//    for(PathInfoCollection::iterator v = hltPaths_.begin(); v!= hltPaths_.end(); ++v ) {

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
      

//       std::string labelname("dummy");
//       labelname = v->getPath() + "_wrt_" + v->getDenomPath();
//       std::string histoname(labelname+"_NOn");
//       std::string title(labelname+" N online");
//       double histEtaMax = 2.5;

//       if (v->getObjectType() == trigger::TriggerMuon || v->getObjectType() == trigger::TriggerL1Mu) {

//         histEtaMax = muonEtaMax_;

//       }
//       else if (v->getObjectType() == trigger::TriggerElectron || v->getObjectType() == trigger::TriggerL1NoIsoEG || v->getObjectType() == trigger::TriggerL1IsoEG )
//       {
//         histEtaMax = electronEtaMax_;
//       }
//         else if (v->getObjectType() == trigger::TriggerTau || v->getObjectType() == trigger::TriggerL1TauJet )
//       {
//         histEtaMax = tauEtaMax_;
//       }
//       else if (v->getObjectType() == trigger::TriggerJet || v->getObjectType() == trigger::TriggerL1CenJet || v->getObjectType() == trigger::TriggerL1ForJet )
//       {
//         histEtaMax = jetEtaMax_; 
//       }
//         else if (v->getObjectType() == trigger::TriggerBJet)
//       {
//         histEtaMax = bjetEtaMax_;
//       }
//       else if (v->getObjectType() == trigger::TriggerMET || v->getObjectType() == trigger::TriggerL1ETM )
//       {
//         histEtaMax = metEtaMax_; 
//       }
//         else if (v->getObjectType() == trigger::TriggerPhoton)
//       {
//         histEtaMax = photonEtaMax_; 
//       }
//       else if (v->getObjectType() == trigger::TriggerTrack)
//       {
//         histEtaMax = trackEtaMax_; 
//       }

//       TString pathfolder = dirname_ + TString("/FourVector/") + v->getPath();
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


//    }  // end for hltPath 

    // HLT_Any
    // book Count vs LS
// //     dbe_->setCurrentFolder(pathsIndividualHLTPathsPerLSFolder_.c_str());
// //     MonitorElement* tempME = dbe_->book1D("HLT_Any_count_per_LS", 
// //                            "HLT_Any rate [Hz]",
// //                            nLS_, 0,nLS_);
// //     tempME->setAxisTitle("Luminosity Section");

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
    //paths.push_back("HLT_"+label+"_L1_Any");
    paths.push_back("");
    paths.push_back("Total "+label);
    //paths.push_back("HLT_Any");

    string h_name; 
    string h_title; 

    dbe_->setCurrentFolder(pathsSummaryFolder_.c_str());

// //     h_name= "HLT_"+label+"_PassPass";
// //     h_title = "HLT_"+label+"_PassPass (x=Pass, y=Pass);;; ";
// //     MonitorElement* ME = dbe_->book2D(h_name.c_str(), h_title.c_str(),
// //                            paths.size(), -0.5, paths.size()-0.5, paths.size(), -0.5, paths.size()-0.5);

    // This is counts per path per for a specific PD
    // it will be corrected for prescales
    h_name= "HLT_"+label+"_Pass_Any";
    h_title = "HLT_"+label+"_Pass -- Prescale*Counts Per Path;Path;PS*Counts";
    MonitorElement* ME_Any = dbe_->book1D(h_name.c_str(), h_title.c_str(),
                           paths.size(), -0.5, paths.size()-0.5);

    // This is RAW counts per path per for a specific PD
    // it will be corrected for
    h_name= "HLT_"+label+"_RawCounts";
    h_title = "HLT_"+label+"_Pass (x=Pass, An) normalized to HLT_Any Pass;;Counts";
    MonitorElement* ME_RawCounts = dbe_->book1D(h_name.c_str(), h_title.c_str(),
                           paths.size(), -0.5, paths.size()-0.5);


    // Make a similar histogram that is xsec per path for a specific PD
    // this is actually a profile of the average xsec per path 
    h_name= "HLT_"+label+"_Xsec";
    h_title = "HLT_"+label+"_Xsec -- Profile shows Average Xsec per path;;#sigma (#mu b)";

    TProfile tempProfile(h_name.c_str(), h_title.c_str(),
                         paths.size(), -0.5, paths.size()-0.5);
    MonitorElement* ME_Xsec = dbe_->bookProfile(h_name.c_str(), &tempProfile);


    // Make a similar histogram that is xsec per path for a specific PD
    // this is actually a profile of the average xsec per path
    // this histogram is scaled to the cross section of a reference path
    h_name= "HLT_"+label+"_XsecScaled";
    h_title = "HLT_"+label+"_Xsec -- Profile shows Average Xsec per path Scaled to Reference;;Ratio (#sigma/#sigma_{ref}";

    TProfile tempProfileScaled(h_name.c_str(), h_title.c_str(),
                         paths.size(), -0.5, paths.size()-0.5);
    MonitorElement* ME_XsecScaled = dbe_->bookProfile(h_name.c_str(), &tempProfileScaled);

    ///////HLT PD rate plot
    h_name= "HLT_"+label+"_Rate";
    h_title = "HLT_"+label+"_Rate -- histogram shows Average Rate per LS;LS;Rate [Hz]";

    // MonitorElement* ME_Rate = dbe_->book1D(h_name.c_str(), h_title.c_str(),nLS_, 0, nLS_);


//     dbe_->setCurrentFolder(pathsSummaryHLTCorrelationsFolder_.c_str());
//     h_name= "HLT_"+label+"_PassPass_Normalized";
//     h_title = "HLT_"+label+"_PassPass (x=Pass, y=Pass) normalized to xBin=Pass";
//     MonitorElement* ME_Normalized = dbe_->book2D(h_name.c_str(), h_title.c_str(),
//                            paths.size(), -0.5, paths.size()-0.5, paths.size(), -0.5, paths.size()-0.5);
//     h_name= "HLT_"+label+"_Pass_Normalized_Any";
//     h_title = "HLT_"+label+"_Pass (x=Pass, Any=Pass) normalized to HLT_Any Pass";
//     MonitorElement* ME_Normalized_Any = dbe_->book1D(h_name.c_str(), h_title.c_str(),
//                            paths.size(), -0.5, paths.size()-0.5);

//     dbe_->setCurrentFolder(pathsSummaryHLTPathsPerLSFolder_.c_str());
//     h_name= "HLT_"+label+"_Total_LS";
//     h_title = label+" HLT paths total combined rate [Hz]";
//     MonitorElement* ME_Total_LS = dbe_->book1D(h_name.c_str(), h_title.c_str(), nLS_, 0, nLS_);
//     ME_Total_LS->setAxisTitle("LS");

//     h_name= "HLT_"+label+"_LS";
//     h_title = label+" HLT paths rate [Hz]";
//     MonitorElement* ME_Group_LS = dbe_->book2D(h_name.c_str(), h_title.c_str(), nLS_, 0, nLS_, paths.size(), -0.5, paths.size()-0.5);
//     ME_Group_LS->setAxisTitle("LS");
//     /// add this path to the vector of 2D LS paths
//     v_ME_HLTAll_LS.push_back(ME_Group_LS);

    /*
    h_name= "HLT_"+label+"_L1_Total_LS";
    h_title = label+" HLT paths total combined rate [Hz]";
    MonitorElement* ME_Total_L1_LS = dbe_->book1D(h_name.c_str(), h_title.c_str(), nLS_, 0, nLS_);
    ME_Total_L1_LS->setAxisTitle("LS");

    h_name= "HLT_"+label+"_L1_LS";
    h_title = label+" HLT L1s rate [Hz]";
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
    */

    for(unsigned int i = 0; i < paths.size(); i++){

// //       ME->getTH2F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
// //       ME->getTH2F()->GetYaxis()->SetBinLabel(i+1, (paths[i]).c_str());
//       ME_Group_LS->getTH2F()->GetYaxis()->SetBinLabel(i+1, (paths[i]).c_str());

//       ME_Normalized->getTH2F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
//       ME_Normalized->getTH2F()->GetYaxis()->SetBinLabel(i+1, (paths[i]).c_str());
//       ME_Normalized_Any->getTH1F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Any->getTH1F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Xsec->getTProfile()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_XsecScaled->getTProfile()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_RawCounts->getTH1F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
    }
    
}


void TrigResRateMon::setupStreamMatrix(const std::string& label, vector<std::string>& paths) {


    paths.push_back("");
    paths.push_back("HLT_"+label+"_Any");

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

    for(unsigned int i = 0; i < paths.size(); i++){

      ME->getTH2F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME->getTH2F()->GetYaxis()->SetBinLabel(i+1, (paths[i]).c_str());

      ME_Normalized->getTH2F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Normalized->getTH2F()->GetYaxis()->SetBinLabel(i+1, (paths[i]).c_str());

      ME_Any->getTH1F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());
      ME_Normalized_Any->getTH1F()->GetXaxis()->SetBinLabel(i+1, (paths[i]).c_str());

    }

}

void TrigResRateMon::fillHltMatrix(const edm::TriggerNames & triggerNames, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

 string fullPathToME; 
 std::vector <std::pair<std::string, bool> > groupAcceptPair;

 // This will store the prescale values
 std::pair<int,int>  psValueCombo;

  // For each dataset
 for (unsigned int mi=0;mi<fGroupNamePathsPair.size()-1;mi++) {  ////


// //   fullPathToME = pathsSummaryFolder_ + "HLT_"+fGroupNamePathsPair[mi].first+"_PassPass";
// //   MonitorElement* ME_2d = dbe_->get(fullPathToME);
  fullPathToME = pathsSummaryFolder_ + "HLT_"+fGroupNamePathsPair[mi].first+"_Pass_Any";
  MonitorElement* ME_1d = dbe_->get(fullPathToME);
//   // if(!ME_2d || !ME_1d) {  
  if(!ME_1d) { 
    LogTrace("TrigResRateMon") << " ME not valid although I gave full path" << endl;
    continue;

  }

// //   TH2F * hist_2d = ME_2d->getTH2F();
  TH1F * hist_1d = ME_1d->getTH1F();

  // Fill HLTPassed Matrix bin (i,j) = (Any,Any)
  // --------------------------------------------------------
// //   int anyBinNumber = hist_2d->GetXaxis()->FindBin("HLT_Any");   

  //string groupBinLabel = "HLT_"+fGroupNamePathsPair[mi].first+"_Any";
// //   string groupBinLabel = "Total "+fGroupNamePathsPair[mi].first;
// //   int groupBinNumber = hist_2d->GetXaxis()->FindBin(groupBinLabel.c_str()); 

  // any triger accepted
  //   if(triggerResults_->accept()){
  
  //     hist_2d->Fill(anyBinNumber-1,anyBinNumber-1);//binNumber1 = 0 = first filter
  
  
  
  //     hist_1d->Fill(anyBinNumber-1);//binNumber1 = 0 = first filter
  
  //   }

    
 

  //if (jmsDebug) std::cout << "Setting histograms to HLT_ZeroBias index " << zbIndex << std::endl;

// //   if (zbIndex < triggerResults_->size() ) {
// //     if (triggerResults_->accept(zbIndex) || jmsFakeZBCounts) {
// //       if (jmsDebug) std::cout << "Filling bin " << (groupBinNumber-1)
// //                               << " (out of " << hist_1d->GetNbinsX()
// //                               << ")  with ZB counts in histo "
// //                               << hist_1d->GetName() << std::endl;
      
// //       hist_1d->Fill(groupBinNumber-1, 50000);
// //       hist_2d->Fill(groupBinNumber-1,groupBinNumber-1, 10000);//binNumber1 = 0 = first filter
// //    }
// //  }
  
  
  bool groupPassed = false;
  //bool groupL1Passed = false;

  // Main loop over paths
  // --------------------

  //for (int i=1; i< hist_2d->GetNbinsX();i++) 
  for (unsigned int i=0; i< fGroupNamePathsPair[mi].second.size(); i++)
  { 

    //string hltPathName =  hist_2d->GetXaxis()->GetBinLabel(i);
    string hltPathName =  fGroupNamePathsPair[mi].second[i];

    // check if this is hlt path name
    //unsigned int pathByIndex = triggerNames.triggerIndex(hltPathName);
    unsigned int pathByIndex = triggerNames.triggerIndex(fGroupNamePathsPair[mi].second[i]);
    if(pathByIndex >= triggerResults_->size() ) continue;

    // check if its L1 passed
    // comment out below but set groupL1Passed to true always
    //if(hasL1Passed(hltPathName,triggerNames)) groupL1Passed = true;
    //groupL1Passed = true;

    // Fill HLTPassed Matrix and HLTPassFail Matrix
    // --------------------------------------------------------

    if(triggerResults_->accept(pathByIndex)){

      groupPassed = true;
      //groupL1Passed = true;

// //       hist_2d->Fill(i,anyBinNumber-1);//binNumber1 = 0 = first filter
// //       hist_2d->Fill(anyBinNumber-1,i);//binNumber1 = 0 = first filter

// //       hist_2d->Fill(i,groupBinNumber-1);//binNumber1 = 0 = first filter
// //       hist_2d->Fill(groupBinNumber-1,i);//binNumber1 = 0 = first filter

      if (jmsDebug) std::cout << "Trying to get prescales... " << std::endl;
  
      psValueCombo =  hltConfig_.prescaleValues(iEvent, iSetup, hltPathName);

      if (jmsDebug) std::cout << "Path " << hltPathName
                              << "  L1 PS " << psValueCombo.first
                              << " and hlt ps " << psValueCombo.second << std::endl;
      
      if ( (psValueCombo.first > 0) && (psValueCombo.second > 0) ){
        hist_1d->Fill(i, psValueCombo.first * psValueCombo.second );//binNumber1 = 0 = first filter
      } else {
        hist_1d->Fill(i);
      }

      //for (int j=1; j< hist_2d->GetNbinsY();j++) 
//*       for (unsigned int j=0; j< fGroupNamePathsPair[mi].second.size(); j++)
//*       { 
  
//*         string crossHltPathName =  fGroupNamePathsPair[mi].second[j];
  
//*         //unsigned int crosspathByIndex = triggerNames.triggerIndex(hist_2d->GetXaxis()->GetBinLabel(j));
//*         //unsigned int crosspathByIndex = triggerNames.triggerIndex(crossHltPathName);
//*         unsigned int crosspathByIndex = triggerNames.triggerIndex(fGroupNamePathsPair[mi].second[j]);
  
//*         if(crosspathByIndex >= triggerResults_->size() ) continue;
  
//*         if(triggerResults_->accept(crosspathByIndex)){
  
//*           hist_2d->Fill(i,j);//binNumber1 = 0 = first filter
  
//*         } // end if j path passed
  
//*       } // end for j 
  
    } // end if i passed
    

  } // end for i

  if(groupPassed) {
  
    rawCountsPerPD[mi]++ ;
    //hist_1d->Fill(groupBinNumber-1);//binNumber1 = 0 = first filter
    //hist_2d->Fill(groupBinNumber-1,groupBinNumber-1);//binNumber1 = 0 = first filter
    //hist_2d->Fill(anyBinNumber-1,groupBinNumber-1);//binNumber1 = 0 = first filter
    //hist_2d->Fill(groupBinNumber-1,anyBinNumber-1);//binNumber1 = 0 = first filter

  }

  // if the group belongs to stream A
  // store groupName and Bool if it has passed 
  bool isGroupFromStreamA = false;

  vector<string> streamDatasetNames =  hltConfig_.streamContent("A") ;
  for (unsigned int g=0;g<streamDatasetNames.size();g++) {

    if(streamDatasetNames[g] == fGroupNamePathsPair[mi].first) 
    {

      isGroupFromStreamA = true;
      break;

    }
  }

  if(isGroupFromStreamA) groupAcceptPair.push_back(make_pair(fGroupNamePathsPair[mi].first,groupPassed));


  // L1 groups  - not used anymore
//   string groupL1BinLabel = "HLT_"+fGroupNamePathsPair[mi].first+"_L1_Any";
// //   int groupL1BinNumber = hist_2d->GetXaxis()->FindBin(groupL1BinLabel.c_str());      

// //  if(groupL1Passed) hist_1d->Fill(groupL1BinNumber-1);//binNumber1 = 0 = first filter

 } // end for mi

// //   fullPathToME = pathsSummaryFolder_ + "HLT_A_PassPass";
// //   MonitorElement* ME_2d_Stream = dbe_->get(fullPathToME);
  fullPathToME = pathsSummaryFolder_ + "HLT_A_Pass_Any";
  MonitorElement* ME_1d_Stream = dbe_->get(fullPathToME);
// //  if(!ME_2d_Stream || !ME_1d_Stream) {  
  if(!ME_1d_Stream) {  

    LogTrace("TrigResRateMon") << " ME not valid although I gave full path" << endl;
    return;

  }
  else {

// //     TH2F * hist_2d_Stream = ME_2d_Stream->getTH2F();
    TH1F * hist_1d_Stream = ME_1d_Stream->getTH1F();
    
    int streamBinNumber = hist_1d_Stream->GetXaxis()->GetLast();

    bool acceptedStreamA = false;
    
    // loop over groups
    for (unsigned int i=0;i<groupAcceptPair.size();i++) {

     if(groupAcceptPair[i].second) {

       acceptedStreamA = true;

// //        int groupBinNumber_i = hist_2d_Stream->GetXaxis()->FindBin(groupAcceptPair[i].first.c_str()); 
       int groupBinNumber_i = hist_1d_Stream->GetXaxis()->FindBin(groupAcceptPair[i].first.c_str()); 
       //LogTrace("TrigResRateMon")  << "Accepted group X " << groupAcceptPair[i].first.c_str() << "    bin number " << groupBinNumber_i << endl;
       // Robin what about prescale for this one?
       hist_1d_Stream->Fill(groupBinNumber_i-1);//binNumber1 = 0 = first filter
// //        hist_2d_Stream->Fill(groupBinNumber_i-1,streamBinNumber-1);//binNumber1 = 0 = first filter
// //        hist_2d_Stream->Fill(streamBinNumber-1,groupBinNumber_i-1);//binNumber1 = 0 = first filter
    
// //        for (unsigned int j=0;j<groupAcceptPair.size();j++) {
    

// //         if(groupAcceptPair[j].second) {
    
// //           int groupBinNumber_j = hist_2d_Stream->GetXaxis()->FindBin(groupAcceptPair[j].first.c_str()); 
// //           //LogTrace("TrigResRateMon") << "Accepted group Y " << groupAcceptPair[j].first.c_str() << "    bin number " << groupBinNumber_j << endl;

// //           // fill StreamMatrix(i,j)
// //           hist_2d_Stream->Fill(groupBinNumber_i-1,groupBinNumber_j-1);//binNumber1 = 0 = first filter

// //         } // end if j-th group accepted
    
// //      } // end for j
    
     } // end if i-th group accepted
    
    } // end for i

    if(acceptedStreamA) {
      
//       hist_2d_Stream->Fill(streamBinNumber-1,streamBinNumber-1);//binNumber1 = 0 = first filter
      hist_1d_Stream->Fill(streamBinNumber-1);//binNumber1 = 0 = first filter
      
      passAny = true ;       //Robin

    }

 } // end else

}



void TrigResRateMon::fillCountsPerPath(const edm::Event& iEvent, const edm::EventSetup& iSetup) {


  if (jmsDebug) std::cout << "Filling counts per path" << std::endl;
  
  if (!triggerResults_.isValid()) {
    if (jmsDebug) std::cout << "Trigger Results not valid, sorry" << std::endl;
    return;
  }
  
  for (unsigned iName = 0; iName < hltConfig_.size(); iName++) {
    if ( triggerResults_ -> accept ( iName ) ){
      rawCountsPerPath[iName]++;

      //---Robin
      std::string thisName = hltConfig_.triggerName(iName);
   
      std::pair<int,int> psValueCombo =  hltConfig_.prescaleValues(iEvent, iSetup, thisName);
      // if ps OK, 
      if ( (psValueCombo.first > 0) && (psValueCombo.second > 0) ){
	finalCountsPerPath[iName] += psValueCombo.first * psValueCombo.second;
      } else {
	finalCountsPerPath[iName]++;
      }
      //-----------

      if ( (iName == referenceTrigIndex_) && (foundReferenceTrigger_) ) {
        // the get the prescales, and increment the PS*counts
        std::pair<int,int> psValueCombo =  hltConfig_.prescaleValues(iEvent, iSetup, referenceTrigName_);
        // if ps OK, 
        if ( (psValueCombo.first > 0) && (psValueCombo.second > 0) ){
          referenceTrigCountsPS_ += psValueCombo.first * psValueCombo.second;
        } else {
          referenceTrigCountsPS_++;
        }

      }// end if this is reference
    
      //Robin
      //     std::string thisName = hltConfig_.triggerName(iName);
//       TString checkName(thisName.c_str());
//       if (checkName.Contains("HLT_IsoMu24_v")){

//         std::pair<int,int> psValueCombo =  hltConfig_.prescaleValues(iEvent, iSetup, thisName);
//         // if ps OK, 
//         if ( (psValueCombo.first > 0) && (psValueCombo.second > 0) ){
//           testTrigCountsPS_ += psValueCombo.first * psValueCombo.second;
//         } else {
//           testTrigCountsPS_++;
//         }

//       }

    } // end if trig fired         
  }// end loop over paths

  // loop over all pds
  for (std::vector<DatasetInfo>::iterator iDS = primaryDataSetInformation.begin();
       iDS != primaryDataSetInformation.end();
       iDS++) {

    // now loop over all paths in the PD
    
    for (std::vector<std::string>::iterator iPath = iDS->pathNames.begin();
         iPath != iDS->pathNames.end();
         iPath++){
      
      unsigned trigIndex = hltConfig_.triggerIndex(*iPath);
      // did you pass the trigger?
      if ( triggerResults_ -> accept ( trigIndex ) ){

        // ok, you passed, increment the raw counts plot
        MonitorElement * thisRawCountsPlot = dbe_->get(iDS->rawCountsPerPathME_Name);
        if (thisRawCountsPlot){
          iDS->fillRawCountsForPath(thisRawCountsPlot, *iPath);
        } else {
          if (jmsDebug) std::cout << "sorry but couldn't find this raw counts plot"<< iDS->datasetName << std::endl;
        }
        


        // the get the prescales, and increment the PS*counts
        std::pair<int,int> psValueCombo =  hltConfig_.prescaleValues(iEvent, iSetup, *iPath);

        // if ps OK, 
        if ( (psValueCombo.first > 0) && (psValueCombo.second > 0) ){
          iDS->incrementCountsForPath(*iPath, psValueCombo.first*psValueCombo.second);
        } else {
          iDS->incrementCountsForPath(*iPath);
        }
        
      } 
    }// end for each path
      
  }// end for each pd

  

  
}

void TrigResRateMon::bookCountsPerPath() {

  
  for (unsigned iName = 0; iName < hltConfig_.size(); iName++) {
    
    rawCountsPerPath.push_back(0);
    finalCountsPerPath.push_back(0);  //Robin
    
  }  

  
}


void TrigResRateMon::findReferenceTriggerIndex() {

  if (jmsDebug) std::cout << "Looking for reference trigger " << referenceTrigInput_ << std::endl;
  for (unsigned iName = 0; iName < hltConfig_.size(); iName++) {
    
    std::string thisName = hltConfig_.triggerName(iName);
    TString tempThisName(thisName.c_str());
    if (tempThisName.Contains(referenceTrigInput_)){
      referenceTrigName_ = thisName;
      if (jmsDebug) std::cout << "Using Reference trigger " << referenceTrigName_ << std::endl;
      referenceTrigIndex_ = iName;
      foundReferenceTrigger_ = true;
      referenceTrigCountsPS_ = 0;
      break;
    }// end if name contains substring
  }  

  if (!foundReferenceTrigger_) {
    std::cout << "Sorry, we couldn't find a trigger like " << referenceTrigInput_ << std::endl;
  }
  
}


void TrigResRateMon::printCountsPerPathThisLumi() {

  std::cout << "===> COUNTS THIS LUMI <===" << std::endl;
  
  for (unsigned iName = 0; iName < hltConfig_.size() ; iName++) {
    std::cout << hltConfig_.triggerName(iName)
              << "  =  " << rawCountsPerPath[iName]
	      << "finalCounts  =  " << finalCountsPerPath[iName]  //Robin
              << std::endl;        
  }

  std::cout << "+++ Reference trigger " << referenceTrigName_ << " index " << referenceTrigIndex_ << " counts " << referenceTrigCountsPS_ << std::endl;

    // loop over all pds
  for (std::vector<DatasetInfo>::const_iterator iDS = primaryDataSetInformation.begin();
       iDS != primaryDataSetInformation.end();
       iDS++) {

    iDS->printCountsPerPath();

  }

  
}

void TrigResRateMon::clearCountsPerPath() {

  
//   for (unsigned iName = 0; iName < hltConfig_.size() ; iName++) {
    
//     rawCountsPerPath[iName] = 0;
//     finalCountsPerPath[iName] = 0;  //Robin
    
//   }

  referenceTrigCountsPS_ = 0 ;
  //  testTrigCountsPS_ = 0 ; //Robin
  
  for (std::vector<DatasetInfo>::iterator iDS = primaryDataSetInformation.begin();
       iDS != primaryDataSetInformation.end();
       iDS++) {
    iDS->clearCountsPerPath();
  }

  
}

void TrigResRateMon::clearLumiAverage() {

  averageInstLumi = 0;
    
}

void TrigResRateMon::addLumiToAverage(double lumi) {

  if (averageInstLumi == 0) {
    averageInstLumi = lumi;
  } else {
    averageInstLumi = (averageInstLumi + lumi) / 2;
  }
    
}

void TrigResRateMon::fillXsecPerDataset(const int& lumi) {

  // calculate the reference cross section

  double refTrigXSec = referenceTrigCountsPS_ / ( averageInstLumi * LSsize_);

  //  string fullpath = pathsSummaryFolder_ + "HLT_A_Pass_Any";
  //  MonitorElement * meStreamA = dbe_->get(fullpath);
  //  if (!meStreamA )  std::cout << "sorry but couldn't get the stream A ME" << std::endl;

  int iPD = 0;

  for (std::vector<DatasetInfo>::iterator iDS = primaryDataSetInformation.begin();
       iDS != primaryDataSetInformation.end();
       iDS++) {
    MonitorElement * thisXsecPlot = dbe_->get(iDS->xsecPerPathME_Name);
    MonitorElement * scaledXsecPlot = dbe_->get(iDS->scaledXsecPerPathME_Name);
    if (thisXsecPlot){
      iDS->fillXsecPlot(thisXsecPlot, averageInstLumi, LSsize_);
    } else {
      if (jmsDebug) std::cout << "sorry but couldn't find this xsec plot"<< iDS->datasetName << std::endl;
    }

    if (scaledXsecPlot){
      iDS->fillXsecPlot(scaledXsecPlot, averageInstLumi, LSsize_, refTrigXSec);
    } else {
      if (jmsDebug) std::cout << "sorry but couldn't find this scaled xsec plot"<< iDS->datasetName << std::endl;
    }

    ///PD rate plot
    MonitorElement * thisRatePlot = dbe_->get(iDS->ratePerLSME_Name);
    if (thisRatePlot) {

      double rate = rawCountsPerPD[iPD] / LSsize_ ;
      
      TH1F* rateHist = thisRatePlot->getTH1F();
      rateHist->SetBinContent(lumi, rate);
    }
    else {
      if (jmsDebug) std::cout << "sorry but couldn't find this rate plot"<< iDS->datasetName << std::endl;
    }
  
    rawCountsPerPD[iPD] = 0 ;
    iPD++;
  }
  
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

  ME_HLTAll_LS  = dbe_->book2D("AllSelectedPaths_count_LS",
                    "AllSelectedPaths paths rate [Hz]",
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
    sprintf(title, "Group %d, paths rate [Hz]",nLSHistos-nh);

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

//Robin
void TrigResRateMon::bookTestHisto()
{
 
  unsigned int npaths = testPaths_.size();

  //pathsSummaryHLTPathsPerLSFolder_ = TString("HLT/TrigResults/PathsSummary/HLT LS/");
  //dbe_->setCurrentFolder(pathsSummaryHLTPathsPerLSFolder_.c_str());
  //dbe_->setCurrentFolder(pathsSummaryHLTPathsPerLSFolder_);
  dbe_->setCurrentFolder(testPathsFolder_);

  TProfile tempProfile("XsecVsTestPath", "Xsec vs Test Path;;Xsec #mub", npaths, 0.5, npaths+0.5);
  meXsecPerTestPath = dbe_->bookProfile("XsecVsTestPath", &tempProfile);


  for(unsigned int i = 0; i < npaths; i++){
    const char* pname = testPaths_[i].c_str() ;
    TString pathname = testPaths_[i].c_str() ;
    meXsecPerTestPath->getTProfile()->GetXaxis()->SetBinLabel(i+1, pathname );
    
    /////
    char name[200];
    char title[200];

    sprintf(name, "path_%s_XsecVsLS",  pname);
    sprintf(title, "path_%s_XsecVsLS;LS;Xsec #mub", pname);

    MonitorElement* tempME  = dbe_->book1D(name,title, nLS_, 0, nLS_ );

    tempME->setAxisTitle("LS");

    v_ME_XsecPerLS.push_back(tempME); 

    char name2[200];
    char title2[200];

    sprintf(name2, "path_%s_countsVsLS",  pname);
    sprintf(title2, "path_%s_countsVsLS;LS;Counts", pname);

    MonitorElement* tempME2  = dbe_->book1D(name2, title2, nLS_, 0, nLS_ );

    tempME2->setAxisTitle("LS");

    v_ME_CountsPerLS.push_back(tempME2); 

  }

  MonitorElement* meXsec1 = dbe_->book1D("Xsec_HLT_IsoMu30_eta2p1", "HLT_IsoMu30_eta2p1 Xsec;Xsec #mub;number of LS", 10, 0.008, 0.012);
  MonitorElement* meXsec2 = dbe_->book1D("Xsec_HLT_Ele65_CaloIdVT_TrkIdT", "HLT_Ele65_CaloIdVT_TrkIdT Xsec;Xsec #mub;number of LS", 10, 0.002, 0.0025);
  MonitorElement* meXsec3 = dbe_->book1D("Xsec_HLT_MET200", "HLT_MET200 Xsec;Xsec #mub;number of LS", 10, 0.0004, 0.0008);
  MonitorElement* meXsec4 = dbe_->book1D("Xsec_HLT_Jet370", "HLT_Jet370 Xsec;Xsec #mub;number of LS", 10, 0.0006, 0.0008);
  MonitorElement* meXsec5 = dbe_->book1D("Xsec_HLT_HT600", "HLT_HT600 Xsec;Xsec #mub;number of LS", 10, 0.004, 0.005);
  MonitorElement* meXsec6 = dbe_->book1D("Xsec_HLT_Photon26_R9Id_Photon18_R9Id", "HLT_Photon26_R9Id_Photon18_R9Id Xsec;Xsec #mub;number of LS", 10, 0.002, 0.004);
  MonitorElement* meXsec7 = dbe_->book1D("Xsec_HLT_IsoMu15_eta2p1_LooseIsoPFTau20", "HLT_IsoMu15_eta2p1_LooseIsoPFTau20 Xsec;Xsec #mub;number of LS", 10, 0.0022, 0.003);
  MonitorElement* meXsec8 = dbe_->book1D("Xsec_HLT_PFMHT150", "HLT_PFMHT150 Xsec;Xsec #mub;number of LS", 10, 0.0005, 0.001);
  MonitorElement* meXsec9 = dbe_->book1D("Xsec_HLT_Photon90_CaloIdVL_IsoL", "HLT_Photon90_CaloIdVL_IsoL Xsec;Xsec #mub;number of LS", 10, 0.0015, 0.0025);


  v_ME_Xsec.push_back(meXsec1); 
  v_ME_Xsec.push_back(meXsec2); 
  v_ME_Xsec.push_back(meXsec3); 
  v_ME_Xsec.push_back(meXsec4); 
  v_ME_Xsec.push_back(meXsec5); 
  v_ME_Xsec.push_back(meXsec6); 
  v_ME_Xsec.push_back(meXsec7); 
  v_ME_Xsec.push_back(meXsec8); 
  v_ME_Xsec.push_back(meXsec9); 


}

void TrigResRateMon::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){   

   //int lumi = int(lumiSeg.id().luminosityBlock());
   //if(lumi < 74 || lumi > 77) fLumiFlag = false;
   //else fLumiFlag = true;

  if (jmsDebug) std::cout << "Inside begin lumi block" << std::endl;
  
  clearCountsPerPath();
  clearLumiAverage();
  nStream_ = 0 ;
  nPass_ = 0 ;

}

void TrigResRateMon::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c)
{

   int lumi = int(lumiSeg.id().luminosityBlock());
   LogTrace("TrigResRateMon") << " end lumiSection number " << lumi << endl;

//   countHLTPathHitsEndLumiBlock(lumi);
//   countHLTGroupHitsEndLumiBlock(lumi);
  //countHLTGroupL1HitsEndLumiBlock(lumi);
  //countHLTGroupBXHitsEndLumiBlock(lumi);

//   normalizeHLTMatrix();

  //if (jmsDebug) printCountsPerPathThisLumi();
  if (jmsDebug) printCountsPerPathThisLumi();
  if (jmsDebug) std::cout << "Average lumi is " << averageInstLumi << std::endl;

  if (averageInstLumi > 500) {
//     MonitorElement* reportSumME = dbe_->get("Info/EventInfo/reportSummaryMap" );
//       TH2F * reportSum = reportSumME->getTH2F();
//       float physDecl = reportSum->GetBinContent(lumi,26);
      
	fillXsecPerDataset(lumi);
	
	filltestHisto(lumi);  //Robin
  }
  //Robin-------Diagnostic plots--------
//   TH1F* tempXsecPerLS = meXsecPerLS->getTH1F();
//   double xsec = 1.0;
   
//   //  std::cout << "counts for HLT_IsoMu15* is " << testTrigCountsPS_ << std::endl;
  
//   if (averageInstLumi > 0) {
//     xsec = testTrigCountsPS_ / (averageInstLumi*LSsize_);
//   }    
//   //  std::cout << "LS is " << lumi << " ; xsec is "<< xsec << std::endl;
//   tempXsecPerLS->SetBinContent(lumi, xsec);
  

//   TH1F* tempXsec = meXsec->getTH1F();
//   tempXsec->Fill(xsec);

//   TProfile* tempXsecPerIL = meXsecPerIL->getTProfile();
//   tempXsecPerIL->Fill(averageInstLumi,xsec);

  //--------- stream counts and xsec
  TH1F* tempCountsStreamPerLS = meCountsStreamPerLS->getTH1F();
  //  std::cout << "number of stream counts is " << nStream_ << std::endl;
  tempCountsStreamPerLS->SetBinContent(lumi, nStream_);

  // dropped events
  MonitorElement* tempDroppedEvents = dbe_->get("SM_SMPS_Stats/droppedEventsCount_HLTTrigerResults DQM Consumer" );
  if (tempDroppedEvents) {
    TH1F* tempDiagnostic = meDiagnostic->getTH1F();
    if (tempDroppedEvents->kind() == MonitorElement::DQM_KIND_INT){
      tempDiagnostic->Fill(2);
      int64_t tempDroppedCounts =  tempDroppedEvents->getIntValue();
      int64_t currentDroppedCounts = tempDroppedCounts - TotalDroppedCounts;
      TotalDroppedCounts = tempDroppedCounts ;
      TH1F* tempCountsDroppedPerLS = meCountsDroppedPerLS->getTH1F();
      tempCountsDroppedPerLS->SetBinContent(lumi, currentDroppedCounts);
    }
    else     tempDiagnostic->Fill(1);
  }
  else {
    TH1F* tempDiagnostic = meDiagnostic->getTH1F();
    tempDiagnostic->Fill(0);
  }

  TH1F* tempXsecStreamPerLS = meXsecStreamPerLS->getTH1F();
  double xsecStream = 1.0 ;
  if (averageInstLumi > 0){
    xsecStream = nStream_ / (averageInstLumi*LSsize_);
    tempXsecStreamPerLS->SetBinContent(lumi, xsecStream);
  }

  TH1F* tempCountsPassPerLS = meCountsPassPerLS->getTH1F();
  //  std::cout << "number of passed stream counts is " << nPass_ << std::endl;
  tempCountsPassPerLS->SetBinContent(lumi, nPass_);

  //-----------

  // keep track of what you thought the lumi was
  // assign an error of 6%
  TH1F* tempLumiPerLS = meAverageLumiPerLS->getTH1F();
  tempLumiPerLS->SetBinContent(lumi, averageInstLumi);
  tempLumiPerLS->SetBinError(lumi, averageInstLumi*0.06);
  

}

//Robin----
void TrigResRateMon::filltestHisto(const int& lumi) {

  averageInstLumi3LS += averageInstLumi;

  if (lumi%3 == 0){
    unsigned int npaths = testPaths_.size();
    for(unsigned int i = 0; i < npaths; i++){
      TString pathname = testPaths_[i].c_str() ;
      pathname += "_v" ;
      
      int index = 0 ;
      int rawCount = 0 ;
      int finalCount = 0;
      double xsec = 0 ;

      
      //find the index for this test trigger path
      for (unsigned iName = 0; iName < hltConfig_.size(); iName++) {
	
	std::string thisName = hltConfig_.triggerName(iName);
	TString checkName(thisName.c_str());
	if (checkName.Contains(pathname)){
	  index = iName ;
	  //	std::cout << "==>test path name is " << checkName << std::endl;
	  break ;
	}
      }
      
      MonitorElement* testME_XsecPerLS = v_ME_XsecPerLS[i];
      MonitorElement* testME_rawCountsPerLS = v_ME_CountsPerLS[i];
      MonitorElement* testME_Xsec = v_ME_Xsec[i];
      //    TProfile tempProfile("XsecVsTestPath", "Xsec vs Test Path", npaths, 0.5, npaths+0.5);
      
      rawCount = rawCountsPerPath[index];
      finalCount = finalCountsPerPath[index];
      
      testME_rawCountsPerLS->getTH1F()->SetBinContent(lumi, rawCount);
      
      if (averageInstLumi > 0 ) {
	xsec = finalCount/ (averageInstLumi3LS*LSsize_); //averageInstLumi ???
	
	testME_XsecPerLS->getTH1F()->SetBinContent(lumi, xsec); 
	//      testME_rawCountsPerLS->getTH1F()->SetBinContent(lumi, rawCount);
	testME_Xsec->getTH1F()->Fill(xsec);    
	
	meXsecPerTestPath->getTProfile()->Fill(i+1,xsec);    
      }
    }
    
    for (unsigned iName = 0; iName < hltConfig_.size() ; iName++) {
      
      rawCountsPerPath[iName] = 0;
      finalCountsPerPath[iName] = 0;  //Robin
    }
    averageInstLumi3LS = 0 ;

  } // end if 3xLS

}
//----------

void TrigResRateMon::countHLTGroupBXHitsEndLumiBlock(const int& lumi)
{

 LogTrace("TrigResRateMon") << " countHLTGroupBXHitsEndLumiBlock() lumiSection number " << lumi << endl;

   if(! ME_HLT_BX) return;

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
   //for (unsigned int j=0;j<fGroupNamePathsPair.size();j++) 
   for (unsigned int j=0;j<v_ME_Total_BX.size();j++) 
   { 

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
   string fullPathToME = pathsSummaryFolder_ +  "HLT_" + fGroupNamePathsPair[i].first+ "_Pass_Any";
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
   string fullPathToME = pathsSummaryFolder_ + "HLT_" + fGroupNamePathsPair[i].first + "_Pass_Any";
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

    string binLabel = "Total "+pathname;
    
    LogTrace("TrigResRateMon") << " Looking for binLabel = " << binLabel <<  endl;

    // get the current count of path up to now
    int pathBin = hist_1d->GetXaxis()->FindBin(binLabel.c_str());      

    LogTrace("TrigResRateMon") << " pathBin = " << pathBin <<  "  out of histogram total number of bins " << hist_1d->GetNbinsX() <<  endl;
    if(pathBin == -1) {
      
      binLabel = pathname;
      int alternativePathBin = hist_1d->GetXaxis()->FindBin(binLabel.c_str());      

      if(alternativePathBin == -1) {

        LogTrace("TrigResRateMon") << " Cannot find the bin for path " << pathname << endl;

        continue;

      }
      else {

        pathBin = alternativePathBin;

      }

    }

    float currCount = hist_1d->GetBinContent(pathBin)/LSsize_;

    // count due to prev lumi sec is a difference bw current and previous
    float diffCount = currCount - prevCount;

    LogTrace("TrigResRateMon") << " lumi = " << lumi << "  path " << pathname << "  diffCount " << diffCount <<  endl;

    // set the counter in the pair to current count
    ip->second = currCount;  

    ////////////////////////////////////////////////////////
    // fill the 1D and 2D gruop and 2d_Stream_A count per LS
    ////////////////////////////////////////////////////////
    string fullPathToME_count = pathsSummaryHLTPathsPerLSFolder_ +"HLT_" + pathname + "_Total_LS";
    MonitorElement* ME_1d = dbe_->get(fullPathToME_count);

    string fullPathToME_2D_count = pathsSummaryHLTPathsPerLSFolder_ +"HLT_" + pathname + "_LS";
    MonitorElement* ME_2d = dbe_->get(fullPathToME_2D_count);

    string fullPathToME_Stream_A_2D_count = pathsSummaryHLTPathsPerLSFolder_ +"HLT_A_LS";
    MonitorElement* ME_Stream_A_2d = dbe_->get(fullPathToME_Stream_A_2D_count);

    if ( ME_1d && ME_2d && ME_Stream_A_2d) { 

      // update  the bin content  (must do that since events don't ncessarily come in the order

      float currentLumiCount = ME_1d->getTH1()->GetBinContent(lumi+1);
      float updatedLumiCount = currentLumiCount + diffCount;
      ME_1d->getTH1()->SetBinContent(lumi+1,updatedLumiCount);

      string groupBinLabel = "Total " + fGroupNamePathsPair[i].first;
      int groupBin = ME_2d->getTH2F()->GetYaxis()->FindBin(groupBinLabel.c_str());      
      if(groupBin != -1) ME_2d->getTH2F()->SetBinContent(lumi+1,groupBin,updatedLumiCount);
      
      // this is to deal with Stream A and bins with names of PDs
      groupBinLabel = fGroupNamePathsPair[i].first;
      groupBin = ME_Stream_A_2d->getTH2F()->GetYaxis()->FindBin(groupBinLabel.c_str());      
      if(groupBin != -1) ME_Stream_A_2d->getTH2F()->SetBinContent(lumi+1,groupBin,updatedLumiCount);

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
   string fullPathToME = pathsSummaryFolder_ + "HLT_AllSelectedPaths_PassPass";
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

void TrigResRateMon::normalizeHLTMatrix() {

  string fullPathToME; 

  // again, get hold of dataset names 
  //vector<string> datasetNames =  hltConfig_.datasetNames() ;
  vector<string> datasetNames =  hltConfig_.streamContent("A") ;

  // fill vectors of MEs needed in  normalization
  for (unsigned int i=0;i<datasetNames.size();i++) {

    fullPathToME = pathsSummaryFolder_ +"HLT_"+datasetNames[i]+"_PassPass";
    v_ME_HLTPassPass.push_back( dbe_->get(fullPathToME));

    fullPathToME = pathsSummaryHLTCorrelationsFolder_+"HLT_"+datasetNames[i]+"_PassPass_Normalized";
    v_ME_HLTPassPass_Normalized.push_back( dbe_->get(fullPathToME));

    fullPathToME = pathsSummaryHLTCorrelationsFolder_+"HLT_"+datasetNames[i]+"_Pass_Normalized_Any";
    v_ME_HLTPass_Normalized_Any.push_back( dbe_->get(fullPathToME));

  }

  // add stream MEs
  fullPathToME = pathsSummaryFolder_ +"HLT_A_PassPass";
  v_ME_HLTPassPass.push_back( dbe_->get(fullPathToME));

  fullPathToME = pathsSummaryHLTCorrelationsFolder_+"HLT_A_PassPass_Normalized";
  v_ME_HLTPassPass_Normalized.push_back( dbe_->get(fullPathToME));

  fullPathToME = pathsSummaryHLTCorrelationsFolder_+"HLT_A_Pass_Normalized_Any";
  v_ME_HLTPass_Normalized_Any.push_back( dbe_->get(fullPathToME));

  for (unsigned int i =0;i<v_ME_HLTPassPass.size();i++) {

    MonitorElement* ME_HLTPassPass = v_ME_HLTPassPass[i]; 
    MonitorElement* ME_HLTPassPass_Normalized = v_ME_HLTPassPass_Normalized[i]; 
    MonitorElement* ME_HLTPass_Normalized_Any = v_ME_HLTPass_Normalized_Any[i]; 

    if(!ME_HLTPassPass || !ME_HLTPassPass_Normalized || !ME_HLTPass_Normalized_Any) return;

    float passCount = 0;
    unsigned int nBinsX = ME_HLTPassPass->getTH2F()->GetNbinsX();
    unsigned int nBinsY = ME_HLTPassPass->getTH2F()->GetNbinsY();

    for(unsigned int binX = 0; binX < nBinsX+1; binX++) {
       
      passCount = ME_HLTPassPass->getTH2F()->GetBinContent(binX,binX);


      for(unsigned int binY = 0; binY < nBinsY+1; binY++) {

        if(passCount != 0) {

          // normalize each bin to number of passCount
          float normalizedBinContentPassPass = (ME_HLTPassPass->getTH2F()->GetBinContent(binX,binY))/passCount;
          //float normalizedBinContentPassFail = (ME_HLTPassFail_->getTH2F()->GetBinContent(binX,binY))/passCount;

          ME_HLTPassPass_Normalized->getTH2F()->SetBinContent(binX,binY,normalizedBinContentPassPass);
          //ME_HLTPassFail_Normalized_->getTH2F()->SetBinContent(binX,binY,normalizedBinContentPassFail);

          if(binX == nBinsX) {

            ME_HLTPass_Normalized_Any->getTH1F()->SetBinContent(binY,normalizedBinContentPassPass);

          }

        }
        else {

          ME_HLTPassPass_Normalized->getTH2F()->SetBinContent(binX,binY,0);
          //ME_HLTPassFail_Normalized_->getTH2F()->SetBinContent(binX,binY,0);

        } // end if else
     
      } // end for binY

    } // end for binX
  
  } // end for i

}

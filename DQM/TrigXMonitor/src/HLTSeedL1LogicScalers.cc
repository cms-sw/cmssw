
// $Id: HLTSeedL1LogicScalers.cc,v 1.9 2010/10/13 13:47:23 eulisse Exp $

#include "DQM/TrigXMonitor/interface/HLTSeedL1LogicScalers.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"



using namespace edm;
using namespace std;
using namespace trigger;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HLTSeedL1LogicScalers::HLTSeedL1LogicScalers(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  LogDebug("HLTSeedL1LogicScalers") << "constructor" ;

  // get untracked parameters
  ///////////////////////////
  fL1BeforeMask = iConfig.getParameter<bool>("l1BeforeMask");
  fProcessname  = iConfig.getParameter<std::string>("processname");

  // input tag for GT DAQ product
  fL1GtDaqReadoutRecordInputTag  = iConfig.getParameter<edm::InputTag>("L1GtDaqReadoutRecordInputTag");

  // input tag for GT lite product
  fL1GtRecordInputTag = iConfig.getParameter<edm::InputTag>("L1GtRecordInputTag");

  // get untracked parameters
  ///////////////////////////
  fDQMFolder    = iConfig.getUntrackedParameter("DQMFolder", string("HLT/HLTSeedL1LogicScalers/HLT_LogicL1"));
  fMonitorPaths = iConfig.getUntrackedParameter<std::vector<std::string > >("monitorPaths");

  fDbe = Service < DQMStore > ().operator->();

  if ( ! fDbe ) {
    LogInfo("HLTSeedL1LogicScalers") << "unable to get DQMStore service?";
  }
  else {
    fDbe->setCurrentFolder(fDQMFolder);
  }
  
}


HLTSeedL1LogicScalers::~HLTSeedL1LogicScalers()
{
}

void
HLTSeedL1LogicScalers::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  LogDebug("HLTSeedL1LogicScalers") << "HLTSeedL1LogicScalers::analyze  event " ;

  // before accessing any result from L1GtUtils, one must retrieve and cache
  // the L1 trigger event setup
  // add this call in the analyze / produce / filter method of your
  // analyzer / producer / filter

  m_l1GtUtils.retrieveL1EventSetup(iSetup);

  // loop over maps of ME-L1Algos
  for (unsigned int i=0;i<fMapMEL1Algos.size();i++) {

    MonitorElement* me = fMapMEL1Algos[i].first;
    LogTrace("HLTSeedL1LogicScalers") << "ME = " << me->getName() << endl;
    
    const vector<string>& l1Algos = fMapMEL1Algos[i].second;

    // word to bit-pack decisions of l1Algos
    unsigned int myL1Word = 0;

    // loop over l1Algos 
    for (unsigned int j=0;j<l1Algos.size();j++) {

      // check if this l1Algo passed
      bool l1Pass = analyzeL1GtUtils(iEvent, iSetup, l1Algos[j]);
      LogTrace("HLTSeedL1LogicScalers") << "l1Algo = " << l1Algos[j] << "  l1Pass = " <<  l1Pass << endl;
      if(l1Pass) {
        
        // bit-wise pack
        myL1Word |= (1<<j);

      }
/*
*/

    } 

    me->Fill(myL1Word);

  } // end for i maps

}



void 
HLTSeedL1LogicScalers::beginRun(const edm::Run& run, const edm::EventSetup& iSetup)
{

  // Get configuration of HLT menu used in this run
  LogTrace("HLTSeedL1LogicScalers") << "beginRun, run " << run.id();
  fDbe->setCurrentFolder(fDQMFolder);
  
  // HLT config does not change within runs!

  bool changed = false;
 
  if (!fHLTConfig.init(run, iSetup, fProcessname, changed)) {

    LogDebug("HLTSeedL1LogicScalers") << "HLTConfigProvider failed to initialize.";
    return;

    // check if trigger name in (new) config
    //  cout << "Available TriggerNames are: " << endl;
    //  fHLTConfig.dump("Triggers");

  }

  const unsigned int n(fHLTConfig.size());

  for (unsigned int j=0; j!=n; ++j) {
  
    LogTrace("HLTSeedL1LogicScalers") << "HLTConfig path " << fHLTConfig.triggerName(j) << endl;

  }

  // book histos for L1 logic of specificified HLT paths
  LogTrace("HLTSeedL1LogicScalers") << "size of vector of paths to monitor = " << fMonitorPaths.size() << endl;
  for (unsigned int iPath=0;iPath<fMonitorPaths.size();iPath++) {

    string monPath = fMonitorPaths[iPath];
    LogTrace("HLTSeedL1LogicScalers") << "monPath = " << monPath << endl;

    string folderName = fDQMFolder + "/" + monPath;
    fDbe->setCurrentFolder(folderName);
    
    // do nothing if monPath is not in the HLT menu
    if(fHLTConfig.triggerIndex(monPath) == fHLTConfig.size()) continue;
    // get L1SeedLogicalExpression of this path
    vector<pair<bool,string> > hltL1GTSeed = fHLTConfig.hltL1GTSeeds(monPath);
    LogTrace("HLTSeedL1LogicScalers") << endl << "size of vector of GTSeedL1LogicalExpression = " << hltL1GTSeed.size() << endl;

    // each GT Seed of each path contains l1Algos
    for (unsigned int iSeed=0;iSeed<hltL1GTSeed.size();iSeed++) {

      LogTrace("HLTSeedL1LogicScalers") << "  TechBit_flag = " << hltL1GTSeed[iSeed].first << "  GTSeedL1LogicalExpression = " << hltL1GTSeed[iSeed].second << endl;;

      istringstream totalSString( hltL1GTSeed[iSeed].second );
      string temp_string;

      vector<string> l1Algos;

      // only if not TechBit flag
      while(! hltL1GTSeed[iSeed].first) {

        totalSString >> temp_string;

        if(! l1Algos.empty()) {
          
          if(temp_string.compare(l1Algos.back()) == 0) break;

        }
        if(temp_string != "OR" && temp_string != "AND" && temp_string != "NOT" && temp_string != "(" && temp_string != ")" && temp_string != ""){

          l1Algos.push_back(temp_string);

        }
      }

      int nL1Algo = l1Algos.size();

      // put an upper limit on the size of l1Algos
      if(nL1Algo > 32) {

        LogWarning("HLTSeedL1LogicScalers") << "  number of l1 Algos grater than 32. Using only the first 32." << endl ;
        l1Algos.resize(32);

      }
      int nBins = 1 << nL1Algo;

      for (unsigned int k=0;k< l1Algos.size();k++) {

        LogTrace("HLTSeedL1LogicScalers") << "  l1 Algo = " << l1Algos[k] << endl;


      } // end for k

      //char title[100];
      //char name[100];

      std::stringstream title;
      std::stringstream name;

      name << monPath << "_Seed_" << iSeed << "_L1BitLogic";
      title << monPath << "  BitPacked L1Algos of GTSeed " << iSeed << ": '" << hltL1GTSeed[iSeed].second << "'";

      //sprintf(name,"%s_Seed_%d_L1BitLogic",monPath.c_str(),iSeed);
      //sprintf(title,"%s BitPacked L1Algos of GTSeed %d: '%s'",monPath.c_str(), iSeed, hltL1GTSeed[iSeed].second.c_str());

      LogTrace("HLTSeedL1LogicScalers") << "  MonitorElement name = " << name << endl;
      LogTrace("HLTSeedL1LogicScalers") << "  MonitorElement title = " << title << endl;
      LogTrace("HLTSeedL1LogicScalers") << "  MonitorElement nBins = " << nBins << endl << endl;

      MonitorElement* me = fDbe->book1D(name.str().c_str(), title.str().c_str(), nBins,0,nBins);
      me->setAxisTitle("bit-packed word L1 Algorithms");
      fMonitorPathsME.push_back(me);

      // pair of 1D hisotgram and vector of l1Algos
      //pair<MonitorElement*, vector<string> > pairMEL1Algo;
      //pairMEL1Algo.first = me;
      //pairMEL1Algo.second = l1Algos;
      //fMapMEL1Algos.push_back(pairMEL1Algo);
      fMapMEL1Algos.push_back(make_pair(me, l1Algos));

   } // end for Seeds



  } // end for monitoring paths

}

void 
HLTSeedL1LogicScalers::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTSeedL1LogicScalers::endJob() {
}

bool HLTSeedL1LogicScalers::analyzeL1GtUtils(const edm::Event& iEvent, const edm::EventSetup& evSetup, const string & l1AlgoName)
{

  LogTrace("HLTSeedL1LogicScalers") << "analyzeL1GtUtils..." << endl;

    // access L1 trigger results using public methods from L1GtUtils
    // always check on error code returned by that method

    int iErrorCode = -1;

    
    LogTrace("HLTSeedL1LogicScalers") << "l1AlgoName = " << l1AlgoName << endl;

    bool decisionAlgTechTrig = false;

    // check flag L1BeforeMask
    if (fL1BeforeMask) {
      
      decisionAlgTechTrig = m_l1GtUtils.decisionBeforeMask(iEvent, fL1GtRecordInputTag, fL1GtDaqReadoutRecordInputTag, l1AlgoName, iErrorCode);

    } 
    else {

     decisionAlgTechTrig = m_l1GtUtils.decisionAfterMask(iEvent, fL1GtRecordInputTag, fL1GtDaqReadoutRecordInputTag, l1AlgoName, iErrorCode);

    }

    LogTrace("HLTSeedL1LogicScalers") << "bool L1BeforeMask = " << fL1BeforeMask << "  decisionAlgTechTrig = " << decisionAlgTechTrig << endl;

    if (iErrorCode == 0) {
 
      return decisionAlgTechTrig; 

    } else if (iErrorCode == 1) {
        
      // algorithm / technical trigger  does not exist in the L1 menu
      LogWarning("HLTSeedL1LogicScalers") << "L1 algorithm " << l1AlgoName << " not in L1 menu, but HLTConfigProvider found it in L1SeedsLogicalExpression of at least one HLT path of the HLT menu." << endl;
      return false;

    } else {

      // error - see error code
      // do whatever needed
      return false;

    }

    return false;

}

bool HLTSeedL1LogicScalers::analyzeL1GtRecord(const edm::Event& iEvent, const edm::EventSetup& evSetup, string l1AlgoName)
{
  LogTrace("HLTSeedL1LogicScalers") << "analyzeL1GtRecord.. " << endl;
/*
  edm::Handle<L1GtTriggerMenu> menuRcd;
  evSetup.get<L1GtTriggerMenuRcd>().get(menuRcd) ;
  const L1GtTriggerMenu* menu = menuRcd.product();

  edm::Handle< L1GlobalTriggerReadoutRecord > gtReadoutRecord;
  iEvent.getByLabel( edm::InputTag("gtDigis"), gtReadoutRecord);
  const  DecisionWord& gtDecisionWordBeforeMask = = gtReadoutRecord->decisionWord();

  // test if the algorithm name is in the menu
  
  const AlgorithmMap& algorithmMap = menu->gtAlgorithmMap();
  CItAlgo itAlgo = algorithmMap.find(l1AlgoName);
  if (itAlgo != algorithmAliasMap.end()) {
      bool l1Result = menu->gtAlgorithmResult( l1AlgoName, gtDecisionWordBeforeMask);
      return l1Result;
  } else {
      // algorithm not in the L1 menu - do whatever needed
      return false;
  }
*/
 return false;

}

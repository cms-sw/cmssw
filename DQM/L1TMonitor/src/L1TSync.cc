/*
 * \file L1TSync.cc
 *
 * $Date: 2011/04/14 13:03:11 $
 * $Revision: 1.2 $
 * \author J. Pela, P. Musella
 *
 */

// 
#include "DQM/L1TMonitor/interface/L1TSync.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
// Not sure if needed
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h" // Parameters associated to Run, LS and Event

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"

// Luminosity Information
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"

// L1TMonitor includes
#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"
#include "DQM/L1TMonitor/interface/L1TOMDSHelper.h"

#include "TList.h"

using namespace edm;
using namespace std;

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
L1TSync::L1TSync(const ParameterSet & pset){

  m_parameters = pset;

  // Mapping parameter input variables
  m_scalersSource       = pset.getParameter         <InputTag>("inputTagScalersResults");
  m_l1GtDataDaqInputTag = pset.getParameter         <InputTag>("inputTagL1GtDataDaq");
  m_verbose             = pset.getUntrackedParameter<bool>    ("verbose",false);
  m_refPrescaleSet      = pset.getParameter         <int>     ("refPrescaleSet");  

  // Getting which categories to monitor
  ParameterSet Categories = pset.getParameter< ParameterSet >("Categories");

  bool forceGlobalParameters = Categories.getParameter<bool>("forceGlobalParameters");
  bool doGlobalAutoSelection = Categories.getParameter<bool>("doGlobalAutoSelection");

  ParameterSet CatMu      = Categories.getParameter<ParameterSet>("Mu");  
  ParameterSet CatEG      = Categories.getParameter<ParameterSet>("EG");  
  ParameterSet CatIsoEG   = Categories.getParameter<ParameterSet>("IsoEG");  
  ParameterSet CatJet     = Categories.getParameter<ParameterSet>("Jet");  
  ParameterSet CatCenJet  = Categories.getParameter<ParameterSet>("CenJet");  
  ParameterSet CatForJet  = Categories.getParameter<ParameterSet>("ForJet");  
  ParameterSet CatTauJet  = Categories.getParameter<ParameterSet>("TauJet");  
  ParameterSet CatETM     = Categories.getParameter<ParameterSet>("ETT");  
  ParameterSet CatETT     = Categories.getParameter<ParameterSet>("ETM");  
  ParameterSet CatHTT     = Categories.getParameter<ParameterSet>("HTT");  
  ParameterSet CatHTM     = Categories.getParameter<ParameterSet>("HTM");  

  // --> Setting parameters related to which algos to monitor
  // If global parameters are forced they take precedence over algo-by-algo parameters 
  if(forceGlobalParameters){

    // If global automatic selection if enable all categories set to be monitored will have
    // their algos auto selected (the lowest prescale algo will be selected) 
    if(doGlobalAutoSelection){
 
      if(CatMu    .getParameter<bool>("monitor")){m_algoAutoSelect["Mu"]     = true;}else{m_algoAutoSelect["Mu"]     = false;}
      if(CatEG    .getParameter<bool>("monitor")){m_algoAutoSelect["EG"]     = true;}else{m_algoAutoSelect["EG"]     = false;}
      if(CatIsoEG .getParameter<bool>("monitor")){m_algoAutoSelect["IsoEG"]  = true;}else{m_algoAutoSelect["IsoEG"]  = false;}
      if(CatJet   .getParameter<bool>("monitor")){m_algoAutoSelect["Jet"]    = true;}else{m_algoAutoSelect["Jet"]    = false;}
      if(CatCenJet.getParameter<bool>("monitor")){m_algoAutoSelect["CenJet"] = true;}else{m_algoAutoSelect["CenJet"] = false;}
      if(CatForJet.getParameter<bool>("monitor")){m_algoAutoSelect["ForJet"] = true;}else{m_algoAutoSelect["ForJet"] = false;}
      if(CatTauJet.getParameter<bool>("monitor")){m_algoAutoSelect["TauJet"] = true;}else{m_algoAutoSelect["TauJet"] = false;}
      if(CatETM   .getParameter<bool>("monitor")){m_algoAutoSelect["ETM"]    = true;}else{m_algoAutoSelect["ETM"]    = false;}
      if(CatETT   .getParameter<bool>("monitor")){m_algoAutoSelect["ETT"]    = true;}else{m_algoAutoSelect["ETT"]    = false;}
      if(CatHTM   .getParameter<bool>("monitor")){m_algoAutoSelect["HTM"]    = true;}else{m_algoAutoSelect["HTM"]    = false;}
      if(CatHTT   .getParameter<bool>("monitor")){m_algoAutoSelect["HTT"]    = true;}else{m_algoAutoSelect["HTT"]    = false;}

    // If global non-automatic selection is enable all categories set to be monitored will use
    // user defined algos
    }else{

      m_algoAutoSelect["Mu"]     = false;
      m_algoAutoSelect["EG"]     = false;
      m_algoAutoSelect["IsoEG"]  = false;
      m_algoAutoSelect["Jet"]    = false;
      m_algoAutoSelect["CenJet"] = false;
      m_algoAutoSelect["ForJet"] = false;
      m_algoAutoSelect["TauJet"] = false;
      m_algoAutoSelect["ETM"]    = false;
      m_algoAutoSelect["ETT"]    = false;
      m_algoAutoSelect["HTM"]    = false;
      m_algoAutoSelect["HTT"]    = false;   

      if(CatMu    .getParameter<bool>("monitor")){m_selectedTriggers["Mu"]        = CatMu    .getParameter<string>("algo");}
      if(CatEG    .getParameter<bool>("monitor")){m_selectedTriggers["EG"]        = CatEG    .getParameter<string>("algo");}
      if(CatIsoEG .getParameter<bool>("monitor")){m_selectedTriggers["IsoEG"]     = CatIsoEG .getParameter<string>("algo");}
      if(CatJet   .getParameter<bool>("monitor")){m_selectedTriggers["Jet"]       = CatJet   .getParameter<string>("algo");}
      if(CatCenJet.getParameter<bool>("monitor")){m_selectedTriggers["CenJet"]    = CatCenJet.getParameter<string>("algo");}
      if(CatForJet.getParameter<bool>("monitor")){m_selectedTriggers["CatForJet"] = CatForJet.getParameter<string>("algo");}
      if(CatTauJet.getParameter<bool>("monitor")){m_selectedTriggers["TauJet"]    = CatTauJet.getParameter<string>("algo");}
      if(CatETM   .getParameter<bool>("monitor")){m_selectedTriggers["ETM"]       = CatETM   .getParameter<string>("algo");}
      if(CatETT   .getParameter<bool>("monitor")){m_selectedTriggers["ETT"]       = CatETT   .getParameter<string>("algo");}
      if(CatHTM   .getParameter<bool>("monitor")){m_selectedTriggers["HTM"]       = CatHTM   .getParameter<string>("algo");}
      if(CatHTT   .getParameter<bool>("monitor")){m_selectedTriggers["HTT"]       = CatHTT   .getParameter<string>("algo");}

    }

  // If we are using algo-by-algo parametes we get them and set what is needed
  }else{

    if(CatMu.getParameter<bool>("monitor")){
      m_algoAutoSelect["Mu"] = CatMu.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["Mu"]){m_selectedTriggers["Mu"] = CatMu.getParameter<string>("algo");}
    }else{m_algoAutoSelect["Mu"] = false;}

    if(CatEG.getParameter<bool>("monitor")){
      m_algoAutoSelect["EG"] = CatEG.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["EG"]){m_selectedTriggers["EG"] = CatEG.getParameter<string>("algo");}
    }else{m_algoAutoSelect["EG"] = false;}

    if(CatIsoEG.getParameter<bool>("monitor")){
      m_algoAutoSelect["IsoEG"] = CatIsoEG.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["IsoEG"]){m_selectedTriggers["IsoEG"] = CatIsoEG.getParameter<string>("algo");}
    }else{m_algoAutoSelect["IsoEG"] = false;}

    if(CatJet.getParameter<bool>("monitor")){
      m_algoAutoSelect["Jet"] = CatJet.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["Jet"]){m_selectedTriggers["Jet"] = CatJet.getParameter<string>("algo");}
    }else{m_algoAutoSelect["Jet"] = false;}

    if(CatCenJet.getParameter<bool>("monitor")){
      m_algoAutoSelect["CenJet"] = CatCenJet.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["CenJet"]){m_selectedTriggers["CenJet"] = CatCenJet.getParameter<string>("algo");}
    }else{m_algoAutoSelect["CenJet"] = false;}

    if(CatForJet.getParameter<bool>("monitor")){
      m_algoAutoSelect["CatForJet"] = CatForJet.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["CatForJet"]){m_selectedTriggers["CatForJet"] = CatForJet.getParameter<string>("algo");}
    }else{m_algoAutoSelect["CatForJet"] = false;}

    if(CatTauJet.getParameter<bool>("monitor")){
      m_algoAutoSelect["TauJet"] = CatTauJet.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["TauJet"]){m_selectedTriggers["TauJet"] = CatTauJet.getParameter<string>("algo");}
    }else{m_algoAutoSelect["TauJet"] = false;}

    if(CatETM.getParameter<bool>("monitor")){
      m_algoAutoSelect["ETM"] = CatETM.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["ETM"]){m_selectedTriggers["ETM"] = CatETM.getParameter<string>("algo");}
    }else{m_algoAutoSelect["ETM"] = false;}

    if(CatETT.getParameter<bool>("monitor")){
      m_algoAutoSelect["ETT"] = CatETT.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["ETT"]){m_selectedTriggers["ETT"] = CatETT.getParameter<string>("algo");}
    }else{m_algoAutoSelect["ETT"] = false;}

    if(CatHTM.getParameter<bool>("monitor")){
      m_algoAutoSelect["HTM"] = CatHTM.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["HTM"]){m_selectedTriggers["HTM"] = CatHTM.getParameter<string>("algo");}
    }else{m_algoAutoSelect["HTM"] = false;}

    if(CatHTT.getParameter<bool>("monitor")){
      m_algoAutoSelect["HTT"] = CatHTT.getParameter<bool>("doAutoSelection");
      if(!m_algoAutoSelect["HTT"]){m_selectedTriggers["HTT"] = CatHTT.getParameter<string>("algo");}
    }else{m_algoAutoSelect["HTT"] = false;}

  }


  if (pset.getUntrackedParameter < bool > ("dqmStore", false)) {
    dbe = Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }

  m_outputFile = pset.getUntrackedParameter < std::string > ("outputFile","");

  if (m_outputFile.size() != 0) {
    std::cout << "L1T Monitoring histograms will be saved to " <<	m_outputFile.c_str() << std::endl;
  }

  bool disable = pset.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {m_outputFile = "";}

  if (dbe != NULL) {dbe->setCurrentFolder("L1T/L1TSync");}

}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
L1TSync::~L1TSync(){}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
void L1TSync::beginJob(void){

  // get hold of back-end interface
  DQMStore *dbe = 0;
  dbe = Service < DQMStore > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1T/L1TSync");
    dbe->rmdir("L1T/L1TSync");
  }
 
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
void L1TSync::endJob(void){

  if (m_verbose)
    cout << "L1TSync: end job...." << endl;

  if (m_outputFile.size() != 0 && dbe)
    dbe->save(m_outputFile);

  return;

}

//-------------------------------------------------------------------------------------
/// BeginRun
//-------------------------------------------------------------------------------------
void L1TSync::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){

  int maxNbins = 2001;

  // Cleaning in the begining of the run
  m_algoBit.clear();

  // Retriving parameters
  string oracleDB   = m_parameters.getParameter<string>("oracleDB");
  string pathCondDB = m_parameters.getParameter<string>("pathCondDB");

  ESHandle<L1GtTriggerMenu> menuRcd;
  iSetup.get<L1GtTriggerMenuRcd>().get(menuRcd);
  const L1GtTriggerMenu* menu = menuRcd.product();

  // Getting fill number for this run
  Handle<ConditionsInRunBlock> runConditions;
  iRun.getByType(runConditions);
  int lhcFillNumber = runConditions->lhcFillNumber;

  L1TOMDSHelper myOMDSHelper = L1TOMDSHelper();
  string conError = "";
  myOMDSHelper.connect(oracleDB,pathCondDB,conError);

  if(conError == ""){
    string errorRetrive = "";
    m_bunchStructure = myOMDSHelper.getBunchStructure(lhcFillNumber,errorRetrive);
    if(errorRetrive != ""){cout << "L1TRate: " << errorRetrive << endl;}
  }else{
    cout << "L1TRate: " << conError << endl;
  }

  //ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;
  //iSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);
  //const L1GtPrescaleFactors* m_l1GtPfAlgo = l1GtPfAlgo.product();

  for (CItAlgo algo = menu->gtAlgorithmAliasMap().begin(); algo!=menu->gtAlgorithmAliasMap().end(); ++algo){
    m_algoBit[(algo->second).algoAlias()] = (algo->second).algoBitNumber();
  }

  // Getting reference prescale factors
  //const vector<int>& RefPrescaleFactors = (*ListsPrescaleFactors).at(m_refPrescaleSet);

  L1TMenuHelper myMenuHelper = L1TMenuHelper(iSetup);  
         
  m_selectedTriggers = myMenuHelper.testAlgos(m_selectedTriggers);

  map<string,string> tAutoSelTrig = myMenuHelper.getLUSOTrigger(m_algoAutoSelect,m_refPrescaleSet);
  m_selectedTriggers.insert(tAutoSelTrig.begin(),tAutoSelTrig.end());

  // Initializing DQM Monitor Elements
  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

    string tCategory = (*i).first;
    string tTrigger  = (*i).second;

    dbe->setCurrentFolder("L1T/L1TSync/AlgoVsBunchStructure/");
    m_algoVsBunchStructure[tTrigger] = dbe->book2D(tCategory,"min #Delta("+tTrigger+",Bunch)",maxNbins,-0.5,double(maxNbins)-0.5,5,-2.5,2.5);
    m_algoVsBunchStructure[tTrigger] ->setAxisTitle("Lumi Section" ,1);
    
    dbe->setCurrentFolder("L1T/L1TSync/Certification/");
    m_algoCertification[tTrigger] = dbe->book1D(tCategory, "% of in of sync: "+tTrigger,maxNbins,-0.5,double(maxNbins)-0.5);
    m_algoCertification[tTrigger] ->setAxisTitle("Lumi Section" ,1);

 }   

}

//-------------------------------------------------------------------------------------
// Function: endLuminosityBlock
// * Fills LS by LS ration of trigger out of sync
//-------------------------------------------------------------------------------------
void L1TSync::endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {

  unsigned int eventLS = lumiBlock.id().luminosityBlock();

  // --> Fill LS by LS - ratio of trigger out of sync
  // Running over all selectec unprescaled algos 
  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

    string tTrigger = (*i).second;

    if(tTrigger != "Undefined" && tTrigger != "Undefined (Wrong Name)"){

      // Getting events with 0 bx difference between BPTX and Algo for current LS
      double CountSync = m_algoVsBunchStructure[tTrigger]->getBinContent(eventLS+1,3);
      double CountAll  = 0;
  
      // Adding all entries for current LS
      for(int a=1 ; a<6 ; a++){CountAll += m_algoVsBunchStructure[tTrigger]->getBinContent(eventLS+1,a);}

      // Filling this LS summary
      if(CountAll > 0){
        int binResult = m_algoCertification[tTrigger]->getTH1()->FindBin(eventLS);
        m_algoCertification[tTrigger]->setBinContent(binResult,CountSync/CountAll);
      }

    }else{
      int ibin = m_algoCertification[tTrigger]->getTH1()->FindBin(eventLS);
      m_algoCertification[tTrigger]->setBinContent(ibin,-1);
    }
  }

}

//_____________________________________________________________________
void L1TSync::endRun(const edm::Run& run, const edm::EventSetup& iSetup){}

//_____________________________________________________________________
void L1TSync::analyze(const Event & iEvent, const EventSetup & eventSetup){

  // Initializing Handles
  edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecordData;
 
  // Retriving data from the event
  iEvent.getByLabel(m_l1GtDataDaqInputTag, gtReadoutRecordData);

  unsigned int eventLS = iEvent.id().luminosityBlock();

  // --> Getting BX difference between BPTX and selected algos
  // Getting Final Decision Logic (FDL) Data from GT
  const vector<L1GtFdlWord>& gtFdlVectorData = gtReadoutRecordData->gtFdlVector();

  // Running over selected triggers
  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

    string tTrigger = (*i).second;

    if(tTrigger != "Undefined" && tTrigger != "Undefined (Wrong Name)"){
      
      bool firedAlgo = false;  // Algo fired in this event
      int  firedInBx = -1;      

      // Running over FDL results to get which bits fired
      for(unsigned int a=0 ; a<gtFdlVectorData.size() ; a++){
   
        // Selecting the 
        if(gtFdlVectorData[a].bxInEvent() == 0){
          if(gtFdlVectorData[a].gtDecisionWord()[ m_algoBit[tTrigger] ]){
            firedAlgo = true;
            firedInBx = gtFdlVectorData[a].localBxNr();
          }
        }
      }

      if(firedAlgo){

        int DifAlgoVsBunchStructure = 5; // Majorated

        for(unsigned int a=0 ; a<gtFdlVectorData.size() ; a++){

          int bxFDL     = gtFdlVectorData[a].localBxNr();
          int bxInEvent = gtFdlVectorData[a].bxInEvent();
          if(m_bunchStructure[bxFDL] && abs(bxInEvent)<abs(DifAlgoVsBunchStructure)){
            DifAlgoVsBunchStructure = -1*bxInEvent;
          }
        }

        m_algoVsBunchStructure[tTrigger]->Fill(eventLS,DifAlgoVsBunchStructure);

      }
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TSync);

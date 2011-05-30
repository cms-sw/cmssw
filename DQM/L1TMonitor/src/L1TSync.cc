/*
 * \file L1TSync.cc
 *
 * $Date: 2011/05/20 19:20:30 $
 * $Revision: 1.5 $
 * \author J. Pela, P. Musella
 *
 */

// 
#include "DQM/L1TMonitor/interface/L1TSync.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "DataFormats/Common/interface/ConditionsInEdm.h" // Parameters associated to Run, LS and Event

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"

// Luminosity Information
//#include "DataFormats/Luminosity/interface/LumiDetails.h"
//#include "DataFormats/Luminosity/interface/LumiSummary.h"

// L1TMonitor includes
#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"

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
  m_l1GtEvmSource       = pset.getParameter         <InputTag>("inputTagtEvmSource");
  m_verbose             = pset.getUntrackedParameter<bool>    ("verbose",false);
  m_refPrescaleSet      = pset.getParameter         <int>     ("refPrescaleSet");  

  // Getting which categories to monitor
  ParameterSet Categories = pset.getParameter< ParameterSet >("Categories");

  bool forceGlobalParameters = Categories.getParameter<bool>("forceGlobalParameters");
  bool doGlobalAutoSelection = Categories.getParameter<bool>("doGlobalAutoSelection");

  ParameterSet CatBPTX    = Categories.getParameter<ParameterSet>("BPTX");
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

    if(CatBPTX.getParameter<bool>("monitor")){
      m_selectedTriggers["BPTX"] = CatBPTX.getParameter<string>("algo");
    }

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

  // Initializing variables
  int maxNbins = 2501;

  // Reseting run dependent variables
  m_lhcFill = 0; 
  m_algoBit.clear();

  // Getting Trigger menu from GT
  ESHandle<L1GtTriggerMenu> menuRcd;
  iSetup.get<L1GtTriggerMenuRcd>().get(menuRcd);
  const L1GtTriggerMenu* menu = menuRcd.product();

  // Filling Alias-Bit Map
  for (CItAlgo algo = menu->gtAlgorithmAliasMap().begin(); algo!=menu->gtAlgorithmAliasMap().end(); ++algo){
    m_algoBit[(algo->second).algoAlias()] = (algo->second).algoBitNumber();
  }

  // Getting fill number for this run
  //Handle<ConditionsInRunBlock> runConditions;
  //iRun.getByType(runConditions);
  //int lhcFillNumber = runConditions->lhcFillNumber;
  //
  //ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;
  //iSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);
  //const L1GtPrescaleFactors* m_l1GtPfAlgo = l1GtPfAlgo.product();

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

  // If we have the information already retrived the information for the bunch 
  // structure we compute the LS by LS syncronization fraction
  if(m_beamConfig.isValid()){

    unsigned int eventLS = lumiBlock.id().luminosityBlock();

    // --> Fill LS by LS - ratio of trigger out of sync
    // Running over all selected unprescaled algos 
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

  // If we already have the LHC Fill Number and still do not have the bunch structure
  // we retry to getting if from OMDS
  if(m_lhcFill != 0 && !m_beamConfig.isValid()){

    // Retriving module parameters
    string oracleDB   = m_parameters.getParameter<string>("oracleDB");
    string pathCondDB = m_parameters.getParameter<string>("pathCondDB");

    // Connecting to OMDS 
    L1TOMDSHelper myOMDSHelper = L1TOMDSHelper();
    string conError = "";
    myOMDSHelper.connect(oracleDB,pathCondDB,conError);

    if(conError == ""){
      string errorRetrive = "";
      m_beamConfig = myOMDSHelper.getBeamConfiguration(m_lhcFill,errorRetrive);
      if(errorRetrive != ""){
        edm::LogError( "L1TSync" ) << errorRetrive << endl;
      }
    }else{
      edm::LogError( "L1TSync" ) << conError;
    }
  }

}

//_____________________________________________________________________
void L1TSync::endRun(const edm::Run& run, const edm::EventSetup& iSetup){}

//_____________________________________________________________________
void L1TSync::analyze(const Event & iEvent, const EventSetup & eventSetup){

  // If LHC Fill is absent for this run:
  // -> Retrive LHC Fill Number from GT
  // -> Retrive from OMDS the bunch structure for this fill
  if(m_lhcFill == 0){

    // Retriving module parameters
    string oracleDB   = m_parameters.getParameter<string>("oracleDB");
    string pathCondDB = m_parameters.getParameter<string>("pathCondDB");

    //Retriving LHC Fill number from GT
    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord;
    iEvent.getByLabel(m_l1GtEvmSource, gtEvmReadoutRecord);

    if(gtEvmReadoutRecord.isValid()){

      const L1GtfeExtWord& gtfeEvmWord = gtEvmReadoutRecord ->gtfeWord();
      m_lhcFill = gtfeEvmWord.lhcFillNumber();

      // Connecting to OMDS 
      L1TOMDSHelper myOMDSHelper = L1TOMDSHelper();
      string conError = "";
      myOMDSHelper.connect(oracleDB,pathCondDB,conError);

      if(conError == ""){
        string errorRetrive = "";
        m_beamConfig = myOMDSHelper.getBeamConfiguration(m_lhcFill,errorRetrive);
        if(errorRetrive != ""){
          edm::LogError( "L1TSync" ) << errorRetrive << endl;
        }
      }else{
        edm::LogError( "L1TSync" ) << conError;
      }
    }
    else{edm::LogError( "L1TSync" ) << "ERROR: Unable to retrive L1GlobalTriggerEvmReadoutRecord";}
  }

  if(m_beamConfig.isValid()){

    unsigned int eventLS = iEvent.id().luminosityBlock();

    // Getting Final Decision Logic (FDL) Data from GT
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecordData;
    iEvent.getByLabel(m_l1GtDataDaqInputTag, gtReadoutRecordData);

    if(gtReadoutRecordData.isValid()){

      const vector<L1GtFdlWord>& gtFdlVectorData = gtReadoutRecordData->gtFdlVector();

      // Running over selected triggers
      for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

        string tTrigger = (*i).second;

        // Analyse only defined triggers
        if(tTrigger != "Undefined" && tTrigger != "Undefined (Wrong Name)"){

          bool beamSingleConfig = false; // Single beam configured for this event
          bool firedAlgo        = false; // Algo fired in this event
          int  eventBx        = -1;      

          // Running over FDL results to get which bits fired
          for(unsigned int a=0 ; a<gtFdlVectorData.size() ; a++){
   
            // Selecting the FDL that triggered
            if(gtFdlVectorData[a].bxInEvent() == 0){
              eventBx = gtFdlVectorData[a].localBxNr();
              if(gtFdlVectorData[a].gtDecisionWord()[ m_algoBit[tTrigger] ]){firedAlgo = true;}
            }
          }

          // Checking beam configuration
          if( m_beamConfig.beam1[eventBx] && !m_beamConfig.beam2[eventBx]){beamSingleConfig = true;}
          if(!m_beamConfig.beam1[eventBx] &&  m_beamConfig.beam2[eventBx]){beamSingleConfig = true;}

          // Analyse only if this trigger fired in this event
          // NOTE: Veto cases where a single beam is configured since
          //       for this cases this could be a real-satelite bunch collision
          // -> Calculate the minimum bx diference between this event and a configured bx
          if(firedAlgo && !beamSingleConfig){

            int DifAlgoVsBunchStructure = 9999; // Majorated

            for(unsigned int a=0 ; a<gtFdlVectorData.size() ; a++){

              int bxFDL     = gtFdlVectorData[a].localBxNr();
              int bxInEvent = gtFdlVectorData[a].bxInEvent();

              if(m_beamConfig.bxConfig(bxFDL) && abs(bxInEvent)<abs(DifAlgoVsBunchStructure)){
                DifAlgoVsBunchStructure = -1*bxInEvent;
              }
            }

            m_algoVsBunchStructure[tTrigger]->Fill(eventLS,DifAlgoVsBunchStructure);

          }
        }
      }
    }
    else{edm::LogError( "L1TSync" ) << "ERROR: Unable to retrive L1GlobalTriggerReadoutRecord";}

  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TSync);

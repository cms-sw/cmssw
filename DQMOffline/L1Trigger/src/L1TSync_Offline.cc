/**
 * \class L1TSync_Offline
 *
 *
 * Description: offline DQM module for L1Trigger/bunchStructure synchronization
 * 
 * Implementation:
 *   <TODO: enter implementation details>
 *
 * \author: Pietro Vischia - LIP Lisbon pietro.vischia@gmail.com
 *
 * Changelog:
 *    2012/08/10 11:01:01: First creation. Dummy module with actual code commented.
 *
 * Todo:
 *  - implement the module in offline
 *  - check if there are user includes specific for offline/online that should be changed
 *
 *
 */

// 

// This class header
#include "DQMOffline/L1Trigger/interface/L1TSync_Offline.h"

// System include files
// --

// User include files
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"

#include "DataFormats/Common/interface/ConditionsInEdm.h" // Parameters associated to Run, LS and Event

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
///
// Luminosity Information
//#include "DataFormats/Luminosity/interface/LumiDetails.h"
//#include "DataFormats/Luminosity/interface/LumiSummary.h"

// L1TMonitor includes
#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"
//#include "DQMOffline/L1Trigger/interface/L1TMenuHelper.h"

#include "TList.h"

using namespace edm;
using namespace std;

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
L1TSync_Offline::L1TSync_Offline(const ParameterSet & pset) :
  m_l1GtUtils(pset, consumesCollector(), false, *this) {

  m_parameters = pset;
  
  // Mapping parameter input variables
  m_l1GtDataDaqInputTag = consumes<L1GlobalTriggerReadoutRecord>(pset.getParameter         <InputTag>("inputTagL1GtDataDaq"));
  m_l1GtEvmSource       = consumes<L1GlobalTriggerEvmReadoutRecord>(pset.getParameter      <InputTag>("inputTagtEvmSource"));
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

  m_outputFile = pset.getUntrackedParameter < std::string > ("outputFile","");

  if (m_outputFile.size() != 0) {
    std::cout << "L1T Monitoring histograms will be saved to " <<	m_outputFile.c_str() << std::endl;
  }

  bool disable = pset.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {m_outputFile = "";}
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
L1TSync_Offline::~L1TSync_Offline(){}

//-------------------------------------------------------------------------------------

void L1TSync_Offline::dqmBeginRun(edm::Run const&, edm::EventSetup const&){

}
//-------------------------------------------------------------------------------------
/// BeginRun
//-------------------------------------------------------------------------------------
void L1TSync_Offline::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& iRun, const edm::EventSetup& iSetup){

  if (m_verbose){cout << "[L1TSync_Offline] Called beginRun." << endl;}

  // Initializing variables
  int maxNbins = 2501;

  // Reseting run dependent variables
  m_lhcFill     = 0; 
  m_currentLS   = 0;
  m_certFirstLS.clear();
  m_certLastLS .clear();
  
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

  m_l1GtUtils.retrieveL1EventSetup(iSetup);
  map<string,string> tAutoSelTrig = myMenuHelper.getLUSOTrigger(m_algoAutoSelect, m_refPrescaleSet, m_l1GtUtils);
  m_selectedTriggers.insert(tAutoSelTrig.begin(),tAutoSelTrig.end());

  // Initializing DQM Monitor Elements
  ibooker.setCurrentFolder("L1T/L1TSync");
  m_ErrorMonitor = ibooker.book1D("ErrorMonitor","ErrorMonitor",7,0,7);
  m_ErrorMonitor->setBinLabel(UNKNOWN                      ,"UNKNOWN");
  m_ErrorMonitor->setBinLabel(WARNING_DB_CONN_FAILED       ,"WARNING_DB_CONN_FAILED");        // Errors from L1TOMDSHelper
  m_ErrorMonitor->setBinLabel(WARNING_DB_QUERY_FAILED      ,"WARNING_DB_QUERY_FAILED");       // Errors from L1TOMDSHelper
  m_ErrorMonitor->setBinLabel(WARNING_DB_INCORRECT_NBUNCHES,"WARNING_DB_INCORRECT_NBUNCHES"); // Errors from L1TOMDSHelper
  m_ErrorMonitor->setBinLabel(ERROR_UNABLE_RETRIVE_PRODUCT ,"ERROR_UNABLE_RETRIVE_PRODUCT"); 
  m_ErrorMonitor->setBinLabel(ERROR_TRIGGERALIAS_NOTVALID  ,"ERROR_TRIGGERALIAS_NOTVALID"); 
  m_ErrorMonitor->setBinLabel(ERROR_LSBLOCK_NOTVALID       ,"ERROR_LSBLOCK_NOTVALID"); 

  // Looping over selected triggers
  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

    string tCategory = (*i).first;
    string tTrigger  = (*i).second;

    // Initializing LS blocks for certification
    m_certFirstLS[(*i).second] = 0;
    m_certLastLS [(*i).second] = 0;

    // Initializing DQM Monitors 
    ibooker.setCurrentFolder("L1T/L1TSync/AlgoVsBunchStructure/");
    m_algoVsBunchStructure[tTrigger] = ibooker.book2D(tCategory,"min #Delta("+tTrigger+",Bunch)",maxNbins,-0.5,double(maxNbins)-0.5,5,-2.5,2.5);
    m_algoVsBunchStructure[tTrigger] ->setAxisTitle("Lumi Section" ,1);
    
    ibooker.setCurrentFolder("L1T/L1TSync/Certification/");
    m_algoCertification[tTrigger] = ibooker.book1D(tCategory, "fraction of in sync: "+tTrigger,maxNbins,-0.5,double(maxNbins)-0.5);
    m_algoCertification[tTrigger] ->setAxisTitle("Lumi Section" ,1);

 }   
}

 //_____________________________________________________________________
 // Function: beginLuminosityBlock
 //_____________________________________________________________________
 void L1TSync_Offline::beginLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
 
   if (m_verbose){cout << "[L1TSync_Offline] Called beginLuminosityBlock." << endl;}
 
   m_currentLSValid = true; 
 
 }
 
//_____________________________________________________________________
void L1TSync_Offline::analyze(const Event & iEvent, const EventSetup & eventSetup){


  if(m_verbose){cout << "[L1TSync_Offline] Called analyze." << endl;}

  // We only start analyzing if current LS is still valid
  if(m_currentLSValid){
  
    if(m_verbose){cout << "[L1TSync_Offline] -> m_currentLSValid=" << m_currentLSValid << endl;}
    
    // Retriving information from GT
    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord;
    iEvent.getByToken(m_l1GtEvmSource, gtEvmReadoutRecord);

    // Determining beam mode and fill number
    if(gtEvmReadoutRecord.isValid()){

      const L1GtfeExtWord& gtfeEvmWord = gtEvmReadoutRecord->gtfeWord();
      unsigned int lhcBeamMode         = gtfeEvmWord.beamMode();          // Updating beam mode

      if(m_lhcFill == 0){
        m_lhcFill = gtfeEvmWord.lhcFillNumber(); // Getting LHC Fill Number from GT

	//I AM HERE
	getBeamConfOffline(iEvent);                       // Getting Beam Configuration from OMDS
	// no OMDS //
	/* (Savannah ticket https://savannah.cern.ch/task/?31857 )
	  Purged the OMDS helper from the module, since apparently we don't have access to that information from Offline.
	  The comparison with the closest BPTX trigger will be performed via conditions objects which are being implemented by Joao Pela.
	  In the meantime, for being able to test the module, he suggested to pass the correct bunch structure by hand for the specific run which is run during the test.
	  The idea of the temporary fix is setting the vector variable that stores the bunch structure either by hand or filling it with the BPTX fires (that is check for the event the BX's where tech0 fired and set those as true in the vector.
	*/
	// no OMDS //    m_beamConfig = myOMDSHelper.getBeamConfiguration(m_lhcFill,errorRetrieve); Yuhuuu!!!OB asked Joao Pela how to fetch that


	
	
      }

      if(lhcBeamMode != STABLE){m_currentLSValid = false;
	if(m_verbose){cout << "[L1TSync_Offline] -> m_currentLSValid=" << m_currentLSValid << "because beams mode not stable, being " << lhcBeamMode << endl;}
      } // If Beams are not stable we invalidate this LS
    }else{
      int eCount = m_ErrorMonitor->getTH1()->GetBinContent(ERROR_UNABLE_RETRIVE_PRODUCT);
      eCount++;
      m_ErrorMonitor->getTH1()->SetBinContent(ERROR_UNABLE_RETRIVE_PRODUCT,eCount);
    }
  }else{
    if(m_verbose){cout << "[L1TSync_Offline] -> m_currentLSValid=" << m_currentLSValid << endl;}
  }

  // Commenting out un-necessary  print outs (S.Dutta)
  /*  for(size_t l=0; l<m_beamConfig.beam1.size(); l++){
    cout << "Beam 1: element " << l << " is in state " << m_beamConfig.beam1[l] << " --- Beam 2: element " << l << " is in state " << m_beamConfig.beam2[l] << endl;
    }*/
  //------------------------------------------------------------------------------
  // If current LS is valid and Beam Configuration is Valid we analyse this event
  //------------------------------------------------------------------------------
  if(m_currentLSValid && m_beamConfig.isValid()){     

    // Getting Final Decision Logic (FDL) Data from GT
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecordData;
    iEvent.getByToken(m_l1GtDataDaqInputTag, gtReadoutRecordData);

    if(gtReadoutRecordData.isValid()){

      const vector<L1GtFdlWord>& gtFdlVectorData = gtReadoutRecordData->gtFdlVector();

      // Running over selected triggers
      for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

        string tTrigger = (*i).second;

        // Analyse only defined triggers
        if(tTrigger != "Undefined" && tTrigger != "Undefined (Wrong Name)"){

          bool beamSingleConfig = false; // Single beam configured for this event
          bool firedAlgo        = false; // Algo fired in this event
          unsigned int  eventBx = ~0;      

          // Running over FDL results to get which bits fired
          for(unsigned int a=0 ; a<gtFdlVectorData.size() ; a++){
   
            // Selecting the FDL that triggered
            if(gtFdlVectorData[a].bxInEvent() == 0){
              eventBx = gtFdlVectorData[a].localBxNr();
              if(gtFdlVectorData[a].gtDecisionWord()[ m_algoBit[tTrigger] ]){firedAlgo = true;}
            }
          }

          // Checking beam configuration
          if ( m_beamConfig.beam1.size() > eventBx && m_beamConfig.beam2.size() > eventBx) {
	    if( m_beamConfig.beam1[eventBx] && !m_beamConfig.beam2[eventBx]){beamSingleConfig = true;}
	    if(!m_beamConfig.beam1[eventBx] &&  m_beamConfig.beam2[eventBx]){beamSingleConfig = true;}
          } 
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

            m_algoVsBunchStructure[tTrigger]->Fill(m_currentLS,DifAlgoVsBunchStructure);

          }
        }
      }
    }
    else{
      int eCount = m_ErrorMonitor->getTH1()->GetBinContent(ERROR_UNABLE_RETRIVE_PRODUCT);
      eCount++;
      m_ErrorMonitor->getTH1()->SetBinContent(ERROR_UNABLE_RETRIVE_PRODUCT,eCount);
    }

  }
}

//_____________________________________________________________________
// Method: getBunchStructureOffline
// Description: Attempt to retrive Beam Configuration from Offline and if 
//              we find error handle it
//_____________________________________________________________________
void L1TSync_Offline::getBeamConfOffline(const Event& iEvent){

  //Getting connection parameters
  //  [11:21:35] Pietro Vischia: pathCondDB and oracleDB are available in offline?
  //  [11:21:51] Joao Pela: no
  //  [11:21:56] Pietro Vischia: cool
  //  [11:21:58] Pietro Vischia: tks
  //  [11:22:00] Joao Pela: well
  //  [11:22:02] Joao Pela: maybe
  //  [11:22:05] Joao Pela: but assume no
  //  [11:22:22] Joao Pela: definitely assume no
  // *** UPDATE: now we have a DB for Rate parameters, it would be useful to have s.th also for L1TSync

  // string oracleDB   = m_parameters.getParameter<string>("oracleDB");
  // string pathCondDB = m_parameters.getParameter<string>("pathCondDB");
  
  //  m_beamConfig = myOMDSHelper.getBeamConfiguration(m_lhcFill,errorRetrieve);
  //  m_lhcFill -> lhcFillNumber
  //  errorRetrieve -> error
  // m_beamConfig -> bConfig
  

  // No error codes at the moment. Taking info from BPTX fires
  //  int errorRetrieve = 0; // NO_ERROR;
  m_beamConfig.m_valid = true;



  //  bool beamSingleConfig = false; // Single beam configured for this event
  bool firedAlgo        = false; // Algo fired in this event
  //  int  eventBx          = -1;      
  
  // Running over FDL results to get which bits fired for BPTX (temporary fix Savannah ticket https://savannah.cern.ch/task/?31857 )
  // Getting Final Decision Logic (FDL) Data from GT
  edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecordData;
  iEvent.getByToken(m_l1GtDataDaqInputTag, gtReadoutRecordData);
  
  if(gtReadoutRecordData.isValid()){
    
    const vector<L1GtFdlWord>& gtFdlVectorData = gtReadoutRecordData->gtFdlVector();
    
    for(unsigned int a=0 ; a<gtFdlVectorData.size() ; a++){
      // Selecting the FDL that triggered
      if(gtFdlVectorData[a].bxInEvent() == 0){
	//eventBx = gtFdlVectorData[a].localBxNr();
	if(gtFdlVectorData[a].gtDecisionWord()[ m_algoBit[ m_selectedTriggers["BPTX"] ] ]){firedAlgo = true;}
      }
      
      // Fill beam status with BPTX fires
      if(firedAlgo){
	m_beamConfig.beam1.push_back(true);
	m_beamConfig.beam2.push_back(true);
      }
      else {
	m_beamConfig.beam1.push_back(false);
	m_beamConfig.beam2.push_back(false);
      }
      // End fill beam status with BPTX fires
    } // End loop on FDL
  } // End if readout is valid  
  
}



//_____________________________________________________________________
// Method: doFractionInSync
// Description: Produce plot with the fraction of in sync trigger for
//              LS blocks with enough statistics.
// Variable: iForce - Forces closing of all blocks and calculation of 
//                    the respective fractions
// Variable: iBad   - (Only works with iForce=true) Forces the current 
//                    all current blocks to be marked as bad 
//_____________________________________________________________________
//void L1TSync_Offline::doFractionInSync(bool iForce,bool iBad){
//  
/////  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){
/////
/////    string theCategory     = (*i).first;
/////    string theTriggerAlias = (*i).second;
/////
/////    // Caching frequently used values from maps
/////    unsigned int fLS = m_certFirstLS[theTriggerAlias];
/////    unsigned int lLS = m_certLastLS [theTriggerAlias];
/////    
/////    // Checking validity of the trigger alias and of the LS block
/////    bool triggerAlias_isValid = theTriggerAlias != "Undefined" && theTriggerAlias != "Undefined (Wrong Name)";
/////    bool lsBlock_exists       = !(fLS == 0 && lLS == 0);
/////    bool lsBlock_isValid      = fLS <= lLS && fLS > 0 && lLS > 0; 
/////
/////    if(triggerAlias_isValid && lsBlock_exists && lsBlock_isValid){
/////
/////      // If we are forced to close blocks and mark them bad
/////      if(iForce && iBad){
/////        certifyLSBlock(theTriggerAlias,fLS,lLS,-1);
/////        m_certFirstLS[theTriggerAlias] = 0;
/////        m_certLastLS [theTriggerAlias] = 0;
/////      }
/////
/////      // If we are not forced to mark bad, we check if we have enough statistics
/////      else{
/////
/////        // Getting events with 0 bx difference between BPTX and Algo for current LS
/////        double CountSync = 0;
/////        double CountAll  = 0;
/////  
/////        // Adding all entries for current LS block
/////        for(unsigned int ls=fLS ; ls<=lLS ; ls++){
/////
/////          CountSync += m_algoVsBunchStructure[theTriggerAlias]->getBinContent(ls+1,3);
/////          for(int a=1 ; a<6 ; a++){
/////            CountAll  += m_algoVsBunchStructure[theTriggerAlias]->getBinContent(ls+1,a);
/////          }
/////        }
/////
/////        if(m_verbose){
/////          cout << "Alias = " << theTriggerAlias 
/////               << " InitLS=" << fLS 
/////               << " EndLS=" <<  lLS 
/////               << " Events=" << CountAll ;
/////        } 
/////
/////        if(iForce ||
/////           CountAll >= m_parameters.getParameter<ParameterSet>("Categories")
/////                                   .getParameter<ParameterSet>(theCategory)
/////                                   .getParameter<int>("CertMinEvents")){
/////
/////          if(m_verbose){cout << " <--------------- Enough Statistics: ";}
/////
/////        
/////          // Calculating fraction of in time 
/////          double fraction = 0;
/////          if(CountAll >0){fraction = CountSync/CountAll;}
/////
/////          // This is to avoid having an entry equal to zero and thus
/////          // disregarded by the automatic tests
/////          if(fraction==0){fraction=0.000001;}
/////        
/////          certifyLSBlock(theTriggerAlias,fLS,lLS,fraction);
/////          m_certFirstLS[theTriggerAlias] = 0;
/////          m_certLastLS [theTriggerAlias] = 0;
/////        }
/////
/////        if(m_verbose){cout << endl;}
/////
/////      }
/////    }
/////
/////    // A problem was found. We report it and set a not physical vale (-1) to the certification plot
/////    else{
/////
/////      // If trigger alias is not valid report it to m_ErrorMonitor
/////      if(!triggerAlias_isValid){
/////        int eCount = m_ErrorMonitor->getTH1()->GetBinContent(ERROR_TRIGGERALIAS_NOTVALID);
/////        eCount++;
/////        m_ErrorMonitor->getTH1()->SetBinContent(ERROR_TRIGGERALIAS_NOTVALID,eCount);
/////        certifyLSBlock(theTriggerAlias,fLS,lLS,-1);
/////        m_certFirstLS[theTriggerAlias] = 0;
/////        m_certLastLS [theTriggerAlias] = 0;
/////      }
/////
/////      // If LS Block is not valid report it to m_ErrorMonitor
/////      if(lsBlock_exists && !lsBlock_isValid){
/////        int eCount = m_ErrorMonitor->getTH1()->GetBinContent(ERROR_LSBLOCK_NOTVALID);
/////        eCount++;
/////        m_ErrorMonitor->getTH1()->SetBinContent(ERROR_LSBLOCK_NOTVALID,eCount);
/////        certifyLSBlock(theTriggerAlias,fLS,lLS,-1);
/////        m_certFirstLS[theTriggerAlias] = 0;
/////        m_certLastLS [theTriggerAlias] = 0;
/////      }
/////
/////    }
/////  }
/////
//}


//_____________________________________________________________________
// Method: certifyLSBlock
// Description: Fill the trigger certification plot by blocks
// Variable: iTrigger - Which trigger to certify
// Variable: iInitLs  - Blocks initial LS
// Variable: iEndLs   - Blocks end LS
// Variable: iValue   - Value to be used to fill
//_____________________________________________________________________
//void L1TSync_Offline::certifyLSBlock(string iTrigger, int iInitLs, int iEndLs ,float iValue){
//  
/////  // Finding correct bins in the histogram for this block
/////  int binInit = m_algoCertification[iTrigger]->getTH1()->FindBin(iInitLs);
/////  int binEnd  = m_algoCertification[iTrigger]->getTH1()->FindBin(iEndLs);
/////
/////  for(int ls=binInit ; ls<=binEnd ; ls++){
/////    m_algoCertification[iTrigger]->setBinContent(ls,iValue);
/////  }
//
//}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TSync_Offline);

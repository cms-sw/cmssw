/*
 * \file L1TSync.cc
 *
 * $Date: 2011/04/06 16:49:34 $
 * $Revision: 1.1 $
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

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"

// Luminosity Information
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"

// My includes
#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"

#include "TList.h"

using namespace edm;
using namespace std;

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
L1TSync::L1TSync(const ParameterSet & pset){

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

  m_outputFile = pset.getUntrackedParameter < std::string > ("outputFile", "");

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
void L1TSync::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){

  ESHandle<L1GtTriggerMenu>     menuRcd;
  //ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;

  iSetup.get<L1GtTriggerMenuRcd>()            .get(menuRcd);
  //iSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);

  const L1GtTriggerMenu*             menu = menuRcd   .product();
  //const L1GtPrescaleFactors* m_l1GtPfAlgo = l1GtPfAlgo.product();

  // Retriving the list of prescale sets
  //ListsPrescaleFactors = &(m_l1GtPfAlgo->gtPrescaleFactors());

  m_algoBit.clear();

  for (CItAlgo algo = menu->gtAlgorithmAliasMap().begin(); algo!=menu->gtAlgorithmAliasMap().end(); ++algo){
    m_algoBit[(algo->second).algoAlias()] = (algo->second).algoBitNumber();
  }

  int maxNbins = 2001;

  // Getting reference prescale factors
  //const vector<int>& RefPrescaleFactors = (*ListsPrescaleFactors).at(m_refPrescaleSet);

  // FIXME:

  L1TMenuHelper myMenuHelper = L1TMenuHelper(iSetup);  
         
  m_selectedTriggers = myMenuHelper.testAlgos(m_selectedTriggers);

  map<string,string> tAutoSelTrig = myMenuHelper.getLUSOTrigger(m_algoAutoSelect,m_refPrescaleSet);
  m_selectedTriggers.insert(tAutoSelTrig.begin(),tAutoSelTrig.end());

  cout << "L1TSync Selected Trigger----------------" << endl;

  // Initializing DQM Monitor Elements
  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

    string tCategory = (*i).first;
    string tTrigger  = (*i).second;

    cout << "Category: " << tCategory << " Algo: " << tTrigger << endl;

    dbe->setCurrentFolder("L1T/L1TSync/AlgoVsBPTX/");
    m_algoVsBPTX[tTrigger] = dbe->book2D(tCategory, tTrigger+" - BPTX",maxNbins,-0.5,double(maxNbins)-0.5,5,-2.5,2.5);
    m_algoVsBPTX[tTrigger] ->setAxisTitle("Lumi Section" ,1);
    
    dbe->setCurrentFolder("L1T/L1TSync/Certification/");
    m_algoVsBPTXSummary[tTrigger] = dbe->book1D(tCategory, "% of in of sync: "+tTrigger,maxNbins,-0.5,double(maxNbins)-0.5);
    m_algoVsBPTXSummary[tTrigger] ->setAxisTitle("Lumi Section" ,1);

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
      double CountSync = m_algoVsBPTX[tTrigger]->getBinContent(eventLS+1,3);
      double CountAll  = 0;
  
      // Adding all entries for current LS
      for(int a=1 ; a<6 ; a++){CountAll += m_algoVsBPTX[tTrigger]->getBinContent(eventLS+1,a);}

      // Filling this LS summary
      if(CountAll > 0){
        int ibin = m_algoVsBPTXSummary[tTrigger]->getTH1()->FindBin(eventLS);
        m_algoVsBPTXSummary[tTrigger]->setBinContent(ibin,CountSync/CountAll);
      }

    }else{
      int ibin = m_algoVsBPTXSummary[tTrigger]->getTH1()->FindBin(eventLS);
      m_algoVsBPTXSummary[tTrigger]->setBinContent(ibin,-1);
    }
  }

}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
void L1TSync::endRun(const edm::Run& run, const edm::EventSetup& iSetup){


}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
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

      int  NAlgo = 0;  // Number of selected Algo triggers
      int  NBPTX = 0;  // Number of BPTX triggers

      bool fdlAlgo[5]; // Position of selected algo triggers
      bool fdlBPTX[5]; // Position of BPTX
     
      // Running over FDL results to get which bits fired
      for(unsigned int a=0 ; a<gtFdlVectorData.size() ; a++){
 
        fdlBPTX[a] = gtFdlVectorData[a].gtTechnicalTriggerWord()[0];
        fdlAlgo[a] = gtFdlVectorData[a].gtDecisionWord()        [ m_algoBit[tTrigger] ];

        if(gtFdlVectorData[a].gtTechnicalTriggerWord()[0])                    {NBPTX++;}
        if(gtFdlVectorData[a].gtDecisionWord()        [ m_algoBit[tTrigger] ]){NAlgo++;}

      }

      // For each event fill only if we have at least trigger from BPTX and the selected Algo
      if(NBPTX >= 1 && NAlgo >= 1){

        // If we have more than one BPTX we need: 
        //  * Choose the closest to the middle has reference
        //  * Remove other BPTX entries and matched triggers
        if(NBPTX > 1){

          int PosBPTX = -1;

          // Finding which BPTX is closer to middle of the FDL array
          if     (fdlBPTX[2]){PosBPTX = 0; break;} // BPTX Position: --X--
          else if(fdlBPTX[1]){PosBPTX = 1; break;} // BPTX Position: -X---
          else if(fdlBPTX[3]){PosBPTX = 3; break;} // BPTX Position: ---X-
          else if(fdlBPTX[0]){PosBPTX = 0; break;} // BPTX Position: X----
          else if(fdlBPTX[4]){PosBPTX = 4; break;} // BPTX Position: ----X
          
          // Removing all BPTX and matched algo triggers that are not the selected 
          for(int a=0 ; a<5 ; a++){
            if(a != PosBPTX && fdlBPTX[a]){
              fdlBPTX[a] = 0;
              fdlAlgo[a] = 0;
            }
          }
        }

        int DifAlgoVsBPTX = 5;  // Difference between the Algo and BPTX (in BX)
        int PosBPTX       = -1; // Position in the FDL array of the reference BPTX 
       
        for(int a=0 ; a<5 ; a++){
          if(fdlBPTX[a]){
            PosBPTX=a; 
            break;
          }
        }

        for(int a=0 ; a<5 ; a++){
          if(fdlAlgo[a] && abs(a-PosBPTX)<abs(DifAlgoVsBPTX)){DifAlgoVsBPTX = a-PosBPTX;}
        }

        //cout << "Algo: " << AlgoName << " Sync=" << DifAlgoVsBPTX << endl;
        m_algoVsBPTX[tTrigger]->Fill(eventLS,DifAlgoVsBPTX);

      }
    }
  }


  if (m_verbose) {cout << "L1TSync: analyze...." <<endl;}

}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TSync);

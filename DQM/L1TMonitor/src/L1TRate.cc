/*
 * \file L1TRate.cc
 *
 * \author J. Pela, P. Musella
 *
 */

// L1TMonitor includes
#include "DQM/L1TMonitor/interface/L1TRate.h"
#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"
#include "DQM/L1TMonitor/interface/L1TOMDSHelper.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ConditionsInEdm.h" // Parameters associated to Run, LS and Event
#include "DataFormats/Luminosity/interface/LumiDetails.h" // Luminosity Information
#include "DataFormats/Luminosity/interface/LumiSummary.h" // Luminosity Information

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"            // L1Gt - Masks
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h" // L1Gt - Masks
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"

#include "TList.h"

using namespace edm;
using namespace std;

//_____________________________________________________________________
L1TRate::L1TRate(const ParameterSet & ps) :
  m_l1GtUtils(ps, consumesCollector(), false, *this) {

  m_maxNbins   = 2500; // Maximum LS for each run (for binning purposes)
  m_parameters = ps;

  // Mapping parameter input variables
  m_scalersSource_colLScal       = consumes<LumiScalersCollection>          (m_parameters.getParameter<InputTag>("inputTagScalersResults"));
  m_scalersSource_triggerScalers = consumes<Level1TriggerScalersCollection> (m_parameters.getParameter<InputTag>("inputTagScalersResults"));
  m_l1GtDataDaqInputTag          = consumes<L1GlobalTriggerReadoutRecord>   (m_parameters.getParameter<InputTag>("inputTagL1GtDataDaq"));
  m_verbose             = m_parameters.getUntrackedParameter<bool>    ("verbose",false);
  m_refPrescaleSet      = m_parameters.getParameter         <int>     ("refPrescaleSet");  
  m_lsShiftGTRates      = m_parameters.getUntrackedParameter<int>     ("lsShiftGTRates",0);
  
  // Getting which categories to monitor
  ParameterSet Categories     = ps.getParameter<ParameterSet>("categories");  
  m_inputCategories["Mu"]     = Categories.getUntrackedParameter<bool>("Mu"); 
  m_inputCategories["EG"]     = Categories.getUntrackedParameter<bool>("EG"); 
  m_inputCategories["IsoEG"]  = Categories.getUntrackedParameter<bool>("IsoEG"); 
  m_inputCategories["Jet"]    = Categories.getUntrackedParameter<bool>("Jet"); 
  m_inputCategories["CenJet"] = Categories.getUntrackedParameter<bool>("CenJet"); 
  m_inputCategories["ForJet"] = Categories.getUntrackedParameter<bool>("ForJet"); 
  m_inputCategories["TauJet"] = Categories.getUntrackedParameter<bool>("TauJet"); 
  m_inputCategories["ETM"]    = Categories.getUntrackedParameter<bool>("ETM"); 
  m_inputCategories["ETT"]    = Categories.getUntrackedParameter<bool>("ETT"); 
  m_inputCategories["HTT"]    = Categories.getUntrackedParameter<bool>("HTT"); 
  m_inputCategories["HTM"]    = Categories.getUntrackedParameter<bool>("HTM"); 

  
  // What to do if we want our output to be saved to a external file
  m_outputFile = ps.getUntrackedParameter < string > ("outputFile", "");
  
  if (!m_outputFile.empty()) {
    cout << "L1T Monitoring histograms will be saved to " << m_outputFile.c_str() << endl;
  }
  
  bool disable = ps.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {m_outputFile = "";}
  
}

//_____________________________________________________________________
L1TRate::~L1TRate(){}

//_____________________________________________________________________
// BeginRun
//_____________________________________________________________________
void L1TRate::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run&, const edm::EventSetup& iSetup){


  ESHandle<L1GtTriggerMenu>     menuRcd;
  ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;

  iSetup.get<L1GtTriggerMenuRcd>()            .get(menuRcd);
  iSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);

  const L1GtTriggerMenu*     menu         = menuRcd   .product();
  const L1GtPrescaleFactors* m_l1GtPfAlgo = l1GtPfAlgo.product();

  // Initializing DQM Monitor Elements
  ibooker.setCurrentFolder("L1T/L1TRate");
  m_ErrorMonitor = ibooker.book1D("ErrorMonitor", "ErrorMonitor",5,0,5);
  m_ErrorMonitor->setBinLabel(1,"WARNING_DB_CONN_FAILED");        // Errors from L1TOMDSHelper
  m_ErrorMonitor->setBinLabel(2,"WARNING_DB_QUERY_FAILED");       // Errors from L1TOMDSHelper
  m_ErrorMonitor->setBinLabel(3,"WARNING_DB_INCORRECT_NBUNCHES"); // Errors from L1TOMDSHelper
  m_ErrorMonitor->setBinLabel(4,"WARNING_PY_MISSING_FIT");
  m_ErrorMonitor->setBinLabel(5,"UNKNOWN");

  // Retriving the list of prescale sets
  m_listsPrescaleFactors = &(m_l1GtPfAlgo->gtPrescaleFactors());
 
  // Getting Lowest Prescale Single Object Triggers from the menu
  L1TMenuHelper myMenuHelper = L1TMenuHelper(iSetup);
  m_l1GtUtils.retrieveL1EventSetup(iSetup);
  m_selectedTriggers = myMenuHelper.getLUSOTrigger(m_inputCategories, m_refPrescaleSet, m_l1GtUtils);

  //-> Getting template fits for the algLo cross sections
  int srcAlgoXSecFit = m_parameters.getParameter<int>("srcAlgoXSecFit");
  if     (srcAlgoXSecFit == 0){getXSexFitsOMDS  (m_parameters);}
  else if(srcAlgoXSecFit == 1){getXSexFitsPython(m_parameters);}

  for (CItAlgo algo = menu->gtAlgorithmMap().begin(); algo!=menu->gtAlgorithmMap().end(); ++algo){
    m_algoBit[(algo->second).algoAlias()] = (algo->second).algoBitNumber();    
  }

  double minInstantLuminosity = m_parameters.getParameter<double>("minInstantLuminosity");
  double maxInstantLuminosity = m_parameters.getParameter<double>("maxInstantLuminosity");

  // Initializing DQM Monitor Elements
  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

    TString tCategory     = (*i).first;
    TString tTrigger      = (*i).second;

    TString tErrorMessage = "";  
    TF1*    tTestFunction;

    if(tTrigger != "Undefined" && m_templateFunctions.find(tTrigger) != m_templateFunctions.end()){
      tTestFunction = m_templateFunctions[tTrigger];
    }
    else if(tTrigger == "Undefined"){
      TString tFunc = "-1";
      tTestFunction = new TF1("FitParametrization_"+tTrigger,tFunc,0,double(m_maxNbins)-0.5);
    }
    else if(m_templateFunctions.find(tTrigger) == m_templateFunctions.end()){
      TString tFunc = "-1";
      tTestFunction = new TF1("FitParametrization_"+tTrigger,tFunc,0,double(m_maxNbins)-0.5);
      tErrorMessage = " (Undefined Test Function)";
    }
    else{
      TString tFunc = "-1";
      tTestFunction = new TF1("FitParametrization_"+tTrigger,tFunc,0,double(m_maxNbins)-0.5);
    }

    if(tTrigger != "Undefined"){

    if(myMenuHelper.getPrescaleByAlias(tCategory,tTrigger) != 1){
      tErrorMessage += " WARNING: Default Prescale = ";
      tErrorMessage += myMenuHelper.getPrescaleByAlias(tCategory,tTrigger);
    }

      if     (tCategory == "Mu"    && myMenuHelper.getEtaRangeByAlias(tCategory,tTrigger) != 4294967295){
        tErrorMessage += " WARNING: Eta Range = ";
        tErrorMessage += myMenuHelper.getEtaRangeByAlias(tCategory,tTrigger);
      }
      else if(tCategory == "EG"    && myMenuHelper.getEtaRangeByAlias(tCategory,tTrigger) != 32639){
        tErrorMessage += " WARNING: Eta Range = ";
        tErrorMessage += myMenuHelper.getEtaRangeByAlias(tCategory,tTrigger);
      }
      else if(tCategory == "IsoEG" && myMenuHelper.getEtaRangeByAlias(tCategory,tTrigger) != 32639){
        tErrorMessage += " WARNING: Eta Range = ";
        tErrorMessage += myMenuHelper.getEtaRangeByAlias(tCategory,tTrigger);
      }

      if(tCategory == "Mu" && myMenuHelper.getQualityAlias(tCategory,tTrigger) != 240){
        tErrorMessage += " WARNING: Quality = ";
        tErrorMessage += myMenuHelper.getQualityAlias(tCategory,tTrigger);      
      }

    }



    ibooker.setCurrentFolder("L1T/L1TRate/TriggerCrossSections");
    m_xSecVsInstLumi[tTrigger] = ibooker.bookProfile(tCategory,
                                                  "Cross Sec. vs Inst. Lumi Algo: "+tTrigger+tErrorMessage,
                                                  m_maxNbins,
                                                  minInstantLuminosity,
                                                  maxInstantLuminosity,0,500); 
    m_xSecVsInstLumi[tTrigger] ->setAxisTitle("Instantaneous Luminosity [10^{30}cm^{-2}s^{-1}]" ,1);
    m_xSecVsInstLumi[tTrigger] ->setAxisTitle("Algorithm #sigma [#mu b]" ,2);
    m_xSecVsInstLumi[tTrigger] ->getTProfile()->GetListOfFunctions()->Add(tTestFunction);
    m_xSecVsInstLumi[tTrigger] ->getTProfile()->SetMarkerStyle(23);

    ibooker.setCurrentFolder("L1T/L1TRate/Certification");
    m_xSecObservedToExpected[tTrigger] = ibooker.book1D(tCategory, "Algo: "+tTrigger+tErrorMessage,m_maxNbins,-0.5,double(m_maxNbins)-0.5);
    m_xSecObservedToExpected[tTrigger] ->setAxisTitle("Lumi Section" ,1);
    m_xSecObservedToExpected[tTrigger] ->setAxisTitle("#sigma_{obs} / #sigma_{exp}" ,2);

  }  

}

void L1TRate::dqmBeginRun(edm::Run const&, edm::EventSetup const&){
  //
  if (m_verbose) {cout << "[L1TRate:] Called beginRun." << endl;}
}
//_____________________________________________________________________
void L1TRate::beginLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {

  if (m_verbose) {cout << "[L1TRate:] Called beginLuminosityBlock at LS=" << lumiBlock.id().luminosityBlock() << endl;}

}

//_____________________________________________________________________
void L1TRate::endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {

  int eventLS = lumiBlock.id().luminosityBlock();  
  if (m_verbose) {cout << "[L1TRate:] Called endLuminosityBlock at LS=" << eventLS << endl;}

  // We can certify LS -1 since we should have available:
  // gt rates: (current LS)-1
  // prescale: current LS
  // lumi    : current LS
  //eventLS--;
  
  // Checking if all necessary quantities are defined for our calculations
  bool isDefRate,isDefLumi,isDefPrescaleIndex;
  map<TString,double>* rates=nullptr;
  double               lumi=0;
  int                  prescalesIndex=0;

  // Reseting MonitorElements so we can refill them
  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){
    string tTrigger      = (*i).second;
    m_xSecObservedToExpected[tTrigger]->getTH1()->Reset("ICE");
    m_xSecVsInstLumi        [tTrigger]->getTH1()->Reset("ICE");
  }
    
  for(map<int,map<TString,double> >::iterator i=m_lsRates.begin() ; i!=m_lsRates.end() ; i++){

    unsigned int ls =  (*i).first;
    rates   = &(*i).second;
    isDefRate=true;

    if(m_lsLuminosity.find(ls)==m_lsLuminosity.end()){isDefLumi=false;}
    else{
      isDefLumi=true;
      lumi=m_lsLuminosity[ls];
    }
  
    if(m_lsPrescaleIndex.find(ls)==m_lsPrescaleIndex.end()){isDefPrescaleIndex=false;}
    else{
      isDefPrescaleIndex=true;
      prescalesIndex=m_lsPrescaleIndex[ls];
    }
    
    if(isDefRate && isDefLumi && isDefPrescaleIndex){
    
      const vector<int>& currentPrescaleFactors = (*m_listsPrescaleFactors).at(prescalesIndex);
     
      for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

        string tTrigger      = (*i).second;
        TF1*   tTestFunction = (TF1*) m_xSecVsInstLumi[tTrigger]->getTProfile()->GetListOfFunctions()->First();

        // If trigger name is defined we get the rate fit parameters 
        if(tTrigger != "Undefined"){

          unsigned int   trigBit      = m_algoBit[tTrigger];
          double trigPrescale = currentPrescaleFactors[trigBit];
          double trigRate     = (*rates)[tTrigger];

          if(lumi!=0 && trigPrescale!=0 && trigRate!=0){

            double AlgoXSec              = (trigPrescale*trigRate)/lumi;
            double TemplateFunctionValue = tTestFunction->Eval(lumi);

            // Checking against Template function
            int ibin = m_xSecObservedToExpected[tTrigger]->getTH1()->FindBin(ls);
            m_xSecObservedToExpected[tTrigger]->setBinContent(ibin,AlgoXSec/TemplateFunctionValue);
            m_xSecVsInstLumi        [tTrigger]->Fill(lumi,AlgoXSec);
  
            if(m_verbose){cout<<"[L1TRate:] ls="<<ls<<" Algo="<<tTrigger<<" XSec="<<AlgoXSec<<" Test="<<AlgoXSec/TemplateFunctionValue<<endl;}

          }
          else{
            int ibin = m_xSecObservedToExpected[tTrigger]->getTH1()->FindBin(ls);
            m_xSecObservedToExpected[tTrigger]->setBinContent(ibin,0.000001);
            if(m_verbose){cout << "[L1TRate:] Algo="<< tTrigger<< " XSec=Failed" << endl;}
          }
        }
      }  
    }    
  }
}

//_____________________________________________________________________
void L1TRate::analyze(const Event & iEvent, const EventSetup & eventSetup){

  edm::Handle<L1GlobalTriggerReadoutRecord>   gtReadoutRecordData;
  edm::Handle<Level1TriggerScalersCollection> triggerScalers;
  edm::Handle<LumiScalersCollection>          colLScal;
 
  iEvent.getByToken(m_l1GtDataDaqInputTag, gtReadoutRecordData);
  iEvent.getByToken(m_scalersSource_colLScal, colLScal);
  iEvent.getByToken(m_scalersSource_triggerScalers, triggerScalers);

  // Integers
  int  EventRun = iEvent.id().run();
  unsigned int eventLS  = iEvent.id().luminosityBlock();

  // Getting the trigger trigger rates from GT and buffering it
  if(triggerScalers.isValid()){
    
    Level1TriggerScalersCollection::const_iterator itL1TScalers = triggerScalers->begin();
    Level1TriggerRates trigRates(*itL1TScalers,EventRun);
    
    int gtLS = (*itL1TScalers).lumiSegmentNr()+m_lsShiftGTRates;
    
    // If we haven't got the data from this LS yet get it
    if(m_lsRates.find(gtLS)==m_lsRates.end()){
    
      if (m_verbose) {cout << "[L1TRate:] Buffering GT Rates for LS=" << gtLS << endl;}
      map<TString,double> bufferRate;
      
      // Buffer the rate informations for all selected bits
      for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

        string tTrigger = (*i).second;

        // If trigger name is defined we store the rate
        if(tTrigger != "Undefined"){

          unsigned int   trigBit  = m_algoBit[tTrigger];
          double trigRate = trigRates.gtAlgoCountsRate()[trigBit]; 
  
          bufferRate[tTrigger] = trigRate;
        }
      }
      m_lsRates[gtLS] = bufferRate;
    }
  }
  
  // Getting from the SCAL the luminosity information and buffering it
  if(colLScal.isValid() && !colLScal->empty()){
    
    LumiScalersCollection::const_iterator itLScal = colLScal->begin();
    unsigned int scalLS  = itLScal->sectionNumber();
    
    // If we haven't got the data from this SCAL LS yet get it
    if(m_lsLuminosity.find(scalLS)==m_lsLuminosity.end()){
    
      if (m_verbose) {cout << "[L1TRate:] Buffering SCAL-HF Lumi for LS=" << scalLS << endl;}
      double instLumi       = itLScal->instantLumi();           // Getting Instant Lumi from HF (via SCAL)   
      double deadTimeNormHF = itLScal->deadTimeNormalization(); // Getting Dead Time Normalization from HF (via SCAL)
       
      // If HF Dead Time Corrections is requested we apply it
      // NOTE: By default this is assumed false since for now WbM fits do NOT assume this correction
      if(m_parameters.getUntrackedParameter<bool>("useHFDeadTimeNormalization",false)){

        // Protecting for deadtime = 0
        if(deadTimeNormHF==0){instLumi = 0;}
        else                 {instLumi = instLumi/deadTimeNormHF;}
      }
      // Buffering the luminosity information
      m_lsLuminosity[scalLS]=instLumi;
    }
  }

  // Getting the prescale index used when this event was triggered
  if(gtReadoutRecordData.isValid()){
    
    // If we haven't got the data from this LS yet get it
    if(m_lsPrescaleIndex.find(eventLS)==m_lsPrescaleIndex.end()){
      
      if (m_verbose) {cout << "[L1TRate:] Buffering Prescale Index for LS=" << eventLS << endl;}

      // Getting Final Decision Logic (FDL) Data from GT
      const vector<L1GtFdlWord>& gtFdlVectorData = gtReadoutRecordData->gtFdlVector();

      // Getting the index for the fdl data for this event
      int indexFDL=0;
      for(unsigned int i=0; i<gtFdlVectorData.size(); i++){
        if(gtFdlVectorData[i].bxInEvent()==0){indexFDL=i; break;}
      }
      
      int CurrentPrescalesIndex  = gtFdlVectorData[indexFDL].gtPrescaleFactorIndexAlgo();
      m_lsPrescaleIndex[eventLS] = CurrentPrescalesIndex;   
    }    
  }
}

//_____________________________________________________________________
// function: getXSexFitsOMDS
// Imputs: 
//   * const edm::ParameterSet& ps = ParameterSet contaning necessary
//     information for the OMDS data extraction
// Outputs:
//   * int error = Number of algos where you did not find a 
//     corresponding fit 
//_____________________________________________________________________
bool L1TRate::getXSexFitsOMDS(const edm::ParameterSet& ps){

  bool noError = true;

  string oracleDB   = ps.getParameter<string>("oracleDB");
  string pathCondDB = ps.getParameter<string>("pathCondDB");

  L1TOMDSHelper myOMDSHelper;
  int conError;
  myOMDSHelper.connect(oracleDB,pathCondDB,conError);
 
  map<string,WbMTriggerXSecFit> wbmFits;

  if(conError == L1TOMDSHelper::NO_ERROR){
    int errorRetrive;
    wbmFits = myOMDSHelper.getWbMAlgoXsecFits(errorRetrive);
    
    // Filling errors if they exist
    if(errorRetrive != L1TOMDSHelper::NO_ERROR){
      noError = false;
      string eName = myOMDSHelper.enumToStringError(errorRetrive);
      m_ErrorMonitor->Fill(eName);
    }
  }else{
    noError = false;
    string eName = myOMDSHelper.enumToStringError(conError);
    m_ErrorMonitor->Fill(eName);
  }  

  double minInstantLuminosity = m_parameters.getParameter<double>("minInstantLuminosity");
  double maxInstantLuminosity = m_parameters.getParameter<double>("maxInstantLuminosity");

  // Getting rate fit parameters for all input triggers
  for(map<string,string>::const_iterator a=m_selectedTriggers.begin() ; a!=m_selectedTriggers.end() ; a++){

    string tTrigger = (*a).second;

    // If trigger name is defined we get the rate fit parameters 
    if(tTrigger != "Undefined"){
      
      if(wbmFits.find(tTrigger) != wbmFits.end()){

        WbMTriggerXSecFit tWbMParameters = wbmFits[tTrigger];

        vector<double> tParameters;
        tParameters.push_back(tWbMParameters.pm1);
        tParameters.push_back(tWbMParameters.p0);
        tParameters.push_back(tWbMParameters.p1);
        tParameters.push_back(tWbMParameters.p2);
	
        // Retriving and populating the m_templateFunctions array
        m_templateFunctions[tTrigger] = new TF1("FitParametrization_"+tWbMParameters.bitName,
                                                tWbMParameters.fitFunction,
                                                minInstantLuminosity,maxInstantLuminosity);
        m_templateFunctions[tTrigger] ->SetParameters(&tParameters[0]);
        m_templateFunctions[tTrigger] ->SetLineWidth(1);
        m_templateFunctions[tTrigger] ->SetLineColor(kRed);

      }else{noError = false;}
    }
  }

  return noError;

}

//_____________________________________________________________________
// function: getXSexFitsPython
// Imputs: 
//   * const edm::ParameterSet& ps = ParameterSet contaning the fit
//     functions and parameters for the selected triggers
// Outputs:
//   * int error = Number of algos where you did not find a 
//     corresponding fit 
//_____________________________________________________________________
bool L1TRate::getXSexFitsPython(const edm::ParameterSet& ps){

  // error meaning
  bool noError = true;

  // Getting fit parameters
  std::vector<edm::ParameterSet>  m_fitParameters = ps.getParameter< vector<ParameterSet> >("fitParameters");

  double minInstantLuminosity = m_parameters.getParameter<double>("minInstantLuminosity");
  double maxInstantLuminosity = m_parameters.getParameter<double>("maxInstantLuminosity");
  
  // Getting rate fit parameters for all input triggers
  for(map<string,string>::const_iterator a=m_selectedTriggers.begin() ; a!=m_selectedTriggers.end() ; a++){

    string tTrigger = (*a).second;

    // If trigger name is defined we get the rate fit parameters 
    if(tTrigger != "Undefined"){

      bool foundFit = false;

      for(unsigned int b=0 ; b<m_fitParameters.size() ; b++){
	
        if(tTrigger == m_fitParameters[b].getParameter<string>("AlgoName")){
	  
          TString        tAlgoName          = m_fitParameters[b].getParameter< string >        ("AlgoName");
          TString        tTemplateFunction  = m_fitParameters[b].getParameter< string >        ("TemplateFunction");
          vector<double> tParameters        = m_fitParameters[b].getParameter< vector<double> >("Parameters");
	  
          // Retriving and populating the m_templateFunctions array
          m_templateFunctions[tTrigger] = new TF1("FitParametrization_"+tAlgoName,tTemplateFunction,
                                                  minInstantLuminosity,maxInstantLuminosity);
          m_templateFunctions[tTrigger] ->SetParameters(&tParameters[0]);
          m_templateFunctions[tTrigger] ->SetLineWidth(1);
          m_templateFunctions[tTrigger] ->SetLineColor(kRed);

          foundFit = true;
          break;
        }

        if(!foundFit){
          noError = false;
          string eName = "WARNING_PY_MISSING_FIT";
          m_ErrorMonitor->Fill(eName);
        }
      }
    }
  }
  
  return noError;

}

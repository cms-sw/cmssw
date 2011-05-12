/*
 * \file L1TRate.cc
 *
 * $Date: 2011/04/14 13:03:11 $
 * $Revision: 1.2 $
 * \author J. Pela, P. Musella
 *
 */

// L1TMonitor includes
#include "DQM/L1TMonitor/interface/L1TRate.h"
#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"
#include "DQM/L1TMonitor/interface/L1TOMDSHelper.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
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

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include "TList.h"

using namespace edm;
using namespace std;

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
L1TRate::L1TRate(const ParameterSet & ps){

  // Maximum LS for each run (for binning purposes)
  m_maxNbins = 2001;

  m_parameters = ps;

  // Mapping parameter input variables
  m_scalersSource       = ps.getParameter         < InputTag >            ("inputTagScalersResults");
  m_l1GtDataDaqInputTag = ps.getParameter         < InputTag >            ("inputTagL1GtDataDaq");
  m_verbose             = ps.getUntrackedParameter< bool >                ("verbose",false);
  m_testEventScalLS     = ps.getUntrackedParameter< bool >                ("testEventScalLS",false);
  m_refPrescaleSet      = ps.getParameter         < int >                 ("refPrescaleSet");  
  
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

  // Inicializing Variables
  dbe         = NULL;
  m_currentLS = 0;

  if (ps.getUntrackedParameter < bool > ("dqmStore", false)) {
    dbe = Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }
  
  m_outputFile = ps.getUntrackedParameter < string > ("outputFile", "");
  
  if (m_outputFile.size() != 0) {
    cout << "L1T Monitoring histograms will be saved to " <<	m_outputFile.c_str() << endl;
  }
  
  bool disable = ps.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {m_outputFile = "";}
  
  if (dbe != NULL) {dbe->setCurrentFolder("L1T/L1TRate");}
  
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
L1TRate::~L1TRate(){}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
void L1TRate::beginJob(void){

  // get hold of back-end interface
  DQMStore *dbe = 0;
  dbe = Service < DQMStore > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1T/L1TRate");
    dbe->rmdir("L1T/L1TRate");
  }
 
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
void L1TRate::endJob(void){

  if (m_verbose)
    cout << "L1TRate: end job...." << endl;

  if (m_outputFile.size() != 0 && dbe)
    dbe->save(m_outputFile);

  return;
}

//-------------------------------------------------------------------------------------
/// BeginRun
//-------------------------------------------------------------------------------------
void L1TRate::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){

  ESHandle<L1GtTriggerMenu>     menuRcd;
  ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;

  iSetup.get<L1GtTriggerMenuRcd>()            .get(menuRcd);
  iSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);

  const L1GtTriggerMenu*     menu         = menuRcd   .product();
  const L1GtPrescaleFactors* m_l1GtPfAlgo = l1GtPfAlgo.product();

  // Retriving the list of prescale sets
  m_listsPrescaleFactors = &(m_l1GtPfAlgo->gtPrescaleFactors());
 
  // Getting Lowest Prescale Single Object Triggers from the menu
  L1TMenuHelper myMenuHelper = L1TMenuHelper(iSetup);
  m_selectedTriggers = myMenuHelper.getLUSOTrigger(m_inputCategories,m_refPrescaleSet);

  //-> Getting template fits for the algLo cross sections
  int srcAlgoXSecFit = m_parameters.getParameter<int>("srcAlgoXSecFit");
  if     (srcAlgoXSecFit == 0){getXSexFitsOMDS  (m_parameters);}
  else if(srcAlgoXSecFit == 1){getXSexFitsPython(m_parameters);}

  for (CItAlgo algo = menu->gtAlgorithmMap().begin(); algo!=menu->gtAlgorithmMap().end(); ++algo){
    m_algoBit[(algo->second).algoAlias()] = (algo->second).algoBitNumber();    
  }

  // Initializing record of which LS were already processed
  m_processedLS = new bool[m_maxNbins];
  for(int i=0 ; i<m_maxNbins ; i++){m_processedLS[i]=false;}

  dbe->setCurrentFolder("L1T/L1TRate");
   
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

    dbe->setCurrentFolder("L1T/L1TRate/TriggerCrossSections");
    m_xSecVsInstLumi[tTrigger] = dbe->bookProfile(tCategory,"Cross Sec. vs Inst. Lumi Algo: "+tTrigger+tErrorMessage,m_maxNbins,10,300,0,500); 
    m_xSecVsInstLumi[tTrigger] ->setAxisTitle("Instantaneous Luminosity [10^{30}cm^{-2}s^{-1}]" ,1);
    m_xSecVsInstLumi[tTrigger] ->setAxisTitle("Algorithm #sigma [#mu b]" ,2);
    m_xSecVsInstLumi[tTrigger] ->getTProfile()->GetListOfFunctions()->Add(tTestFunction);
    m_xSecVsInstLumi[tTrigger] ->getTProfile()->SetMarkerStyle(23);

    dbe->setCurrentFolder("L1T/L1TRate/Certification");
    m_xSecObservedToExpected[tTrigger] = dbe->book1D(tCategory, "Algo: "+tTrigger+tErrorMessage,m_maxNbins,-0.5,double(m_maxNbins)-0.5);
    m_xSecObservedToExpected[tTrigger] ->setAxisTitle("Lumi Section" ,1);
    m_xSecObservedToExpected[tTrigger] ->setAxisTitle("#sigma_{obs} / #sigma_{exp}" ,2);
  }
  
}

void L1TRate::endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {

  //unsigned int eventLS = lumiBlock.id().luminosityBlock();

  //cout << "Called endLuminosityBlock: " << eventLS << " will store rate for LS " <<  m_currentLS << endl;

  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

    string tTrigger = (*i).second;

    //cout << "Trigger: " << tTrigger << endl;

    // If trigger name is defined we get the rate fit parameters 
    if(tTrigger != "Undefined" && m_bufferInstLumi > 0){

      double AlgoXSec = m_bufferRate[tTrigger]/m_bufferInstLumi;

      //cout << "AlgoXSec: " << AlgoXSec << " Rate: " << m_bufferRate[tTrigger] << " InstLumi: " << m_bufferInstLumi << endl;

      // Checking against Template function
      TF1* tTestFunction = (TF1*) m_xSecVsInstLumi[tTrigger]->getTProfile()->GetListOfFunctions()->First();
      double TemplateFunctionValue = tTestFunction->Eval(m_bufferInstLumi);
      
      
      int ibin = m_xSecObservedToExpected[tTrigger]->getTH1()->FindBin(m_currentLS);
      m_xSecObservedToExpected[tTrigger]->setBinContent(ibin,AlgoXSec/TemplateFunctionValue);
      m_xSecVsInstLumi        [tTrigger]->Fill(m_bufferInstLumi,AlgoXSec);

    }
  }

/*
  Handle<LumiSummary> lumiSummary; 
  lumiBlock.getByLabel("lumiProducer", lumiSummary);

  cout << "avgInsDelLumi=" << lumiSummary->avgInsDelLumi() << " err=" << lumiSummary->avgInsDelLumiErr() << endl;

  Handle<LumiDetails> lumiDetails;
  lumiBlock.getByLabel("lumiProducer", lumiDetails);

  bool   BunchStructure[3564]; // The bunch structure
  int    NBunch = 0;           // Total number of bunches
  double InstLumiBxMax = 0;    // Value for the max. instant luminosity in a single bunch

  for (int bx=0 ; bx<3564 ; bx++){

    double LumiBx    = lumiDetails->lumiValue(LumiDetails::kOCC1, bx);
    double LumiErrBx = lumiDetails->lumiError(LumiDetails::kOCC1, bx);

    // clearing the bunch structure
    BunchStructure[bx] = false;

    // Filling bx by bx instant luminosity
    InstLumiBX->setBinContent(bx,LumiBx);
    InstLumiBX->setBinError  (bx,LumiErrBx);

    // If needed update maximum
    if(InstLumiBxMax < LumiBx){InstLumiBxMax = LumiBx;}

  }

  for (int bx=0 ; bx<3564 ; bx++){

    double LumiBx = lumiDetails->lumiValue(LumiDetails::kOCC1, bx);

    if(LumiBx > 0.1*InstLumiBxMax){

      //cout << "bx " << bx << " lumi = " << LumiBx << " err=" << lumiDetails->lumiError(LumiDetails::kOCC1, bx) << endl;
      BunchStructure[bx] = true;
      NBunch++;
      InstLumiBX2->setBinContent(bx,0.1);
      
    }

  }

  cout << "Total number of bunches = " << NBunch << endl;
*/
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
void L1TRate::endRun(const edm::Run& run, const edm::EventSetup& iSetup){

  delete[] m_processedLS;

}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
void L1TRate::analyze(const Event & iEvent, const EventSetup & eventSetup){

   edm::Handle<L1GlobalTriggerReadoutRecord>   gtReadoutRecordData;
   edm::Handle<Level1TriggerScalersCollection> triggerScalers;
   edm::Handle<LumiScalersCollection>          colLScal;
 
   iEvent.getByLabel(m_l1GtDataDaqInputTag, gtReadoutRecordData);
   iEvent.getByLabel(m_scalersSource       , colLScal);
   iEvent.getByLabel(m_scalersSource       , triggerScalers);

   Level1TriggerScalersCollection::const_iterator it      = triggerScalers->begin();
   LumiScalersCollection         ::const_iterator itLScal = colLScal->begin();

   // Integers
   int EventRun = iEvent.id().run();

   // --> Accessing Instant Luminosity via LScal
   if(colLScal->size()){ 

     unsigned int scalLS  = itLScal->sectionNumber();
     unsigned int eventLS = iEvent.id().luminosityBlock();
     
     bool testEventScalLS; // Checks if the SCAL LS is the same as Event LS 
     
     if(m_testEventScalLS){testEventScalLS = scalLS == eventLS-1;}
     else                 {testEventScalLS = true;}
     
     // We only run this code once per LS
     if(testEventScalLS && m_currentLS != scalLS && !m_processedLS[scalLS]){
       
       m_currentLS                = scalLS;                    // Updating current LS
       m_processedLS[m_currentLS] = true;                      // Current LS as processed 
       m_bufferInstLumi           = itLScal->instantLumi();    // Getting instant lumi    
       //double tInstantLumiErr     = itLScal->instantLumiErr(); // Getting instant lumi error

       //cout << "EventLS=" << eventLS << " scalLS = " << scalLS << " Inst Lumi = " << m_bufferInstLumi << " +/- " << tInstantLumiErr << endl;

       Level1TriggerRates trigRates(*it,EventRun);
       
       if(m_bufferInstLumi > 0){

         // --> Getting current L1 prescales
         // Getting Final Decision Logic (FDL) Data from GT
         const vector<L1GtFdlWord>& gtFdlVectorData = gtReadoutRecordData->gtFdlVector();

         // Getting vector mid-entry and accessing CurrentPrescalesIndex
         // NOTE: This gets the middle L1GtFdlWord from the vector (we assume vector is ordered by time)
         int FdlVectorCurrentEvent = gtFdlVectorData.size()/2;
         int CurrentPrescalesIndex = gtFdlVectorData[FdlVectorCurrentEvent].gtPrescaleFactorIndexAlgo();

         const vector<int>& CurrentPrescaleFactors = (*m_listsPrescaleFactors).at(CurrentPrescalesIndex);
	 
        // Buffer the rate informations for all selected bits
        for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

          string tTrigger = (*i).second;

          // If trigger name is defined we store the rate
          if(tTrigger != "Undefined"){
		  
            L1GtUtils tUtils;
            tUtils.retrieveL1GtTriggerMenuLite(iEvent);
            // int       tError;

            double tAlgoRate = trigRates.gtAlgoCountsRate()[m_algoBit[tTrigger]]; 
            double tPrescale = CurrentPrescaleFactors[m_algoBit[tTrigger]];
            //double tPrescale = tUtils.prescaleFactor(iEvent,tTrigger,tError);

            //cout << "Algo: " << tTrigger << " Prescale: " << tPrescale << " rate: " << tAlgoRate << " error: " // << tError << endl;

            m_bufferRate[tTrigger] = tPrescale*tAlgoRate; 

          }
        }
      }
    }
  }

  if (m_verbose) {cout << "L1TRate: analyze...." <<endl;}

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
int L1TRate::getXSexFitsOMDS(const edm::ParameterSet& ps){

  int error = 0;

  string oracleDB   = ps.getParameter<string>("oracleDB");
  string pathCondDB = ps.getParameter<string>("pathCondDB");

  L1TOMDSHelper myOMDSHelper = L1TOMDSHelper();
  string conError = "";
  myOMDSHelper.connect(oracleDB,pathCondDB,conError);
 
  map<string,WbMTriggerXSecFit> wbmFits;

  if(conError == ""){
    string errorRetrive = "";
    wbmFits = myOMDSHelper.getWbMAlgoXsecFits(errorRetrive);
    if(errorRetrive != ""){cout << "L1TSync: " << errorRetrive << endl;}
  }else{
    cout << "L1TSync: " << conError << endl;    
  }  

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
                                                0,double(m_maxNbins)-0.5);
        m_templateFunctions[tTrigger] ->SetParameters(&tParameters[0]);
        m_templateFunctions[tTrigger] ->SetLineWidth(1);
        m_templateFunctions[tTrigger] ->SetLineColor(kRed);

      }else{error++;}
    }
  }

  return error;

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
int L1TRate::getXSexFitsPython(const edm::ParameterSet& ps){

  // error meaning
  // 0 = No error
  int error = 0;

  // Getting fit parameters
  std::vector<edm::ParameterSet>  m_fitParameters = ps.getParameter< vector<ParameterSet> >("fitParameters");

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
          m_templateFunctions[tTrigger] = new TF1("FitParametrization_"+tAlgoName,tTemplateFunction,0,double(m_maxNbins)-0.5);
          m_templateFunctions[tTrigger] ->SetParameters(&tParameters[0]);
          m_templateFunctions[tTrigger] ->SetLineWidth(1);
          m_templateFunctions[tTrigger] ->SetLineColor(kRed);

          foundFit = true;
          break;
        }

        if(!foundFit){error++;}

      }
    }
  }
  
  return error;

}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TRate);

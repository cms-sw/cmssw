 /*
 * \file L1TRate_Offline.cc
 *
 * $Date: 2012/12/03 15:03:34 $
 * $Revision: 1.6 $
 * \author J. Pela, P. Musella
 *
 */

// L1TMonitor includes
#include "DQMOffline/L1Trigger/interface/L1TRate_Offline.h"

#include "DQMOffline/L1Trigger/interface/L1TMenuHelper.h"

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

#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

#include "TList.h"

using namespace edm;
using namespace std;

//_____________________________________________________________________
L1TRate_Offline::L1TRate_Offline(const ParameterSet & ps){

  m_maxNbins   = 2500; // Maximum LS for each run (for binning purposes)
  m_parameters = ps;

  // Mapping parameter input variables
  m_scalersSource       = m_parameters.getParameter         <InputTag>("inputTagScalersResults");
  m_l1GtDataDaqInputTag = m_parameters.getParameter         <InputTag>("inputTagL1GtDataDaq");
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

  // Inicializing Variables
  if (m_verbose) {
    cout << "[L1TRate_Offline:] ____________ Storage inicialization ____________ " << endl;
    cout << "[L1TRate_Offline:] Setting up dbe folder: L1T/L1TRate_Offline" << endl;
  }

  dbe = Service < DQMStore > ().operator->();
  dbe->setVerbose(0);
  dbe->setCurrentFolder("L1T/L1TRate_Offline");

  // Inicializing Variables
  if (m_verbose) {cout << "[L1TRate_Offline:] Pointer for DQM Store: " << dbe << endl;}
}

//_____________________________________________________________________
L1TRate_Offline::~L1TRate_Offline(){}

//_____________________________________________________________________
void L1TRate_Offline::beginJob(void){

  if (m_verbose) {cout << "[L1TRate_Offline:] Called beginJob." << endl;}

  // get hold of back-end interface
  DQMStore *dbe = 0;
  dbe = Service < DQMStore > ().operator->();

  if (dbe) {
    dbe->setCurrentFolder("L1T/L1TRate_Offline");
    dbe->rmdir("L1T/L1TRate_Offline");
  }

}

//_____________________________________________________________________
void L1TRate_Offline::endJob(void){

  if (m_verbose) {cout << "[L1TRate_Offline:] Called endJob." << endl;}

  if (m_outputFile.size() != 0 && dbe)
    dbe->save(m_outputFile);

  return;

}

//_____________________________________________________________________
// BeginRun: as input you get filtered events...
//_____________________________________________________________________
void L1TRate_Offline::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){

  if (m_verbose) {cout << "[L1TRate_Offline:] Called beginRun." << endl;}

  ESHandle<L1GtTriggerMenu>     menuRcd;
  ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;

  iSetup.get<L1GtTriggerMenuRcd>()            .get(menuRcd);
  iSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);

  const L1GtTriggerMenu*     menu         = menuRcd   .product();
  const L1GtPrescaleFactors* m_l1GtPfAlgo = l1GtPfAlgo.product();

  // Initializing DQM Monitor Elements
  dbe->setCurrentFolder("L1T/L1TRate_Offline");
  m_ErrorMonitor = dbe->book1D("ErrorMonitor", "ErrorMonitor",2,0,2);
  m_ErrorMonitor->setBinLabel(UNKNOWN               ,"UNKNOWN");
  m_ErrorMonitor->setBinLabel(WARNING_PY_MISSING_FIT,"WARNING_PY_MISSING_FIT");

  cout << "[L1TRate_Offline:] m_ErrorMonitor: " << m_ErrorMonitor << endl;

  // Retriving the list of prescale sets
  m_listsPrescaleFactors = &(m_l1GtPfAlgo->gtPrescaleFactors());

  // Getting Lowest Prescale Single Object Triggers from the menu
  L1TMenuHelper myMenuHelper = L1TMenuHelper(iSetup);
  m_selectedTriggers = myMenuHelper.getLUSOTrigger(m_inputCategories,m_refPrescaleSet);

  //-> Getting template fits for the algLo cross sections
  getXSexFitsPython(m_parameters);

  for (CItAlgo algo = menu->gtAlgorithmMap().begin(); algo!=menu->gtAlgorithmMap().end(); ++algo){
    m_algoBit[(algo->second).algoAlias()] = (algo->second).algoBitNumber();
  }

  double minInstantLuminosity = m_parameters.getParameter<double>("minInstantLuminosity");
  double maxInstantLuminosity = m_parameters.getParameter<double>("maxInstantLuminosity");

  // Initializing DQM Monitor Elements
  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){

    TString tCategory = (*i).first;
    TString tTrigger  = (*i).second;

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

    if     (tCategory == "Mu"    && myMenuHelper.getEtaRangeByAlias(tCategory,tTrigger) != 4294967295){ //hexadecimal of the whole range
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

    dbe->setCurrentFolder("L1T/L1TRate_Offline/TriggerCounts"); // trigger counts...
    m_CountsVsLS[tTrigger] = dbe->bookProfile(tCategory,
                                                  "Cross Sec. vs Inst. Lumi Algo: "+tTrigger+tErrorMessage,
                                                  m_maxNbins,
                                                  minInstantLuminosity,
                                                  maxInstantLuminosity,0,500);
    m_CountsVsLS[tTrigger] ->setAxisTitle("Instantaneous Luminosity [10^{30}cm^{-2}s^{-1}]" ,1);
    m_CountsVsLS[tTrigger] ->setAxisTitle("Algorithm #sigma [#mu b]" ,2);
    m_CountsVsLS[tTrigger] ->getTProfile()->GetListOfFunctions()->Add(tTestFunction);
    m_CountsVsLS[tTrigger] ->getTProfile()->SetMarkerStyle(23);

    m_algoFit[tTrigger] = (TF1*) tTestFunction->Clone("Fit_"+tTrigger); // NOTE: Workaround

    dbe->setCurrentFolder("L1T/L1TRate_Offline/Ratio");
    m_xSecObservedToExpected[tTrigger] = dbe->book1D(tCategory, "Algo: "+tTrigger+tErrorMessage,m_maxNbins,-0.5,double(m_maxNbins)-0.5);
    m_xSecObservedToExpected[tTrigger] ->setAxisTitle("Lumi Section" ,1);
    m_xSecObservedToExpected[tTrigger] ->setAxisTitle("#sigma_{obs} / #sigma_{exp}" ,2);


  }

}

//_____________________________________________________________________
void L1TRate_Offline::endRun(const edm::Run& run, const edm::EventSetup& iSetup){

  if (m_verbose) {cout << "[L1TRate_Offline:] Called endRun." << endl;}
}

//_____________________________________________________________________
void L1TRate_Offline::beginLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {

  if (m_verbose) {cout << "[L1TRate_Offline:] Called beginLuminosityBlock at LS=" << lumiBlock.id().luminosityBlock() << endl;}

}

//_____________________________________________________________________
void L1TRate_Offline::endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {

  int eventLS = lumiBlock.id().luminosityBlock();
  if (m_verbose) {cout << "[L1TRate_Offline:] Called endLuminosityBlock at LS=" << eventLS << endl;}

  // We can certify LS -1 since we should have available:
  // gt rates: (current LS)-1
  // prescale: current LS
  // lumi    : current LS
  //eventLS--;

  // Checking if all necessary quantities are defined for our calculations
  //bool isDefRate,isDefLumi,isDefPrescaleIndex;
  bool isDefLumi,isDefPrescaleIndex;
  //map<TString,double>* rates=0;
  double               lumi=0;
  int                  prescalesIndex=0;

  bool isDefCount;
  map<TString,double>* counts=0;

  // Resetting MonitorElements so we can refill them
  for(map<string,string>::const_iterator i=m_selectedTriggers.begin() ; i!=m_selectedTriggers.end() ; i++){
    string tTrigger      = (*i).second;
    m_CountsVsLS            [tTrigger]->getTH1()->Reset("ICE");
    m_xSecObservedToExpected[tTrigger]->getTH1()->Reset("ICE");

  }

  //Trying to do the same with Counts....
  for(map<int,map<TString,double> >::iterator j=m_lsRates.begin() ; j!=m_lsRates.end() ; j++){

    unsigned int lsOffline =  (*j).first;
    counts   = &(*j).second;
    isDefCount=true;

    unsigned int lsPreInd;

    if(m_lsLuminosity.find(lsOffline)==m_lsLuminosity.end()){isDefLumi=false;}
    else{
      isDefLumi=true;
      lumi=m_lsLuminosity[lsOffline];
    }

    lsPreInd = lsOffline + 1; // NOTE: Workaround

    if(m_lsPrescaleIndex.find(lsPreInd)==m_lsPrescaleIndex.end()){isDefPrescaleIndex=false;}
    else{
      isDefPrescaleIndex=true;
      prescalesIndex=m_lsPrescaleIndex[lsPreInd];
    }

    if(isDefCount && isDefLumi && isDefPrescaleIndex){

      //const vector<int>& currentPrescaleFactors = (*m_listsPrescaleFactors).at(prescalesIndex);

      for(map<string,string>::const_iterator j=m_selectedTriggers.begin() ; j!=m_selectedTriggers.end() ; j++){

        string tTrigger      = (*j).second;
        double trigCount     = (*counts)[tTrigger];

        //   TF1*   tTestFunction = (TF1*) m_CountsVsLS[tTrigger]->getTProfile()->GetListOfFunctions()->First();
        TF1* tTestFunction = m_algoFit[tTrigger]; // NOTE: Workaround....


        // If trigger name is defined we get the rate fit parameters
        if(tTrigger != "Undefined"){


          if(lumi!=0 && trigCount!=0 && prescalesIndex!=0){

            double AlgoXSec              = (prescalesIndex*trigCount)/lumi;
            double TemplateFunctionValue = tTestFunction->Eval(lumi);

            // Checking against Template function
            m_CountsVsLS  [tTrigger]->Fill(lumi,AlgoXSec);

            int ibin = m_xSecObservedToExpected[tTrigger]->getTH1()->FindBin(lsOffline);
            m_xSecObservedToExpected[tTrigger]->setBinContent(ibin,AlgoXSec/TemplateFunctionValue);


          }
          else {
            m_CountsVsLS  [tTrigger]->Fill(0.000001,0.000001);

            int ibin = m_xSecObservedToExpected[tTrigger]->getTH1()->FindBin(lsOffline);
            m_xSecObservedToExpected[tTrigger]->setBinContent(ibin,0.000001);
          }
        }
      }
    }
  }
}


//_____________________________________________________________________
void L1TRate_Offline::analyze(const Event & iEvent, const EventSetup & eventSetup){

  edm::Handle<L1GlobalTriggerReadoutRecord>   gtReadoutRecordData;
  edm::Handle<Level1TriggerScalersCollection> triggerScalers;
  edm::Handle<LumiScalersCollection>          colLScal;

  iEvent.getByLabel(m_l1GtDataDaqInputTag, gtReadoutRecordData);
  iEvent.getByLabel(m_scalersSource      , colLScal);
  iEvent.getByLabel(m_scalersSource      , triggerScalers);

  // Integers
  //int  EventRun = iEvent.id().run();
  unsigned int eventLS  = iEvent.id().luminosityBlock();

  // Getting the trigger trigger rates from GT and buffering it
  if(triggerScalers.isValid() && triggerScalers->size()){

    Level1TriggerScalersCollection::const_iterator itL1TScalers = triggerScalers->begin();
    //Level1TriggerRates trigRates(*itL1TScalers,EventRun);

    // Trying to get the trigger counts
    const std::vector<unsigned int> gtAlgoCounts =  itL1TScalers->gtAlgoCounts();

    int gtLS = (*itL1TScalers).lumiSegmentNr()+m_lsShiftGTRates;

    // If we haven't got the data from this LS yet get it
    if(m_lsRates.find(gtLS)==m_lsRates.end()){

      map<TString,double> bufferCount;

      // Buffer the rate informations for all selected bits
      for(map<string,string>::const_iterator i=m_selectedTriggers.begin(); i!=m_selectedTriggers.end() ; i++){

        string tTrigger = (*i).second;

        // If trigger name is defined we store the rate
        if(tTrigger != "Undefined"){

          unsigned int   trigBit  = m_algoBit[tTrigger];
          double trigCount = gtAlgoCounts[trigBit];

          bufferCount[tTrigger] = trigCount;

	}
      }
      m_lsRates[gtLS] = bufferCount;
    }
  }


  // Getting from the SCAL the luminosity information and buffering it
  if(colLScal.isValid() && colLScal->size()){

    LumiScalersCollection::const_iterator itLScal = colLScal->begin();
    unsigned int scalLS  = itLScal->sectionNumber();

    // If we haven't got the data from this SCAL LS yet get it
    if(m_lsLuminosity.find(scalLS)==m_lsLuminosity.end()){

      if (m_verbose) {cout << "[L1TRate_Offline:] Buffering SCAL-HF Lumi for LS=" << scalLS << endl;}
      double instLumi       = itLScal->instantLumi();           // Getting Instant Lumi from HF (via SCAL) // <###### WE NEED TO STORE THIS
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

      if (m_verbose) {cout << "[L1TRate_Offline:] Buffering Prescale Index for LS=" << eventLS << endl;}

      // Getting Final Decision Logic (FDL) Data from GT
      const vector<L1GtFdlWord>& gtFdlVectorData = gtReadoutRecordData->gtFdlVector();

      // Getting the index for the fdl data for this event
      int indexFDL=0;
      for(unsigned int i=0; i<gtFdlVectorData.size(); i++){
        if(gtFdlVectorData[i].bxInEvent()==0){indexFDL=i; break;}
      }

      int CurrentPrescalesIndex  = gtFdlVectorData[indexFDL].gtPrescaleFactorIndexAlgo(); // <###### WE NEED TO STORE THIS
      m_lsPrescaleIndex[eventLS] = CurrentPrescalesIndex;

    }

  }


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
bool L1TRate_Offline::getXSexFitsPython(const edm::ParameterSet& ps){

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

          int eCount = m_ErrorMonitor->getTH1()->GetBinContent(WARNING_PY_MISSING_FIT);
          eCount++;
          m_ErrorMonitor->getTH1()->SetBinContent(WARNING_PY_MISSING_FIT,eCount);

        }
      }
    }
  }

   return noError;

}



//define this as a plug-in
DEFINE_FWK_MODULE(L1TRate_Offline);

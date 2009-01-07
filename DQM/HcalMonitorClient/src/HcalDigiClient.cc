#include <DQM/HcalMonitorClient/interface/HcalDigiClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalDigiClient::HcalDigiClient(){}

void HcalDigiClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  //errorFrac_=ps.getUntrackedParameter<double>("digiErrorFrac",0.05);

  hbHists.shape   =0;
  heHists.shape   =0;
  hoHists.shape   =0;
  hfHists.shape   =0;
  hbHists.shapeThresh   =0;
  heHists.shapeThresh   =0;
  hoHists.shapeThresh   =0;
  hfHists.shapeThresh   =0;
  hbHists.presample   =0;
  heHists.presample   =0;
  hoHists.presample   =0;
  hfHists.presample   =0;
  hbHists.BQ   =0;
  heHists.BQ   =0;
  hoHists.BQ   =0;
  hfHists.BQ   =0;
  hbHists.BQFrac   =0;
  heHists.BQFrac   =0;
  hoHists.BQFrac   =0;
  hfHists.BQFrac   =0;
  hbHists.DigiFirstCapID   =0;
  heHists.DigiFirstCapID   =0;
  hoHists.DigiFirstCapID   =0;
  hfHists.DigiFirstCapID   =0;
  hbHists.DVerr   =0;
  heHists.DVerr   =0;
  hoHists.DVerr   =0;
  hfHists.DVerr   =0;
  hbHists.CapID   =0;
  heHists.CapID   =0;
  hoHists.CapID   =0;
  hfHists.CapID   =0;
  hbHists.ADC   =0;
  heHists.ADC   =0;
  hoHists.ADC   =0;
  hfHists.ADC   =0;
  hbHists.ADCsum   =0;
  heHists.ADCsum   =0;
  hoHists.ADCsum   =0;
  hfHists.ADCsum   =0;
  DigiSize    =0;
  DigiOccupancyEta    =0;
  DigiOccupancyPhi    =0;
  DigiNum    =0;
  DigiBQ    =0;
  DigiBQFrac    =0;
  ProblemDigis    =0;
  DigiOccupancyByDepth    =0;
  DigiOccupancyVME    =0;
  DigiOccupancySpigot    =0;
  DigiErrorEtaPhi    =0;
  DigiErrorVME    =0;
  DigiErrorSpigot    =0;
  for (int i=0;i<6;++i)
    {
      ProblemDigisByDepth[i]    =0;
      DigiErrorsBadCapID[i]    =0;
      DigiErrorsBadDigiSize[i]    =0;
      DigiErrorsBadADCSum[i]    =0;
      DigiErrorsNoDigi[i]    =0;
      DigiErrorsDVErr[i]    =0;
    } // for (int i=0;i<6;++i)


  subdets_.push_back("HB HF Depth 1 ");
  subdets_.push_back("HB HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO ZDC ");
  subdets_.push_back("HE Depth 1 ");
  subdets_.push_back("HE Depth 2 ");

} // void HcalDigiClient::init(...)

HcalDigiClient::~HcalDigiClient(){
  cleanup();
}

void HcalDigiClient::beginJob(void){
  
  if ( debug_ ) cout << "HcalDigiClient: beginJob" << endl;
  
  ievt_ = 0;
  jevt_ = 0;
  setup();
  resetAllME();
  return;
}

void HcalDigiClient::beginRun(void){

  if ( debug_ ) cout << "HcalDigiClient: beginRun" << endl;

  jevt_ = 0;
  setup();
  resetAllME();
  return;
}

void HcalDigiClient::endJob(void) {

  if ( debug_ ) cout << "HcalDigiClient: endJob, ievt = " << ievt_ << endl;

  cleanup(); 
  return;
}

void HcalDigiClient::endRun(void) {

  if ( debug_ ) cout << "HcalDigiClient: endRun, jevt = " << jevt_ << endl;

  cleanup();  
  return;
}

void HcalDigiClient::setup(void) {
  
  return;
}

void HcalDigiClient::cleanup(void) {

  if ( cloneME_ ) 
    {
      if (hbHists.shape) delete hbHists.shape;
      if (heHists.shape) delete heHists.shape;
      if (hoHists.shape) delete hoHists.shape;
      if (hfHists.shape) delete hfHists.shape;
      if (hbHists.shapeThresh) delete hbHists.shapeThresh;
      if (heHists.shapeThresh) delete heHists.shapeThresh;
      if (hoHists.shapeThresh) delete hoHists.shapeThresh;
      if (hfHists.shapeThresh) delete hfHists.shapeThresh;
      if (hbHists.presample) delete hbHists.presample;
      if (heHists.presample) delete heHists.presample;
      if (hoHists.presample) delete hoHists.presample;
      if (hfHists.presample) delete hfHists.presample;
      if (hbHists.BQ) delete hbHists.BQ;
      if (heHists.BQ) delete heHists.BQ;
      if (hoHists.BQ) delete hoHists.BQ;
      if (hfHists.BQ) delete hfHists.BQ;
      if (hbHists.BQFrac) delete hbHists.BQFrac;
      if (heHists.BQFrac) delete heHists.BQFrac;
      if (hoHists.BQFrac) delete hoHists.BQFrac;
      if (hfHists.BQFrac) delete hfHists.BQFrac;
      if (hbHists.DigiFirstCapID) delete hbHists.DigiFirstCapID;
      if (heHists.DigiFirstCapID) delete heHists.DigiFirstCapID;
      if (hoHists.DigiFirstCapID) delete hoHists.DigiFirstCapID;
      if (hfHists.DigiFirstCapID) delete hfHists.DigiFirstCapID;
      if (hbHists.DVerr) delete hbHists.DVerr;
      if (heHists.DVerr) delete heHists.DVerr;
      if (hoHists.DVerr) delete hoHists.DVerr;
      if (hfHists.DVerr) delete hfHists.DVerr;
      if (hbHists.CapID) delete hbHists.CapID;
      if (heHists.CapID) delete heHists.CapID;
      if (hoHists.CapID) delete hoHists.CapID;
      if (hfHists.CapID) delete hfHists.CapID;
      if (hbHists.ADC) delete hbHists.ADC;
      if (heHists.ADC) delete heHists.ADC;
      if (hoHists.ADC) delete hoHists.ADC;
      if (hfHists.ADC) delete hfHists.ADC;
      if (hbHists.ADCsum) delete hbHists.ADCsum;
      if (heHists.ADCsum) delete heHists.ADCsum;
      if (hoHists.ADCsum) delete hoHists.ADCsum;
      if (hfHists.ADCsum) delete hfHists.ADCsum;
      if (DigiSize) delete DigiSize;
      if (DigiOccupancyEta) delete DigiOccupancyEta;
      if (DigiOccupancyPhi) delete DigiOccupancyPhi;
      if (DigiNum) delete DigiNum;
      if (DigiBQ) delete DigiBQ;
      if (DigiBQFrac) delete DigiBQFrac;
      if (ProblemDigis) delete ProblemDigis;
      if (DigiOccupancyByDepth) delete DigiOccupancyByDepth;
      if (DigiOccupancyVME) delete DigiOccupancyVME;
      if (DigiOccupancySpigot) delete DigiOccupancySpigot;
      if (DigiErrorEtaPhi) delete DigiErrorEtaPhi;
      if (DigiErrorVME) delete DigiErrorVME;
      if (DigiErrorSpigot) delete DigiErrorSpigot;
      for (int i=0;i<6;++i)
	{
	  if (ProblemDigisByDepth[i]) delete ProblemDigisByDepth[i];
	  if (DigiErrorsBadCapID[i]) delete DigiErrorsBadCapID[i];
	  if (DigiErrorsBadDigiSize[i]) delete DigiErrorsBadDigiSize[i];
	  if (DigiErrorsBadADCSum[i]) delete DigiErrorsBadADCSum[i];
	  if (DigiErrorsNoDigi[i]) delete DigiErrorsNoDigi[i];
	  if (DigiErrorsDVErr[i]) delete DigiErrorsDVErr[i];
	} // for (int i=0;i<6;++i)

    } // if (cloneME_)

  hbHists.shape   =0;
  heHists.shape   =0;
  hoHists.shape   =0;
  hfHists.shape   =0;
  hbHists.shapeThresh   =0;
  heHists.shapeThresh   =0;
  hoHists.shapeThresh   =0;
  hfHists.shapeThresh   =0;
  hbHists.presample   =0;
  heHists.presample   =0;
  hoHists.presample   =0;
  hfHists.presample   =0;
  hbHists.BQ   =0;
  heHists.BQ   =0;
  hoHists.BQ   =0;
  hfHists.BQ   =0;
  hbHists.BQFrac   =0;
  heHists.BQFrac   =0;
  hoHists.BQFrac   =0;
  hfHists.BQFrac   =0;
  hbHists.DigiFirstCapID   =0;
  heHists.DigiFirstCapID   =0;
  hoHists.DigiFirstCapID   =0;
  hfHists.DigiFirstCapID   =0;
  hbHists.DVerr   =0;
  heHists.DVerr   =0;
  hoHists.DVerr   =0;
  hfHists.DVerr   =0;
  hbHists.CapID   =0;
  heHists.CapID   =0;
  hoHists.CapID   =0;
  hfHists.CapID   =0;
  hbHists.ADC   =0;
  heHists.ADC   =0;
  hoHists.ADC   =0;
  hfHists.ADC   =0;
  hbHists.ADCsum   =0;
  heHists.ADCsum   =0;
  hoHists.ADCsum   =0;
  hfHists.ADCsum   =0;
  DigiSize    =0;
  DigiOccupancyEta    =0;
  DigiOccupancyPhi    =0;
  DigiNum    =0;
  DigiBQ    =0;
  DigiBQFrac    =0;
  ProblemDigis    =0;
  DigiOccupancyByDepth    =0;
  DigiOccupancyVME    =0;
  DigiOccupancySpigot    =0;
  DigiErrorEtaPhi    =0;
  DigiErrorVME    =0;
  DigiErrorSpigot    =0;
  for (int i=0;i<6;++i)
    {
      ProblemDigisByDepth[i]    =0;
      DigiErrorsBadCapID[i]    =0;
      DigiErrorsBadDigiSize[i]    =0;
      DigiErrorsBadADCSum[i]    =0;
      DigiErrorsNoDigi[i]    =0;
      DigiErrorsDVErr[i]    =0;
    } // for (int i=0;i<6;++i)

  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  return;
} // void HcalDigiClient::cleanup(void)


void HcalDigiClient::report(){

  if ( debug_ ) cout << "HcalDigiClient: report" << endl;
  
  char name[256];
  sprintf(name, "%sHcal/DigiMonitor/Digi Task Event Number",process_.c_str());
  MonitorElement* me = 0;
  if(dbe_) me = dbe_->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_ ) cout << "Found '" << name << "'" << endl;
  }

  getHistograms();
  return;
}

void HcalDigiClient::analyze(void){

  jevt_++;
  int updates = 0;

  if ( updates % 10 == 0 ) {
    if ( debug_ ) cout << "HcalDigiClient: " << updates << " updates" << endl;
  }
  
  return;
}

void HcalDigiClient::getHistograms(){
  if(!dbe_) return;

  ostringstream name;
  TH2F* dummy2D = new TH2F();
  TH1F* dummy1D = new TH1F();

  // Get Histograms
  name<<process_.c_str()<<"hbHists.shape";  //DigiMonitor_Hcal/digi_info/HB/HB Digi Shape
  hbHists.shape = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"heHists.shape";  //DigiMonitor_Hcal/digi_info/HE/HE Digi Shape
  heHists.shape = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hoHists.shape";  //DigiMonitor_Hcal/digi_info/HO/HO Digi Shape
  hoHists.shape = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hfHists.shape";  //DigiMonitor_Hcal/digi_info/HF/HF Digi Shape
  hfHists.shape = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hbHists.shapeThresh";  //DigiMonitor_Hcal/digi_info/HB/HB Digi Shape - over thresh
  hbHists.shapeThresh = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"heHists.shapeThresh";  //DigiMonitor_Hcal/digi_info/HE/HE Digi Shape - over thresh
  heHists.shapeThresh = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hoHists.shapeThresh";  //DigiMonitor_Hcal/digi_info/HO/HO Digi Shape - over thresh
  hoHists.shapeThresh = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hfHists.shapeThresh";  //DigiMonitor_Hcal/digi_info/HF/HF Digi Shape - over thresh
  hfHists.shapeThresh = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hbHists.presample";  //DigiMonitor_Hcal/digi_info/HB/HB Digi Presamples
  hbHists.presample = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"heHists.presample";  //DigiMonitor_Hcal/digi_info/HE/HE Digi Presamples
  heHists.presample = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hoHists.presample";  //DigiMonitor_Hcal/digi_info/HO/HO Digi Presamples
  hoHists.presample = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hfHists.presample";  //DigiMonitor_Hcal/digi_info/HF/HF Digi Presamples
  hfHists.presample = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hbHists.BQ";  //DigiMonitor_Hcal/digi_info/HB/HB Bad Quality Digis
  hbHists.BQ = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"heHists.BQ";  //DigiMonitor_Hcal/digi_info/HE/HE Bad Quality Digis
  heHists.BQ = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hoHists.BQ";  //DigiMonitor_Hcal/digi_info/HO/HO Bad Quality Digis
  hoHists.BQ = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hfHists.BQ";  //DigiMonitor_Hcal/digi_info/HF/HF Bad Quality Digis
  hfHists.BQ = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hbHists.BQFrac";  //DigiMonitor_Hcal/digi_info/HB/HB Bad Quality Digi Fraction
  hbHists.BQFrac = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"heHists.BQFrac";  //DigiMonitor_Hcal/digi_info/HE/HE Bad Quality Digi Fraction
  heHists.BQFrac = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hoHists.BQFrac";  //DigiMonitor_Hcal/digi_info/HO/HO Bad Quality Digi Fraction
  hoHists.BQFrac = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hfHists.BQFrac";  //DigiMonitor_Hcal/digi_info/HF/HF Bad Quality Digi Fraction
  hfHists.BQFrac = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hbHists.DigiFirstCapID";  //DigiMonitor_Hcal/digi_info/HB/HB Capid 1st Time Slice
  hbHists.DigiFirstCapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"heHists.DigiFirstCapID";  //DigiMonitor_Hcal/digi_info/HE/HE Capid 1st Time Slice
  heHists.DigiFirstCapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hoHists.DigiFirstCapID";  //DigiMonitor_Hcal/digi_info/HO/HO Capid 1st Time Slice
  hoHists.DigiFirstCapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hfHists.DigiFirstCapID";  //DigiMonitor_Hcal/digi_info/HF/HF Capid 1st Time Slice
  hfHists.DigiFirstCapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hbHists.DVerr";  //DigiMonitor_Hcal/digi_info/HB/HB Data Valid Err Bits
  hbHists.DVerr = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"heHists.DVerr";  //DigiMonitor_Hcal/digi_info/HE/HE Data Valid Err Bits
  heHists.DVerr = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hoHists.DVerr";  //DigiMonitor_Hcal/digi_info/HO/HO Data Valid Err Bits
  hoHists.DVerr = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hfHists.DVerr";  //DigiMonitor_Hcal/digi_info/HF/HF Data Valid Err Bits
  hfHists.DVerr = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hbHists.CapID";  //DigiMonitor_Hcal/digi_info/HB/HB CapID
  hbHists.CapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"heHists.CapID";  //DigiMonitor_Hcal/digi_info/HE/HE CapID
  heHists.CapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hoHists.CapID";  //DigiMonitor_Hcal/digi_info/HO/HO CapID
  hoHists.CapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hfHists.CapID";  //DigiMonitor_Hcal/digi_info/HF/HF CapID
  hfHists.CapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hbHists.ADC";  //DigiMonitor_Hcal/digi_info/HB/HB ADC count per time slice
  hbHists.ADC = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"heHists.ADC";  //DigiMonitor_Hcal/digi_info/HE/HE ADC count per time slice
  heHists.ADC = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hoHists.ADC";  //DigiMonitor_Hcal/digi_info/HO/HO ADC count per time slice
  hoHists.ADC = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hfHists.ADC";  //DigiMonitor_Hcal/digi_info/HF/HF ADC count per time slice
  hfHists.ADC = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hbHists.ADCsum";  //DigiMonitor_Hcal/digi_info/HB/HB ADC sum
  hbHists.ADCsum = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"heHists.ADCsum";  //DigiMonitor_Hcal/digi_info/HE/HE ADC sum
  heHists.ADCsum = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hoHists.ADCsum";  //DigiMonitor_Hcal/digi_info/HO/HO ADC sum
  hoHists.ADCsum = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"hfHists.ADCsum";  //DigiMonitor_Hcal/digi_info/HF/HF ADC sum
  hfHists.ADCsum = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiSize";   //DigiMonitor_Hcal/problem_digis/baddigisize/ Digis with Bad Size
  DigiSize = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiOccupancyEta";   //DigiMonitor_Hcal/digi_occupancy/Digi Eta Occupancy Map
  DigiOccupancyEta = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiOccupancyPhi";   //DigiMonitor_Hcal/digi_occupancy/Digi Phi Occupancy Map
  DigiOccupancyPhi = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiNum";   //DigiMonitor_Hcal/digi_info/# of Digis
  DigiNum = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiBQ";   //DigiMonitor_Hcal/digi_errors/# Bad Qual Digis
  DigiBQ = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiBQFrac";   //DigiMonitor_Hcal/digi_errors/Bad Digi Fraction
  DigiBQFrac = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"ProblemDigis";   //DigiMonitor_Hcal/ ProblemDigis
  ProblemDigis = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiOccupancyByDepth";   //DigiMonitor_Hcal/digi_occupancy/Digi Eta-Phi Occupancy Map
  DigiOccupancyByDepth = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiOccupancyVME";   //DigiMonitor_Hcal/digi_occupancy/Digi VME Occupancy Map
  DigiOccupancyVME = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiOccupancySpigot";   //DigiMonitor_Hcal/digi_occupancy/Digi Spigot Occupancy Map
  DigiOccupancySpigot = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiErrorEtaPhi";   //DigiMonitor_Hcal/digi_errors/Digi Geo Error Map
  DigiErrorEtaPhi = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiErrorVME";   //DigiMonitor_Hcal/digi_errors/Digi VME Error Map
  DigiErrorVME = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiErrorSpigot";   //DigiMonitor_Hcal/digi_errors/Digi Spigot Error Map
  DigiErrorSpigot = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");



  getSJ6histos("DigiMonitor_Hcal/problem_digis/"," Problem Digi Rate",ProblemDigisByDepth);
  getSJ6histos("DigiMonitor_Hcal/problem_digis/badcapID/"," Digis with Bad Cap ID Rotation",DigiErrorsBadCapID);
  getSJ6histos("DigiMonitor_Hcal/problem_digis/baddigisize/"," Digis with Bad Size",DigiErrorsBadDigiSize);
  getSJ6histos("DigiMonitor_Hcal/problem_digis/badADCsum/","Digis with ADC sum below threshold ADC counts",DigiErrorsBadADCSum);

  getSJ6histos("DigiMonitor_Hcal/problem_digis/nodigis/"," Digis Missing for a Number of Consecutive Events",DigiErrorsNoDigi);
  getSJ6histos("DigiMonitor_Hcal/problem_digis/data_invalid_error/"," Digis with Data Invalid or Error Bit Set",DigiErrorsDVErr);
  return;
} // void HcalDigiClient::getHistograms()

void HcalDigiClient::resetAllME(){
  
  if(!dbe_) return;
  
  ostringstream name;
  // Reset
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Shape";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Shape";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Shape";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Shape";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Shape - over thresh";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Shape - over thresh";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Shape - over thresh";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Shape - over thresh";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Presamples";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Presamples";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Presamples";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Presamples";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Bad Quality Digis";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Bad Quality Digis";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Bad Quality Digis";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Bad Quality Digis";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Bad Quality Digi Fraction";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Bad Quality Digi Fraction";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Bad Quality Digi Fraction";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Bad Quality Digi Fraction";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Capid 1st Time Slice";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Capid 1st Time Slice";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Capid 1st Time Slice";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Capid 1st Time Slice";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Data Valid Err Bits";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Data Valid Err Bits";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Data Valid Err Bits";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Data Valid Err Bits";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB CapID";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE CapID";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO CapID";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF CapID";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB ADC count per time slice";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE ADC count per time slice";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO ADC count per time slice";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF ADC count per time slice";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB ADC sum";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE ADC sum";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO ADC sum";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF ADC sum";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/baddigisize/ Digis with Bad Size";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Eta Occupancy Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Phi Occupancy Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/# of Digis";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/# Bad Qual Digis";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Bad Digi Fraction";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/ ProblemDigis";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Eta-Phi Occupancy Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi VME Occupancy Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Spigot Occupancy Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Digi Geo Error Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Digi VME Error Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Digi Spigot Error Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  for (int i=0;i<6;++i)
    {
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/"<<subdets_[i]<<" Problem Digi Rate";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/badcapID/"<<subdets_[i]<<" Digis with Bad Cap ID Rotation";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/baddigisize/"<<subdets_[i]<<" Digis with Bad Size";
      resetME(name.str().c_str(),dbe_);
      name.str(""); 
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/badADCsum/"<<subdets_[i]<<" Digis with ADC sum below threshold ADC counts";
      resetME(name.str().c_str(),dbe_);
      name.str(""); 
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/nodigis/"<<subdets_[i]<<" Digis Missing for a Number of Consecutive Events";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/data_invalid_error/"<<subdets_[i]<<" Digis with Data Invalid or Error Bit Set";
      resetME(name.str().c_str(),dbe_);
      name.str("");
    } // for (int i=0;i<6;++i)

  return;
} // void HcalDigiClient::resetAllME()

void HcalDigiClient::htmlExpertOutput(int runNo, string htmlDir, string htmlName){

  
  if (debug_)
    cout << "Preparing HcalDigiClient Expert html output ..." << endl;

  string client = "DigiMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  ofstream htmlFile;
  htmlFile.open((htmlDir +"Expert_"+ htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Digi Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Digis</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table width=100%  border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"DigiMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"DigiMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"DigiMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal Digi Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  if(subDetsOn_[0]) htmlFile << "<a href=\"#HB_Plots\">HB Plots </a></br>" << endl;
  if(subDetsOn_[1]) htmlFile << "<a href=\"#HE_Plots\">HE Plots </a></br>" << endl;
  if(subDetsOn_[2]) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  if(subDetsOn_[3]) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile <<"<a href=\"#Pedestal_Plots\">Pedestal Plots </a></br>"<<endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;


  htmlFile << "<br>" << endl;

  

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  return;
}
void HcalDigiClient::createTests(){
  if(!dbe_) return;

  char meTitle[250], name[250];    
  vector<string> params;
  
  if(debug_) cout <<"Creating Digi tests..."<<endl;
  
  for(int i=0; i<4; ++i){
    if(!subDetsOn_[i]) continue;

    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO";
    
    sprintf(meTitle,"%sHcal/DigiMonitor/%s/%s Digi Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s Digi Errors by Geo_metry",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = dbe_->get(meTitle);
      if(me){
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back((string)meTitle); params.push_back((string)name);  //hist and qtest titles
	params.push_back("0"); params.push_back("1e-10");  //mean ranges
	params.push_back("0"); params.push_back("1e-10");  //rms ranges
	createH2ContentTest(dbe_, params);
      }
    }

    /*
    sprintf(meTitle,"%sHcal/DigiMonitor/%s/%s QIE Cap-ID",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s QIE CapID",type.c_str());
    if(dqmQtests_.find(name) == dqmQtests_.end()){	
      MonitorElement* me = dbe_->get(meTitle);
      if(me){	
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("1.0"); params.push_back("0.975");  //warn, err probs
	params.push_back("0"); params.push_back("3");  //xmin, xmax
	createXRangeTest(dbe_, params);
      }
    }
    */
  } // for (int i=0;i<4;++i)

  return;
} //void HcalDigiClient::createTests()

void HcalDigiClient::loadHistograms(TFile* infile){


  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/DigiMonitor/Digi Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  
  ostringstream name;

  //Load Histograms
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Shape";
  hbHists.shape = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Shape";
  heHists.shape = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Shape";
  hoHists.shape = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Shape";
  hfHists.shape = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Shape - over thresh";
  hbHists.shapeThresh = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Shape - over thresh";
  heHists.shapeThresh = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Shape - over thresh";
  hoHists.shapeThresh = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Shape - over thresh";
  hfHists.shapeThresh = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Presamples";
  hbHists.presample = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Presamples";
  heHists.presample = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Presamples";
  hoHists.presample = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Presamples";
  hfHists.presample = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Bad Quality Digis";
  hbHists.BQ = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Bad Quality Digis";
  heHists.BQ = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Bad Quality Digis";
  hoHists.BQ = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Bad Quality Digis";
  hfHists.BQ = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Bad Quality Digi Fraction";
  hbHists.BQFrac = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Bad Quality Digi Fraction";
  heHists.BQFrac = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Bad Quality Digi Fraction";
  hoHists.BQFrac = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Bad Quality Digi Fraction";
  hfHists.BQFrac = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Capid 1st Time Slice";
  hbHists.DigiFirstCapID = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Capid 1st Time Slice";
  heHists.DigiFirstCapID = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Capid 1st Time Slice";
  hoHists.DigiFirstCapID = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Capid 1st Time Slice";
  hfHists.DigiFirstCapID = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Data Valid Err Bits";
  hbHists.DVerr = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Data Valid Err Bits";
  heHists.DVerr = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Data Valid Err Bits";
  hoHists.DVerr = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Data Valid Err Bits";
  hfHists.DVerr = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB CapID";
  hbHists.CapID = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE CapID";
  heHists.CapID = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO CapID";
  hoHists.CapID = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF CapID";
  hfHists.CapID = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB ADC count per time slice";
  hbHists.ADC = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE ADC count per time slice";
  heHists.ADC = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO ADC count per time slice";
  hoHists.ADC = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF ADC count per time slice";
  hfHists.ADC = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB ADC sum";
  hbHists.ADCsum = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE ADC sum";
  heHists.ADCsum = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO ADC sum";
  hoHists.ADCsum = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF ADC sum";
  hfHists.ADCsum = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/baddigisize/ Digis with Bad Size";
  DigiSize = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Eta Occupancy Map";
  DigiOccupancyEta = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Phi Occupancy Map";
  DigiOccupancyPhi = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/# of Digis";
  DigiNum = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/# Bad Qual Digis";
  DigiBQ = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Bad Digi Fraction";
  DigiBQFrac = static_cast<TH1F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/ ProblemDigis";
  ProblemDigis = static_cast<TH2F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Eta-Phi Occupancy Map";
  DigiOccupancyByDepth = static_cast<TH2F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi VME Occupancy Map";
  DigiOccupancyVME = static_cast<TH2F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Spigot Occupancy Map";
  DigiOccupancySpigot = static_cast<TH2F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Digi Geo Error Map";
  DigiErrorEtaPhi = static_cast<TH2F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Digi VME Error Map";
  DigiErrorVME = static_cast<TH2F*>(infile->Get(name.str().c_str()));
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Digi Spigot Error Map";
  DigiErrorSpigot = static_cast<TH2F*>(infile->Get(name.str().c_str()));
  name.str("");
  for (int i=0;i<6;++i)
    {
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/"<<subdets_[i]<<" Problem Digi Rate";
      ProblemDigisByDepth[i] = static_cast<TH2F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/badcapID/"<<subdets_[i]<<" Digis with Bad Cap ID Rotation";
      DigiErrorsBadCapID[i] = static_cast<TH2F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/baddigisize/"<<subdets_[i]<<" Digis with Bad Size";
      DigiErrorsBadDigiSize[i] = static_cast<TH2F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/badADCsum/"<<subdets_[i]<<" Digis with ADC sum below threshold ADC counts";
      DigiErrorsBadADCSum[i] = static_cast<TH2F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/nodigis/"<<subdets_[i]<<" Digis Missing for a Number of Consecutive Events";
      DigiErrorsNoDigi[i] = static_cast<TH2F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/data_invalid_error/"<<subdets_[i]<<" Digis with Data Invalid or Error Bit Set";
      DigiErrorsDVErr[i] = static_cast<TH2F*>(infile->Get(name.str().c_str()));
      name.str("");
    } // for (int i=0;i<6;++i)

  return;
} // void HcalDigiClient::loadHistograms()




void HcalDigiClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  
  if (debug_) cout << "Preparing HcalDigiClient html output ..." << endl;

  string client = "DigiMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Digi Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Digis</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table width=100%  border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"DigiMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"DigiMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"DigiMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal Digi Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  // Need to implement these later
  //if(subDetsOn_[0]) htmlFile << "<a href=\"#HB_Plots\">HB Plots </a></br>" << endl;
  //if(subDetsOn_[1]) htmlFile << "<a href=\"#HE_Plots\">HE Plots </a></br>" << endl;
  //if(subDetsOn_[2]) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  //if(subDetsOn_[3]) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  /*
  // Scale to number of events
  ProblemDigiCells->Scale(1./ievt_); 
  ProblemDigiCells->SetMinimum(errorFrac_); 

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  histoHTML2(runNo,ProblemDigiCells,"iEta","iPhi", 92, htmlFile,htmlDir);
  htmlFile<<"</tr>"<<endl;
  htmlFile<<"<tr><td> A digi cell is considered bad if there was no digi for that cell in the event, if the digi size was wrong (<=1), if the capid rotation for that digi was incorrect, or if the sum of ADC counts over all time slices for the digi is 0.  <br> If zero-suppression of the HCAL is enabled for a run, this plot may have high occupancy, and you should check the expert plots for more detailed information.</td></tr>"<<endl;
  htmlFile<<"<tr><td><a href=\"Expert_"<< htmlName<<"\">ExpertPlots </a></br></td>"<<endl;
  htmlFile<<"</tr></table><br>"<<endl;

  //ProblemDigiCells->Scale(ievt_); 
  ProblemDigiCells->SetMinimum(0);
  htmlFile <<"<h2>List of Problem Digi Cells</h2>"<<endl;
  htmlFile <<"<table width=75%align = \"center\"><tr align=\"center\">" <<endl; 
  htmlFile <<"<td> Problem Cells</td><td align=\"center\"> Fraction of Events in wh\
ich cells are bad (%)</td></tr>"<<endl; 

  int etabins = ProblemDigiCells->GetNbinsX(); 
  int phibins = ProblemDigiCells->GetNbinsY(); 
  float etaMin=ProblemDigiCells->GetXaxis()->GetXmin(); 
  float phiMin=ProblemDigiCells->GetYaxis()->GetXmin(); 
  
  int eta,phi; 

  // HB problem cells
  for (int depth=0;depth<4; ++depth)
    {
      for (int ieta=1;ieta<=etabins;++ieta) 
	{ 
	  for (int iphi=1; iphi<=phibins;++iphi) 
	    {
	      eta=ieta+int(etaMin)-1; 
	      phi=iphi+int(phiMin)-1; 
	      if (abs(eta)>20 && phi%2!=1) continue;
	      if (abs(eta)>39 && phi%4!=3) continue;
	      if (ProblemDigiCellsHB_DEPTH[depth]->GetBinContent(ieta,iphi)>errorFrac_)
		htmlFile<<"<td align=\"center\">HB ("<<eta<<", "<<phi<<", "<<depth+1<<") </td><td align=\"center\"> "<<100.*ProblemDigiCells_DEPTH[depth]->GetBinContent(ieta,iphi)/ievt_<<"</td></tr>"<<endl; 
	    } // for (int iphi...)
	} // for (int ieta...)
    } // for (int depth...)

  // HE problem cells
  for (int depth=0;depth<4; ++depth)
    {
      for (int ieta=1;ieta<=etabins;++ieta) 
	{ 
	  for (int iphi=1; iphi<=phibins;++iphi) 
	    {
	      eta=ieta+int(etaMin)-1; 
	      phi=iphi+int(phiMin)-1; 
	      if (ProblemDigiCellsHE_DEPTH[depth]->GetBinContent(ieta,iphi)>errorFrac_)
		htmlFile<<"<td align=\"center\">HE ("<<eta<<", "<<phi<<", "<<depth+1<<") </td><td align=\"center\"> "<<100.*ProblemDigiCells_DEPTH[depth]->GetBinContent(ieta,iphi)/ievt_<<"</td></tr>"<<endl; 
	    } // for (int iphi...)
	} // for (int ieta...)
    } // for (int depth...)

  // HO problem cells
  for (int depth=0;depth<4; ++depth)
    {
      for (int ieta=1;ieta<=etabins;++ieta) 
	{ 
	  for (int iphi=1; iphi<=phibins;++iphi) 
	    {
	      eta=ieta+int(etaMin)-1; 
	      phi=iphi+int(phiMin)-1; 
	      if (ProblemDigiCellsHO_DEPTH[depth]->GetBinContent(ieta,iphi)>errorFrac_)
		htmlFile<<"<td align=\"center\">HO ("<<eta<<", "<<phi<<", "<<depth+1<<") </td><td align=\"center\"> "<<100.*ProblemDigiCells_DEPTH[depth]->GetBinContent(ieta,iphi)/ievt_<<"</td></tr>"<<endl; 
	    } // for (int iphi...)
	} // for (int ieta...)
    } // for (int depth...)

  // HF problem cells
  for (int depth=0;depth<4; ++depth)
    {
      for (int ieta=1;ieta<=etabins;++ieta) 
	{ 
	  for (int iphi=1; iphi<=phibins;++iphi) 
	    {
	      eta=ieta+int(etaMin)-1; 
	      phi=iphi+int(phiMin)-1; 
	      if (ProblemDigiCellsHF_DEPTH[depth]->GetBinContent(ieta,iphi)>errorFrac_)
		htmlFile<<"<td align=\"center\">HF ("<<eta<<", "<<phi<<", "<<depth+1<<") </td><td align=\"center\"> "<<100.*ProblemDigiCells_DEPTH[depth]->GetBinContent(ieta,iphi)/ievt_<<"</td></tr>"<<endl; 
	    } // for (int iphi...)
	} // for (int ieta...)
    } // for (int depth...)
  */

  htmlFile << "</table>" <<endl; 

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();


  htmlExpertOutput(runNo, htmlDir, htmlName);
  return;
}

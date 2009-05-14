#include <DQM/HcalMonitorClient/interface/HcalDigiClient.h>
#include <math.h>
#include <iostream>
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

HcalDigiClient::HcalDigiClient(){}

void HcalDigiClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName)
{
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>0)
    cout <<"<HcalDigiClient> init(const ParameterSet& ps, QMStore* dbe, string clientName)"<<endl;

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
      DigiOccupancyByDepth[i]    =0;
    } // for (int i=0;i<6;++i)

  for (int i=0;i<9;++i)
    {
      hbHists.TS_sum_plus[i]=0;
      hbHists.TS_sum_minus[i]=0;
      heHists.TS_sum_plus[i]=0;
      heHists.TS_sum_minus[i]=0;
      hoHists.TS_sum_plus[i]=0;
      hoHists.TS_sum_minus[i]=0;
      hfHists.TS_sum_plus[i]=0;
      hfHists.TS_sum_minus[i]=0;
    }

  subdets_.push_back("HB HF Depth 1 ");
  subdets_.push_back("HB HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO ZDC ");
  subdets_.push_back("HE Depth 1 ");
  subdets_.push_back("HE Depth 2 ");

  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiClient INIT  -> "<<cpu_timer.cpuTime()<<endl;
    }
  return;
} // void HcalDigiClient::init(...)

HcalDigiClient::~HcalDigiClient(){
  cleanup();
}

void HcalDigiClient::beginJob(void){
  
  if ( debug_>0 ) 
    cout << "HcalDigiClient: beginJob" << endl;
  
  ievt_ = 0;
  jevt_ = 0;
  setup();
  resetAllME();
  return;
}

void HcalDigiClient::beginRun(void){

  if ( debug_>0 ) 
    cout << "HcalDigiClient: beginRun" << endl;

  jevt_ = 0;
  setup();
  resetAllME();
  return;
}

void HcalDigiClient::endJob(void) {

  if ( debug_>0 )
    cout << "HcalDigiClient: endJob, ievt = " << ievt_ << endl;

  cleanup(); 
  return;
}

void HcalDigiClient::endRun(void) {

  if ( debug_ >0) 
    cout << "HcalDigiClient: endRun, jevt = " << jevt_ << endl;

  cleanup();  
  return;
}

void HcalDigiClient::setup(void) {
  
  return;
}

void HcalDigiClient::cleanup(void) 
{
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
	  if (DigiOccupancyByDepth[i]) delete DigiOccupancyByDepth[i];
	} // for (int i=0;i<6;++i)
      for (int i=0;i<9;++i)
	{
	  if (hbHists.TS_sum_plus[i])  delete hbHists.TS_sum_plus[i];
	  if (hbHists.TS_sum_minus[i]) delete hbHists.TS_sum_minus[i];
	  if (heHists.TS_sum_plus[i])  delete heHists.TS_sum_plus[i];
	  if (heHists.TS_sum_minus[i]) delete heHists.TS_sum_minus[i];
	  if (hoHists.TS_sum_plus[i])  delete hoHists.TS_sum_plus[i];
	  if (hoHists.TS_sum_minus[i]) delete hoHists.TS_sum_minus[i];
	  if (hfHists.TS_sum_plus[i])  delete hfHists.TS_sum_plus[i];
	  if (hfHists.TS_sum_minus[i]) delete hfHists.TS_sum_minus[i];
	}
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
      DigiOccupancyByDepth[i]    =0;
    } // for (int i=0;i<6;++i)
  for (int i=0;i<9;++i)
   {
      hbHists.TS_sum_plus[i]=0;
      hbHists.TS_sum_minus[i]=0;
      heHists.TS_sum_plus[i]=0;
      heHists.TS_sum_minus[i]=0;
      hoHists.TS_sum_plus[i]=0;
      hoHists.TS_sum_minus[i]=0;
      hfHists.TS_sum_plus[i]=0;
      hfHists.TS_sum_minus[i]=0;
    }

  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  return;
} // void HcalDigiClient::cleanup(void)


void HcalDigiClient::report(){
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if ( debug_ ) cout << "HcalDigiClient: report" << endl;
  
  ostringstream name;
  name<<process_.c_str()<<"Hcal/DigiMonitor_Hcal/Digi Task Event Number";
  MonitorElement* me = 0;
  if(dbe_) me = dbe_->get(name.str().c_str());
  if ( me ) 
    {
      string s = me->valueString();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
      if ( debug_ ) cout << "Found '" << name.str().c_str() << "'" << endl;
    }
  name.str("");
  getHistograms();

  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiClient REPORT  -> "<<cpu_timer.cpuTime()<<endl;
    }
  return;
}

void HcalDigiClient::analyze(void){

  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  jevt_++;
  int updates = 0;

  if ( updates % 10 == 0 ) {
    if ( debug_ ) cout << "HcalDigiClient: " << updates << " updates" << endl;
  }
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiClient ANALYZE  -> "<<cpu_timer.cpuTime()<<endl;
    }
  return;
}

void HcalDigiClient::getHistograms(){
  if(!dbe_) return;
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (debug_>0) cout <<"HcalDigiClient> getHistograms()"<<endl;

  ostringstream name;
  TH2F* dummy2D = new TH2F();
  TH1F* dummy1D = new TH1F();

  // Get Histograms
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Shape";  //hbHists.shape
  hbHists.shape = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Shape";  //heHists.shape
  heHists.shape = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Shape";  //hoHists.shape
  hoHists.shape = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Shape";  //hfHists.shape
  hfHists.shape = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Shape - over thresh";  //hbHists.shapeThresh
  hbHists.shapeThresh = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Shape - over thresh";  //heHists.shapeThresh
  heHists.shapeThresh = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Shape - over thresh";  //hoHists.shapeThresh
  hoHists.shapeThresh = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Shape - over thresh";  //hfHists.shapeThresh
  hfHists.shapeThresh = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Presamples";  //hbHists.presample
  hbHists.presample = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Presamples";  //heHists.presample
  heHists.presample = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Presamples";  //hoHists.presample
  hoHists.presample = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Presamples";  //hfHists.presample
  hfHists.presample = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Bad Quality Digis";  //hbHists.BQ
  hbHists.BQ = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Bad Quality Digis";  //heHists.BQ
  heHists.BQ = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Bad Quality Digis";  //hoHists.BQ
  hoHists.BQ = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Bad Quality Digis";  //hfHists.BQ
  hfHists.BQ = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Bad Quality Digi Fraction";  //hbHists.BQFrac
  hbHists.BQFrac = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Bad Quality Digi Fraction";  //heHists.BQFrac
  heHists.BQFrac = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Bad Quality Digi Fraction";  //hoHists.BQFrac
  hoHists.BQFrac = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Bad Quality Digi Fraction";  //hfHists.BQFrac
  hfHists.BQFrac = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Capid 1st Time Slice";  //hbHists.DigiFirstCapID
  hbHists.DigiFirstCapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Capid 1st Time Slice";  //heHists.DigiFirstCapID
  heHists.DigiFirstCapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Capid 1st Time Slice";  //hoHists.DigiFirstCapID
  hoHists.DigiFirstCapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Capid 1st Time Slice";  //hfHists.DigiFirstCapID
  hfHists.DigiFirstCapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Data Valid Err Bits";  //hbHists.DVerr
  hbHists.DVerr = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Data Valid Err Bits";  //heHists.DVerr
  heHists.DVerr = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Data Valid Err Bits";  //hoHists.DVerr
  hoHists.DVerr = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Data Valid Err Bits";  //hfHists.DVerr
  hfHists.DVerr = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB CapID";  //hbHists.CapID
  hbHists.CapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE CapID";  //heHists.CapID
  heHists.CapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO CapID";  //hoHists.CapID
  hoHists.CapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF CapID";  //hfHists.CapID
  hfHists.CapID = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB ADC count per time slice";  //hbHists.ADC
  hbHists.ADC = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE ADC count per time slice";  //heHists.ADC
  heHists.ADC = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO ADC count per time slice";  //hoHists.ADC
  hoHists.ADC = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF ADC count per time slice";  //hfHists.ADC
  hfHists.ADC = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB ADC sum";  //hbHists.ADCsum
  hbHists.ADCsum = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE ADC sum";  //heHists.ADCsum
  heHists.ADCsum = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO ADC sum";  //hoHists.ADCsum
  hoHists.ADCsum = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF ADC sum";  //hfHists.ADCsum
  hfHists.ADCsum = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/problem_digis/baddigisize/Digi Size";   //DigiSize
  DigiSize = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Eta Occupancy Map";   //DigiOccupancyEta
  DigiOccupancyEta = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Phi Occupancy Map";   //DigiOccupancyPhi
  DigiOccupancyPhi = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/# of Digis";   //DigiNum
  DigiNum = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/# Bad Qual Digis";   //DigiBQ
  DigiBQ = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Bad Digi Fraction";   //DigiBQFrac
  DigiBQFrac = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/ ProblemDigis";   //ProblemDigis
  ProblemDigis = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi VME Occupancy Map";   //DigiOccupancyVME
  DigiOccupancyVME = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/Digi Spigot Occupancy Map";   //DigiOccupancySpigot
  DigiOccupancySpigot = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Digi Geo Error Map";   //DigiErrorEtaPhi
  DigiErrorEtaPhi = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Digi VME Error Map";   //DigiErrorVME
  DigiErrorVME = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_errors/Digi Spigot Error Map";   //DigiErrorSpigot
  DigiErrorSpigot = getAnyHisto(dummy2D, name.str(), process_, dbe_, debug_, cloneME_);
  name.str("");
  for (int i=0;i<9;++i)
    {
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Plus Time Slices "<<i<<" and "<<i+1;  //hbHists.TS_sum_plus[i]
      hbHists.TS_sum_plus[i] = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Plus Time Slices "<<i<<" and "<<i+1;  //heHists.TS_sum_plus[i]
      heHists.TS_sum_plus[i] = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Plus Time Slices "<<i<<" and "<<i+1;  //hoHists.TS_sum_plus[i]
      hoHists.TS_sum_plus[i] = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Plus Time Slices "<<i<<" and "<<i+1;  //hfHists.TS_sum_plus[i]
      hfHists.TS_sum_plus[i] = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Minus Time Slices "<<i<<" and "<<i+1;  //hbHists.TS_sum_minus[i]
      hbHists.TS_sum_minus[i] = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Minus Time Slices "<<i<<" and "<<i+1;  //heHists.TS_sum_minus[i]
      heHists.TS_sum_minus[i] = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Minus Time Slices "<<i<<" and "<<i+1;  //hoHists.TS_sum_minus[i]
      hoHists.TS_sum_minus[i] = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Minus Time Slices "<<i<<" and "<<i+1;  //hfHists.TS_sum_minus[i]
      hfHists.TS_sum_minus[i] = getAnyHisto(dummy1D, name.str(), process_, dbe_, debug_, cloneME_);
      name.str("");
    } // for (int i=0;i<9;++i)

  getSJ6histos("DigiMonitor_Hcal/digi_occupancy/"," Digi Eta-Phi Occupancy Map",DigiOccupancyByDepth);
  getSJ6histos("DigiMonitor_Hcal/problem_digis/"," Problem Digi Rate",ProblemDigisByDepth);
  getSJ6histos("DigiMonitor_Hcal/problem_digis/badcapID/"," Digis with Bad Cap ID Rotation",DigiErrorsBadCapID);
  getSJ6histos("DigiMonitor_Hcal/problem_digis/baddigisize/"," Digis with Bad Size",DigiErrorsBadDigiSize);
  getSJ6histos("DigiMonitor_Hcal/problem_digis/badADCsum/"," Digis with ADC sum below threshold ADC counts",DigiErrorsBadADCSum);
  getSJ6histos("DigiMonitor_Hcal/problem_digis/nodigis/"," Digis Missing for a Number of Consecutive Events",DigiErrorsNoDigi);
  getSJ6histos("DigiMonitor_Hcal/problem_digis/data_invalid_error/"," Digis with Data Invalid or Error Bit Set",DigiErrorsDVErr);
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiClient GET HISTOGRAMS  -> "<<cpu_timer.cpuTime()<<endl;
    }
  return;
} // void HcalDigiClient::getHistograms()


void HcalDigiClient::resetAllME()
{
  if (debug_>0) cout <<"HcalDigiClient> resetAllME()"<<endl;
  if(!dbe_) return;
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
 
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

  for (int i=0;i<9;++i)
    {
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Plus Time Slices "<<i<<" and "<<i+1;  //hbHists.TS_sum_plus[i]
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Plus Time Slices "<<i<<" and "<<i+1;  //heHists.TS_sum_plus[i]
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Plus Time Slices "<<i<<" and "<<i+1;  //hoHists.TS_sum_plus[i]
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Plus Time Slices "<<i<<" and "<<i+1;  //hfHists.TS_sum_plus[i]
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Minus Time Slices "<<i<<" and "<<i+1;  //hbHists.TS_sum_minus[i]
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Minus Time Slices "<<i<<" and "<<i+1;  //heHists.TS_sum_minus[i]
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Minus Time Slices "<<i<<" and "<<i+1;  //hoHists.TS_sum_minus[i]
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Minus Time Slices "<<i<<" and "<<i+1;  //hfHists.TS_sum_minus[i]
      resetME(name.str().c_str(),dbe_);
      name.str("");
    } // for (int i=0;i<9;++i)

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
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiClient RESET ALL ME  -> "<<cpu_timer.cpuTime()<<endl;
    }
  return;
} // void HcalDigiClient::resetAllME()

void HcalDigiClient::htmlExpertOutput(int runNo, string htmlDir, string htmlName){
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (debug_>1)
    cout << "<HcalDigiClient> Preparing HcalDigiClient Expert html output ..." << endl;
  
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
  htmlFile <<"<a name=\"EXPERT_DIGI_TOP\" href = \".\"> Back to Main HCAL DQM Page </a><br>"<<endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Digi Status Page </a><br>"<<endl;
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

  htmlFile << "<table width=100%  border = 1>"<<endl;
  htmlFile << "<tr><td align=\"center\" colspan=2><a href=\"#OVERALL_PROBLEMS\">PROBLEM CELLS BY DEPTH </a></td></tr>"<<endl;
  htmlFile << "<tr><td align=\"center\">"<<endl;
  htmlFile<<"<br><a href=\"#OCCUPANCY\">Digi Occupancy Plots </a>"<<endl;
  htmlFile<<"<br><a href=\"#DIGISIZE\">Digi Bad Size Plots</a>"<<endl;
  htmlFile<<"<br><a href=\"#DIGICAPID\">Digi Bad Cap ID Plots </a>"<<endl;
  htmlFile<<"</td><td align=\"center\">"<<endl;
  htmlFile<<"<br><a href=\"#DIGIADCSUM\">Digi Bad ADC Sum Plots </a>"<<endl;
  htmlFile<<"<br><a href=\"#DIGIERRORBIT\">Digi Bad Error Bit Plots </a>"<<endl;
  htmlFile<<"<br><a href=\"#NODIGI\">Missing Digi Plots </a>"<<endl;
  htmlFile<<"</td></tr><tr><td align=\"center\">"<<endl;
  htmlFile<<"<br><a href=\"#HBDIGI\">HB Digi Plots </a>"<<endl;
  htmlFile<<"<br><a href=\"#HEDIGI\">HE Digi Plots </a>"<<endl;
  htmlFile<<"</td><td align=\"center\">"<<endl;
  htmlFile<<"<br><a href=\"#HODIGI\">HO Digi Plots </a>"<<endl;
  htmlFile<<"<br><a href=\"#HFDIGI\">HF Digi Plots </a>"<<endl;

  htmlFile << "</td></tr>"<<endl;
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><br>"<<endl;

  // Plot overall errors
  htmlFile << "<h2><strong><a name=\"OVERALL_PROBLEMS\">Eta-Phi Maps of Problem Cells By Depth</strong></h2>"<<endl;
  htmlFile <<" These plots of problem cells combine results from all digi tests<br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  
  // Depths are stored as:  0:  HF/HF depth 1, 1:  HF/HF 2, 2:  HE 3, 3:  HO/ZDC, 4: HE 1, 5:  HE2
  // remap so that HE depths are plotted consecutively
  int mydepth[6]={0,1,4,5,2,3};
  for (int i=0;i<3;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,ProblemDigisByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ProblemDigisByDepth[mydepth[2*i]+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }

  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><hr><br>"<<endl;


  // Occupancy Plots
  htmlFile << "<h2><strong><a name=\"OCCUPANCY\">Occupancy Plots</strong></h2>"<<endl;
  htmlFile <<"This shows average digi occupancy of each cell per event<br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  for (int i=0;i<3;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,DigiOccupancyByDepth[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiOccupancyByDepth[mydepth[2*i+1]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile<<"<tr align=\"left\">"<<endl;
  htmlAnyHisto(runNo,DigiOccupancyEta,"","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,DigiOccupancyPhi,"","", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<endl;
  htmlFile<<"<tr align=\"left\">"<<endl;
  htmlAnyHisto(runNo,DigiOccupancyVME,"","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,DigiOccupancySpigot,"","", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<endl;
  htmlFile <<"</table>"<<endl;

  // Digi Size Plots
  htmlFile << "<h2><strong><a name=\"DIGISIZE\">Digi Size Plots</strong></h2>"<<endl;
  htmlFile <<"This shows the fraction of events for each digi in which the digi's size is outside the expected range.<br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  for (int i=0;i<3;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,DigiErrorsBadDigiSize[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsBadDigiSize[mydepth[2*i+1]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile<<"<tr align=\"left\">"<<endl;
  htmlAnyHisto(runNo,DigiSize,"","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,DigiNum,"","", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<endl;
  htmlFile <<"</table>"<<endl;

  // Digi Cap ID Plots
  htmlFile << "<h2><strong><a name=\"DIGICAPID\">Digi Cap ID Plots</strong></h2>"<<endl;
  htmlFile <<"This shows the fraction of events for each digi in which the digi's capacitor-ID rotation is incorrect.<br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  for (int i=0;i<3;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,DigiErrorsBadCapID[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsBadCapID[mydepth[2*i+1]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;

  // Digi ADC SUM Plots
  htmlFile << "<h2><strong><a name=\"DIGIADCSUM\">Digi ADC Sum Plots</strong></h2>"<<endl;
  htmlFile <<"This shows the fraction of events for each digi in which the digi's ADC sum is below threshold.<br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  for (int i=0;i<3;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,DigiErrorsBadADCSum[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsBadADCSum[mydepth[2*i+1]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;

  // Digi Error Bit Plots
  htmlFile << "<h2><strong><a name=\"DIGIERRORBIT\">Digi Error Bit Plots</strong></h2>"<<endl;
  htmlFile <<"This shows average number of digi errors/data invalids of each cell per event<br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  for (int i=0;i<3;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,DigiErrorsDVErr[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsDVErr[mydepth[2*i+1]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlAnyHisto(runNo,DigiBQ,"","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,DigiBQFrac,"","", 92, htmlFile, htmlDir);
  htmlFile<<"</tr></table>"<<endl;


  // Missing Digi
  htmlFile << "<h2><strong><a name=\"NODIGI\">Missing digi Plots</strong></h2>"<<endl;
  htmlFile <<"This shows digis that are not present for a number of consecutive events <br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  for (int i=0;i<3;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,DigiErrorsNoDigi[mydepth[2*i]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsNoDigi[mydepth[2*i+1]],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;


  // HB Plots
  htmlFile << "<h2><strong><a name=\"HBDIGI\">HB digi Plots</strong></h2>"<<endl;
  htmlFile <<"This shows expert-level information for the HB subdetector digis <br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hbHists.shape,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hbHists.shapeThresh,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hbHists.BQ,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hbHists.BQFrac,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hbHists.DigiFirstCapID,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hbHists.CapID,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hbHists.ADC,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hbHists.ADCsum,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hbHists.presample,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hbHists.DVerr,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  for (int i=0;i<9;++i)
    {
      htmlFile << "</tr>"<<endl;
      htmlAnyHisto(runNo,hbHists.TS_sum_plus[i],"","",92,htmlFile,htmlDir);
      htmlAnyHisto(runNo,hbHists.TS_sum_minus[i],"","",92,htmlFile,htmlDir);
      htmlFile << "</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;


  // HE Plots
  htmlFile << "<h2><strong><a name=\"HEDIGI\">HE digi Plots</strong></h2>"<<endl;
  htmlFile <<"This shows expert-level information for the HE subdetector digis <br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,heHists.shape,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,heHists.shapeThresh,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,heHists.BQ,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,heHists.BQFrac,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,heHists.DigiFirstCapID,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,heHists.CapID,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,heHists.ADC,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,heHists.ADCsum,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,heHists.presample,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,heHists.DVerr,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  for (int i=0;i<9;++i)
    {
      htmlFile << "</tr>"<<endl;
      htmlAnyHisto(runNo,heHists.TS_sum_plus[i],"","",92,htmlFile,htmlDir);
      htmlAnyHisto(runNo,heHists.TS_sum_minus[i],"","",92,htmlFile,htmlDir);
      htmlFile << "</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;

  // HO Plots
  htmlFile << "<h2><strong><a name=\"HODIGI\">HO digi Plots</strong></h2>"<<endl;
  htmlFile <<"This shows expert-level information for the HO subdetector digis <br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hoHists.shape,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hoHists.shapeThresh,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hoHists.BQ,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hoHists.BQFrac,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hoHists.DigiFirstCapID,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hoHists.CapID,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hoHists.ADC,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hoHists.ADCsum,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hoHists.presample,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hoHists.DVerr,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  for (int i=0;i<9;++i)
    {
      htmlFile << "</tr>"<<endl;
      htmlAnyHisto(runNo,hoHists.TS_sum_plus[i],"","",92,htmlFile,htmlDir);
      htmlAnyHisto(runNo,hoHists.TS_sum_minus[i],"","",92,htmlFile,htmlDir);
      htmlFile << "</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;

  // HF Plots
  htmlFile << "<h2><strong><a name=\"HFDIGI\">HF digi Plots</strong></h2>"<<endl;
  htmlFile <<"This shows expert-level information for the HF subdetector digis <br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hfHists.shape,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hfHists.shapeThresh,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hfHists.BQ,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hfHists.BQFrac,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hfHists.DigiFirstCapID,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hfHists.CapID,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hfHists.ADC,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hfHists.ADCsum,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,hfHists.presample,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hfHists.DVerr,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<endl;
  for (int i=0;i<9;++i)
    {
      htmlFile << "</tr>"<<endl;
      htmlAnyHisto(runNo,hfHists.TS_sum_plus[i],"","",92,htmlFile,htmlDir);
      htmlAnyHisto(runNo,hfHists.TS_sum_minus[i],"","",92,htmlFile,htmlDir);
      htmlFile << "</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();
  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiClient HTML EXPERT  -> "<<cpu_timer.cpuTime()<<endl;
    }
  return;
}

void HcalDigiClient::createTests(){
  if(!dbe_) return;
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
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

  if (debug_>0) cout <<"HcalDigiClient> loadHistograms(TFile* infile)"<<endl;
  
  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/DigiMonitor_Hcal/Digi Task Event Number");
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
  DigiSize = static_cast<TH2F*>(infile->Get(name.str().c_str()));
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

  for (int i=0;i<9;++i)
    {
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Plus Time Slices "<<i<<" and "<<i+1;  //hbHists.TS_sum_plus[i]
      hbHists.TS_sum_plus[i] = static_cast<TH1F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Plus Time Slices "<<i<<" and "<<i+1;  //heHists.TS_sum_plus[i]
      heHists.TS_sum_plus[i] = static_cast<TH1F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Plus Time Slices "<<i<<" and "<<i+1;  //hoHists.TS_sum_plus[i]
      hoHists.TS_sum_plus[i] = static_cast<TH1F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Plus Time Slices "<<i<<" and "<<i+1;  //hfHists.TS_sum_plus[i]
      hfHists.TS_sum_plus[i] = static_cast<TH1F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Minus Time Slices "<<i<<" and "<<i+1;  //hbHists.TS_sum_minus[i]
      hbHists.TS_sum_minus[i] = static_cast<TH1F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Minus Time Slices "<<i<<" and "<<i+1;  //heHists.TS_sum_minus[i]
      heHists.TS_sum_minus[i] = static_cast<TH1F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Minus Time Slices "<<i<<" and "<<i+1;  //hoHists.TS_sum_minus[i]
      hoHists.TS_sum_minus[i] = static_cast<TH1F*>(infile->Get(name.str().c_str()));
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Minus Time Slices "<<i<<" and "<<i+1;  //hfHists.TS_sum_minus[i]
      hfHists.TS_sum_minus[i] = static_cast<TH1F*>(infile->Get(name.str().c_str()));
      name.str("");
    } // for (int i=0;i<9;++i)

  for (int i=0;i<6;++i)
    {
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_occupancy/"<<subdets_[i]<<" Digi Eta-Phi Occupancy Map";
      DigiOccupancyByDepth[i] = static_cast<TH2F*>(infile->Get(name.str().c_str()));
      name.str("");
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




void HcalDigiClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  if (debug_>0) cout << "<HcalDigiClient::htmlOutput> Preparing html output ..." << endl;

  getHistograms(); // only do this here; no need to do it in regular analyze() method?

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

  htmlFile << "<h2><strong>Hcal Digi Status</strong></h2>" << endl;

  htmlFile << "<table align=\"center\" border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  htmlAnyHisto(runNo,ProblemDigis,"i#eta","i#phi", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<endl;
  htmlFile<<"<tr align=\"center\"><td> A digi is considered bad if the digi size is incorrect, the cap ID rotation is incorrect, or the ADC sum for the digi is less than some threshold value.  It is also considered bad if its error bit is on or its data valid bit is off."<<endl;

  htmlFile<<"</td>"<<endl;
  htmlFile<<"</tr></table>"<<endl;
  htmlFile<<"<hr><table align=\"center\" border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile<<"<tr><td align=center><a href=\"Expert_"<< htmlName<<"\"><h2>Detailed Digi Plots</h2> </a></br></td>"<<endl;
  htmlFile<<"</tr></table><br><hr>"<<endl;

 // Now print out problem cells
  htmlFile <<"<br>"<<endl;
  htmlFile << "<h2><strong>Hcal Problem Digis</strong></h2>" << endl;
  htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile <<"<td> Problem Digis<br>(ieta, iphi, depth)</td><td align=\"center\"> Fraction of Events <br>in which cells are bad (%)</td></tr>"<<endl;

  if (ProblemDigis==0)
    {
      if (debug_>0) cout <<"<HcalDigiClient::htmlOutput>  ERROR: can't find Problem Rec Hit plot!"<<endl;
      return;
    }
  int etabins  = ProblemDigis->GetNbinsX();
  int phibins  = ProblemDigis->GetNbinsY();
  float etaMin = ProblemDigis->GetXaxis()->GetXmin();
  float phiMin = ProblemDigis->GetYaxis()->GetXmin();

  int eta,phi;

  ostringstream name;
  for (int depth=0;depth<6; ++depth)
    {
      for (int ieta=1;ieta<=etabins;++ieta)
        {
          for (int iphi=1; iphi<=phibins;++iphi)
            {
              eta=ieta+int(etaMin)-1;
              phi=iphi+int(phiMin)-1;
	      if (abs(eta)>20 && phi%2!=1) continue;
	      if (abs(eta)>39 && phi%4!=3) continue;
	      int mydepth=depth+1;
	      if (mydepth>4) mydepth-=4; // last two depth values are for HE depth 1,2
	      if (ProblemDigisByDepth[depth]==0)
		{
		  continue;
		}
	      if (ProblemDigisByDepth[depth]->GetBinContent(ieta,iphi)>0)
		{
		  if (depth<2)
		    (fabs(eta)<29) ? name<<"HB" : name<<"HF";
		  else if (depth==3)
		    (fabs(eta)<42) ? name<<"HO" : name<<"ZDC";
		  else name <<"HE";
		  htmlFile<<"<td>"<<name.str().c_str()<<" ("<<eta<<", "<<phi<<", "<<mydepth<<")</td><td align=\"center\">"<<ProblemDigisByDepth[depth]->GetBinContent(ieta,iphi)*100.<<"</td></tr>"<<endl;

		  name.str("");
		}
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // for (int depth=0;...)
  
  
  // html page footer
  htmlFile <<"</table> " << endl;
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();
  htmlExpertOutput(runNo, htmlDir, htmlName);

  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiClient HTMLOUTPUT  -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;

}

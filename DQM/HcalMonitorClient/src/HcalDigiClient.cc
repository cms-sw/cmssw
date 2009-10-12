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
    std::cout <<"<HcalDigiClient> init(const ParameterSet& ps, QMStore* dbe, string clientName)"<<std::endl;

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
  DigiOccupancyVME    =0;
  DigiOccupancySpigot    =0;
  DigiErrorEtaPhi    =0;
  DigiErrorVME    =0;
  DigiErrorSpigot    =0;
  for (int i=0;i<4;++i)
    {
      BadDigisByDepth[i]    =0;
      DigiErrorsBadCapID[i]    =0;
      DigiErrorsBadDigiSize[i]    =0;
      DigiErrorsBadADCSum[i]    =0;
      //DigiErrorsNoDigi[i]    =0;
      DigiErrorsDVErr[i]    =0;
      DigiOccupancyByDepth[i]    =0;
      DigiErrorsUnpacker[i] = 0;
      DigiErrorsBadFibBCNOff[i] = 0;
    } // for (int i=0;i<4;++i)

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

  subdets_.push_back("HB HE HF Depth 1 ");
  subdets_.push_back("HB HE HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO Depth 4 ");

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiClient INIT  -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalDigiClient::init(...)

HcalDigiClient::~HcalDigiClient(){
  //cleanup();
}

void HcalDigiClient::beginJob()
{
  if ( debug_>0 ) 
    std::cout << "HcalDigiClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;
  setup();
  resetAllME();

  if (!dbe_) return;

  stringstream mydir;
  mydir<<rootFolder_<<"/DigiMonitor_Hcal";

  //cout <<"DIGI ROOT FOLDER = "<<rootFolder_<<endl;
  dbe_->setCurrentFolder(mydir.str().c_str());
  ProblemCells=dbe_->book2D(" ProblemDigis",
			   " Problem Digi Rate for all HCAL;i#eta;i#phi",
			   85,-42.5,42.5,
			   72,0.5,72.5);
  SetEtaPhiLabels(ProblemCells);
  (ProblemCells->getTH2F())->SetMinimum(0);
  (ProblemCells->getTH2F())->SetMaximum(1);
  mydir<<"/problem_digis";
  dbe_->setCurrentFolder(mydir.str().c_str());
  ProblemCellsByDepth.setup(dbe_," Problem Digi Rate");
  for  (unsigned int d=0;d<ProblemCellsByDepth.depth.size();++d)
    {
      if (ProblemCellsByDepth.depth[d]!=0) 
	{
	  (ProblemCellsByDepth.depth[d]->getTH2F())->SetMinimum(0);
	  (ProblemCellsByDepth.depth[d]->getTH2F())->SetMaximum(1);
	}
    }


  return;
}

void HcalDigiClient::beginRun(void){

  if ( debug_>0 ) 
    std::cout << "HcalDigiClient: beginRun" << std::endl;

  jevt_ = 0;
  setup();
  resetAllME();
  return;
}

void HcalDigiClient::endJob(void) {

  if ( debug_>0 )
    std::cout << "HcalDigiClient: endJob, ievt = " << ievt_ << std::endl;

  cleanup(); 
  return;
}

void HcalDigiClient::endRun(void) {

  if ( debug_ >0) 
    std::cout << "HcalDigiClient: endRun, jevt = " << jevt_ << std::endl;

  calculateProblems();
  cleanup();  
  return;
}

void HcalDigiClient::setup(void) {
  
  return;
}

void HcalDigiClient::cleanup(void) 
{
  // let framework deal with deleting pointers
  if ( 1<0 && cloneME_ ) 
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
      if (DigiOccupancyVME) delete DigiOccupancyVME;
      if (DigiOccupancySpigot) delete DigiOccupancySpigot;
      if (DigiErrorEtaPhi) delete DigiErrorEtaPhi;
      if (DigiErrorVME) delete DigiErrorVME;
      if (DigiErrorSpigot) delete DigiErrorSpigot;
      for (int i=0;i<4;++i)
	{
	  if (BadDigisByDepth[i]) delete BadDigisByDepth[i];
	  if (DigiErrorsBadCapID[i]) delete DigiErrorsBadCapID[i];
	  if (DigiErrorsBadDigiSize[i]) delete DigiErrorsBadDigiSize[i];
	  if (DigiErrorsBadADCSum[i]) delete DigiErrorsBadADCSum[i];
	  //if (DigiErrorsNoDigi[i]) delete DigiErrorsNoDigi[i];
	  if (DigiErrorsDVErr[i]) delete DigiErrorsDVErr[i];
	  if (DigiErrorsUnpacker[i]) delete DigiErrorsUnpacker[i];
	  if (DigiOccupancyByDepth[i]) delete DigiOccupancyByDepth[i];
	  if (DigiErrorsBadFibBCNOff[i]) delete DigiErrorsBadFibBCNOff[i];
	} // for (int i=0;i<4;++i)
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

  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  return;
} // void HcalDigiClient::cleanup(void)


void HcalDigiClient::calculateProblems()
{
  getProblemHistograms();
  
  double problemvalue=0;
  int etabins=0, phibins=0, zside=0;

  // Clear away old problems
  if (ProblemCells!=0)
      ProblemCells->Reset();

  for  (unsigned int d=0;d<ProblemCellsByDepth.depth.size();++d)
    {
      if (ProblemCellsByDepth.depth[d]!=0) 
	  ProblemCellsByDepth.depth[d]->Reset();
    }
  for (unsigned int d=0;d<ProblemCellsByDepth.depth.size();++d)
    {
      // bad digi rate = bad digis/(good+bad)
      if (ProblemCellsByDepth.depth[d]==0) continue;
      if (BadDigisByDepth[d]==0 || DigiOccupancyByDepth[d]==0) continue; //need both good & bad histograms to calculate bad rate
      etabins=ProblemCellsByDepth.depth[d]->getNbinsX();
      phibins=ProblemCellsByDepth.depth[d]->getNbinsY();

      for (int eta=0;eta<etabins;++eta)
	{
	  int ieta=CalcIeta(eta,d+1);
	  if (ieta==-9999) continue;
	  zside=0;
	  if (isHF(eta,d+1))
	     ieta<0 ? zside = -1 : zside = 1;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      if (BadDigisByDepth[d]->GetBinContent(eta+1,phi+1) > 0)
		{
		  problemvalue=(BadDigisByDepth[d]->GetBinContent(eta+1,phi+1)*1./(BadDigisByDepth[d]->GetBinContent(eta+1,phi+1)+DigiOccupancyByDepth[d]->GetBinContent(eta+1,phi+1)));
		  ProblemCellsByDepth.depth[d]->setBinContent(eta+1,phi+1,problemvalue);
		  // temporary fill; doesn't yet show a meaningful rate
		  if (ProblemCells!=0)
		    ProblemCells->Fill(ieta+zside,phi+1,problemvalue);
		} // bad digis found
	    } // phi loop
	} // eta loop
    } // depth loop


  if (ProblemCells==0)
      if (debug_>0) std::cout <<"<HcalDeadCellClient::analyze> ProblemCells histogram does not exist!"<<endl;
  // no need to renormalize ProblemCells; it's min and max were already set
  return;

} // calculateProblems()

void HcalDigiClient::report()
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if ( debug_ ) std::cout << "HcalDigiClient: report" << std::endl;
  getHistograms();
  stringstream name;
  name<<process_.c_str()<<rootFolder_.c_str()<<"/DigiMonitor_Hcal/Digi Task Event Number";
  MonitorElement* me = 0;
  if(dbe_) me = dbe_->get(name.str().c_str());
  if ( me ) 
    {
      string s = me->valueString();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
      if ( debug_ ) std::cout << "Found '" << name.str().c_str() << "'" << std::endl;
    }
  else
    std::cout <<"Didn't find "<<name.str().c_str()<<endl;
  name.str("");

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiClient REPORT  -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
}

void HcalDigiClient::analyze(void){
  // analyze function only runs every N events to save time
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  jevt_++;
  int updates = 0;

  if ( updates % 10 == 0 ) {
    if ( debug_ ) std::cout << "HcalDigiClient: " << updates << " updates" << std::endl;
  }
  calculateProblems();
  //report();

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiClient ANALYZE  -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
}

void HcalDigiClient::getProblemHistograms()
{
  /*  August 28, 2009 -- the standard 'getHistograms() method has 
      a memory leak.  This probably affects more than just Digi Client.
      For now, when running online, get only the histograms needed
      to evaluate problems.
  */
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (debug_>0) std::cout <<"HcalDigiClient> getProblemHistograms()"<<std::endl;

  stringstream name;
  name<<process_.c_str()<<rootFolder_<<"/DigiMonitor_Hcal/Digi Task Event Number";
  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) 
    {
      string s = me->valueString();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
      if ( debug_>1 ) std::cout << "Found '" << name.str().c_str() << "'" << std::endl;
    }
  name.str("");
  
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/good_digis/digi_occupancy/"," Digi Eta-Phi Occupancy Map",DigiOccupancyByDepth);
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/bad_digi_occupancy/","Bad Digi Map",BadDigisByDepth);
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/baddigisize/"," Digis with Bad Size",DigiErrorsBadDigiSize);
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/bad_reportUnpackerErrors/", " Bad Unpacker Digis",DigiErrorsUnpacker);
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/badfibBCNoff/",
		 " Digis with non-zero Fiber Orbit Msg Idle BCN Offsets", DigiErrorsBadFibBCNOff);

  // The following two histogram sets are only created if diagnostics turned on
  
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/badcapID/"," Digis with Bad Cap ID Rotation",DigiErrorsBadCapID);
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/data_invalid_error/"," Digis with Data Invalid or Error Bit Set",DigiErrorsDVErr);
  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiClient GETPROBLEMHISTOGRAMS  -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
}


void HcalDigiClient::getHistograms()
{
  if(!dbe_) return;
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (debug_>0) std::cout <<"HcalDigiClient> getHistograms()"<<std::endl;

  stringstream name;
  name<<process_.c_str()<<rootFolder_<<"/DigiMonitor_Hcal/Digi Task Event Number";
  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) 
    {
      string s = me->valueString();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
      if ( debug_>1 ) std::cout << "Found '" << name.str().c_str() << "'" << std::endl;
    }
  name.str("");

  // Get Histograms
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Shape";  //hbHists.shape
  hbHists.shape = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_ );
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Shape";  //heHists.shape
  heHists.shape = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Shape";  //hoHists.shape
  hoHists.shape = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Shape";  //hfHists.shape
  hfHists.shape = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Shape - over thresh";  //hbHists.shapeThresh
  hbHists.shapeThresh = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Shape - over thresh";  //heHists.shapeThresh
  heHists.shapeThresh = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Shape - over thresh";  //hoHists.shapeThresh
  hoHists.shapeThresh = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Shape - over thresh";  //hfHists.shapeThresh
  hfHists.shapeThresh = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Digi Presamples";  //hbHists.presample
  hbHists.presample = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Digi Presamples";  //heHists.presample
  heHists.presample = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Digi Presamples";  //hoHists.presample
  hoHists.presample = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Digi Presamples";  //hfHists.presample
  hfHists.presample = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Bad Quality Digis";  //hbHists.BQ
  hbHists.BQ = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Bad Quality Digis";  //heHists.BQ
  heHists.BQ = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Bad Quality Digis";  //hoHists.BQ
  hoHists.BQ = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Bad Quality Digis";  //hfHists.BQ
  hfHists.BQ = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Bad Quality Digi Fraction";  //hbHists.BQFrac
  hbHists.BQFrac = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Bad Quality Digi Fraction";  //heHists.BQFrac
  heHists.BQFrac = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Bad Quality Digi Fraction";  //hoHists.BQFrac
  hoHists.BQFrac = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Bad Quality Digi Fraction";  //hfHists.BQFrac
  hfHists.BQFrac = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Capid 1st Time Slice";  //hbHists.DigiFirstCapID
  hbHists.DigiFirstCapID = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Capid 1st Time Slice";  //heHists.DigiFirstCapID
  heHists.DigiFirstCapID = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Capid 1st Time Slice";  //hoHists.DigiFirstCapID
  hoHists.DigiFirstCapID = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Capid 1st Time Slice";  //hfHists.DigiFirstCapID
  hfHists.DigiFirstCapID = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Data Valid Err Bits";  //hbHists.DVerr
  hbHists.DVerr = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Data Valid Err Bits";  //heHists.DVerr
  heHists.DVerr = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Data Valid Err Bits";  //hoHists.DVerr
  hoHists.DVerr = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Data Valid Err Bits";  //hfHists.DVerr
  hfHists.DVerr = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB CapID";  //hbHists.CapID
  hbHists.CapID = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE CapID";  //heHists.CapID
  heHists.CapID = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO CapID";  //hoHists.CapID
  hoHists.CapID = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF CapID";  //hfHists.CapID
  hfHists.CapID = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB ADC count per time slice";  //hbHists.ADC
  hbHists.ADC = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE ADC count per time slice";  //heHists.ADC
  heHists.ADC = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO ADC count per time slice";  //hoHists.ADC
  hoHists.ADC = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF ADC count per time slice";  //hfHists.ADC
  hfHists.ADC = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB ADC sum";  //hbHists.ADCsum
  hbHists.ADCsum = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE ADC sum";  //heHists.ADCsum
  heHists.ADCsum = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO ADC sum";  //hoHists.ADCsum
  hoHists.ADCsum = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF ADC sum";  //hfHists.ADCsum
  hfHists.ADCsum = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/Digi Size";   //DigiSize
  DigiSize = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/digi_occupancy/Digi Eta Occupancy Map";   //DigiOccupancyEta
  DigiOccupancyEta = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/digi_occupancy/Digi Phi Occupancy Map";   //DigiOccupancyPhi
  DigiOccupancyPhi = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/# of Good Digis";   //DigiNum
  DigiNum = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/# Bad Qual Digis";   //DigiBQ
  DigiBQ = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/Bad Digi Fraction";   //DigiBQFrac
  DigiBQFrac = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);

  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/digi_occupancy/Digi VME Occupancy Map";   //DigiOccupancyVME
  DigiOccupancyVME = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/digi_occupancy/Digi Spigot Occupancy Map";   //DigiOccupancySpigot
  DigiOccupancySpigot = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);


  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/bad_digi_occupancy/Digi VME Error Map";   //DigiErrorVME
  DigiErrorVME = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/bad_digi_occupancy/Digi Spigot Error Map";   //DigiErrorSpigot
  DigiErrorSpigot = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  for (int i=0;i<9;++i)
    {
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Plus Time Slices "<<i<<" and "<<i+1;  //hbHists.TS_sum_plus[i]
      hbHists.TS_sum_plus[i] = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Plus Time Slices "<<i<<" and "<<i+1;  //heHists.TS_sum_plus[i]
      heHists.TS_sum_plus[i] = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Plus Time Slices "<<i<<" and "<<i+1;  //hoHists.TS_sum_plus[i]
      hoHists.TS_sum_plus[i] = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Plus Time Slices "<<i<<" and "<<i+1;  //hfHists.TS_sum_plus[i]
      hfHists.TS_sum_plus[i] = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HB/HB Minus Time Slices "<<i<<" and "<<i+1;  //hbHists.TS_sum_minus[i]
      hbHists.TS_sum_minus[i] = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HE/HE Minus Time Slices "<<i<<" and "<<i+1;  //heHists.TS_sum_minus[i]
      heHists.TS_sum_minus[i] = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HO/HO Minus Time Slices "<<i<<" and "<<i+1;  //hoHists.TS_sum_minus[i]
      hoHists.TS_sum_minus[i] = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/HF/HF Minus Time Slices "<<i<<" and "<<i+1;  //hfHists.TS_sum_minus[i]
      hfHists.TS_sum_minus[i] = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
      name.str("");
    } // for (int i=0;i<9;++i)

  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/good_digis/digi_occupancy/"," Digi Eta-Phi Occupancy Map",DigiOccupancyByDepth);
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/bad_digi_occupancy/","Bad Digi Map",BadDigisByDepth);

  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/badcapID/"," Digis with Bad Cap ID Rotation",DigiErrorsBadCapID);
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/baddigisize/"," Digis with Bad Size",DigiErrorsBadDigiSize);
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/data_invalid_error/"," Digis with Data Invalid or Error Bit Set",DigiErrorsDVErr);
  getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/bad_reportUnpackerErrors/", " Bad Unpacker Digis",DigiErrorsUnpacker);
 getEtaPhiHists(rootFolder_,"DigiMonitor_Hcal/bad_digis/badfibBCNoff/",
		 " Digis with non-zero Fiber Orbit Msg Idle BCN Offsets", DigiErrorsBadFibBCNOff); 
 if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiClient GET HISTOGRAMS  -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalDigiClient::getHistograms()


void HcalDigiClient::getSubdetHists(DigiClientHists& h, std::string subdet)
{
  
  stringstream name;
  // Get Histograms
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Digi Shape";  //h.shape
  h.shape = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Digi Shape - over thresh"; 
  h.shapeThresh = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Digi Presamples";  //h.presample
  h.presample = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Bad Quality Digis";  //h.BQ
  h.BQ = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Bad Quality Digi Fraction";  //h.BQFrac
  h.BQFrac = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Capid 1st Time Slice";  
  h.DigiFirstCapID = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Data Valid Err Bits";  //h.DVerr
  h.DVerr = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" CapID";  //h.CapID
  h.CapID = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" ADC count per time slice";  //h.ADC
  h.ADC = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" ADC sum";  //h.ADCsum
  h.ADCsum = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");

  for (int i=0;i<9;++i)
    {
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Plus Time Slices "<<i<<" and "<<i+1;  
      h.TS_sum_plus[i] = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Minus Time Slices "<<i<<" and "<<i+1; 
      h.TS_sum_minus[i] = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
      name.str("");
    } // for (int i=0;i<9;++i)

  return;
} // void getSubdetHists(...)

void HcalDigiClient::resetAllME()
{
  if (debug_>0) std::cout <<"HcalDigiClient> resetAllME()"<<std::endl;
  if(!dbe_) return;
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
 

  stringstream name;
  resetSubdetHists(hbHists, "HB");
  resetSubdetHists(hbHists, "HE");
  resetSubdetHists(hbHists, "HO");
  resetSubdetHists(hbHists, "HF");

  // Reset

  name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/baddigisize/ Digis with Bad Size";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/digi_occupancy/Digi Eta Occupancy Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/digi_occupancy/Digi Phi Occupancy Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/# of Good Digis";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/# Bad Qual Digis";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/Bad Digi Fraction";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/digi_occupancy/Digi Eta-Phi Occupancy Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/digi_occupancy/Digi VME Occupancy Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/good_digis/digi_occupancy/Digi Spigot Occupancy Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/bad_digi_occupancy/Digi VME Error Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/bad_digi_occupancy/Digi Spigot Error Map";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  for (int i=0;i<4;++i)
    {
      name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/bad_digi_occupancy/"<<subdets_[i]<<"Bad Digi Map";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/badcapID/"<<subdets_[i]<<" Digis with Bad Cap ID Rotation";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/baddigisize/"<<subdets_[i]<<" Digis with Bad Size";
      resetME(name.str().c_str(),dbe_);
      name.str(""); 
      /*
      name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/badADCsum/"<<subdets_[i]<<" Digis with ADC sum below threshold ADC counts";
      resetME(name.str().c_str(),dbe_);
      name.str(""); 
      */
      name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/data_invalid_error/"<<subdets_[i]<<" Digis with Data Invalid or Error Bit Set";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/bad_reportUnpackerErrors/"<<subdets_[i]<<" Bad Unpacker Digis";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/badfibBCNoff/"<<subdets_[i]<<" Digis with non-zero Fiber Orbit Msg Idle BCN Offsets";
      resetME(name.str().c_str(),dbe_);
      name.str("");
    } // for (int i=0;i<4;++i)
  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiClient RESET ALL ME  -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalDigiClient::resetAllME()

void HcalDigiClient::resetSubdetHists(DigiClientHists& h, std::string subdet)
{
  stringstream name;
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Digi Shape";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Digi Shape - over thresh";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Digi Presamples";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Bad Quality Digis";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Bad Quality Digi Fraction";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Capid 1st Time Slice";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Data Valid Err Bits";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" CapID";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" ADC count per time slice";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" ADC sum";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"DigiMonitor_Hcal/bad_digis/baddigisize/ Digis with Bad Size";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  for (int i=0;i<9;++i)
    {
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Plus Time Slices "<<i<<" and "<<i+1;  //hbHists.TS_sum_plus[i]
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<"DigiMonitor_Hcal/digi_info/"<<subdet<<"/"<<subdet<<" Minus Time Slices "<<i<<" and "<<i+1;  //hbHists.TS_sum_minus[i]
      resetME(name.str().c_str(),dbe_);
      name.str("");
    } // for (int i=0;i<9;++i)
  return;

} // void HcalDigiClient::resetSubdetHists(...)


void HcalDigiClient::htmlExpertOutput(int runNo, string htmlDir, string htmlName){
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (debug_>1)
    std::cout << "<HcalDigiClient> Preparing HcalDigiClient Expert html output ..." << std::endl;
  
  ofstream htmlFile;
  htmlFile.open((htmlDir +"Expert_"+ htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal Digi Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile <<"<a name=\"EXPERT_DIGI_TOP\" href = \".\"> Back to Main HCAL DQM Page </a><br>"<<std::endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Digi Status Page </a><br>"<<std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Digis</span></h2> " << std::endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<table width=100%  border = 1>"<<std::endl;
  htmlFile << "<tr><td align=\"center\" colspan=2><a href=\"#OVERALL_PROBLEMS\">PROBLEM CELLS BY DEPTH </a></td></tr>"<<std::endl;
  htmlFile << "<tr><td align=\"center\">"<<std::endl;
  htmlFile<<"<br><a href=\"#OCCUPANCY\">Digi Occupancy Plots </a>"<<std::endl;
  htmlFile<<"<br><a href=\"#DIGISIZE\">Digi Bad Size Plots</a>"<<std::endl;
  htmlFile<<"<br><a href=\"#DIGICAPID\">Digi Bad Cap ID Plots </a>"<<std::endl;
  htmlFile<<"</td><td align=\"center\">"<<std::endl;
  htmlFile<<"<br><a href=\"#DIGIADCSUM\">Digi Bad ADC Sum Plots </a>"<<std::endl;
  htmlFile<<"<br><a href=\"#DIGIERRORBIT\">Digi Bad Error Bit Plots </a>"<<std::endl;
  //htmlFile<<"<br><a href=\"#NODIGI\">Missing Digi Plots </a>"<<std::endl;
  htmlFile<<"<br><a href=\"#IDLEBCN\">Bad Idle BCN Offset Plots </a>"<<std::endl;
  htmlFile<<"<br><a href=\"#REPORTDIGI\">Bad Unpacker Report Digi Plots </a>"<<std::endl;
  htmlFile<<"</td></tr><tr><td align=\"center\">"<<std::endl;
  htmlFile<<"<br><a href=\"#HBDIGI\">HB Digi Plots </a>"<<std::endl;
  htmlFile<<"<br><a href=\"#HEDIGI\">HE Digi Plots </a>"<<std::endl;
  htmlFile<<"</td><td align=\"center\">"<<std::endl;
  htmlFile<<"<br><a href=\"#HODIGI\">HO Digi Plots </a>"<<std::endl;
  htmlFile<<"<br><a href=\"#HFDIGI\">HF Digi Plots </a>"<<std::endl;

  htmlFile << "</td></tr>"<<std::endl;
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><br>"<<std::endl;

  // Plot overall errors
  htmlFile << "<h2><strong><a name=\"OVERALL_PROBLEMS\">Eta-Phi Maps of Problem Cells By Depth</strong></h2>"<<std::endl;
  htmlFile <<" These plots of problem cells combine results from all digi tests<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  
  // Depths are stored as:  0:  HB/HE/HF depth 1, 1:  HB/HE/HF depth 2, 2:  HE depth 3, 3:  HO
  
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,BadDigisByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,BadDigisByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }

  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;

  // General diagnostics
  htmlFile << "<h2><strong><a name=\"DIAGNOSTIC\">General Diagnostic Plots</strong></h2>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "This shows number of digis/event, digi size, and number/fraction of bad digis per event.<br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile<<"<tr align=\"left\">"<<std::endl;
  htmlAnyHisto(runNo,DigiSize,"","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,DigiNum,"","", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"</table>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlAnyHisto(runNo,DigiBQ,"","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,DigiBQFrac,"","", 92, htmlFile, htmlDir);
  htmlFile<<"</tr></table>"<<std::endl;

  // Occupancy Plots
  htmlFile << "<h2><strong><a name=\"OCCUPANCY\">Occupancy Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows average digi occupancy of each cell per event<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,DigiOccupancyByDepth[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiOccupancyByDepth[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile<<"<tr align=\"left\">"<<std::endl;
  htmlAnyHisto(runNo,DigiOccupancyEta,"","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,DigiOccupancyPhi,"","", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile<<"<tr align=\"left\">"<<std::endl;
  htmlAnyHisto(runNo,DigiOccupancyVME,"","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,DigiOccupancySpigot,"","", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile <<"</table>"<<std::endl;

  // Digi Size Plots
  htmlFile << "<h2><strong><a name=\"DIGISIZE\">Digi Size Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows the fraction of events for each digi in which the digi's size is outside the expected range.<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,DigiErrorsBadDigiSize[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsBadDigiSize[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;

  // Digi Cap ID Plots
  htmlFile << "<h2><strong><a name=\"DIGICAPID\">Digi Cap ID Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows the fraction of events for each digi in which the digi's capacitor-ID rotation is incorrect.  (This plot is only made if diagnostics flag turned on, since digis with bad cap ID rotation are not kept in the default digi collection.)<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (unsigned int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,DigiErrorsBadCapID[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsBadCapID[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;

  /*
  // Digi ADC SUM Plots
  htmlFile << "<h2><strong><a name=\"DIGIADCSUM\">Digi ADC Sum Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows the fraction of events for each digi in which the digi's ADC sum is below threshold.<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      if (DigiErrorsBadADCSum[2*i]!=0)
	htmlAnyHisto(runNo,DigiErrorsBadADCSum[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      if (DigiErrorsBadADCSum[2*i+1]!=0)
	htmlAnyHisto(runNo,DigiErrorsBadADCSum[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;
  */

  // Digi Error Bit Plots
  htmlFile << "<h2><strong><a name=\"DIGIERRORBIT\">Digi Error Bit Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows average number of digi errors/data invalids of each cell per event.  (These plots are only generated if diagnostics have been turned on, since invalid/error data is normally excluded from the digi collection.)<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,DigiErrorsDVErr[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsDVErr[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;

  // Missing Digi
  /*
  htmlFile << "<h2><strong><a name=\"NODIGI\">Missing digi Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows digis that are not present for a number of consecutive events.  (More detailed information on missing channels is found in the dead cell monitor.)<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,DigiErrorsNoDigi[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsNoDigi[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;
  */
  
// report unpacker errors
  htmlFile << "<h2><strong><a name=\"IDLEBCN\">Idle BCN Error Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows digis that have a non-zero fiber orbit message idle BCN offset. <br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,DigiErrorsBadFibBCNOff[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsBadFibBCNOff[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;

  // report unpacker errors
  htmlFile << "<h2><strong><a name=\"REPORTDIGI\">Report Unpacker Error Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows digis that are reported as bad by the unpacker (and thus aren't included in the base digi collection) <br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,DigiErrorsUnpacker[2*i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,DigiErrorsUnpacker[2*i+1],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;


  // HB Plots
  htmlFile << "<h2><strong><a name=\"HBDIGI\">HB digi Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows expert-level information for the HB subdetector digis <br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hbHists.shape,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hbHists.shapeThresh,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hbHists.BQ,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hbHists.BQFrac,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hbHists.DigiFirstCapID,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hbHists.CapID,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hbHists.ADC,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hbHists.ADCsum,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hbHists.presample,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hbHists.DVerr,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  for (int i=0;i<9;++i)
    {
      htmlFile << "</tr>"<<std::endl;
      htmlAnyHisto(runNo,hbHists.TS_sum_plus[i],"","",92,htmlFile,htmlDir);
      htmlAnyHisto(runNo,hbHists.TS_sum_minus[i],"","",92,htmlFile,htmlDir);
      htmlFile << "</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;


  // HE Plots
  htmlFile << "<h2><strong><a name=\"HEDIGI\">HE digi Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows expert-level information for the HE subdetector digis <br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,heHists.shape,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,heHists.shapeThresh,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,heHists.BQ,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,heHists.BQFrac,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,heHists.DigiFirstCapID,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,heHists.CapID,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,heHists.ADC,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,heHists.ADCsum,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,heHists.presample,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,heHists.DVerr,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  for (int i=0;i<9;++i)
    {
      htmlFile << "</tr>"<<std::endl;
      htmlAnyHisto(runNo,heHists.TS_sum_plus[i],"","",92,htmlFile,htmlDir);
      htmlAnyHisto(runNo,heHists.TS_sum_minus[i],"","",92,htmlFile,htmlDir);
      htmlFile << "</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;

  // HO Plots
  htmlFile << "<h2><strong><a name=\"HODIGI\">HO digi Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows expert-level information for the HO subdetector digis <br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hoHists.shape,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hoHists.shapeThresh,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hoHists.BQ,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hoHists.BQFrac,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hoHists.DigiFirstCapID,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hoHists.CapID,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hoHists.ADC,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hoHists.ADCsum,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hoHists.presample,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hoHists.DVerr,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  for (int i=0;i<9;++i)
    {
      htmlFile << "</tr>"<<std::endl;
      htmlAnyHisto(runNo,hoHists.TS_sum_plus[i],"","",92,htmlFile,htmlDir);
      htmlAnyHisto(runNo,hoHists.TS_sum_minus[i],"","",92,htmlFile,htmlDir);
      htmlFile << "</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;

  // HF Plots
  htmlFile << "<h2><strong><a name=\"HFDIGI\">HF digi Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows expert-level information for the HF subdetector digis <br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_DIGI_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hfHists.shape,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hfHists.shapeThresh,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hfHists.BQ,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hfHists.BQFrac,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hfHists.DigiFirstCapID,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hfHists.CapID,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hfHists.ADC,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hfHists.ADCsum,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo,hfHists.presample,"","",92,htmlFile,htmlDir);
  htmlAnyHisto(runNo,hfHists.DVerr,"","",92,htmlFile,htmlDir);
  htmlFile << "</tr>"<<std::endl;
  for (int i=0;i<9;++i)
    {
      htmlFile << "</tr>"<<std::endl;
      htmlAnyHisto(runNo,hfHists.TS_sum_plus[i],"","",92,htmlFile,htmlDir);
      htmlAnyHisto(runNo,hfHists.TS_sum_minus[i],"","",92,htmlFile,htmlDir);
      htmlFile << "</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;

  // html page footer
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;

  htmlFile.close();
  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiClient HTML EXPERT  -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
}

void HcalDigiClient::createTests(){
  if(!dbe_) return;
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  if(debug_) std::cout <<"Creating Digi tests..."<<std::endl;
  
  /*
  char meTitle[250], name[250];   
  vector<string> params;
  */
  for(int i=0; i<4; ++i){
    if(!subDetsOn_[i]) continue;

    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO";
    /*
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
    */
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

  //  deprecated function; no longer needed
  if (debug_>0) std::cout <<"<HcalDigiClient> loadHistograms(TFile* infile) -- DEPRECATED!"<<std::endl;
  return;
} // void HcalDigiClient::loadHistograms()


void HcalDigiClient::loadSubdetHists(TFile* infile,DigiClientHists& h, std::string subdet)
{
  // Deprecated function; no longer needed
  return;
} // void HcalDigiClient::loadSubdetHists(...)


void HcalDigiClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  if (debug_>0) std::cout << "<HcalDigiClient::htmlOutput> Preparing html output ..." << std::endl;

  getHistograms();

  string client = "DigiMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal Digi Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Digis</span></h2> " << std::endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;
  htmlFile << "<table width=100%  border=1><tr>" << std::endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"DigiMonitorErrors.html\">Errors in this task</a></td>" << std::endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << std::endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"DigiMonitorWarnings.html\">Warnings in this task</a></td>" << std::endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << std::endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"DigiMonitorMessages.html\">Messages in this task</a></td>" << std::endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << std::endl;
  htmlFile << "</tr></table>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<h2><strong>Hcal Digi Status</strong></h2>" << std::endl;

  htmlFile << "<table align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  htmlAnyHisto(runNo,(ProblemCells->getTH2F()),"i#eta","i#phi", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile<<"<tr align=\"center\"><td> A digi is considered bad if the digi size is incorrect, the cap ID rotation is incorrect, or the ADC sum for the digi is less than some threshold value.  It is also considered bad if its error bit is on or its data valid bit is off."<<std::endl;

  htmlFile<<"</td>"<<std::endl;
  htmlFile<<"</tr></table>"<<std::endl;
  htmlFile<<"<hr><table align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile<<"<tr><td align=center><a href=\"Expert_"<< htmlName<<"\"><h2>Detailed Digi Plots</h2> </a></br></td>"<<std::endl;
  htmlFile<<"</tr></table><br><hr>"<<std::endl;

 // Now print out problem cells
  htmlFile <<"<br>"<<std::endl;
  htmlFile << "<h2><strong>Hcal Problem Digis</strong></h2>" << std::endl;
  htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile <<"<td> Problem Digis<br>(ieta, iphi, depth)</td><td align=\"center\"> Fraction of Events <br>in which cells are bad (%)</td></tr>"<<std::endl;

  if (ProblemCells==0)
    {
      if (debug_>0) std::cout <<"<HcalDigiClient::htmlOutput>  ERROR: can't find Problem Digi plot!"<<std::endl;
      return;
    }

  int ieta,iphi;

  stringstream name;
  int etabins=0, phibins=0;
  for (unsigned int depth=0;depth<ProblemCellsByDepth.depth.size(); ++depth)
    {
      etabins=(ProblemCellsByDepth.depth[depth]->getTH2F())->GetNbinsX();
      for (int eta=0;eta<etabins;++eta)
        {
	  ieta=CalcIeta(eta,depth+1);
	  if (ieta==-9999) continue;
	  phibins=(ProblemCellsByDepth.depth[depth]->getTH2F())->GetNbinsY();
          for (int phi=0;phi<phibins;++phi)
            {
	      iphi=phi+1;
	      if (abs(eta)>20 && phi%2!=1) continue;
	      if (abs(eta)>39 && phi%4!=3) continue;
	      if (BadDigisByDepth[depth]==0)
		  continue;
	      if (ProblemCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1)>0)
		{
		  if (depth<2)
		    {
                      if (isHB(eta,depth+1)) name <<"HB";
                      else if (isHE(eta,depth+1)) name<<"HE";
                      else if (isHF(eta,depth+1)) name<<"HF";
                    }
                  else if (depth==2) name <<"HE";
                  else if (depth==3) name<<"HO";

		  htmlFile<<"<td>"<<name.str().c_str()<<" ("<<ieta<<", "<<iphi<<", "<<depth+1<<")</td><td align=\"center\">"<<ProblemCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1)*100.<<"</td></tr>"<<std::endl;

		  name.str("");
		}
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // for (int depth=0;...)
  
  
  // html page footer
  htmlFile <<"</table> " << std::endl;
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;

  htmlFile.close();
  htmlExpertOutput(runNo, htmlDir, htmlName);

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiClient HTMLOUTPUT  -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;

} // void HcalDigiClient::htmlOutput()




bool HcalDigiClient::hasErrors_Temp()
{
  int problemcount=0;
  int etabins=0;
  int phibins=0;

  for (int depth=0;depth<4; ++depth)
    {
      if (BadDigisByDepth[depth]==0) continue;
      etabins  = BadDigisByDepth[depth]->GetNbinsX();
      phibins  = BadDigisByDepth[depth]->GetNbinsY();
      for (int ieta=0;ieta<etabins;++ieta)
        {
          for (int iphi=0; iphi<phibins;++iphi)
            {
	      if (BadDigisByDepth[depth]->GetBinContent(ieta+1,iphi+1)>0)
		problemcount++;
	    } // for (int iphi=0;...)
	} // for (int ieta=0;...)
    } // for (int depth=0;...)

  if (problemcount>=100) return true;
  return false;

} // bool HcalDigiClient::hasErrors_Temp()


bool HcalDigiClient::hasWarnings_Temp()
{
  int problemcount=0;
  int etabins=0;
  int phibins=0;

  for (int depth=0;depth<4; ++depth)
    {
      if (BadDigisByDepth[depth]==0) continue;
      etabins  = BadDigisByDepth[depth]->GetNbinsX();
      phibins  = BadDigisByDepth[depth]->GetNbinsY();
      for (int ieta=0;ieta<etabins;++ieta)
        {
          for (int iphi=0; iphi<phibins;++iphi)
            {
	      if (BadDigisByDepth[depth]->GetBinContent(ieta+1,iphi+1)>0)
		problemcount++;
	    } // for (int iphi=0;...)
	} // for (int ieta=0;...)
    } // for (int depth=0;...)

  if (problemcount>0) return true;
  return false;

} // bool HcalDigiClient::hasWarnings_Temp()

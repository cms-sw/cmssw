#include "DQM/HcalMonitorTasks/interface/HcalDigiMonitor.h"
#include <cmath>


// constructor

HcalDigiMonitor::HcalDigiMonitor() {
  doPerChannel_ = false;
  occThresh_ = 1;
  ievt_=0;
  shape_=NULL;
}

// destructor
HcalDigiMonitor::~HcalDigiMonitor() {}

void HcalDigiMonitor::reset(){}

// Checks capid rotation; returns false if no problems with rotation
static bool bitUpset(int last, int now){
  if(last ==-1) return false;
  int v = last+1; 
  if(v==4) v=0;
  if(v==now) return false;
  return true;
} // static bool bitUpset(...)


void HcalDigiMonitor::setup(const edm::ParameterSet& ps, 
			    DQMStore* dbe)
{
  // Call base class setup
  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"DigiMonitor_Hcal";

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
 
  // Get digi-specific parameters

  shapeThresh_ = ps.getUntrackedParameter<int>("DigiMonitor_ShapeThresh", -1);
  //shapeThresh_ is used for plotting pulse shapes for all digis with ADC sum > shapeThresh_;
  occThresh_ = ps.getUntrackedParameter<int>("DigiMonitor_ADCsumThresh", 0);
  //occThresh_ is used to determine when checking ADC sums of digis
  if (fVerbosity>0)
    {
      cout << "<HcalDigiMonitor> Digi ADC occupancy threshold set to: >" << occThresh_ << endl;
      cout <<"<HcalDigiMonitor> Digi shape ADC threshold set to: >" << shapeThresh_ << endl;
    }
  makeDiagnostics = ps.getUntrackedParameter<bool>("DigiMonitor_MakeDiagnosticPlots",false); // not yet used

  doPerChannel_ = ps.getUntrackedParameter<bool>("DigiMonitor_DigisPerchannel",false); // not yet used -- never will be?
  if (fVerbosity>1)
    cout << "<HcalDigiMonitor> Digi phi min/max set to " << phiMin_ << "/" <<phiMax_ << endl;

  digi_checkNevents_ = ps.getUntrackedParameter<int>("DigiMonitor_checkNevents",checkNevents_); 
  if (fVerbosity>1)
    cout <<"<HcalDigiMonitor>  Perform checks and histogram fills every "<<digi_checkNevents_<<" events"<<endl;

  // Specify which tests to run when looking for problem digis
  digi_checkoccupancy_ = ps.getUntrackedParameter<bool>("DigiMonitor_problems_checkForMissingDigis",false);
  digi_checkcapid_     = ps.getUntrackedParameter<bool>("DigiMonitor_problems_checkCapID",true);
  digi_checkdigisize_  = ps.getUntrackedParameter<bool>("DigiMonitor_problems_checkDigiSize",true);
  digi_checkadcsum_    = ps.getUntrackedParameter<bool>("DigiMonitor_problems_checkADCsum",true);
  digi_checkdverr_     = ps.getUntrackedParameter<bool>("DigiMonitor_problems_checkDVerr",true);
  mindigisize_ = ps.getUntrackedParameter<int>("DigiMonitor_minDigiSize",1);
  maxdigisize_ = ps.getUntrackedParameter<int>("DigiMonitor_maxDigiSize",20);

  hbHists.check=ps.getUntrackedParameter<bool>("checkHB",true);
  heHists.check=ps.getUntrackedParameter<bool>("checkHE",true);
  hoHists.check=ps.getUntrackedParameter<bool>("checkHO",true);
  hfHists.check=ps.getUntrackedParameter<bool>("checkHF",true);

  if (fVerbosity>1)
    {
      cout <<"<HcalDigiMonitor> Checking for the following problems:"<<endl; 
      if (digi_checkoccupancy_) cout <<"\tChecking that digi present at least once every "<<digi_checkNevents_<<" events;"<<endl;
      if (digi_checkcapid_) cout <<"\tChecking that cap ID rotation is correct;"<<endl;
      if (digi_checkdigisize_) cout <<"\tChecking that digi size is between ["<<mindigisize_<<" - "<<maxdigisize_<<"];"<<endl;
      if (digi_checkadcsum_) cout <<"\tChecking that ADC sum of digi is greater than 0;"<<endl; 
      if (digi_checkdverr_) cout <<"\tChecking that data valid bit is true and digi error bit is false;"<<endl;
      cout <<"\tChecking digis for the following subdetectors:"<<endl;
      if (hbHists.check) cout <<"\tHB";
      if (heHists.check) cout <<"\tHE";
      if (hoHists.check) cout <<"\tHO";
      if (hfHists.check) cout <<"\tHF";
      cout <<endl;
    }

  ievt_=0;

  /******** Zero all counters *******/
  
  zeroCounters();

  /******* Set up all histograms  ********/

  if (m_dbe)
    {
      ostringstream name;
      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("Digi Task Event Number");    
      meEVT_->Fill(ievt_);

      MonitorElement* checkN = m_dbe->bookInt("DigiCheckNevents");
      checkN->Fill(digi_checkNevents_);
      MonitorElement* occT = m_dbe->bookInt("DigiOccThresh");
      occT->Fill(occThresh_);
      MonitorElement* shapeT = m_dbe->bookInt("DigiShapeThresh");
      shapeT->Fill(shapeThresh_);
      ProblemDigis = m_dbe->book2D(" ProblemDigis",
				   " Problem Digi Rate for all HCAL",
				   etaBins_,etaMin_,etaMax_,
				   phiBins_,phiMin_,phiMax_);
      
      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis");
      setupDepthHists2D(ProblemDigisByDepth," Problem Digi Rate","");
      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis/badcapID");
      setupDepthHists2D(DigiErrorsBadCapID," Digis with Bad Cap ID Rotation", "");
      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis/baddigisize");
      setupDepthHists2D(DigiErrorsBadDigiSize," Digis with Bad Size", "");
      DigiSize = m_dbe->book2D("Digi Size", "Digi Size",4,0,4,20,-0.5,19.5);
      DigiSize->setBinLabel(1,"HB",1);
      DigiSize->setBinLabel(2,"HE",1);
      DigiSize->setBinLabel(3,"HO",1);
      DigiSize->setBinLabel(4,"HF",1);
      DigiSize->setAxisTitle("Subdetector",1);
      DigiSize->setAxisTitle("Digi Size",2);

      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis/badADCsum");
      name<<" Digis with ADC sum below threshold ADC counts"; // make the name variable at some point?  Or just change title to specify ADC threshold?
      setupDepthHists2D(DigiErrorsBadADCSum," Digis with ADC sum below threshold ADC counts", "");
      name.str("");
      for (int i=0;i<6;++i)
	{
	  name<<DigiErrorsBadADCSum[i]->getTitle()<<"(ADC sum > "<<occThresh_<<" events)";
	  DigiErrorsBadADCSum[i]->setTitle(static_cast<const string>(name.str().c_str()));
	  name.str("");
	}

      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis/data_invalid_error");
      setupDepthHists2D(DigiErrorsDVErr," Digis with Data Invalid or Error Bit Set", "");

      m_dbe->setCurrentFolder(baseFolder_+"/digi_occupancy");
      setupDepthHists2D(DigiOccupancyByDepth," Digi Eta-Phi Occupancy Map","");
      DigiOccupancyPhi= m_dbe->book1D("Digi Phi Occupancy Map",
				      "Digi Phi Occupancy Map",
				      phiBins_,phiMin_,phiMax_);
      DigiOccupancyPhi->setAxisTitle("i#phi",1);
      DigiOccupancyPhi->setAxisTitle("# of Events",2);
      DigiOccupancyEta= m_dbe->book1D("Digi Eta Occupancy Map",
				      "Digi Eta Occupancy Map",
				      etaBins_,etaMin_,etaMax_);
      DigiOccupancyEta->setAxisTitle("i#eta",1);
      DigiOccupancyEta->setAxisTitle("# of Events",2);

      DigiOccupancyVME = m_dbe->book2D("Digi VME Occupancy Map",
				       "Digi VME Occupancy Map",
				       40,-0.25,19.75,18,-0.5,17.5);
      DigiOccupancyVME -> setAxisTitle("HTR Slot",1);  
      DigiOccupancyVME -> setAxisTitle("VME Crate Id",2);
      
      DigiOccupancySpigot = m_dbe->book2D("Digi Spigot Occupancy Map",
					  "Digi Spigot Occupancy Map",
					  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					  36,-0.5,35.5);
      DigiOccupancySpigot -> setAxisTitle("Spigot",1);  
      DigiOccupancySpigot -> setAxisTitle("DCC Id",2);
      
      m_dbe->setCurrentFolder(baseFolder_+"/digi_errors");
      /*
      DigiErrorEtaPhi = m_dbe->book2D("Digi Geo Error Map","Digi Geo Error Map",
				  etaBins_,etaMin_,etaMax_,
				  phiBins_,phiMin_,phiMax_);
      DigiErrorEtaPhi -> setAxisTitle("i#eta",1);  
      DigiErrorEtaPhi -> setAxisTitle("i#phi",2);
      */

      DigiErrorVME = m_dbe->book2D("Digi VME Error Map",
				  "Digi VME Error Map",
				  40,-0.25,19.75,18,-0.5,17.5);
      DigiErrorVME -> setAxisTitle("HTR Slot",1);  
      DigiErrorVME -> setAxisTitle("VME Crate Id",2);
      
      DigiErrorSpigot = m_dbe->book2D("Digi Spigot Error Map",
				  "Digi Spigot Error Map",
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  36,-0.5,35.5);
      DigiErrorSpigot -> setAxisTitle("Spigot",1);  
      DigiErrorSpigot -> setAxisTitle("DCC Id",2);
      
      DigiBQ = m_dbe->book1D("# Bad Qual Digis","# Bad Qual Digis",DIGI_NUM+500,-0.5,DIGI_NUM+500-0.5);
      DigiBQ -> setAxisTitle("# Bad Quality Digis",1);  
      DigiBQ -> setAxisTitle("# of Events",2);
      
      DigiBQFrac =  m_dbe->book1D("Bad Digi Fraction","Bad Digi Fraction",DIGI_BQ_FRAC_NBINS,(0-0.5/(DIGI_BQ_FRAC_NBINS-1)),1+0.5/(DIGI_BQ_FRAC_NBINS-1));
      DigiBQFrac -> setAxisTitle("Bad Quality Digi Fraction",1);  
      DigiBQFrac -> setAxisTitle("# of Events",2);


      m_dbe->setCurrentFolder(baseFolder_+"/digi_info");
      DigiNum = m_dbe->book1D("# of Digis","# of Digis",DIGI_NUM+500,-0.5,DIGI_NUM+500-0.5);
      DigiNum -> setAxisTitle("# of Digis",1);  
      DigiNum -> setAxisTitle("# of Events",2);
      
      // Individual subdetector histograms
      m_dbe->setCurrentFolder(baseFolder_+"/digi_info/HB");
      hbHists.shape = m_dbe->book1D("HB Digi Shape","HB Digi Shape",10,-0.5,9.5);
      hbHists.shapeThresh = m_dbe->book1D("HB Digi Shape - over thresh",
					  "HB Digi Shape - over thresh",
					  10,-0.5,9.5);
      hbHists.shape->setAxisTitle("Time Slice",1);
      hbHists.shapeThresh->setAxisTitle("Time Slice",1);

      // Create plots of sums of adjacent time slices
      for (int ts=0;ts<9;++ts)
	{
	  name<<"HB Plus Time Slices "<<ts<<" and "<<ts+1;
	  hbHists.TS_sum_plus.push_back(m_dbe->book1D(name.str().c_str(),name.str().c_str(),50,-5.5,44.5));
	  name.str("");
	  name<<"HB Minus Time Slices "<<ts<<" and "<<ts+1;
	  hbHists.TS_sum_minus.push_back(m_dbe->book1D(name.str().c_str(),name.str().c_str(),50,-5.5,44.5));
	  name.str("");
	}
      hbHists.presample= m_dbe->book1D("HB Digi Presamples","HB Digi Presamples",50,-0.5,49.5);
      hbHists.BQ = m_dbe->book1D("HB Bad Quality Digis","HB Bad Quality Digis",DIGI_SUBDET_NUM,-0.5,DIGI_SUBDET_NUM-0.5);
      hbHists.BQFrac = m_dbe->book1D("HB Bad Quality Digi Fraction","HB Bad Quality Digi Fraction",DIGI_BQ_FRAC_NBINS,(0-0.5/(DIGI_BQ_FRAC_NBINS-1)),1+0.5/(DIGI_BQ_FRAC_NBINS-1));
      hbHists.DigiFirstCapID = m_dbe->book1D("HB Capid 1st Time Slice","HB Capid for 1st Time Slice",7,-3.5,3.5);
      hbHists.DigiFirstCapID -> setAxisTitle("CapID (T0) - 1st CapId (T0)",1);  
      hbHists.DigiFirstCapID -> setAxisTitle("# of Events",2);
      hbHists.DVerr = m_dbe->book1D("HB Data Valid Err Bits","HB QIE Data Valid Err Bits",4,-0.5,3.5);
      hbHists.DVerr ->setBinLabel(1,"Err=0, DV=0",1);
      hbHists.DVerr ->setBinLabel(2,"Err=0, DV=1",1);
      hbHists.DVerr ->setBinLabel(3,"Err=1, DV=0",1);
      hbHists.DVerr ->setBinLabel(4,"Err=1, DV=1",1);
      hbHists.CapID = m_dbe->book1D("HB CapID","HB CapID",4,-0.5,3.5);
      hbHists.ADC = m_dbe->book1D("HB ADC count per time slice","HB ADC count per time slice",200,-0.5,199.5);
      hbHists.ADCsum = m_dbe->book1D("HB ADC sum", "HB ADC sum",200,-0.5,199.5);

      m_dbe->setCurrentFolder(baseFolder_+"/digi_info/HE");
      heHists.shape = m_dbe->book1D("HE Digi Shape","HE Digi Shape",10,-0.5,9.5);
      heHists.shapeThresh = m_dbe->book1D("HE Digi Shape - over thresh",
					  "HE Digi Shape - over thresh",
					  10,-0.5,9.5);
      heHists.shape->setAxisTitle("Time Slice",1);
      heHists.shapeThresh->setAxisTitle("Time Slice",1);
      // Create plots of sums of adjacent time slices
      for (int ts=0;ts<9;++ts)
	{
	  name<<"HE Plus Time Slices "<<ts<<" and "<<ts+1;
	  heHists.TS_sum_plus.push_back(m_dbe->book1D(name.str().c_str(),name.str().c_str(),50,-5.5,44.5));
	  name.str("");
	  name<<"HE Minus Time Slices "<<ts<<" and "<<ts+1;
	  heHists.TS_sum_minus.push_back(m_dbe->book1D(name.str().c_str(),name.str().c_str(),50,-5.5,44.5));
	  name.str("");
	}

      heHists.presample= m_dbe->book1D("HE Digi Presamples","HE Digi Presamples",50,-0.5,49.5);
      heHists.BQ = m_dbe->book1D("HE Bad Quality Digis","HE Bad Quality Digis",DIGI_SUBDET_NUM,-0.5,DIGI_SUBDET_NUM-0.5);
      heHists.BQFrac = m_dbe->book1D("HE Bad Quality Digi Fraction","HE Bad Quality Digi Fraction",DIGI_BQ_FRAC_NBINS,(0-0.5/(DIGI_BQ_FRAC_NBINS-1)),1+0.5/(DIGI_BQ_FRAC_NBINS-1));
      heHists.DigiFirstCapID = m_dbe->book1D("HE Capid 1st Time Slice","HE Capid for 1st Time Slice",7,-3.5,3.5);
      heHists.DigiFirstCapID -> setAxisTitle("CapID (T0) - 1st CapId (T0)",1);  
      heHists.DigiFirstCapID -> setAxisTitle("# of Events",2);
      heHists.DVerr = m_dbe->book1D("HE Data Valid Err Bits","HE QIE Data Valid Err Bits",4,-0.5,3.5);
      heHists.DVerr ->setBinLabel(1,"Err=0, DV=0",1);
      heHists.DVerr ->setBinLabel(2,"Err=0, DV=1",1);
      heHists.DVerr ->setBinLabel(3,"Err=1, DV=0",1);
      heHists.DVerr ->setBinLabel(4,"Err=1, DV=1",1);
      heHists.CapID = m_dbe->book1D("HE CapID","HE CapID",4,-0.5,3.5);
      heHists.ADC = m_dbe->book1D("HE ADC count per time slice","HE ADC count per time slice",200,-0.5,199.5);
      heHists.ADCsum = m_dbe->book1D("HE ADC sum", "HE ADC sum",200,-0.5,199.5);

      m_dbe->setCurrentFolder(baseFolder_+"/digi_info/HO");
      hoHists.shape = m_dbe->book1D("HO Digi Shape","HO Digi Shape",10,-0.5,9.5);
      hoHists.shapeThresh = m_dbe->book1D("HO Digi Shape - over thresh",
					  "HO Digi Shape - over thresh",
					  10,-0.5,9.5);
      hoHists.shape->setAxisTitle("Time Slice",1);
      hoHists.shapeThresh->setAxisTitle("Time Slice",1);
      // Create plots of sums of adjacent time slices
      for (int ts=0;ts<9;++ts)
	{
	  name<<"HO Plus Time Slices "<<ts<<" and "<<ts+1;
	  hoHists.TS_sum_plus.push_back(m_dbe->book1D(name.str().c_str(),name.str().c_str(),50,-5.5,44.5));
	  name.str("");
	  name<<"HO Minus Time Slices "<<ts<<" and "<<ts+1;
	  hoHists.TS_sum_minus.push_back(m_dbe->book1D(name.str().c_str(),name.str().c_str(),50,-5.5,44.5));
	  name.str("");
	}
      hoHists.presample= m_dbe->book1D("HO Digi Presamples","HO Digi Presamples",50,-0.5,49.5);
      hoHists.BQ = m_dbe->book1D("HO Bad Quality Digis","HO Bad Quality Digis",DIGI_SUBDET_NUM,-0.5,DIGI_SUBDET_NUM-0.5);
      hoHists.BQFrac = m_dbe->book1D("HO Bad Quality Digi Fraction","HO Bad Quality Digi Fraction",DIGI_BQ_FRAC_NBINS,(0-0.5/(DIGI_BQ_FRAC_NBINS-1)),1+0.5/(DIGI_BQ_FRAC_NBINS-1));
      hoHists.DigiFirstCapID = m_dbe->book1D("HO Capid 1st Time Slice","HO Capid for 1st Time Slice",7,-3.5,3.5);
      hoHists.DigiFirstCapID -> setAxisTitle("CapID (T0) - 1st CapId (T0)",1);  
      hoHists.DigiFirstCapID -> setAxisTitle("# of Events",2);
      hoHists.DVerr = m_dbe->book1D("HO Data Valid Err Bits","HO QIE Data Valid Err Bits",4,-0.5,3.5);
      hoHists.DVerr ->setBinLabel(1,"Err=0, DV=0",1);
      hoHists.DVerr ->setBinLabel(2,"Err=0, DV=1",1);
      hoHists.DVerr ->setBinLabel(3,"Err=1, DV=0",1);
      hoHists.DVerr ->setBinLabel(4,"Err=1, DV=1",1);
      hoHists.CapID = m_dbe->book1D("HO CapID","HO CapID",4,-0.5,3.5);
      hoHists.ADC = m_dbe->book1D("HO ADC count per time slice","HO ADC count per time slice",200,-0.5,199.5);
      hoHists.ADCsum = m_dbe->book1D("HO ADC sum", "HO ADC sum",200,-0.5,199.5);

      m_dbe->setCurrentFolder(baseFolder_+"/digi_info/HF");
      hfHists.shape = m_dbe->book1D("HF Digi Shape","HF Digi Shape",10,-0.5,9.5);
      hfHists.shapeThresh = m_dbe->book1D("HF Digi Shape - over thresh",
					  "HF Digi Shape - over thresh",
					  10,-0.5,9.5);
      // Create plots of sums of adjacent time slices
      for (int ts=0;ts<9;++ts)
	{
	  name<<"HF Plus Time Slices "<<ts<<" and "<<ts+1;
	  hfHists.TS_sum_plus.push_back(m_dbe->book1D(name.str().c_str(),name.str().c_str(),50,-5.5,44.5));
	  name.str("");
	  name<<"HF Minus Time Slices "<<ts<<" and "<<ts+1;
	  hfHists.TS_sum_minus.push_back(m_dbe->book1D(name.str().c_str(),name.str().c_str(),50,-5.5,44.5));
	  name.str("");
	}
      hfHists.shape->setAxisTitle("Time Slice",1);
      hfHists.shapeThresh->setAxisTitle("Time Slice",1);
      hfHists.presample= m_dbe->book1D("HF Digi Presamples","HF Digi Presamples",50,-0.5,49.5);
      hfHists.BQ = m_dbe->book1D("HF Bad Quality Digis","HF Bad Quality Digis",DIGI_SUBDET_NUM,-0.5,DIGI_SUBDET_NUM-0.5);
      hfHists.BQFrac = m_dbe->book1D("HF Bad Quality Digi Fraction","HF Bad Quality Digi Fraction",DIGI_BQ_FRAC_NBINS,(0-0.5/(DIGI_BQ_FRAC_NBINS-1)),1+0.5/(DIGI_BQ_FRAC_NBINS-1));
      hfHists.DigiFirstCapID = m_dbe->book1D("HF Capid 1st Time Slice","HF Capid for 1st Time Slice",7,-3.5,3.5);
      hfHists.DigiFirstCapID -> setAxisTitle("CapID (T0) - 1st CapId (T0)",1);  
      hfHists.DigiFirstCapID -> setAxisTitle("# of Events",2);
      hfHists.DVerr = m_dbe->book1D("HF Data Valid Err Bits","HF QIE Data Valid Err Bits",4,-0.5,3.5);
      hfHists.DVerr ->setBinLabel(1,"Err=0, DV=0",1);
      hfHists.DVerr ->setBinLabel(2,"Err=0, DV=1",1);
      hfHists.DVerr ->setBinLabel(3,"Err=1, DV=0",1);
      hfHists.DVerr ->setBinLabel(4,"Err=1, DV=1",1);
      hfHists.CapID = m_dbe->book1D("HF CapID","HF CapID",4,-0.5,3.5);
      hfHists.ADC = m_dbe->book1D("HF ADC count per time slice","HF ADC count per time slice",200,-0.5,199.5);
      hfHists.ADCsum = m_dbe->book1D("HF ADC sum", "HF ADC sum",200,-0.5,199.5);

    } // if (m_dbe) // ends histogram setup
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiMonitor Setup -> "<<cpu_timer.cpuTime()<<endl;
    }

} // void HcalDigiMonitor::setup(...)


void HcalDigiMonitor::processEvent(const HBHEDigiCollection& hbhe,
				   const HODigiCollection& ho,
				   const HFDigiCollection& hf,
				   const HcalDbService& cond,
				   const HcalUnpackerReport& report)
{ 
  if(!m_dbe) 
    { 
      if(fVerbosity) 
	cout <<"HcalDigiMonitor::processEvent   DQMStore not instantiated!!!"<<endl; 
      return; 
    }
  

  ++ievt_;
  meEVT_->Fill(ievt_);
  
  int iEta, iPhi, iDepth;

  int err;
  bool occ, bitUp;

  hbHists.count_bad=0;
  hbHists.count_all=0;
  heHists.count_bad=0;
  heHists.count_all=0;
  hoHists.count_bad=0;
  hoHists.count_all=0;
  hfHists.count_bad=0;
  hfHists.count_all=0;

  int tssum=0;

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  ///////////////////////////////////////// Loop over HBHE

  int firsthbcap=-1; int firsthecap=-1;
  for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); ++j)
    {
	const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
	iEta = digi.id().ieta();
	iPhi = digi.id().iphi();
	iDepth = digi.id().depth();

	err=0x0;
	occ=false;
	bitUp=false;

	int ADCcount=0;

	// Check HB 
	if ((HcalSubdetector)(digi.id().subdet())==HcalBarrel)
	  {
	    if (!hbHists.check) continue;
	    ++hbHists.count_all;
	    // Check that digi size is correct
	    if (digi.size()<mindigisize_ || digi.size()>maxdigisize_)
	      {
		if (digi_checkdigisize_) err|=0x1;
		++baddigisize[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	      }
	    // Check digi size; if > 20, increment highest bin of digisize array
	    if (digi.size()<20)
	      ++digisize[static_cast<int>(digi.size())][0];
	    else
	      ++digisize[19][0];
	    // loop over time slices of digi to check capID and errors
	    ++hbHists.count_presample[digi.presamples()];


	    // Compare starting cap ID with first capID found
	    if (firsthbcap==-1) firsthbcap = digi.sample(0).capid();
	    int capdif = digi.sample(0).capid() - firsthbcap;
	    //capdif = capdif%3 - capdif/3; // unnecessary?
	    // capdif should run from -3 to +3
	    if (capdif >-4 && capdif<4)
	      ++hbHists.capIDdiff[capdif+3];
	    else
	      {
		++hbHists.capIDdiff[7];
		if (fVerbosity > 1)
		  cout <<"<HcalDigiMonitor> Odd behavior of HB capIDs:  capID diff = "<<capdif<<" = "<<digi.sample(0).capid()<< " - "<<firsthbcap<<endl;
	      }

	    int last=-1;
	    for (int i=0;i<digi.size();++i)
	      {
		// Check capid rotation
		int thisCapid = digi.sample(i).capid();
		if (thisCapid<4) ++hbHists.capid[thisCapid];
		if(bitUpset(last,thisCapid)) bitUp=true;
		last = thisCapid;

		// Check for digi error bits
		if (digi_checkdverr_)
		  {
		    if(digi.sample(i).er()) err=(err|0x2);
		    if(!digi.sample(i).dv()) err=(err|0x2);
		  }
		if (digi.sample(i).er() || !digi.sample(i).dv())
		  ++digierrorsdverr[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
		++hbHists.dverr[static_cast<int>(2*digi.sample(i).er()+digi.sample(i).dv())];
		//  Store ADC value and make ADC sum over whole digi sample
		ADCcount+=digi.sample(i).adc();
		if (digi.sample(i).adc()<200) ++hbHists.adc[digi.sample(i).adc()];
		hbHists.count_shape[i]+=digi.sample(i).adc();
		// Calculate ADC sum of adjacent samples
		if (i==digi.size()-1) continue;
		tssum= digi.sample(i).adc()+digi.sample(i+1).adc();
		if (tssum<45 && tssum>=-5)
		  {
		    if (iEta>0)
		      ++hbHists.tssumplus[tssum+5][i];
		    else
		      ++hbHists.tssumminus[tssum+5][i];
		  }
	      } // for (int i=0;i<digi.size();++i)
	    if(ADCcount>occThresh_) occ=true; 
	    if (ADCcount<200)
	      ++hbHists.adcsum[ADCcount];
	    if (ADCcount>shapeThresh_)
	      {
		for (int i=0;i<digi.size();++i)
		  hbHists.count_shapeThresh[i]+=digi.sample(i).adc();
	      }
	    if(bitUp) 
	      {
		if (digi_checkcapid_) err=(err|0x4);
		++badcapID[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	      }

	    ++occupancyEtaPhi[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	    ++occupancyEta[static_cast<int>(iEta+(etaBins_-2)/2)];
	    ++occupancyPhi[iPhi-1];
	    
	    // htr Slots run from 0-20, incremented by 0.5 for top/bottom
	    ++occupancyVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	    ++occupancySpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	    if (!occ)
	      {
		if (digi_checkadcsum_) err=err|0x8;
		++badADCsum[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	      }
	    if (err>0)
	      {
		++hbHists.count_bad;
		++problemdigis[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
		++errorVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
		++errorSpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	      }
	  } // if ((HcalSubdetector)(digi.id().subdet())==HcalBarrel)
	else
	  {
	    if (!heHists.check) continue;
	    ++heHists.count_all;
	    if (iDepth<3)
	      iDepth=iDepth+4;

	    // Check that digi size is correct
	    if (digi.size()<mindigisize_ || digi.size()>maxdigisize_)
	      {
		if (digi_checkdigisize_) err|=0x1;
		++baddigisize[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	      }
	    // Check digi size; if > 20, increment highest bin of digisize array
	    if (digi.size()<20)
	      ++digisize[static_cast<int>(digi.size())][1];
	    else
	      ++digisize[19][1];
	    // loop over time slices of digi to check capID and errors
	    ++heHists.count_presample[digi.presamples()];
	    
	    // Check CapID rotation
	    if (firsthecap==-1) firsthecap = digi.sample(0).capid();
	    int capdif = digi.sample(0).capid() - firsthecap;
	    //capdif = capdif%3 - capdif/3; // unnecessary?
	    // capdif should run from -3 to +3
	    if (capdif >-4 && capdif<4)
	      ++heHists.capIDdiff[capdif+3];
	    else
	      {
		++heHists.capIDdiff[7];
		if (fVerbosity > 1)
		  cout <<"<HcalDigiMonitor> Odd behavior of HB capIDs:  capID diff = "<<capdif<<" = "<<digi.sample(0).capid()<< " - "<<firsthbcap<<endl;
	      }
	    int last=-1;
	    for (int i=0;i<digi.size();++i)
	      {
		int thisCapid = digi.sample(i).capid();
		if (thisCapid<4) ++heHists.capid[thisCapid];
		if(bitUpset(last,thisCapid)) bitUp=true;
		last = thisCapid;
		// Check for digi error bits
		if (digi_checkdverr_)
		  {
		    if(digi.sample(i).er()) err=(err|0x2);
		    if(!digi.sample(i).dv()) err=(err|0x2);
		  }
		if (digi.sample(i).er() || !digi.sample(i).dv())
		  ++digierrorsdverr[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
		++heHists.dverr[static_cast<int>(2*digi.sample(i).er()+digi.sample(i).dv())];
		ADCcount+=digi.sample(i).adc();
		if (digi.sample(i).adc()<200) ++heHists.adc[digi.sample(i).adc()];
		heHists.count_shape[i]+=digi.sample(i).adc();
		// Calculate ADC sum of adjacent samples
		if (i==digi.size()-1) continue;
		tssum= digi.sample(i).adc()+digi.sample(i+1).adc();
		if (tssum<45 && tssum>=-5)
		  {
		    if (iEta>0)
		      ++heHists.tssumplus[tssum+5][i];
		    else
		      ++heHists.tssumminus[tssum+5][i];
		  }
	      } //for (int i=0;i<digi.size();++i)
	    if(ADCcount>occThresh_) occ=true; 
	    if (ADCcount<200)
	      ++heHists.adcsum[ADCcount];
	    if (ADCcount>shapeThresh_)
	      {
		for (int i=0;i<digi.size();++i)
		  heHists.count_shapeThresh[i]+=digi.sample(i).adc();
	      }
	    if(bitUp) 
	      {
		if (digi_checkcapid_) err=(err|0x4);
		++badcapID[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	      }
	    
	    ++occupancyEtaPhi[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	    ++occupancyEta[static_cast<int>(iEta+(etaBins_-2)/2)];
	    ++occupancyPhi[iPhi-1];
	    // htr Slots run from 0-20, incremented by 0.5 for top/bottom
	    ++occupancyVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	    ++occupancySpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	    if (!occ)
	      {
		if (digi_checkadcsum_) err=err|0x8;
		++badADCsum[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	      }
	    if (err>0)
	      {
		++heHists.count_bad;
		++problemdigis[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
		++errorVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
		++errorSpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	      }
	  } // else // HE loop
    } // loop over HBHE collection
  
  // Calculate number of bad quality cells and bad quality fraction
  if (hbHists.check && hbHists.count_all>0)
    {
      ++hbHists.count_BQ[static_cast<int>(hbHists.count_bad)];
      //if (hbHists.count_bad>0)
	++hbHists.count_BQFrac[static_cast<int>(hbHists.count_bad/hbHists.count_all)*DIGI_BQ_FRAC_NBINS];
    }
  if (heHists.check && heHists.count_all>0)
    {
      ++heHists.count_BQ[static_cast<int>(heHists.count_bad)];
      //if (heHists.count_bad>0)
	++heHists.count_BQFrac[static_cast<int>(heHists.count_bad/heHists.count_all)*DIGI_BQ_FRAC_NBINS];
    }

  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiMonitor DIGI HBHE -> "<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }


  //////////////////////////////////// Loop over HO collection
  if (hoHists.check)
    {
      int firsthocap=-1;
      for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); ++j)
	{
	  const HODataFrame digi = (const HODataFrame)(*j);
	  iEta = digi.id().ieta();
	  iPhi = digi.id().iphi();
	  iDepth = digi.id().depth();
	  
	  err=0x0;
	  occ=false;
	  bitUp=false;

	  int ADCcount=0;
	  ++hoHists.count_all;
	  
	  // Check that digi size is correct
	  if (digi.size()<mindigisize_ || digi.size()>maxdigisize_)
	    {
	      if (digi_checkdigisize_) err|=0x1;
	      ++baddigisize[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	    }
	  // Check digi size; if > 20, increment highest bin of digisize array
	  if (digi.size()<20)
	    ++digisize[static_cast<int>(digi.size())][2];
	  else
	    ++digisize[19][2];
	  // loop over time slices of digi to check capID and errors
	  ++hoHists.count_presample[digi.presamples()];

	  // Check CapID rotation
	  if (firsthocap==-1) firsthocap = digi.sample(0).capid();
	  int capdif = digi.sample(0).capid() - firsthocap;
	  // capdif should run from -3 to +3
	  if (capdif >-4 && capdif<4)
	    ++hoHists.capIDdiff[capdif+3];
	  else
	    {
	      ++hoHists.capIDdiff[7];
	      if (fVerbosity > 1)
		cout <<"<HcalDigiMonitor> Odd behavior of HB capIDs:  capID diff = "<<capdif<<" = "<<digi.sample(0).capid()<< " - "<<firsthbcap<<endl;
	    }
	  int last=-1;
	  for (int i=0;i<digi.size();++i)
	    {
	      int thisCapid = digi.sample(i).capid();
	      if (thisCapid<4) ++hoHists.capid[thisCapid];
	      if(bitUpset(last,thisCapid)) bitUp=true;
	      last = thisCapid;
	      // Check for digi error bits
	      if (digi_checkdverr_)
		{
		  if(digi.sample(i).er()) err=(err|0x2);
		  if(!digi.sample(i).dv()) err=(err|0x2);
		}
	      if (digi.sample(i).er() || !digi.sample(i).dv())
		++digierrorsdverr[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	      ++hoHists.dverr[static_cast<int>(2*digi.sample(i).er()+digi.sample(i).dv())];
	      ADCcount+=digi.sample(i).adc();
	      if (digi.sample(i).adc()<200) ++hoHists.adc[digi.sample(i).adc()];
	      hoHists.count_shape[i]+=digi.sample(i).adc();
	      // Calculate ADC sum of adjacent samples
		if (i==digi.size()-1) continue;
		tssum= digi.sample(i).adc()+digi.sample(i+1).adc();
		if (tssum<45 && tssum>=-5)
		  {
		    if (iEta>0)
		      ++hoHists.tssumplus[tssum+5][i];
		    else
		      ++hoHists.tssumminus[tssum+5][i];
		  }
	    } //for (int i=0;i<digi.size();++i)
	  if(ADCcount>occThresh_) occ=true;
	  if (ADCcount<200)
	    ++hoHists.adcsum[ADCcount];
	  if (ADCcount>shapeThresh_)
	    {
	      for (int i=0;i<digi.size();++i)
		hoHists.count_shapeThresh[i]+=digi.sample(i).adc();
	    }
	  if(bitUp) 
	    {
	      if (digi_checkcapid_) err=(err|0x4);
	      ++badcapID[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	    }
	  
	  ++occupancyEtaPhi[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	  ++occupancyEta[static_cast<int>(iEta+(etaBins_-2)/2)];
	  ++occupancyPhi[iPhi-1];
	  // htr Slots run from 0-20, incremented by 0.5 for top/bottom
	  ++occupancyVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	  ++occupancySpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	  if (!occ)
	    {
	      if (digi_checkadcsum_) err=err|0x8;
	      ++badADCsum[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	    }
	  if (err>0)
	    {
	      ++hoHists.count_bad;
	      ++problemdigis[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	      ++errorVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	      ++errorSpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	    }
	} // for (HODigiCollection)
   
      if (hoHists.count_all>0)
	{
	  ++hoHists.count_BQ[static_cast<int>(hoHists.count_bad)];
	  // if (hoHists.count_bad>0)
	    ++hoHists.count_BQFrac[static_cast<int>(hoHists.count_bad/hoHists.count_all)*DIGI_BQ_FRAC_NBINS];
	}
    } // if (hoHists.check)

  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiMonitor DIGI HO -> "<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  /////////////////////////////////////// Loop over HF collection
  if (hfHists.check)
    {
      int firsthfcap=-1;
      for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); ++j)
	{
	  const HFDataFrame digi = (const HFDataFrame)(*j);
	  iEta = digi.id().ieta();
	  iPhi = digi.id().iphi();
	  iDepth = digi.id().depth();
	  
	  err=0x0;
	  occ=false;
	  bitUp=false;

	  int ADCcount=0;
	  ++hfHists.count_all;
	  
	  // Check that digi size is correct
	  if (digi.size()<mindigisize_ || digi.size()>maxdigisize_)
	    {
	      if (digi_checkdigisize_) err|=0x1;
	      ++baddigisize[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	    }
	  // Check digi size; if > 20, increment highest bin of digisize array
	  if (digi.size()<20)
	    ++digisize[static_cast<int>(digi.size())][3];
	  else
	    ++digisize[19][3];
	  // loop over time slices of digi to check capID and errors
	  ++hfHists.count_presample[digi.presamples()];

	  // Check CapID rotation
	  if (firsthfcap==-1) firsthfcap = digi.sample(0).capid();
	  int capdif = digi.sample(0).capid() - firsthfcap;
	  //capdif = capdif%3 - capdif/3; // unnecessary?
	  // capdif should run from -3 to +3
	  if (capdif >-4 && capdif<4)
	    ++hoHists.capIDdiff[capdif+3];
	  else
	    {
	      ++hoHists.capIDdiff[7];
	      if (fVerbosity > 1)
		cout <<"<HcalDigiMonitor> Odd behavior of HB capIDs:  capID diff = "<<capdif<<" = "<<digi.sample(0).capid()<< " - "<<firsthbcap<<endl;
	    }
	  int last=-1;
	  for (int i=0;i<digi.size();++i)
	    {
	      int thisCapid = digi.sample(i).capid();
	      if (thisCapid<4) ++hfHists.capid[thisCapid];
	      if(bitUpset(last,thisCapid)) bitUp=true;
	      last = thisCapid;
	      // Check for digi error bits
	      if (digi_checkdverr_)
		{
		  if(digi.sample(i).er()) err=(err|0x2);
		  if(!digi.sample(i).dv()) err=(err|0x2);
		}
	      if (digi.sample(i).er() || !digi.sample(i).dv())
		++digierrorsdverr[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	      ++hfHists.dverr[static_cast<int>(2*digi.sample(i).er()+digi.sample(i).dv())];
	      ADCcount+=digi.sample(i).adc();
	      if (digi.sample(i).adc()<200) ++hfHists.adc[digi.sample(i).adc()];
	      hfHists.count_shape[i]+=digi.sample(i).adc();
	      // Calculate ADC sum of adjacent samples
		if (i==digi.size()-1) continue;
		tssum= digi.sample(i).adc()+digi.sample(i+1).adc();
		if (tssum<45 && tssum>=-5)
		  {
		    if (iEta>0)
		      ++hfHists.tssumplus[tssum+5][i];
		    else
		      ++hfHists.tssumminus[tssum+5][i];
		  }
	    } // for (int i=0;i<digi.size();++i)
	  if(ADCcount>occThresh_) occ=true; 
	  if (ADCcount<200)
	    ++hfHists.adcsum[ADCcount];
	  if (ADCcount>shapeThresh_)
	    {
	      for (int i=0;i<digi.size();++i)
		hfHists.count_shapeThresh[i]+=digi.sample(i).adc();
	    }
	  if(bitUp) 
	    {
	      if (digi_checkcapid_) err=(err|0x4);
	      ++badcapID[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	    }
	  
	  ++occupancyEtaPhi[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	  ++occupancyEta[static_cast<int>(iEta+(etaBins_-2)/2)];
	  ++occupancyPhi[iPhi-1];
	  // htr Slots run from 0-20, incremented by 0.5 for top/bottom
	  ++occupancyVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	  ++occupancySpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	  if (!occ)
	    {
	      if (digi_checkadcsum_) err=err|0x8;
	      ++badADCsum[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	    }
	  if (err>0)
	    {
	      ++hfHists.count_bad;
	      ++problemdigis[static_cast<int>(iEta+(etaBins_-2)/2)][iPhi-1][iDepth-1];
	      ++errorVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	      ++errorSpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	    }
	} // for (HFDigiCollection)
   
      if (hfHists.count_all>0)
	{
	  ++hfHists.count_BQ[static_cast<int>(hfHists.count_bad)];
	  // if (hfHists.count_bad>0)
	    ++hfHists.count_BQFrac[static_cast<int>(hfHists.count_bad/hfHists.count_all)*DIGI_BQ_FRAC_NBINS];
	}
    } // if (hfHists.check)

 if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiMonitor DIGI HF -> "<<cpu_timer.cpuTime()<<endl;
    }

  // This only counts digis that are present but bad somehow; it does not count digis that are missing
  int count_all=hbHists.count_all+heHists.count_all+hoHists.count_all+hfHists.count_all;
  int count_bad=hbHists.count_bad+heHists.count_bad+hoHists.count_bad+hfHists.count_bad;

  ++digiBQ[count_bad];
  ++diginum[count_all];
  if (count_all>0)
    ++digiBQfrac[static_cast<int>(count_bad/count_all)*DIGI_BQ_FRAC_NBINS];

  if (ievt_%digi_checkNevents_==0)
    fill_Nevents();
  
  return;
} // void HcalDigiMonitor::processEvent(...)


void HcalDigiMonitor::fill_Nevents()
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    cout <<"<HcalDigiMonitor> Calling fill_Nevents for event # "<<ievt_<<endl;
  int iPhi, iEta, iDepth;
  double problemvalue=0;
  double problemsum=0;
  bool valid=false;


  // Fill plots of sums of adjacent digi samples
  for (int i=0;i<10;++i)
    {
      for (int j=0;j<50;++j)
	{
	  if (hbHists.tssumplus[j][i]>0) hbHists.TS_sum_plus[i]->Fill(j, hbHists.tssumplus[j][i]);
	  if (hbHists.tssumminus[j][i]>0) hbHists.TS_sum_minus[i]->Fill(j, hbHists.tssumminus[j][i]);
	  if (heHists.tssumplus[j][i]>0) heHists.TS_sum_plus[i]->Fill(j, heHists.tssumplus[j][i]);
	  if (heHists.tssumminus[j][i]>0) heHists.TS_sum_minus[i]->Fill(j, heHists.tssumminus[j][i]);
	  if (hoHists.tssumplus[j][i]>0) hoHists.TS_sum_plus[i]->Fill(j, hoHists.tssumplus[j][i]);
	  if (hoHists.tssumminus[j][i]>0) hoHists.TS_sum_minus[i]->Fill(j, hoHists.tssumminus[j][i]);
	  if (hfHists.tssumplus[j][i]>0) hfHists.TS_sum_plus[i]->Fill(j, hfHists.tssumplus[j][i]);
	  if (hfHists.tssumminus[j][i]>0) hfHists.TS_sum_minus[i]->Fill(j, hfHists.tssumminus[j][i]);
	}
    } // for (int i=0;i<10;++i)

  // Fill plots of number of digis found
  for (int i=0;i<DIGI_NUM;++i)
    {
      if (diginum[i]>0) DigiNum->Fill(i, diginum[i]);
      if (digiBQ[i]>0) DigiBQ->Fill(i, digiBQ[i]);
      if (i>=DIGI_SUBDET_NUM) continue;
      if (hbHists.count_BQ[i]>0) hbHists.BQ->Fill(i, hbHists.count_BQ[i]);
      if (heHists.count_BQ[i]>0) heHists.BQ->Fill(i, heHists.count_BQ[i]);
      if (hoHists.count_BQ[i]>0) hoHists.BQ->Fill(i, hoHists.count_BQ[i]);
      if (hfHists.count_BQ[i]>0) hfHists.BQ->Fill(i, hfHists.count_BQ[i]);

    }//for int i=0;i<DIGI_NUM;++i)

  // Fill data-valid/error plots and capid plots
  for (int i=0;i<4;++i)
    {
      if (hbHists.dverr[i]>0) hbHists.DVerr->Fill(i, hbHists.dverr[i]);
      if (heHists.dverr[i]>0) heHists.DVerr->Fill(i, heHists.dverr[i]);
      if (hoHists.dverr[i]>0) hoHists.DVerr->Fill(i, hoHists.dverr[i]);
      if (hfHists.dverr[i]>0) hfHists.DVerr->Fill(i, hfHists.dverr[i]);
      if (hbHists.capid[i]>0) hbHists.CapID->Fill(i, hbHists.capid[i]);
      if (heHists.capid[i]>0) heHists.CapID->Fill(i, heHists.capid[i]);
      if (hoHists.capid[i]>0) hoHists.CapID->Fill(i, hoHists.capid[i]);
      if (hfHists.capid[i]>0) hfHists.CapID->Fill(i, hfHists.capid[i]);
    }
  for (int i=0;i<200;++i)
    {
      if (hbHists.adc[i]>0) hbHists.ADC->Fill(i, hbHists.adc[i]);
      if (heHists.adc[i]>0) heHists.ADC->Fill(i, heHists.adc[i]);
      if (hoHists.adc[i]>0) hoHists.ADC->Fill(i, hoHists.adc[i]);
      if (hfHists.adc[i]>0) hfHists.ADC->Fill(i, hfHists.adc[i]);
      if (hbHists.adcsum[i]>0) hbHists.ADCsum->Fill(i, hbHists.adcsum[i]);
      if (heHists.adcsum[i]>0) heHists.ADCsum->Fill(i, heHists.adcsum[i]);
      if (hoHists.adcsum[i]>0) hoHists.ADCsum->Fill(i, hoHists.adcsum[i]);
      if (hfHists.adcsum[i]>0) hfHists.ADCsum->Fill(i, hfHists.adcsum[i]);
    }


  // Fill plots of bad fraction of digis found
  for (int i=0;i<DIGI_BQ_FRAC_NBINS;++i)
    {
      if (digiBQfrac[i]>0) DigiBQFrac->Fill(i, digiBQfrac[i]);
      if (hbHists.count_BQFrac[i]>0) hbHists.BQFrac->Fill(i, hbHists.count_BQFrac[i]);
      if (heHists.count_BQFrac[i]>0) heHists.BQFrac->Fill(i, heHists.count_BQFrac[i]);
      if (hoHists.count_BQFrac[i]>0) hoHists.BQFrac->Fill(i, hoHists.count_BQFrac[i]);
      if (hfHists.count_BQFrac[i]>0) hfHists.BQFrac->Fill(i, hfHists.count_BQFrac[i]);

    }//for (int i=0;i<DIGI_BQ_FRAC_NBINS;++i)

  // Fill presample plots
  for (int i=0;i<50;++i)
    {
      if (hbHists.count_presample[i]>0) hbHists.presample->Fill(i, hbHists.count_presample[i]);
      if (heHists.count_presample[i]>0) heHists.presample->Fill(i, heHists.count_presample[i]);
      if (hoHists.count_presample[i]>0) hoHists.presample->Fill(i, hoHists.count_presample[i]);
      if (hfHists.count_presample[i]>0) hfHists.presample->Fill(i, hfHists.count_presample[i]);
    } //for (int i=0;i<50;++i)

  // Fill shape plots
  for (int i=0;i<10;++i)
    {
      if (hbHists.count_shape[i]>0) hbHists.shape->Fill(i, hbHists.count_shape[i]);
      if (hbHists.count_shapeThresh[i]>0) hbHists.shapeThresh->Fill(i, hbHists.count_shapeThresh[i]);
      if (heHists.count_shape[i]>0) heHists.shape->Fill(i, heHists.count_shape[i]);
      if (heHists.count_shapeThresh[i]>0) heHists.shapeThresh->Fill(i, heHists.count_shapeThresh[i]);
      if (hoHists.count_shape[i]>0) hoHists.shape->Fill(i, hoHists.count_shape[i]);
      if (hoHists.count_shapeThresh[i]>0) hoHists.shapeThresh->Fill(i, hoHists.count_shapeThresh[i]);
      if (hfHists.count_shape[i]>0) hfHists.shape->Fill(i, hfHists.count_shape[i]);
      if (hfHists.count_shapeThresh[i]>0) hfHists.shapeThresh->Fill(i, hfHists.count_shapeThresh[i]);
    }//  for (int i=0;i<10;++i)

  // Fill capID difference plots
  for (int i=0;i<8;++i)
    {
      if (hbHists.capIDdiff[i]>0) hbHists.DigiFirstCapID->Fill(i, hbHists.capIDdiff[i]);
      if (heHists.capIDdiff[i]>0) heHists.DigiFirstCapID->Fill(i, heHists.capIDdiff[i]);
      if (hoHists.capIDdiff[i]>0) hoHists.DigiFirstCapID->Fill(i, hoHists.capIDdiff[i]);
      if (hfHists.capIDdiff[i]>0) hfHists.DigiFirstCapID->Fill(i, hfHists.capIDdiff[i]);
    }


  // Fill VME plots
  for (int i=0;i<40;++i)
    {
      for (int j=0;j<18;++j)
	{
	  if (errorVME[i][j]>0) DigiErrorVME->Fill(i, j,errorVME[i][j]);
	  if (occupancyVME[i][j]>0) DigiOccupancyVME->Fill(i, j,occupancyVME[i][j]);
	}
    } //for (int i=0;i<40;++i)
  
  // Fill VME plots
  for (int i=0;i<HcalDCCHeader::SPIGOT_COUNT;++i)
    {
      for (int j=0;j<36;++j)
	{
	  if (errorSpigot[i][j]>0) DigiErrorSpigot->Fill(i, j,errorSpigot[i][j]);
	  if (occupancySpigot[i][j]>0) DigiOccupancySpigot->Fill(i, j,occupancySpigot[i][j]);
	}
    } //for (int i=0;i<HcalDCCHeader::SPIGOT_COUNT;++i)

  // Loop over subdetectors
  for (int sub=0;sub<4;++sub)
    {
      for (int dsize=0;dsize<20;++dsize)
	{
	  if (digisize[dsize][sub]>0)
	    DigiSize->Fill(sub,dsize,digisize[dsize][sub]);
	}
    } // for (int sub=0;sub<4;++sub)


  // Loop over eta, phi, depth
  for (int phi=0;phi<(phiBins_-2);++phi)
    {
      iPhi=phi+1;
      DigiOccupancyPhi->Fill(iPhi,occupancyPhi[phi]);
      for (int eta=0;eta<(etaBins_-2);++eta)
	{
	  iEta=eta-int((etaBins_-2)/2);
	  if (phi==0)
	    DigiOccupancyEta->Fill(iEta,occupancyEta[eta]);
	  problemsum=0;  
	  valid=false;

	  for (int d=0;d<6;++d)
	    {
	      iDepth=d+1;
	      ProblemDigisByDepth[d]->setBinContent(0,0,ievt_); // underflow bin contains event counter
	      // HB
	      if (validDetId(HcalBarrel, iEta, iPhi, iDepth))
		{
		  valid=true;
		  if (hbHists.check)
		    {
		      if (occupancyEtaPhi[eta][phi][d]==0 && digi_checkoccupancy_)
			{
			  problemdigis[eta][phi][d]+=digi_checkNevents_;
			}
		      
		      // Fill plots as fractions of total # of events
		      
		      // Occupancy plot needs to get old occupancy value, since counter gets reset
		      DigiOccupancyByDepth[d]->Fill(iEta, iPhi,
						    occupancyEtaPhi[eta][phi][d]);
		      
		      DigiErrorsBadCapID[d]->Fill(iEta, iPhi,
						  badcapID[eta][phi][d]);
		      DigiErrorsBadDigiSize[d]->Fill(iEta, iPhi,
						     baddigisize[eta][phi][d]);
		      DigiErrorsBadADCSum[d]->Fill(iEta, iPhi,
						   badADCsum[eta][phi][d]);
		      DigiErrorsDVErr[d]->Fill(iEta, iPhi,
					       digierrorsdverr[eta][phi][d]);
		      problemsum+=problemdigis[eta][phi][d];
		      problemvalue=min(ievt_,problemdigis[eta][phi][d]);
		      ProblemDigisByDepth[d]->Fill(iEta, iPhi,
						   problemvalue);
		      // Use this for testing purposes only
		      //ProblemDigisByDepth[d]->Fill(iEta, iPhi, ievt_);
		    } // if (hbHists.check)
		} 
	      // HE (depth=3 only)
	      if (d==2 && validDetId(HcalEndcap, iEta, iPhi, iDepth))
		{
		  valid=true;
		  if (heHists.check)
		    {
		      if (occupancyEtaPhi[eta][phi][d]==0 && digi_checkoccupancy_)
			{
			  problemdigis[eta][phi][d]+=digi_checkNevents_;
			}
		      
		      // Fill plots as fractions of total # of events
		      // Update -- making fractional plots needs to take place in Client; otherwise offline jobs split among processes won't calculate fractions correctly
		      
		      // Occupancy plot needs to get old occupancy value, since counter gets reset
		      DigiOccupancyByDepth[d]->Fill(iEta, iPhi,
						    occupancyEtaPhi[eta][phi][d]);
		      
		      DigiErrorsBadCapID[d]->Fill(iEta, iPhi,
						  badcapID[eta][phi][d]);
		      DigiErrorsBadDigiSize[d]->Fill(iEta, iPhi,
						     baddigisize[eta][phi][d]);
		      DigiErrorsBadADCSum[d]->Fill(iEta, iPhi,
						   badADCsum[eta][phi][d]);
		      DigiErrorsDVErr[d]->Fill(iEta, iPhi,
					       digierrorsdverr[eta][phi][d]);
		      problemsum+=problemdigis[eta][phi][d];
		      problemvalue=problemdigis[eta][phi][d];
		      ProblemDigisByDepth[d]->Fill(iEta, iPhi,
						   problemvalue);
		    } // if (heHists.check)
		} 
	      // HO 
	      if (validDetId(HcalOuter,iEta,iPhi,iDepth))
		{
		  valid=true;
		  if (hoHists.check)
		    {
		      if (occupancyEtaPhi[eta][phi][d]==0 && digi_checkoccupancy_)
			{
			  problemdigis[eta][phi][d]+=digi_checkNevents_;
			}
		      
		      // Fill plots as fractions of total # of events
		      
		      // Occupancy plot needs to get old occupancy value, since counter gets reset
		      DigiOccupancyByDepth[d]->Fill(iEta, iPhi,
						    occupancyEtaPhi[eta][phi][d]);

		      DigiErrorsBadCapID[d]->Fill(iEta, iPhi,
						  badcapID[eta][phi][d]);
		      DigiErrorsBadDigiSize[d]->Fill(iEta, iPhi,
						     baddigisize[eta][phi][d]);
		      DigiErrorsBadADCSum[d]->Fill(iEta, iPhi,
						   badADCsum[eta][phi][d]);
		      DigiErrorsDVErr[d]->Fill(iEta, iPhi,
					       digierrorsdverr[eta][phi][d]);
		      problemsum+=problemdigis[eta][phi][d];
		      problemvalue=problemdigis[eta][phi][d];
		      ProblemDigisByDepth[d]->Fill(iEta, iPhi,
						   problemvalue);
		    } // if (hoHists.check)
		}
	      // HF
	      if (validDetId(HcalForward,iEta,iPhi,iDepth))
		{
		  valid=true;
		  if (hfHists.check)
		    {
		      if (occupancyEtaPhi[eta][phi][d]==0 && digi_checkoccupancy_)
			{
		      problemdigis[eta][phi][d]+=digi_checkNevents_;
			}
		      
		      // Fill plots as fractions of total # of events
		      
		      // Occupancy plot needs to get old occupancy value, since counter gets reset
		      DigiOccupancyByDepth[d]->Fill(iEta, iPhi,
						    occupancyEtaPhi[eta][phi][d]);

		      DigiErrorsBadCapID[d]->Fill(iEta, iPhi,
						  badcapID[eta][phi][d]);
		      DigiErrorsBadDigiSize[d]->Fill(iEta, iPhi,
						     baddigisize[eta][phi][d]);
		      DigiErrorsBadADCSum[d]->Fill(iEta, iPhi,
						   badADCsum[eta][phi][d]);
		      DigiErrorsDVErr[d]->Fill(iEta, iPhi,
					       digierrorsdverr[eta][phi][d]);
		      problemsum+=problemdigis[eta][phi][d];
		      problemvalue=problemdigis[eta][phi][d];
		      ProblemDigisByDepth[d]->Fill(iEta, iPhi,
							    problemvalue);
		    } // if (hfHists.check)
		}
	      // HE (depths 1 & 2)
	      if (d>3)
		{
		  if (!heHists.check) continue;
		  iDepth=d-3; // iDepth=d+1, but shift down by 4 for HE
		  if (validDetId(HcalEndcap,iEta,iPhi,iDepth))
		    {
		      valid=true;
		      if (occupancyEtaPhi[eta][phi][d]==0 && digi_checkoccupancy_)
			{
			  problemdigis[eta][phi][d]+=digi_checkNevents_;
			}
		      
		      DigiOccupancyByDepth[d]->Fill(iEta, iPhi,
						    occupancyEtaPhi[eta][phi][d]);
		      
		      DigiErrorsBadCapID[d]->Fill(iEta, iPhi,
						  badcapID[eta][phi][d]);
		      DigiErrorsBadDigiSize[d]->Fill(iEta, iPhi,
						     baddigisize[eta][phi][d]);
		      DigiErrorsBadADCSum[d]->Fill(iEta, iPhi,
						   badADCsum[eta][phi][d]);
		      DigiErrorsDVErr[d]->Fill(iEta, iPhi,
					       digierrorsdverr[eta][phi][d]);
		      problemsum+=problemdigis[eta][phi][d];
		      problemvalue=problemdigis[eta][phi][d];
		      ProblemDigisByDepth[d]->Fill(iEta, iPhi,
						   problemvalue);
		    } // if (validDetId(HcalEndcap,iEta,iPhi,iDepth)
		} //if (d>3)
	     
	      occupancyEtaPhi[eta][phi][d]=0; //reset counter
	    } // for (int d=0;...)
	  if (valid==true) // only fill overall problem plot if the (eta,phi) value was valid for some depth
	    {
	      //problemvalue=min(1.,problemsum/ievt_);
	      problemsum=min((double)ievt_,problemsum);
	      ProblemDigis->Fill(iEta, iPhi,problemsum);
	      ProblemDigis->setBinContent(0,0,ievt_);
	    }
	} // for (int phi=0;...)
    } // for (int eta=0;...)

  // Now fill all the unphysical cell values
  FillUnphysicalHEHFBins(ProblemDigis);
  FillUnphysicalHEHFBins(ProblemDigisByDepth);
  FillUnphysicalHEHFBins(DigiErrorsBadCapID);
  FillUnphysicalHEHFBins(DigiErrorsDVErr);
  FillUnphysicalHEHFBins(DigiErrorsBadDigiSize);
  FillUnphysicalHEHFBins(DigiErrorsBadADCSum);
  FillUnphysicalHEHFBins(DigiOccupancyByDepth);

  zeroCounters();
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDigiMonitor DIGI fill_Nevents -> "<<cpu_timer.cpuTime()<<endl;
    }
  return;
} // void HcalDigiMonitor::fill_Nevents()

void HcalDigiMonitor::setSubDetectors(bool hb, bool he, bool ho, bool hf)
{
  hbHists.check&=hb;
  heHists.check&=he;
  hoHists.check&=ho;
  hfHists.check&=hf;
  
  return;
} // void HcalDigiMonitor::setSubDetectors(...)

void HcalDigiMonitor::zeroCounters()
{
  // Set all histogram counters back to 0
  /******** Zero all counters *******/
  
  for (int i=0;i<87;++i)
    {
      occupancyEta[i]=0;
      if (i<72)
	occupancyPhi[i]=0;
      for (int j=0;j<72;++j)
	{
	  for (int k=0;k<6;++k)
	    {
	      problemdigis[i][j][k]=0;
	      badcapID[i][j][k]=0;
	      baddigisize[i][j][k]=0;
	      badADCsum[i][j][k]=0;
	      occupancyEtaPhi[i][j][k]=0;
	      digierrorsdverr[i][j][k]=0;
	    }
	} // for (int j=0;j<72;++i)
    } // for (int i=0;i<87;++i)

  for (int i=0;i<40;++i)
    {
      for (int j=0;j<18;++j)
	{
	  occupancyVME[i][j]=0;
	  errorVME[i][j]=0;
	}
    }

  for (int i=0;i<HcalDCCHeader::SPIGOT_COUNT;++i)
    {
      for (int j=0;j<36;++j)
	{
	  occupancySpigot[i][j]=0;
	  errorSpigot[i][j]=0;
	}
    }


  for (int i=0;i<20;++i)
    {
      for (int j=0;j<4;++j)
	digisize[i][j]=0;
    }

  for (int i=0;i<DIGI_NUM;++i)
    {
      if (i<DIGI_BQ_FRAC_NBINS)
	digiBQfrac[i]=0;
      digiBQ[i]=0;
      diginum[i]=0;
      
      // set all DigiHists counters to 0
      if (i<4)
	{
	  hbHists.dverr[i]=0;
	  heHists.dverr[i]=0;
	  hoHists.dverr[i]=0;
	  hfHists.dverr[i]=0;
	  hbHists.capid[i]=0;
	  heHists.capid[i]=0;
	  hoHists.capid[i]=0;
	  hfHists.capid[i]=0;
	}
      if (i<8)
	{
	  hbHists.capIDdiff[i]=0;
	  heHists.capIDdiff[i]=0;
	  hoHists.capIDdiff[i]=0;
	  hfHists.capIDdiff[i]=0;
	}

      if (i<10)
	{
	  hbHists.count_shape[i]=0;
	  heHists.count_shape[i]=0;
	  hoHists.count_shape[i]=0;
	  hfHists.count_shape[i]=0;
	  hbHists.count_shapeThresh[i]=0;
	  heHists.count_shapeThresh[i]=0;
	  hoHists.count_shapeThresh[i]=0;
	  hfHists.count_shapeThresh[i]=0;
	}
      if (i<50)
	{
	  hbHists.count_presample[i]=0;
	  heHists.count_presample[i]=0;
	  hoHists.count_presample[i]=0;
	  hfHists.count_presample[i]=0;
	  for (int j=0;j<10;++j)
	    {
	      hbHists.tssumplus[i][j]=0;
	      heHists.tssumplus[i][j]=0;
	      hoHists.tssumplus[i][j]=0;
	      hfHists.tssumplus[i][j]=0;
	      hbHists.tssumminus[i][j]=0;
	      heHists.tssumminus[i][j]=0;
	      hoHists.tssumminus[i][j]=0;
	      hfHists.tssumminus[i][j]=0;
	    }
	}
      if (i<200)
	{
	  hbHists.adc[i]=0;
	  heHists.adc[i]=0;
	  hoHists.adc[i]=0;
	  hfHists.adc[i]=0;
	  hbHists.adcsum[i]=0;
	  heHists.adcsum[i]=0;
	  hoHists.adcsum[i]=0;
	  hfHists.adcsum[i]=0;
	}
      if (i<DIGI_SUBDET_NUM)
	{
	  hbHists.count_BQ[i]=0;
	  heHists.count_BQ[i]=0;
	  hoHists.count_BQ[i]=0;
	  hfHists.count_BQ[i]=0;
	}
      if (i<DIGI_BQ_FRAC_NBINS)
	{
	  hbHists.count_BQFrac[i]=0;
	  heHists.count_BQFrac[i]=0;
	  hoHists.count_BQFrac[i]=0;
	  hfHists.count_BQFrac[i]=0;
	}
    } // for (int i=0;i<DIGI_NUM;++i)


  return;
}

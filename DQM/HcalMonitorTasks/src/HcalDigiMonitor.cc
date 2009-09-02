#include "DQM/HcalMonitorTasks/interface/HcalDigiMonitor.h"
#include <cmath>


// constructor

HcalDigiMonitor::HcalDigiMonitor() {
  doPerChannel_ = false;
  occThresh_ = 1;
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
      std::cout << "<HcalDigiMonitor> Digi ADC occupancy threshold set to: >" << occThresh_ << std::endl;
      std::cout <<"<HcalDigiMonitor> Digi shape ADC threshold set to: >" << shapeThresh_ << std::endl;
    }
  makeDiagnostics = ps.getUntrackedParameter<bool>("DigiMonitor_MakeDiagnosticPlots",false); // not yet used

  doPerChannel_ = ps.getUntrackedParameter<bool>("DigiMonitor_DigisPerchannel",false); // not yet used -- never will be?
  if (fVerbosity>1)
    std::cout << "<HcalDigiMonitor> Digi phi min/max set to " << phiMin_ << "/" <<phiMax_ << std::endl;

  digi_checkNevents_ = ps.getUntrackedParameter<int>("DigiMonitor_checkNevents",checkNevents_); 
  if (fVerbosity>1)
    std::cout <<"<HcalDigiMonitor>  Perform checks and histogram fills every "<<digi_checkNevents_<<" events"<<std::endl;

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
  zdcHists.check=ps.getUntrackedParameter<bool>("checkZDC",true);

  if (fVerbosity>1)
    {
      std::cout <<"<HcalDigiMonitor> Checking for the following problems:"<<std::endl; 
      if (digi_checkoccupancy_) std::cout <<"\tChecking that digi present at least once every "<<digi_checkNevents_<<" events;"<<std::endl;
      if (digi_checkcapid_) std::cout <<"\tChecking that cap ID rotation is correct;"<<std::endl;
      if (digi_checkdigisize_) std::cout <<"\tChecking that digi size is between ["<<mindigisize_<<" - "<<maxdigisize_<<"];"<<std::endl;
      if (digi_checkadcsum_) std::cout <<"\tChecking that ADC sum of digi is greater than 0;"<<std::endl; 
      if (digi_checkdverr_) std::cout <<"\tChecking that data valid bit is true and digi error bit is false;"<<std::endl;
      std::cout <<"\tChecking digis for the following subdetectors:"<<std::endl;
      if (hbHists.check) std::cout <<"\tHB";
      if (heHists.check) std::cout <<"\tHE";
      if (hoHists.check) std::cout <<"\tHO";
      if (hfHists.check) std::cout <<"\tHF";
      if (zdcHists.check) std::cout <<"\tZDC";
      std::cout <<std::endl;
    }


  /******** Zero all counters *******/
  
  zeroCounters();

  /******* Set up all histograms  ********/

  if (m_dbe)
    {
      ostringstream name;
      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("Digi Task Event Number");    
      meEVT_->Fill(ievt_);
      meTOTALEVT_ = m_dbe->bookInt("Digi Task Total Events Processed");
      meTOTALEVT_->Fill(tevt_);

      MonitorElement* checkN = m_dbe->bookInt("DigiCheckNevents");
      checkN->Fill(digi_checkNevents_);
      MonitorElement* occT = m_dbe->bookInt("DigiOccThresh");
      occT->Fill(occThresh_);
      MonitorElement* shapeT = m_dbe->bookInt("DigiShapeThresh");
      shapeT->Fill(shapeThresh_);
      ProblemCells = m_dbe->book2D(" ProblemDigis",
				   " Problem Digi Rate for all HCAL",
				   85, -42.5, 42.5,
				   72, 0.5, 72.5);
      ProblemCells->setAxisTitle("i#eta",1);
      ProblemCells->setAxisTitle("i#phi",2);
      SetEtaPhiLabels(ProblemCells);

      
      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis");
      SetupEtaPhiHists(ProblemCellsByDepth," Problem Digi Rate","");
      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis/badcapID");
      SetupEtaPhiHists(DigiErrorsBadCapID," Digis with Bad Cap ID Rotation", "");
      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis/baddigisize");
      SetupEtaPhiHists(DigiErrorsBadDigiSize," Digis with Bad Size", "");
      DigiSize = m_dbe->book2D("Digi Size", "Digi Size",4,0,4,20,-0.5,19.5);
      DigiSize->setBinLabel(1,"HB",1);
      DigiSize->setBinLabel(2,"HE",1);
      DigiSize->setBinLabel(3,"HO",1);
      DigiSize->setBinLabel(4,"HF",1);
      DigiSize->setAxisTitle("Subdetector",1);
      DigiSize->setAxisTitle("Digi Size",2);

      /*
	// This is really a dead cell check.  If we need a check on ADC counts, we should put this in DeadCellMonitor.  Otherwise, let's delete this.
      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis/badADCsum");
      name<<" Digis with ADC sum below threshold ADC counts"; 
      SetupEtaPhiHists(DigiErrorsBadADCSum," Digis with ADC sum below threshold ADC counts", "");
      name.str("");
      for (int i=0;i<4;++i)
	{
	  name<<DigiErrorsBadADCSum.depth[i]->getTitle()<<"(ADC sum > "<<occThresh_<<" events)";
	  DigiErrorsBadADCSum.depth[i]->setTitle(static_cast<const string>(name.str().c_str()));
	  name.str("");
	}
      */

      m_dbe->setCurrentFolder(baseFolder_+"/1D_digi_plots");
      HBocc_vs_LB=m_dbe->bookProfile("HBoccVsLB","HB digi occupancy vs Luminosity Block",
				     Nlumiblocks_,0.5,Nlumiblocks_+0.5,
				     100,0,2600);
      HEocc_vs_LB=m_dbe->bookProfile("HEoccVsLB","HE digi occupancy vs Luminosity Block",
				     Nlumiblocks_,0.5,Nlumiblocks_+0.5,
				     100,0,2600);
      HOocc_vs_LB=m_dbe->bookProfile("HOoccVsLB","HO digi occupancy vs Luminosity Block",
				     Nlumiblocks_,0.5,Nlumiblocks_+0.5,
				     100,0,2200);
      HFocc_vs_LB=m_dbe->bookProfile("HFoccVsLB","HF digi occupancy vs Luminosity Block",
				     Nlumiblocks_,0.5,Nlumiblocks_+0.5,
				     100,0,2000);

      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis/badfibBCNoff");
      SetupEtaPhiHists(DigiErrorsBadFibBCNOff," Digis with non-zero Fiber Orbit Msg Offsets", "");

      m_dbe->setCurrentFolder(baseFolder_+"/problem_digis/data_invalid_error");
      SetupEtaPhiHists(DigiErrorsDVErr," Digis with Data Invalid or Error Bit Set", "");

      m_dbe->setCurrentFolder(baseFolder_+"/digi_occupancy");
      SetupEtaPhiHists(DigiOccupancyByDepth," Digi Eta-Phi Occupancy Map","");
      DigiOccupancyPhi= m_dbe->book1D("Digi Phi Occupancy Map",
				      "Digi Phi Occupancy Map",
				      72,0.5,72.5);
      DigiOccupancyPhi->setAxisTitle("i#phi",1);
      DigiOccupancyPhi->setAxisTitle("# of Events",2);
      DigiOccupancyEta= m_dbe->book1D("Digi Eta Occupancy Map",
				      "Digi Eta Occupancy Map",
				      83,-41.5,41.5);
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
                                      83, -41.5, 41.5, 72, 0.5, 72.5);
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
      
      float bins_cellcount[]={-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5,
			      11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5,
			      21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5,
			      31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5,
			      41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5,
			      60.5, 70.5, 80.5, 90.5, 100.5, 150.5, 200.5, 250.5, 300.5,
			      400.5, 500.5, 600.5, 700.5, 800.5, 900.5, 1000.5, 1100.5,
			      1200.5, 1300.5, 1400.5, 1500.5, 1600.5, 1700.5, 1800.5, 1900.5,
			      2000.5, 2100.5, 2200.5, 2300.5, 2400.5, 2500.5, 2600.5, 2700.5,
			      2800.5, 2900.5, 3000.5, 3100.5, 3200.5, 3300.5, 3400.5, 3500.5,
			      3600.5, 3700.5, 3800.5, 3900.5, 4000.5, 4100.5, 4200.5, 4300.5,
			      4400.5, 4500.5, 4600.5, 4700.5, 4800.5, 4900.5, 5000.5, 5100.5,
			      5200.5, 5300.5, 5400.5, 5500.5, 5600.5, 5700.5, 5800.5, 5900.5,
			      6000.5, 6100.5, 6200.5, 6300.5, 6400.5, 6500.5, 6600.5, 6700.5,
			      6800.5, 6900.5, 7000.5, 7100.5, 7200.5, 7300.5, 7400.5, 7500.5,
			      7600.5, 7700.5, 7800.5, 7900.5, 8000.5, 8100.5, 8200.5, 8300.5,
			      8400.5, 8500.5, 8600.5, 8700.5, 8800.5, 8900.5, 9000.5, 9100.5};

      

      DigiBQ = m_dbe->book1D("# Bad Qual Digis","# Bad Qual Digis",148, bins_cellcount);
      ProblemsVsLB=m_dbe->bookProfile("BadDigisVsLB","# Bad Digis vs Luminosity block", Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      ProblemsVsLB -> setAxisTitle("Lumi block",1);  
      ProblemsVsLB -> setAxisTitle("# of Bad digis",2);
      
      DigiBQFrac =  m_dbe->book1D("Bad Digi Fraction","Bad Digi Fraction",DIGI_BQ_FRAC_NBINS,(0-0.5/(DIGI_BQ_FRAC_NBINS-1)),1+0.5/(DIGI_BQ_FRAC_NBINS-1));
      DigiBQFrac -> setAxisTitle("Bad Quality Digi Fraction",1);  
      DigiBQFrac -> setAxisTitle("# of Events",2);


      m_dbe->setCurrentFolder(baseFolder_+"/digi_info");
      DigiNum = m_dbe->book1D("# of Digis","# of Digis",DIGI_NUM+1,-0.5,DIGI_NUM+1-0.5);
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
      ProblemsVsLB_HB=m_dbe->bookProfile("HB Bad Quality Digis vs LB","HB Bad Quality Digis vs Luminosity Block",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,
					 100,0,10000);
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
      hbHists.fibBCNOff = m_dbe->book1D("HB Fiber Orbit Message BCN Offset", "HB Fiber Orbit Message BCN Offset",
					15, -7.5, 7.5);
      hbHists.fibBCNOff->setAxisTitle("Offset from Expected", 1);
      

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
      ProblemsVsLB_HE=m_dbe->bookProfile("HE Bad Quality Digis vs LB","HE Bad Quality Digis vs Luminosity Block",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,
					 100,0,10000);

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
      heHists.fibBCNOff = m_dbe->book1D("HB Fiber Orbit Message BCN Offset", "HB Fiber Orbit Message BCN Offset",
					15, -7.5, 7.5);
      heHists.fibBCNOff->setAxisTitle("Offset from Expected", 1);

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
      ProblemsVsLB_HO=m_dbe->bookProfile("HO Bad Quality Digis vs LB","HO Bad Quality Digis vs Luminosity Block",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,
					 100,0,10000);
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
      hoHists.fibBCNOff = m_dbe->book1D("HB Fiber Orbit Message BCN Offset", "HB Fiber Orbit Message BCN Offset",
					15, -7.5, 7.5);
      hoHists.fibBCNOff->setAxisTitle("Offset from Expected", 1);

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
      ProblemsVsLB_HF=m_dbe->bookProfile("HF Bad Quality Digis vs LB","HF Bad Quality Digis vs Luminosity Block",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,
					 100,0,10000);
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
      hfHists.fibBCNOff = m_dbe->book1D("HB Fiber Orbit Message BCN Offset", "HB Fiber Orbit Message BCN Offset",
					15, -7.5, 7.5);
      hfHists.fibBCNOff->setAxisTitle("Offset from Expected", 1);


      m_dbe->setCurrentFolder(baseFolder_+"/digi_info/ZDC");
      zdcHists.shape = m_dbe->book1D("ZDC Digi Shape","ZDC Digi Shape",10,-0.5,9.5);
      zdcHists.shapeThresh = m_dbe->book1D("ZDC Digi Shape - over thresh",
					  "ZDC Digi Shape - over thresh",
					  10,-0.5,9.5);
      // Create plots of sums of adjacent time slices
      for (int ts=0;ts<9;++ts)
	{
	  name<<"ZDC Plus Time Slices "<<ts<<" and "<<ts+1;
	  zdcHists.TS_sum_plus.push_back(m_dbe->book1D(name.str().c_str(),name.str().c_str(),50,-5.5,44.5));
	  name.str("");
	  name<<"ZDC Minus Time Slices "<<ts<<" and "<<ts+1;
	  zdcHists.TS_sum_minus.push_back(m_dbe->book1D(name.str().c_str(),name.str().c_str(),50,-5.5,44.5));
	  name.str("");
	}
      zdcHists.shape->setAxisTitle("Time Slice",1);
      zdcHists.shapeThresh->setAxisTitle("Time Slice",1);
      zdcHists.presample= m_dbe->book1D("ZDC Digi Presamples","ZDC Digi Presamples",50,-0.5,49.5);
      zdcHists.BQ = m_dbe->book1D("ZDC Bad Quality Digis","ZDC Bad Quality Digis",DIGI_SUBDET_NUM,-0.5,DIGI_SUBDET_NUM-0.5);
      zdcHists.BQFrac = m_dbe->book1D("ZDC Bad Quality Digi Fraction","ZDC Bad Quality Digi Fraction",DIGI_BQ_FRAC_NBINS,(0-0.5/(DIGI_BQ_FRAC_NBINS-1)),1+0.5/(DIGI_BQ_FRAC_NBINS-1));
      zdcHists.DigiFirstCapID = m_dbe->book1D("ZDC Capid 1st Time Slice","ZDC Capid for 1st Time Slice",7,-3.5,3.5);
      zdcHists.DigiFirstCapID -> setAxisTitle("CapID (T0) - 1st CapId (T0)",1);  
      zdcHists.DigiFirstCapID -> setAxisTitle("# of Events",2);
      zdcHists.DVerr = m_dbe->book1D("ZDC Data Valid Err Bits","ZDC QIE Data Valid Err Bits",4,-0.5,3.5);
      zdcHists.DVerr ->setBinLabel(1,"Err=0, DV=0",1);
      zdcHists.DVerr ->setBinLabel(2,"Err=0, DV=1",1);
      zdcHists.DVerr ->setBinLabel(3,"Err=1, DV=0",1);
      zdcHists.DVerr ->setBinLabel(4,"Err=1, DV=1",1);
      zdcHists.CapID = m_dbe->book1D("ZDC CapID","ZDC CapID",4,-0.5,3.5);
      zdcHists.ADC = m_dbe->book1D("ZDC ADC count per time slice","ZDC ADC count per time slice",200,-0.5,199.5);
      zdcHists.ADCsum = m_dbe->book1D("ZDC ADC sum", "ZDC ADC sum",200,-0.5,199.5);
    } // if (m_dbe) // ends histogram setup
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiMonitor Setup -> "<<cpu_timer.cpuTime()<<std::endl;
    }

} // void HcalDigiMonitor::setup(...)


void HcalDigiMonitor::processEvent(const HBHEDigiCollection& hbhe,
				   const HODigiCollection& ho,
				   const HFDigiCollection& hf,
				   const ZDCDigiCollection& zdc,
				   const HcalDbService& cond,
				   const HcalUnpackerReport& report)
{ 
  if(!m_dbe) 
    { 
      if(fVerbosity) 
	std::cout <<"HcalDigiMonitor::processEvent   DQMStore not instantiated!!!"<<std::endl; 
      return; 
    }
  
  HcalBaseMonitor::processEvent();
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
  int hbcount=0;
  int hecount=0;
  for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); ++j)
    {
	const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
	iEta = digi.id().ieta();
	iPhi = digi.id().iphi();
	iDepth = digi.id().depth();
        HcalSubdetector subdet = digi.id().subdet();
        int calcEta = CalcEtaBin(subdet,iEta,iDepth);

	err=0x0;
	occ=false;
	bitUp=false;

	int ADCcount=0;

	// Check HB 
	if (subdet==HcalBarrel)
	  {
	    if (!hbHists.check) continue;
	    ++hbHists.count_all;
	    ++hbcount;
	    // Check that digi size is correct
	    if (digi.size()<mindigisize_ || digi.size()>maxdigisize_)
	      {
		if (digi_checkdigisize_) err|=0x1;
		++baddigisize[calcEta][iPhi-1][iDepth-1];
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
		  std::cout <<"<HcalDigiMonitor> Odd behavior of HB capIDs:  capID diff = "<<capdif<<" = "<<digi.sample(0).capid()<< " - "<<firsthbcap<<std::endl;
	      }

	    int last=-1;

	    int offset = digi.fiberIdleOffset();
	    if (offset != -1000) {
	      ++hbHists.fibbcnoff[offset + 7];
	      if (offset != 0) {
		++badFibBCNOff[calcEta][iPhi-1][iDepth-1];
		err |= 0xF;
	      }
	    }

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
		  ++digierrorsdverr[calcEta][iPhi-1][iDepth-1];
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
		++badcapID[calcEta][iPhi-1][iDepth-1];
	      }

	    ++occupancyEtaPhi[calcEta][iPhi-1][iDepth-1];
	    ++occupancyEta[iEta+41];
	    ++occupancyPhi[iPhi-1];
	    
	    // htr Slots run from 0-20, incremented by 0.5 for top/bottom
	    ++occupancyVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	    ++occupancySpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];

	    /*
	      // This is a dead cell check; we shouldn't need it here
	    if (!occ)
	      {
		if (digi_checkadcsum_) err=err|0x8;
		++badADCsum[calcEta][iPhi-1][iDepth-1];
	      }
	    */
	    if (err>0)
	      {
		++hbHists.count_bad;
		++problemdigis[calcEta][iPhi-1][iDepth-1];
		++errorVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
		++errorSpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	      }
	  } // if ((HcalSubdetector)(digi.id().subdet())==HcalBarrel)
	else
	  {
	    if (!heHists.check) continue;
	    ++heHists.count_all;
	    ++hecount;
	    // Check that digi size is correct
	    if (digi.size()<mindigisize_ || digi.size()>maxdigisize_)
	      {
		if (digi_checkdigisize_) err|=0x1;
		++baddigisize[calcEta][iPhi-1][iDepth-1];
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
		  std::cout <<"<HcalDigiMonitor> Odd behavior of HE capIDs:  capID diff = "<<capdif<<" = "<<digi.sample(0).capid()<< " - "<<firsthbcap<<std::endl;
	      }
	    int last=-1;
	    int offset = digi.fiberIdleOffset();

	    if (offset != -1000) {
	      ++heHists.fibbcnoff[offset + 7];
	      if (offset != 0) {
		++badFibBCNOff[calcEta][iPhi-1][iDepth-1];
		err |= 0xF;
	      }
	    }

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
		  ++digierrorsdverr[calcEta][iPhi-1][iDepth-1];
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
		++badcapID[calcEta][iPhi-1][iDepth-1];
	      }
	    
	    ++occupancyEtaPhi[calcEta][iPhi-1][iDepth-1];
	    ++occupancyEta[iEta+41];
	    ++occupancyPhi[iPhi-1];
	    // htr Slots run from 0-20, incremented by 0.5 for top/bottom
	    ++occupancyVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	    ++occupancySpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	    /*
	    if (!occ)
	      {
		if (digi_checkadcsum_) err=err|0x8;
		++badADCsum[calcEta][iPhi-1][iDepth-1];
	      }
	    */
	    if (err>0)
	      {
		++heHists.count_bad;
		++problemdigis[calcEta][iPhi-1][iDepth-1];
		++errorVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
		++errorSpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	      }
	  } // else // HE loop
    } // loop over HBHE collection
  
  HBocc_vs_LB->Fill(lumiblock,hbcount);
  HEocc_vs_LB->Fill(lumiblock,hecount);

  // Calculate number of bad quality cells and bad quality fraction
  if (hbHists.check && hbHists.count_all>0 && hbHists.count_bad>0)
    {
      ++hbHists.count_BQ[static_cast<int>(hbHists.count_bad)];
      //if (hbHists.count_bad>0)
	++hbHists.count_BQFrac[static_cast<int>(hbHists.count_bad/hbHists.count_all)*DIGI_BQ_FRAC_NBINS];
    }
  if (heHists.check && heHists.count_all>0 && heHists.count_bad>0)
    {
      ++heHists.count_BQ[static_cast<int>(heHists.count_bad)];
      //if (heHists.count_bad>0)
	++heHists.count_BQFrac[static_cast<int>(heHists.count_bad/heHists.count_all)*DIGI_BQ_FRAC_NBINS];
    }

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiMonitor DIGI HBHE -> "<<cpu_timer.cpuTime()<<std::endl;
      cpu_timer.reset(); cpu_timer.start();
    }


  //////////////////////////////////// Loop over HO collection
  if (hoHists.check)
    {
      int firsthocap=-1;
      int hocount=0;
      for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); ++j)
	{
	  ++hocount;
	  const HODataFrame digi = (const HODataFrame)(*j);
	  iEta = digi.id().ieta();
	  iPhi = digi.id().iphi();
	  iDepth = digi.id().depth();
          int calcEta = CalcEtaBin(HcalOuter,iEta,iDepth);

	  err=0x0;
	  occ=false;
	  bitUp=false;

	  int ADCcount=0;
	  ++hoHists.count_all;
	  
	  // Check that digi size is correct
	  if (digi.size()<mindigisize_ || digi.size()>maxdigisize_)
	    {
	      if (digi_checkdigisize_) err|=0x1;
	      ++baddigisize[calcEta][iPhi-1][iDepth-1];
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
		std::cout <<"<HcalDigiMonitor> Odd behavior of HO capIDs:  capID diff = "<<capdif<<" = "<<digi.sample(0).capid()<< " - "<<firsthbcap<<std::endl;
	    }
	  int last=-1;

	  int offset = digi.fiberIdleOffset();
	  if (offset != -1000) {
	    ++hoHists.fibbcnoff[offset + 7];
	    if (offset != 0) {
	      ++badFibBCNOff[calcEta][iPhi-1][iDepth-1];
	      err |= 0xF;
	    }
	  }

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
		++digierrorsdverr[calcEta][iPhi-1][iDepth-1];
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
	      ++badcapID[calcEta][iPhi-1][iDepth-1];
	    }
	  
	  ++occupancyEtaPhi[calcEta][iPhi-1][iDepth-1];
	  ++occupancyEta[iEta+41];
	  ++occupancyPhi[iPhi-1];
	  // htr Slots run from 0-20, incremented by 0.5 for top/bottom
	  ++occupancyVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	  ++occupancySpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	  /*
	    if (!occ)
	    {
	    if (digi_checkadcsum_) err=err|0x8;
	    ++badADCsum[calcEta][iPhi-1][iDepth-1];
	    }
	  */
	  if (err>0)
	    {
	      ++hoHists.count_bad;
	      ++problemdigis[calcEta][iPhi-1][iDepth-1];
	      ++errorVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	      ++errorSpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	    }
	} // for (HODigiCollection)
   
      if (hoHists.count_bad>0 && hoHists.count_all>0)
	{
	  ++hoHists.count_BQ[static_cast<int>(hoHists.count_bad)];
	  // if (hoHists.count_bad>0)
	    ++hoHists.count_BQFrac[static_cast<int>(hoHists.count_bad/hoHists.count_all)*DIGI_BQ_FRAC_NBINS];
	}
      HOocc_vs_LB->Fill(lumiblock,hocount);
    } // if (hoHists.check)

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiMonitor DIGI HO -> "<<cpu_timer.cpuTime()<<std::endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  /////////////////////////////////////// Loop over HF collection
  if (hfHists.check)
    {
      int firsthfcap=-1;
      int hfcount=0;
      for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); ++j)
	{
	  ++hfcount;
	  const HFDataFrame digi = (const HFDataFrame)(*j);
	  iEta = digi.id().ieta();
	  iPhi = digi.id().iphi();
	  iDepth = digi.id().depth();
          int calcEta = CalcEtaBin(HcalForward,iEta,iDepth);
	  
	  err=0x0;
	  occ=false;
	  bitUp=false;

	  int ADCcount=0;
	  ++hfHists.count_all;
	  
	  // Check that digi size is correct
	  if (digi.size()<mindigisize_ || digi.size()>maxdigisize_)
	    {
	      if (digi_checkdigisize_) err|=0x1;
	      ++baddigisize[calcEta][iPhi-1][iDepth-1];
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
	    ++hfHists.capIDdiff[capdif+3];
	  else
	    {
	      ++hfHists.capIDdiff[7];
	      if (fVerbosity > 1)
		std::cout <<"<HcalDigiMonitor> Odd behavior of HF capIDs:  capID diff = "<<capdif<<" = "<<digi.sample(0).capid()<< " - "<<firsthbcap<<std::endl;
	    }
	  int last=-1;

	  int offset = digi.fiberIdleOffset();
	  if (offset != -1000) {
	    ++hfHists.fibbcnoff[offset + 7];
	    if (offset != 0) {
	      ++badFibBCNOff[calcEta][iPhi-1][iDepth-1];
	      err |= 0xF;
	    }
	  }

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
		++digierrorsdverr[calcEta][iPhi-1][iDepth-1];
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
	      ++badcapID[calcEta][iPhi-1][iDepth-1];
	    }
	  
	  ++occupancyEtaPhi[calcEta][iPhi-1][iDepth-1];
	  ++occupancyEta[iEta+41];
	  ++occupancyPhi[iPhi-1];
	  // htr Slots run from 0-20, incremented by 0.5 for top/bottom
	  ++occupancyVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	  ++occupancySpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	  /*
	  if (!occ)
	    {
	      if (digi_checkadcsum_) err=err|0x8;
	      ++badADCsum[calcEta][iPhi-1][iDepth-1];
	    }
	  */
	  if (err>0)
	    {
	      ++hfHists.count_bad;
	      ++problemdigis[calcEta][iPhi-1][iDepth-1];
	      ++errorVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	      ++errorSpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	    }
	} // for (HFDigiCollection)
   
      if (hfHists.count_bad>0 && hfHists.count_all>0)
	{
	  ++hfHists.count_BQ[static_cast<int>(hfHists.count_bad)];
	  // if (hfHists.count_bad>0)
	    ++hfHists.count_BQFrac[static_cast<int>(hfHists.count_bad/hfHists.count_all)*DIGI_BQ_FRAC_NBINS];
	}
      HFocc_vs_LB->Fill(lumiblock,hfcount);
    } // if (hfHists.check)
  
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiMonitor DIGI HF -> "<<cpu_timer.cpuTime()<<std::endl;
    }

 
  int zside, zsection, zdepth, zchannel;
  /////////////////////////////////////// Loop over ZDC collection
  if (zdcHists.check)
    {
      int firstzdccap=-1;
      for (ZDCDigiCollection::const_iterator j=zdc.begin(); j!=zdc.end(); ++j)
	{
	  const ZDCDataFrame digi = (const ZDCDataFrame)(*j);
	  zside=digi.id().zside();
	  zsection=digi.id().section();
	  zdepth=digi.id().depth();
	  zchannel=digi.id().channel();
	  err=0x0;
	  occ=false;
	  bitUp=false;

	  int ADCcount=0;
	  ++zdcHists.count_all;
	  
	  // Check that digi size is correct
	  if (digi.size()<mindigisize_ || digi.size()>maxdigisize_)
	    {
	      if (digi_checkdigisize_) err|=0x1;
	      //++baddigisize[calcEta][iPhi-1][iDepth-1];
	    }
	  // Check digi size; if > 20, increment highest bin of digisize array
	  if (digi.size()<20)
	    ++digisize[static_cast<int>(digi.size())][3];
	  else
	    ++digisize[19][3];
	  // loop over time slices of digi to check capID and errors
	  //++zdcHists.count_presample[digi.presamples()];
	  

	  // Check CapID rotation
	  if (firstzdccap==-1) firstzdccap = digi.sample(0).capid();
	  int capdif = digi.sample(0).capid() - firstzdccap;
	  //capdif = capdif%3 - capdif/3; // unnecessary?
	  // capdif should run from -3 to +3
	  if (capdif >-4 && capdif<4)
	    ++zdcHists.capIDdiff[capdif+3];
	  else
	    {
	      ++zdcHists.capIDdiff[7];
	      if (fVerbosity > 1)
		std::cout <<"<HcalDigiMonitor> Odd behavior of ZDC capIDs:  capID diff = "<<capdif<<" = "<<digi.sample(0).capid()<< " - "<<firsthbcap<<std::endl;
	    }

	  int last=-1;
	  for (int i=0;i<digi.size();++i)
	    {
	      int thisCapid = digi.sample(i).capid();
	      if (thisCapid<4) ++zdcHists.capid[thisCapid];
	      if(bitUpset(last,thisCapid)) bitUp=true;
	      last = thisCapid;
	      // Check for digi error bits
	      if (digi_checkdverr_)
		{
		  if(digi.sample(i).er()) err=(err|0x2);
		  if(!digi.sample(i).dv()) err=(err|0x2);
		}
	      //if (digi.sample(i).er() || !digi.sample(i).dv())
	      //	++digierrorsdverr[calcEta][iPhi-1][iDepth-1];
	      ++zdcHists.dverr[static_cast<int>(2*digi.sample(i).er()+digi.sample(i).dv())];
	      ADCcount+=digi.sample(i).adc();
	      if (digi.sample(i).adc()<200) ++zdcHists.adc[digi.sample(i).adc()];
	      zdcHists.count_shape[i]+=digi.sample(i).adc();
	      // Calculate ADC sum of adjacent samples
		if (i==digi.size()-1) continue;
		tssum= digi.sample(i).adc()+digi.sample(i+1).adc();
		if (tssum<45 && tssum>=-5)
		  {
		    if (zside>0)
		      ++zdcHists.tssumplus[tssum+5][i];
		    else
		      ++zdcHists.tssumminus[tssum+5][i];
		  }
	    } // for (int i=0;i<digi.size();++i)
	  if(ADCcount>occThresh_) occ=true; 
	  if (ADCcount<200)
	    ++zdcHists.adcsum[ADCcount];
	  if (ADCcount>shapeThresh_)
	    {
	      for (int i=0;i<digi.size();++i)
		zdcHists.count_shapeThresh[i]+=digi.sample(i).adc();
	    }
	  if(bitUp) 
	    {
	      if (digi_checkcapid_) err=(err|0x4);
	      //++badcapID[calcEta][iPhi-1][iDepth-1];
	    }
	  
	  //++occupancyEtaPhi[calcEta][iPhi-1][iDepth-1];
	  //++occupancyEta[iEta+41];
	  //++occupancyPhi[iPhi-1];
	  // htr Slots run from 0-20, incremented by 0.5 for top/bottom
	  ++occupancyVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	  ++occupancySpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	  /*
	  if (!occ)
	    {
	      if (digi_checkadcsum_) err=err|0x8;
	      ++badADCsum[calcEta][iPhi-1][iDepth-1];
	    }
	  */
	  if (err>0)
	    {
	      ++zdcHists.count_bad;
	      //++problemdigis[calcEta][iPhi-1][iDepth-1];
	      ++errorVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
	      ++errorSpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
	    }
	} // for (ZDCDigiCollection)
   
      if (zdcHists.count_all>0)
	{
	  ++zdcHists.count_BQ[static_cast<int>(zdcHists.count_bad)];
	  // if (zdcHists.count_bad>0)
	    ++zdcHists.count_BQFrac[static_cast<int>(zdcHists.count_bad/zdcHists.count_all)*DIGI_BQ_FRAC_NBINS];
	}
    } // if (zdcHists.check)





  // This only counts digis that are present but bad somehow; it does not count digis that are missing
  int count_all=hbHists.count_all+heHists.count_all+hoHists.count_all+hfHists.count_all;
  int count_bad=hbHists.count_bad+heHists.count_bad+hoHists.count_bad+hfHists.count_bad;

  ++digiBQ[count_bad];
  ++diginum[count_all];
  if (count_all>0)
    ++digiBQfrac[static_cast<int>(count_bad/count_all)*DIGI_BQ_FRAC_NBINS];

  //Jeff's dummy fills to make sure plots update.  Hmm...
  hbHists.fibBCNOff->Fill(-1,0);
  heHists.fibBCNOff->Fill(-1,0);
  hoHists.fibBCNOff->Fill(-1,0);
  hfHists.fibBCNOff->Fill(-1,0);

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
    std::cout <<"<HcalDigiMonitor> Calling fill_Nevents for event # "<<ievt_<<std::endl;
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
	  if (zdcHists.tssumplus[j][i]>0) zdcHists.TS_sum_plus[i]->Fill(j, zdcHists.tssumplus[j][i]); 
	  if (zdcHists.tssumminus[j][i]>0) zdcHists.TS_sum_minus[i]->Fill(j, zdcHists.tssumminus[j][i]); 
	}
    } // for (int i=0;i<10;++i)

  // Fill plots of number of digis found
  NumBadHB=0;
  NumBadHE=0;
  NumBadHO=0;
  NumBadHF=0;
  NumBadZDC=0;

  for (int i=0;i<DIGI_NUM;++i)
    {
      if (diginum[i]>0) DigiNum->Fill(i, diginum[i]);
      if (digiBQ[i]>0) DigiBQ->Fill(i, digiBQ[i]);
      if (i>=DIGI_SUBDET_NUM) continue;
      if (hbHists.count_BQ[i]>0) hbHists.BQ->Fill(i, hbHists.count_BQ[i]);
      if (heHists.count_BQ[i]>0) heHists.BQ->Fill(i, heHists.count_BQ[i]);
      if (hoHists.count_BQ[i]>0) hoHists.BQ->Fill(i, hoHists.count_BQ[i]);
      if (hfHists.count_BQ[i]>0) hfHists.BQ->Fill(i, hfHists.count_BQ[i]);
      if (zdcHists.count_BQ[i]>0) zdcHists.BQ->Fill(i, zdcHists.count_BQ[i]);
      if (hbHists.count_BQ[i]>0) ++NumBadHB;
      if (heHists.count_BQ[i]>0) ++NumBadHE;
      if (hoHists.count_BQ[i]>0) ++NumBadHO;
      if (hfHists.count_BQ[i]>0) ++NumBadHF;
    }//for int i=0;i<DIGI_NUM;++i)

  ProblemsVsLB->Fill(lumiblock,NumBadHB+NumBadHE+NumBadHO+NumBadHF);
  ProblemsVsLB_HB->Fill(lumiblock,NumBadHB);
  ProblemsVsLB_HE->Fill(lumiblock,NumBadHE);
  ProblemsVsLB_HO->Fill(lumiblock,NumBadHO);
  ProblemsVsLB_HF->Fill(lumiblock,NumBadHF);
 

  // Fill data-valid/error plots and capid plots
  for (int i=0;i<4;++i)
    {
      if (hbHists.dverr[i]>0) hbHists.DVerr->Fill(i, hbHists.dverr[i]);
      if (heHists.dverr[i]>0) heHists.DVerr->Fill(i, heHists.dverr[i]);
      if (hoHists.dverr[i]>0) hoHists.DVerr->Fill(i, hoHists.dverr[i]);
      if (hfHists.dverr[i]>0) hfHists.DVerr->Fill(i, hfHists.dverr[i]);
      if (zdcHists.dverr[i]>0) zdcHists.DVerr->Fill(i, zdcHists.dverr[i]);

      if (hbHists.capid[i]>0) hbHists.CapID->Fill(i, hbHists.capid[i]);
      if (heHists.capid[i]>0) heHists.CapID->Fill(i, heHists.capid[i]);
      if (hoHists.capid[i]>0) hoHists.CapID->Fill(i, hoHists.capid[i]);
      if (hfHists.capid[i]>0) hfHists.CapID->Fill(i, hfHists.capid[i]);
      if (zdcHists.capid[i]>0) zdcHists.CapID->Fill(i, zdcHists.capid[i]);

    }

  for (int i=0;i<200;++i)
    {
      if (hbHists.adc[i]>0) hbHists.ADC->Fill(i, hbHists.adc[i]);
      if (heHists.adc[i]>0) heHists.ADC->Fill(i, heHists.adc[i]);
      if (hoHists.adc[i]>0) hoHists.ADC->Fill(i, hoHists.adc[i]);
      if (hfHists.adc[i]>0) hfHists.ADC->Fill(i, hfHists.adc[i]);
      if (zdcHists.adc[i]>0) zdcHists.ADC->Fill(i, zdcHists.adc[i]);
      if (hbHists.adcsum[i]>0) hbHists.ADCsum->Fill(i, hbHists.adcsum[i]);
      if (heHists.adcsum[i]>0) heHists.ADCsum->Fill(i, heHists.adcsum[i]);
      if (hoHists.adcsum[i]>0) hoHists.ADCsum->Fill(i, hoHists.adcsum[i]);
      if (hfHists.adcsum[i]>0) hfHists.ADCsum->Fill(i, hfHists.adcsum[i]);
      if (zdcHists.adcsum[i]>0) zdcHists.ADCsum->Fill(i, zdcHists.adcsum[i]);

    }

  for (int i = 0; i < 15; ++i)
    {
      if (hbHists.fibbcnoff[i]>0) hbHists.fibBCNOff->Fill(i, hbHists.fibbcnoff[i]);
      if (heHists.fibbcnoff[i]>0) heHists.fibBCNOff->Fill(i, heHists.fibbcnoff[i]);
      if (hfHists.fibbcnoff[i]>0) hfHists.fibBCNOff->Fill(i, hfHists.fibbcnoff[i]);
      if (hoHists.fibbcnoff[i]>0) hoHists.fibBCNOff->Fill(i, hoHists.fibbcnoff[i]);
    }

  // Fill plots of bad fraction of digis found
  for (int i=0;i<DIGI_BQ_FRAC_NBINS;++i)
    {
      if (digiBQfrac[i]>0) DigiBQFrac->Fill(i, digiBQfrac[i]);
      if (hbHists.count_BQFrac[i]>0) hbHists.BQFrac->Fill(i, hbHists.count_BQFrac[i]);
      if (heHists.count_BQFrac[i]>0) heHists.BQFrac->Fill(i, heHists.count_BQFrac[i]);
      if (hoHists.count_BQFrac[i]>0) hoHists.BQFrac->Fill(i, hoHists.count_BQFrac[i]);
      if (hfHists.count_BQFrac[i]>0) hfHists.BQFrac->Fill(i, hfHists.count_BQFrac[i]);
      if (zdcHists.count_BQFrac[i]>0) zdcHists.BQFrac->Fill(i, zdcHists.count_BQFrac[i]);

    }//for (int i=0;i<DIGI_BQ_FRAC_NBINS;++i)

  // Fill presample plots
  for (int i=0;i<50;++i)
    {
      if (hbHists.count_presample[i]>0) hbHists.presample->Fill(i, hbHists.count_presample[i]);
      if (heHists.count_presample[i]>0) heHists.presample->Fill(i, heHists.count_presample[i]);
      if (hoHists.count_presample[i]>0) hoHists.presample->Fill(i, hoHists.count_presample[i]);
      if (hfHists.count_presample[i]>0) hfHists.presample->Fill(i, hfHists.count_presample[i]);
      if (zdcHists.count_presample[i]>0) zdcHists.presample->Fill(i, zdcHists.count_presample[i]);
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
      if (zdcHists.count_shape[i]>0) zdcHists.shape->Fill(i, zdcHists.count_shape[i]);
      if (zdcHists.count_shapeThresh[i]>0) zdcHists.shapeThresh->Fill(i, zdcHists.count_shapeThresh[i]);
    }//  for (int i=0;i<10;++i)

  // Fill capID difference plots
  for (int i=0;i<8;++i)
    {
      if (hbHists.capIDdiff[i]>0) hbHists.DigiFirstCapID->Fill(i, hbHists.capIDdiff[i]);
      if (heHists.capIDdiff[i]>0) heHists.DigiFirstCapID->Fill(i, heHists.capIDdiff[i]);
      if (hoHists.capIDdiff[i]>0) hoHists.DigiFirstCapID->Fill(i, hoHists.capIDdiff[i]);
      if (hfHists.capIDdiff[i]>0) hfHists.DigiFirstCapID->Fill(i, hfHists.capIDdiff[i]);
      if (zdcHists.capIDdiff[i]>0) zdcHists.DigiFirstCapID->Fill(i, zdcHists.capIDdiff[i]);

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
  for (int phi=0;phi<72;++phi)
    {
      iPhi=phi+1;
      DigiOccupancyPhi->Fill(iPhi,occupancyPhi[phi]);
      for (int eta=0;eta<83;++eta)
	{
	  iEta=eta-41;
	  if (phi==0)
	    DigiOccupancyEta->Fill(iEta,occupancyEta[eta]);
	  problemsum=0;  
	  valid=false;

	  for (int d=0;d<4;++d)
	    {
	      iDepth=d+1;
	      ProblemCellsByDepth.depth[d]->setBinContent(0,0,ievt_); // underflow bin contains event counter
	      // HB
	      if (validDetId(HcalBarrel, iEta, iPhi, iDepth))
		{
		  valid=true;
		  if (hbHists.check)
		    {
                      int calcEta = CalcEtaBin(HcalBarrel,iEta,iDepth);

		      // This is a dead cell check; no longer needed within digi monitor
		      //if (occupancyEtaPhi[calcEta][phi][d]==0 && digi_checkoccupancy_)
		      //{
		      //  problemdigis[calcEta][phi][d]+=digi_checkNevents_;
		      //}

		      DigiOccupancyByDepth.depth[d]->Fill(iEta, iPhi,
						    occupancyEtaPhi[calcEta][phi][d]);
		      
		      DigiErrorsBadCapID.depth[d]->Fill(iEta, iPhi,
						  badcapID[calcEta][phi][d]);
		      DigiErrorsBadDigiSize.depth[d]->Fill(iEta, iPhi,
						     baddigisize[calcEta][phi][d]);
		      //DigiErrorsBadADCSum.depth[d]->Fill(iEta, iPhi,
		      //			   badADCsum[calcEta][phi][d]);
		      DigiErrorsDVErr.depth[d]->Fill(iEta, iPhi,
					       digierrorsdverr[calcEta][phi][d]);
		      DigiErrorsBadFibBCNOff.depth[d]->Fill(iEta, iPhi,
							    badFibBCNOff[calcEta][phi][d]);
		      problemsum+=problemdigis[calcEta][phi][d];
		      problemvalue=min(ievt_,problemdigis[calcEta][phi][d]);
		      ProblemCellsByDepth.depth[d]->Fill(iEta, iPhi,
						   problemvalue);
		      // Use this for testing purposes only
		      //ProblemCellsByDepth[d]->Fill(iEta, iPhi, ievt_);
		    } // if (hbHists.check)
		} 
	      // HE
	      if (validDetId(HcalEndcap, iEta, iPhi, iDepth))
		{
		  valid=true;
		  if (heHists.check)
		    {
                      int calcEta = CalcEtaBin(HcalEndcap,iEta,iDepth);

		      // This is a dead cell check; no longer needed within digi monitor
		      //if (occupancyEtaPhi[calcEta][phi][d]==0 && digi_checkoccupancy_)
		      //{
		      //  problemdigis[calcEta][phi][d]+=digi_checkNevents_;
		      //}

		      DigiOccupancyByDepth.depth[d]->Fill(iEta, iPhi,
						    occupancyEtaPhi[calcEta][phi][d]);
		      
		      DigiErrorsBadCapID.depth[d]->Fill(iEta, iPhi,
						  badcapID[calcEta][phi][d]);
		      DigiErrorsBadDigiSize.depth[d]->Fill(iEta, iPhi,
						     baddigisize[calcEta][phi][d]);
		      //DigiErrorsBadADCSum.depth[d]->Fill(iEta, iPhi,
		      //badADCsum[calcEta][phi][d]);
		      DigiErrorsDVErr.depth[d]->Fill(iEta, iPhi,
					       digierrorsdverr[calcEta][phi][d]);
		      DigiErrorsBadFibBCNOff.depth[d]->Fill(iEta, iPhi,
							 badFibBCNOff[calcEta][phi][d]);
		      problemsum+=problemdigis[calcEta][phi][d];
		      problemvalue=problemdigis[calcEta][phi][d];
		      ProblemCellsByDepth.depth[d]->Fill(iEta, iPhi,
						   problemvalue);
		    } // if (heHists.check)
		} 
	      // HO
	      if (validDetId(HcalOuter,iEta,iPhi,iDepth))
		{
		  valid=true;
		  if (hoHists.check)
		    {
                      int calcEta = CalcEtaBin(HcalOuter,iEta,iDepth);

		      // This is a dead cell check; no longer needed within digi monitor
		      //if (occupancyEtaPhi[calcEta][phi][d]==0 && digi_checkoccupancy_)
		      //{
		      //  problemdigis[calcEta][phi][d]+=digi_checkNevents_;
		      //}

		      DigiOccupancyByDepth.depth[d]->Fill(iEta, iPhi,
						    occupancyEtaPhi[calcEta][phi][d]);
		      
		      DigiErrorsBadCapID.depth[d]->Fill(iEta, iPhi,
						  badcapID[calcEta][phi][d]);
		      DigiErrorsBadDigiSize.depth[d]->Fill(iEta, iPhi,
						     baddigisize[calcEta][phi][d]);
		      //DigiErrorsBadADCSum.depth[d]->Fill(iEta, iPhi,
		      //				   badADCsum[calcEta][phi][d]);
		      DigiErrorsDVErr.depth[d]->Fill(iEta, iPhi,
					       digierrorsdverr[calcEta][phi][d]);
		      DigiErrorsBadFibBCNOff.depth[d]->Fill(iEta, iPhi,
							 badFibBCNOff[calcEta][phi][d]);
		      problemsum+=problemdigis[calcEta][phi][d];
		      problemvalue=problemdigis[calcEta][phi][d];
		      ProblemCellsByDepth.depth[d]->Fill(iEta, iPhi,
						   problemvalue);
		    } // if (hoHists.check)
		}
	      // HF
	      if (validDetId(HcalForward,iEta,iPhi,iDepth))
		{
		  valid=true;
		  if (hfHists.check)
		    {
                      int calcEta = CalcEtaBin(HcalForward,iEta,iDepth);
                      int zside = iEta/abs(iEta);
		      
		      // This is a dead cell check; no longer needed within digi monitor
		      //if (occupancyEtaPhi[calcEta][phi][d]==0 && digi_checkoccupancy_)
		      //{
		      //  problemdigis[calcEta][phi][d]+=digi_checkNevents_;
		      //}

		      DigiOccupancyByDepth.depth[d]->Fill(iEta+zside, iPhi,
						    occupancyEtaPhi[calcEta][phi][d]);
		      
		      DigiErrorsBadCapID.depth[d]->Fill(iEta+zside, iPhi,
						  badcapID[calcEta][phi][d]);
		      DigiErrorsBadDigiSize.depth[d]->Fill(iEta+zside, iPhi,
						     baddigisize[calcEta][phi][d]);
		      //DigiErrorsBadADCSum.depth[d]->Fill(iEta+zside, iPhi,
		      //				   badADCsum[calcEta][phi][d]);
		      DigiErrorsDVErr.depth[d]->Fill(iEta+zside, iPhi,
					       digierrorsdverr[calcEta][phi][d]);
		      DigiErrorsBadFibBCNOff.depth[d]->Fill(iEta, iPhi,
							 badFibBCNOff[calcEta][phi][d]);
		      problemsum+=problemdigis[calcEta][phi][d];
		      problemvalue=problemdigis[calcEta][phi][d];
		      ProblemCellsByDepth.depth[d]->Fill(iEta+zside, iPhi,
						   problemvalue);
		    } // if (hfHists.check)
		}
	    } // for (int d=0;...)
	  if (valid==true) // only fill overall problem plot if the (eta,phi) value was valid for some depth
	    {
	      //problemvalue=min(1.,problemsum/ievt_);
	      problemsum=min((double)ievt_,problemsum);
	      ProblemCells->Fill(iEta, iPhi,problemsum);
	      ProblemCells->setBinContent(0,0,ievt_);
	    }
	} // for (int phi=0;...)
    } // for (int eta=0;...)

  // Now fill all the unphysical cell values
  FillUnphysicalHEHFBins(ProblemCells);
  FillUnphysicalHEHFBins(ProblemCellsByDepth);
  FillUnphysicalHEHFBins(DigiErrorsBadCapID);
  FillUnphysicalHEHFBins(DigiErrorsDVErr);
  FillUnphysicalHEHFBins(DigiErrorsBadDigiSize);
  //FillUnphysicalHEHFBins(DigiErrorsBadADCSum);
  FillUnphysicalHEHFBins(DigiOccupancyByDepth);

  zeroCounters();
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDigiMonitor DIGI fill_Nevents -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalDigiMonitor::fill_Nevents()

void HcalDigiMonitor::setSubDetectors(bool hb, bool he, bool ho, bool hf, bool zdc)
{
  hbHists.check&=hb;
  heHists.check&=he;
  hoHists.check&=ho;
  hfHists.check&=hf;
  zdcHists.check&=zdc;
  return;
} // void HcalDigiMonitor::setSubDetectors(...)

void HcalDigiMonitor::zeroCounters()
{
  // Set all histogram counters back to 0
  /******** Zero all counters *******/
  
  for (int i=0;i<85;++i)
    {
      occupancyEta[i]=0;
      if (i<72)
        occupancyPhi[i]=0;
      for (int j=0;j<72;++j)
        {
          for (int k=0;k<4;++k)
            {
              problemdigis[i][j][k]=0;
              badcapID[i][j][k]=0;
              baddigisize[i][j][k]=0;
              //badADCsum[i][j][k]=0;
              occupancyEtaPhi[i][j][k]=0;
              digierrorsdverr[i][j][k]=0;
            }
        } // for (int j=0;j<72;++i)
    } // for (int i=0;i<85;++i)

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
	  zdcHists.dverr[i]=0;
	  hbHists.capid[i]=0;
	  heHists.capid[i]=0;
	  hoHists.capid[i]=0;
	  hfHists.capid[i]=0;
	  zdcHists.capid[i]=0;
	}
      if (i<8)
	{
	  hbHists.capIDdiff[i]=0;
	  heHists.capIDdiff[i]=0;
	  hoHists.capIDdiff[i]=0;
	  hfHists.capIDdiff[i]=0;
	  zdcHists.capIDdiff[i]=0;
	}

      if (i<10)
	{
	  hbHists.count_shape[i]=0;
	  heHists.count_shape[i]=0;
	  hoHists.count_shape[i]=0;
	  hfHists.count_shape[i]=0;
	  zdcHists.count_shape[i]=0;
	  hbHists.count_shapeThresh[i]=0;
	  heHists.count_shapeThresh[i]=0;
	  hoHists.count_shapeThresh[i]=0;
	  hfHists.count_shapeThresh[i]=0;
	  zdcHists.count_shapeThresh[i]=0;
	}
      if (i<50)
	{
	  hbHists.count_presample[i]=0;
	  heHists.count_presample[i]=0;
	  hoHists.count_presample[i]=0;
	  hfHists.count_presample[i]=0;
	  zdcHists.count_presample[i]=0;
	  for (int j=0;j<10;++j)
	    {
	      hbHists.tssumplus[i][j]=0;
	      heHists.tssumplus[i][j]=0;
	      hoHists.tssumplus[i][j]=0;
	      hfHists.tssumplus[i][j]=0;
	      zdcHists.tssumplus[i][j]=0;
	      hbHists.tssumminus[i][j]=0;
	      heHists.tssumminus[i][j]=0;
	      hoHists.tssumminus[i][j]=0;
	      hfHists.tssumminus[i][j]=0;
	      zdcHists.tssumminus[i][j]=0;
	    }
	}
      if (i<200)
	{
	  hbHists.adc[i]=0;
	  heHists.adc[i]=0;
	  hoHists.adc[i]=0;
	  hfHists.adc[i]=0;
	  zdcHists.adc[i]=0;
	  hbHists.adcsum[i]=0;
	  heHists.adcsum[i]=0;
	  hoHists.adcsum[i]=0;
	  hfHists.adcsum[i]=0;
	  zdcHists.adcsum[i]=0;
	}
      if (i<DIGI_SUBDET_NUM)
	{
	  hbHists.count_BQ[i]=0;
	  heHists.count_BQ[i]=0;
	  hoHists.count_BQ[i]=0;
	  hfHists.count_BQ[i]=0;
	  zdcHists.count_BQ[i]=0;
	}
      if (i<DIGI_BQ_FRAC_NBINS)
	{
	  hbHists.count_BQFrac[i]=0;
	  heHists.count_BQFrac[i]=0;
	  hoHists.count_BQFrac[i]=0;
	  hfHists.count_BQFrac[i]=0;
	  zdcHists.count_BQFrac[i]=0;
	}
    } // for (int i=0;i<DIGI_NUM;++i)


  return;
}

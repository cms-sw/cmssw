#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"

#define OUT if(fverbosity_)cout
#define TIME_MIN -250
#define TIME_MAX 250

using namespace std;

HcalRecHitMonitor::HcalRecHitMonitor()
{
  ievt_=0;
} //constructor

HcalRecHitMonitor::~HcalRecHitMonitor()
{
} //destructor


/* ------------------------------------ */ 

void HcalRecHitMonitor::setup(const edm::ParameterSet& ps,
				DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (fVerbosity>0)
    cout <<"<HcalRecHitMonitor::setup>  Setting up histograms"<<endl;

  baseFolder_ = rootFolder_+"RecHitMonitor_Hcal";

  // Assume subdetectors not present until shown otherwise
  HBpresent_=false;
  HEpresent_=false;
  HOpresent_=false;
  HFpresent_=false;

  // Rec Hit Monitor - specific cfg variables

  if (fVerbosity>1)
    cout <<"<HcalRecHitMonitor::setup>  Getting variable values from cfg files"<<endl;
  
  // rechit_makeDiagnostics_ will take on base task value unless otherwise specified
  rechit_makeDiagnostics_ = ps.getUntrackedParameter<bool>("RecHitMonitor_makeDiagnosticPlots",makeDiagnostics);
  
  // Set checkNevents values
  rechit_checkNevents_ = ps.getUntrackedParameter<int>("RecHitMonitor_checkNevents",checkNevents_);
  rechit_minErrorFlag_ = ps.getUntrackedParameter<double>("RecHitMonitor_minErrorFlag",0.0);

  energyThreshold_       = ps.getUntrackedParameter<double>("RecHitMonitor_energyThreshold",                  0);
  HBenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HB_energyThreshold",energyThreshold_);
  HEenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HE_energyThreshold",energyThreshold_);
  HOenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HO_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HF_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HF_energyThreshold",            -999);
  ZDCenergyThreshold_    = ps.getUntrackedParameter<double>("RecHitMonitor_ZDC_energyThreshold",           -999);

  // Set initial event # to 0
  ievt_=0;

  // zero all counters

  zeroCounters();

  // Set up histograms
  if (m_dbe)
    {
      if (fVerbosity>1)
	cout <<"<HcalRecHitMonitor::setup>  Setting up histograms"<<endl;

      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("RecHit Event Number");
      meEVT_->Fill(ievt_);

      // Create problem cell plots
      // Overall plot gets an initial " " in its name
      ProblemRecHits=m_dbe->book2D(" ProblemRecHits",
                                     " Problem Rec Hit Rate for all HCAL",
                                     etaBins_,etaMin_,etaMax_,
                                     phiBins_,phiMin_,phiMax_);
      ProblemRecHits->setAxisTitle("i#eta",1);
      ProblemRecHits->setAxisTitle("i#phi",2);
      
      // Overall Problem plot appears in main directory; plots by depth appear \in subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/problem_rechits");
      setupDepthHists2D(ProblemRecHitsByDepth, " Problem RecHit Rate","");

      m_dbe->setCurrentFolder(baseFolder_+"/rechit_info");
      setupDepthHists2D(EnergyByDepth,"Rec Hit Average Energy","GeV");
      setupDepthHists2D(OccupancyByDepth,"Rec Hit Occupancy","");
      setupDepthHists2D(TimeByDepth,"Rec Hit Average Time","nS");

      m_dbe->setCurrentFolder(baseFolder_+"/rechit_energy/sumplots");
      setupDepthHists2D(SumEnergyByDepth,"Rec Hit Summed Energy","GeV");
      setupDepthHists2D(SumTimeByDepth,"Rec Hit Summed Time","nS");
      
      m_dbe->setCurrentFolder(baseFolder_+"/rechit_info_threshold");
      setupDepthHists2D(EnergyThreshByDepth,"Above Threshold Rec Hit Average Energy","GeV");
      setupDepthHists2D(OccupancyThreshByDepth,"Above Threshold Rec Hit Occupancy","");
      setupDepthHists2D(TimeThreshByDepth,"Above Threshold Rec Hit Average Time","nS");

      m_dbe->setCurrentFolder(baseFolder_+"/rechit_info_threshold/sumplots");
      setupDepthHists2D(SumEnergyThreshByDepth,"Above Threshold Rec Hit Summed Energy","GeV");
      setupDepthHists2D(SumTimeThreshByDepth,"Above Threshold Rec Hit Summed Time","nS");

      if (rechit_makeDiagnostics_)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/hb");
	  h_HBEnergy=m_dbe->book1D("HB_energy","HB Rec Hit Energy",200,-5,5);
	  h_HBThreshEnergy=m_dbe->book1D("HB_energy_thresh", "HB Rec Hit Energy Above Threshold",200,-5,5);
	  h_HBTotalEnergy=m_dbe->book1D("HB_total_energy","HB Rec Hit Total Energy",200,-200,200);
	  h_HBThreshTotalEnergy=m_dbe->book1D("HB_total_energy_thresh", "HB Rec Hit Total Energy Above Threshold",200,-200,200);
	  h_HBTime=m_dbe->book1D("HB_time","HB Rec Hit Time",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HBThreshTime=m_dbe->book1D("HB_time_thresh", "HB Rec Hit Time Above Threshold",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HBOccupancy=m_dbe->book1D("HB_occupancy","HB Rec Hit Occupancy",2593,-0.5,2592.5);
	  h_HBThreshOccupancy=m_dbe->book1D("HB_occupancy_thresh","HB Rec Hit Occupancy Above Threshold",2593,-0.5,2592.5);
	  // hb

	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/he");	
	  h_HEEnergy=m_dbe->book1D("HE_energy","HE Rec Hit Energy",200,-5,5);
	  h_HEThreshEnergy=m_dbe->book1D("HE_energy_thresh", "HE Rec Hit Energy Above Threshold",200,-5,5);
	  h_HETotalEnergy=m_dbe->book1D("HE_total_energy","HE Rec Hit Total Energy",200,-200,200);
	  h_HEThreshTotalEnergy=m_dbe->book1D("HE_total_energy_thresh", "HE Rec Hit Total Energy Above Threshold",200,-200,200);
	  h_HETime=m_dbe->book1D("HE_time","HE Rec Hit Time",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HEThreshTime=m_dbe->book1D("HE_time_thresh", "HE Rec Hit Time Above Threshold",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HEOccupancy=m_dbe->book1D("HE_occupancy","HE Rec Hit Occupancy",2593,-0.5,2592.5);
	  h_HEThreshOccupancy=m_dbe->book1D("HE_occupancy_thresh","HE Rec Hit Occupancy Above Threshold",2593,-0.5,2592.5);
	  // he

	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/ho");	
	  h_HOEnergy=m_dbe->book1D("HO_energy","HO Rec Hit Energy",200,-5,5);
	  h_HOThreshEnergy=m_dbe->book1D("HO_energy_thresh", "HO Rec Hit Energy Above Threshold",200,-5,5);
	  h_HOTotalEnergy=m_dbe->book1D("HO_total_energy","HO Rec Hit Total Energy",200,-200,200);
	  h_HOThreshTotalEnergy=m_dbe->book1D("HO_total_energy_thresh", "HO Rec Hit Total Energy Above Threshold",200,-200,200);
	  h_HOTime=m_dbe->book1D("HO_time","HO Rec Hit Time",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HOThreshTime=m_dbe->book1D("HO_time_thresh", "HO Rec Hit Time Above Threshold",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HOOccupancy=m_dbe->book1D("HO_occupancy","HO Rec Hit Occupancy",2161,-0.5,2160.5);
	  h_HOThreshOccupancy=m_dbe->book1D("HO_occupancy_thresh","HO Rec Hit Occupancy Above Threshold",2161,-0.5,2160.5);
	  // ho

	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/hf");	
	  h_HFEnergy=m_dbe->book1D("HF_energy","HF Rec Hit Energy",200,-5,5);
	  h_HFThreshEnergy=m_dbe->book1D("HF_energy_thresh", "HF Rec Hit Energy Above Threshold",200,-5,5);
	  h_HFTotalEnergy=m_dbe->book1D("HF_total_energy","HF Rec Hit Total Energy",200,-200,200);
	  h_HFThreshTotalEnergy=m_dbe->book1D("HF_total_energy_thresh", "HF Rec Hit Total Energy Above Threshold",200,-200,200);
	  h_HFTime=m_dbe->book1D("HF_time","HF Rec Hit Time",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HFThreshTime=m_dbe->book1D("HF_time_thresh", "HF Rec Hit Time Above Threshold",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HFOccupancy=m_dbe->book1D("HF_occupancy","HF Rec Hit Occupancy",1729,-0.5,1728.5);
	  h_HFThreshOccupancy=m_dbe->book1D("HF_occupancy_thresh","HF Rec Hit Occupancy Above Threshold",1729,-0.5,1728.5);
	  // hf
	  
	} // if (rechit_Diagnostics_)
    } // if (m_dbe)

  return;
  
} //void HcalRecHitMonitor::setup(...)


/* --------------------------- */

void HcalRecHitMonitor::reset(){}  // reset function is empty for now


/* ------------------------- */

void HcalRecHitMonitor::done()
{
  // Can eventually dump bad rec hit info here, when we decide on a definition for bad rec hits
  return;
  
} // void HcalRecHitMonitor::done()



/* --------------------------------- */

void HcalRecHitMonitor::clearME()
{
  // I don't think this function gets cleared any more.  
  // And need to add code to clear out subfolders as well?
  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
    }
  return;
} // void HcalRecHitMonitor::clearME()

/* -------------------------------- */


void HcalRecHitMonitor::processEvent(const HBHERecHitCollection& hbHits,
				     const HORecHitCollection& hoHits,
				     const HFRecHitCollection& hfHits
				     //const ZDCRecHitCollection& zdcHits,
				     )
{

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  ++ievt_;
  if (m_dbe) meEVT_->Fill(ievt_);
  
  if (hoHits.size()>0) HOpresent_=true;
  if (hfHits.size()>0) HFpresent_=true;

  if (fVerbosity>1) cout <<"<HcalRecHitMonitor::processEvent> Processing event..."<<endl;

  processEvent_rechit(hbHits, hoHits, hfHits);
  
  // Fill problem cells
  if (ievt_%rechit_checkNevents_ ==0)
    {
      fillNevents();
    }

  return;
} // void HcalRecHitMonitor::processEvent(...)


/* --------------------------------------- */


void HcalRecHitMonitor::fillRecHitHistosAtEndRun()
{
  // Fill histograms one last time at endRun call
  
  /*
    I'm not sure I like this feature.  Suppose checkNevents=500, and the end run occurs at 501?
    Then the occupancy plot would create errors for whichever digis were not found in a single event.
    That's not desired behavior.
    We could just exclude the occupancy test from running here, but I'm not sure that's the best solution either.
    For now (28 Oct. 2008), just disable this functionality.  We'll come back to it if necessary.
  */
  return;

  fillNevents();
}

/* --------------------------------------- */


void HcalRecHitMonitor::processEvent_rechit( const HBHERecHitCollection& hbheHits,
					     const HORecHitCollection& hoHits,
					     const HFRecHitCollection& hfHits)
  
{
  // Gather rechit info
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>1) cout <<"<HcalRecHitMonitor::processEvent_rechitenergy> Processing rechits..."<<endl;
  
  // loop over HBHE
  
  int     hbocc=0;
  int     heocc=0;
  int     hboccthresh=0;
  int     heoccthresh=0;
  double  hbenergy=0;
  double  heenergy=0;
  double  hbenergythresh=0;
  double  heenergythresh=0;
  int HEbindepth=0;

  for (HBHERecHitCollection::const_iterator HBHEiter=hbheHits.begin(); HBHEiter!=hbheHits.end(); ++HBHEiter) 
    { // loop over all hits
      float en = HBHEiter->energy();
      float ti = HBHEiter->time();

      HcalDetId id(HBHEiter->detid().rawId());
      int ieta = id.ieta();
      int iphi = id.iphi();
      int depth = id.depth();
      if (id.subdet()==HcalBarrel)
	{
	  HBpresent_=true;
	  if (!checkHB_) continue;
	
	  ++occupancy_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1];
	  energy_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=en;
	  time_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=ti;
	  if (en>=HBenergyThreshold_)
	    {
	      ++occupancy_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1];
	      energy_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=en;
	      time_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=ti;
	    }
	  if (rechit_makeDiagnostics_)
	    {
	      ++hbocc;
	      hbenergy+=en;
	      if (ti<TIME_MIN || ti>TIME_MAX)
		h_HBTime->Fill(ti);
	      else
		++HBtime_[int(ti-TIME_MIN)];
	      if (en<-5 || en>-5)
		h_HBEnergy->Fill(en);
	      else
		++HBenergy_[20*int(en+5)];
	      if (en>=HBenergyThreshold_)
		{
		  ++hboccthresh;
		  hbenergythresh+=en;
		  if (ti<TIME_MIN || ti>TIME_MAX)
		    h_HBThreshTime->Fill(ti);
		  else
		    ++HBtime_thresh_[int(ti-TIME_MIN)];
		  if (en<-5 || en>-5)
		    h_HBThreshEnergy->Fill(en);
		  else
		    ++HBenergy_thresh_[20*int(en+5)];
		} // if (en>=HBenergyThreshold_)
	    } // if (rechit_makeDiagnostics_)
	} // if (id.subdet()==HcalBarrel)

      else if (id.subdet()==HcalEndcap)
	{
	  HEpresent_=true;
	  if (!checkHE_) continue;
	  HEbindepth=depth-1;
	  if (depth<=2) HEbindepth+=4; 
	  
	  ++occupancy_[ieta+(int)((etaBins_-2)/2)][iphi-1][HEbindepth];
	  energy_[ieta+(int)((etaBins_-2)/2)][iphi-1][HEbindepth]+=en;
	  time_[ieta+(int)((etaBins_-2)/2)][iphi-1][HEbindepth]+=ti;
	  if (en>=HEenergyThreshold_)
	    {
	      ++occupancy_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][HEbindepth];
	      energy_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][HEbindepth]+=en;
	      time_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][HEbindepth]+=ti;
	    }
	  if (rechit_makeDiagnostics_)
	    {
	      ++heocc;
	      heenergy+=en;
	      if (ti<-100 || ti>200)
		h_HETime->Fill(ti);
	      else
		++HEtime_[int(ti+100)];
	      if (en<-5 || en>-5)
		h_HEEnergy->Fill(en);
	      else
		++HEenergy_[20*int(en+5)];
	      if (en>=HEenergyThreshold_)
		{
		  ++heoccthresh;
		  heenergythresh+=en;
		  if (ti<-100 || ti>200)
		    h_HEThreshTime->Fill(ti);
		  else
		    ++HEtime_thresh_[int(ti+100)];
		  if (en<-5 || en>-5)
		    h_HEThreshEnergy->Fill(en);
		  else
		    ++HEenergy_thresh_[20*int(en+5)];
		} // if (en>=HEenergyThreshold_)
	    } // if (rechit_makeDiagnostics_)


	} // else if (id.subdet()==HcalEndcap)
     
    } //for (HBHERecHitCollection::const_iterator HBHEiter=...)

  if (rechit_makeDiagnostics_)
    {
      ++HB_occupancy_[hbocc];
      ++HE_occupancy_[heocc];
      ++HB_occupancy_thresh_[hboccthresh];
      ++HE_occupancy_thresh_[heoccthresh];
      h_HBTotalEnergy->Fill(hbenergy);
      h_HETotalEnergy->Fill(heenergy);
      h_HBThreshTotalEnergy->Fill(hbenergythresh);
      h_HEThreshTotalEnergy->Fill(heenergythresh);
    }

  // loop over HO

  if (checkHO_)
   {
     int hoocc=0;
     int hooccthresh=0;
     double hoenergy=0;
     double hoenergythresh=0;
     for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) 
       { // loop over all hits
	 float en = HOiter->energy();
	 float ti = HOiter->time();

	 HcalDetId id(HOiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();

	 ++occupancy_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1];
	 energy_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=en;
	 time_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=ti;

	 if (en>=HOenergyThreshold_)
	   {
	     ++occupancy_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1];
	     energy_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=en;
	     time_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=ti;
	   }
	 if (rechit_makeDiagnostics_)
	   {
	     ++hoocc;
	     hoenergy+=en;
	     if (ti<-100 || ti>200)
	       h_HOTime->Fill(ti);
	     else
	       ++HOtime_[int(ti+100)];
	     if (en<-5 || en>-5)
	       h_HOEnergy->Fill(en);
	     else
	       ++HOenergy_[20*int(en+5)];
	     if (en>=HOenergyThreshold_)
	       {
		 ++hooccthresh;
		 hoenergythresh+=en;
		 if (ti<-100 || ti>200)
		   h_HOThreshTime->Fill(ti);
		 else
		   ++HOtime_thresh_[int(ti+100)];
		 if (en<-5 || en>-5)
		   h_HOThreshEnergy->Fill(en);
		 else
		   ++HOenergy_thresh_[20*int(en+5)];
	       } // if (en>=HOenergyThreshold_)
	   } // if (rechit_makeDiagnostics_)
       } // loop over all HO hits
     if (rechit_makeDiagnostics_)
       {
	 ++HO_occupancy_[hoocc];
	 ++HO_occupancy_thresh_[hooccthresh];
	 h_HOTotalEnergy->Fill(hoenergy);
	 h_HOThreshTotalEnergy->Fill(hoenergythresh);
       }

   } // if (checkHO_)
 
  // loop over HF
  if (checkHF_)
   {
     int hfocc=0;
     int hfoccthresh=0;
     double hfenergy=0;
     double hfenergythresh=0;
     for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter) 
       { // loop over all hits
	 float en = HFiter->energy();
	 float ti = HFiter->time();

	 HcalDetId id(HFiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();

	 ++occupancy_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1];
	 energy_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=en;
	 time_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=ti;

	 if (en>=HFenergyThreshold_)
	   {
	     ++occupancy_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1];
	     energy_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=en;
	     time_thresh_[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]+=ti;
	   }
	 if (rechit_makeDiagnostics_)
	   {
	     ++hfocc;
	     hfenergy+=en;
	     if (ti<-100 || ti>200)
	       h_HFTime->Fill(ti);
	     else
	       ++HFtime_[int(ti+100)];
	     if (en<-5 || en>-5)
	       h_HFEnergy->Fill(en);
	     else
	       ++HFenergy_[20*int(en+5)];
	     if (en>=HFenergyThreshold_)
	       {
		 ++hfoccthresh;
		 hfenergythresh+=en;
		 if (ti<-100 || ti>200)
		   h_HFThreshTime->Fill(ti);
		 else
		   ++HFtime_thresh_[int(ti+100)];
		 if (en<-5 || en>-5)
		   h_HFThreshEnergy->Fill(en);
		 else
		   ++HFenergy_thresh_[20*int(en+5)];
	       } // if (en>=HFenergyThreshold_)
	   } // if (rechit_makeDiagnostics_)
       } // loop over all HF hits
     if (rechit_makeDiagnostics_)
       {
	 ++HF_occupancy_[hfocc];
	 ++HF_occupancy_thresh_[hfoccthresh];
	 h_HFTotalEnergy->Fill(hfenergy);
	 h_HFThreshTotalEnergy->Fill(hfenergythresh);
       }
   } // if (checkHF_)
 
 if (showTiming)
   {
     cpu_timer.stop();  cout <<"TIMER:: HcalRecHitMonitor PROCESSEVENT_RECHITENERGY -> "<<cpu_timer.cpuTime()<<endl;
   }
 return;
} // void HcalRecHitMonitor::processEvent_rechitenergy

/* --------------------------------------- */




void HcalRecHitMonitor::fillNevents(void)
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  int ieta=0;
  int iphi=0;
  int mydepth=0;

  ProblemRecHits->setBinContent(0,0,ievt_);
  for (int i=0;i<6;++i)
    ProblemRecHitsByDepth[i]->setBinContent(0,0,ievt_);

  // Fill Occupancy & Average Energy,Time plots
  if (ievt_>0)
    {
      for (int eta=0;eta<(etaBins_-2);++eta)
	{
	  ieta=eta-int((etaBins_-2)/2);
	  for (int phi=0;phi<72;++phi)
	    {
	      iphi=phi+1;
	      for (int depth=0;depth<6;++depth)
		{
		  mydepth=depth;
		  OccupancyByDepth[mydepth]->Fill(ieta,iphi,occupancy_[eta][phi][mydepth]);
		  OccupancyThreshByDepth[mydepth]->Fill(ieta,iphi,occupancy_thresh_[eta][phi][mydepth]);
		  OccupancyByDepth[mydepth]->setBinContent(0,0,ievt_);
		  OccupancyThreshByDepth[mydepth]->setBinContent(0,0,ievt_);
		  SumEnergyByDepth[mydepth]->Fill(ieta,iphi,energy_[eta][phi][mydepth]);
		  SumEnergyThreshByDepth[mydepth]->Fill(ieta,iphi,energy_thresh_[eta][phi][mydepth]);
		  SumTimeByDepth[mydepth]->Fill(ieta,iphi,time_[eta][phi][mydepth]);
		  SumTimeThreshByDepth[mydepth]->Fill(ieta,iphi,time_thresh_[eta][phi][mydepth]);

		  // This won't work with offline DQM, since tasks get split
		  if (occupancy_[eta][phi][mydepth]>0)
		    {
		      EnergyByDepth[mydepth]->setBinContent(eta+2, phi+2, energy_[eta][phi][mydepth]/occupancy_[eta][phi][mydepth]);
		      TimeByDepth[mydepth]->setBinContent(eta+2, phi+2, time_[eta][phi][mydepth]/occupancy_[eta][phi][mydepth]);
		    }
		  if (occupancy_thresh_[eta][phi][mydepth]>0)
		    {
		      EnergyThreshByDepth[mydepth]->setBinContent(eta+2, phi+2, energy_thresh_[eta][phi][mydepth]/occupancy_thresh_[eta][phi][mydepth]);
		      TimeThreshByDepth[mydepth]->setBinContent(eta+2, phi+2, time_thresh_[eta][phi][mydepth]/occupancy_thresh_[eta][phi][mydepth]);
		    }

		} // for (int depth=0;depth<6;++depth)
	    } // for (int phi=0;phi<72;++phi)
	} // for (int eta=0;eta<(etaBins_-2);++eta)


      FillUnphysicalHEHFBins(OccupancyByDepth);
      FillUnphysicalHEHFBins(OccupancyThreshByDepth);
      FillUnphysicalHEHFBins(EnergyByDepth);
      FillUnphysicalHEHFBins(EnergyThreshByDepth);
      FillUnphysicalHEHFBins(TimeByDepth);
      FillUnphysicalHEHFBins(TimeThreshByDepth);
      FillUnphysicalHEHFBins(SumEnergyByDepth);
      FillUnphysicalHEHFBins(SumEnergyThreshByDepth);
      FillUnphysicalHEHFBins(SumTimeByDepth);
      FillUnphysicalHEHFBins(SumTimeThreshByDepth);
    } // if (ievt_>0)

  // Fill subdet plots

  for (int i=0;i<200;++i)
    {
      if (HBenergy_[i]!=0) 
	{
	  h_HBEnergy->Fill(i,HBenergy_[i]);
	}
      if (HBenergy_thresh_[i]!=0) 
	{
	  h_HBThreshEnergy->Fill(i,HBenergy_thresh_[i]);
	}
      if (HEenergy_[i]!=0) 
	{
	  h_HEEnergy->Fill(i,HEenergy_[i]);
	}
      if (HEenergy_thresh_[i]!=0) 
	{
	  h_HEThreshEnergy->Fill(i,HEenergy_thresh_[i]);
	}
      if (HOenergy_[i]!=0) 
	{
	  h_HOEnergy->Fill(i,HOenergy_[i]);
	}
      if (HOenergy_thresh_[i]!=0) 
	{
	  h_HOThreshEnergy->Fill(i,HOenergy_thresh_[i]);
	}
      if (HFenergy_[i]!=0) 
	{
	  h_HFEnergy->Fill(i,HFenergy_[i]);
	}
      if (HFenergy_thresh_[i]!=0) 
	{
	  h_HFThreshEnergy->Fill(i,HFenergy_thresh_[i]);
	}
    }// for (int i=0;i<200;++i) // Jeff

  for (int i=0;i<(TIME_MAX-TIME_MIN);++i)
    {
      if (HBtime_[i]!=0)
	{
	  h_HBTime->Fill(i,HBtime_[i]);
	}
      if (HBtime_thresh_[i]!=0)
	{
	  h_HBThreshTime->Fill(i,HBtime_thresh_[i]);
	}
      if (HEtime_[i]!=0)
	{

	  h_HETime->Fill(i,HEtime_[i]);
	}
      if (HEtime_thresh_[i]!=0)
	{
	  h_HEThreshTime->Fill(i,HEtime_thresh_[i]);
	}
      if (HOtime_[i]!=0)
	{
	  h_HOTime->Fill(i,HOtime_[i]);
	}
      if (HOtime_thresh_[i]!=0)
	{
	  h_HOThreshTime->Fill(i,HOtime_thresh_[i]);
	}
      if (HFtime_[i]!=0)
	{
	  h_HFTime->Fill(i,HFtime_[i]);
	}
      if (HFtime_thresh_[i]!=0)
	{
	  h_HFThreshTime->Fill(i,HFtime_thresh_[i]);
	}
    } // for (int  i=0;i<(TIME_MAX-TIME_MIN);++i)

  for (int i=0;i<2593;++i)
    {
      if (HB_occupancy_[i]>0)
	{
	  h_HBOccupancy->Fill(i,HB_occupancy_[i]);
	}
      if (HB_occupancy_thresh_[i]>0)
	{
	  h_HBThreshOccupancy->Fill(i,HB_occupancy_thresh_[i]);
	}
      if (HE_occupancy_[i]>0)
	{
	  h_HEOccupancy->Fill(i,HE_occupancy_[i]);
	}
      if (HE_occupancy_thresh_[i]>0)
	{
	  h_HEThreshOccupancy->Fill(i,HE_occupancy_thresh_[i]);
	}
    }//for (int i=0;i<2593;++i)

  for (int i=0;i<2161;++i)
    {
      if (HO_occupancy_[i]>0)
	{
	  h_HOOccupancy->Fill(i,HO_occupancy_[i]);
	}
      if (HO_occupancy_thresh_[i]>0)
	{
	  h_HOThreshOccupancy->Fill(i,HO_occupancy_thresh_[i]);
	}
    }//  for (int i=0;i<2161;++i)

  for (int i=0;i<1729;++i)
    {
      if (HF_occupancy_[i]>0)
	{
	  h_HFOccupancy->Fill(i,HF_occupancy_[i]);
	}
      if (HF_occupancy_thresh_[i]>0)
	{
	  h_HFThreshOccupancy->Fill(i,HF_occupancy_thresh_[i]);
	}
    }//  for (int i=0;i<2161;++i)

  zeroCounters();

  if (fVerbosity>0)
    cout <<"<HcalRecHitMonitor::fillNevents_problemCells> FILLED REC HIT CELL PLOTS"<<endl;
    
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalRecHitMonitor FILLNEVENTS -> "<<cpu_timer.cpuTime()<<endl;
    }

} // void HcalRecHitMonitor::fillNevents(void)


void HcalRecHitMonitor::zeroCounters(void)
{
  // Set all histogram counters back to zero

  // TH2F counters
  for (int i=0;i<ETABINS;++i)
    {
      for (int j=0;j<PHIBINS;++j)
	{
	  for (int k=0;k<6;++k)
	    {
	      occupancy_[i][j][k]=0;
	      occupancy_thresh_[i][j][k]=0;
	      energy_[i][j][k]=0;
	      energy_thresh_[i][j][k]=0;
	      time_[i][j][k]=0;
	      time_thresh_[i][j][k]=0;
	    }
	} // for (int j=0;j<PHIBINS;++j)
    } // for (int i=0;i<ETABINS;++i)

  // TH1F counters
  
  // energy
  for (int i=0;i<200;++i)
    {
      HBenergy_[i]=0;
      HBenergy_thresh_[i]=0;
      HEenergy_[i]=0;
      HEenergy_thresh_[i]=0;
      HOenergy_[i]=0;
      HOenergy_thresh_[i]=0;
      HFenergy_[i]=0;
      HFenergy_thresh_[i]=0;
      HFenergyLong_[i]=0;
      HFenergyLong_thresh_[i]=0;
      HFenergyShort_[i]=0;
      HFenergyShort_thresh_[i]=0;
    }

  // time
  for (int i=0;i<(TIME_MAX-TIME_MIN);++i)
    {
      HBtime_[i]=0;
      HBtime_thresh_[i]=0;
      HEtime_[i]=0;
      HEtime_thresh_[i]=0;
      HOtime_[i]=0;
      HOtime_thresh_[i]=0;
      HFtime_[i]=0;
      HFtime_thresh_[i]=0;
      HFtimeLong_[i]=0;
      HFtimeLong_thresh_[i]=0;
      HFtimeShort_[i]=0;
      HFtimeShort_thresh_[i]=0;
    }

  // occupancy
  for (int i=0;i<2593;++i)
    {
      HB_occupancy_[i]=0;
      HE_occupancy_[i]=0;
      HB_occupancy_thresh_[i]=0;
      HE_occupancy_thresh_[i]=0;
      if (i<=2160)
	{
	  HO_occupancy_[i]=0;
	  HO_occupancy_thresh_[i]=0;
	}
      if (i<=1728)
	{
	  HF_occupancy_[i]=0;
	  HF_occupancy_thresh_[i]=0;
	}
      if (i<=864)
	{
	  HFlong_occupancy_[i] =0;
	  HFshort_occupancy_[i]=0;
	  HFlong_occupancy_thresh_[i] =0;
	  HFshort_occupancy_thresh_[i]=0;
	}
    } // for (int i=0;i<2592;++i)

  return;
} //void HcalRecHitMonitor::zeroCounters(void)

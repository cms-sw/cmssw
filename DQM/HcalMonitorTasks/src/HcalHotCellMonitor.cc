#include "DQM/HcalMonitorTasks/interface/HcalHotCellMonitor.h"

#define OUT if(fverbosity_)std::cout
#define BITSHIFT 6

using namespace std;

HcalHotCellMonitor::HcalHotCellMonitor()
{
  // Default initialization
  hotmon_makeDiagnostics_   = false;
  hotmon_test_energy_       = false;
  hotmon_test_neighbor_     = false;
  hotmon_test_persistent_   = false;
  showTiming                = false;
  fVerbosity                = 0;
} //constructor

HcalHotCellMonitor::~HcalHotCellMonitor()
{
} //destructor


/* ------------------------------------ */ 

void HcalHotCellMonitor::setup(const edm::ParameterSet& ps,
				DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  baseFolder_ = rootFolder_+"HotCellMonitor_Hcal";

  if (fVerbosity>0)
    std::cout <<"<HcalHotCellMonitor::setup>  Setting up histograms"<<std::endl;
  
  // Hot Cell Monitor - specific cfg variables

  if (fVerbosity>1)
    std::cout <<"<HcalHotCellMonitor::setup>  Getting variable values from cfg files"<<std::endl;

  // hotmon_makeDiagnostics_ will take on base task value unless otherwise specified
  hotmon_makeDiagnostics_ = ps.getUntrackedParameter<bool>("HotCellMonitor_makeDiagnosticPlots",makeDiagnostics);
  
  // Set checkNevents values
  hotmon_checkNevents_ = ps.getUntrackedParameter<int>("HotCellMonitor_checkNevents",checkNevents_);
   
  // Set which hot cell checks will be performed
  hotmon_test_persistent_         = ps.getUntrackedParameter<bool>("HotCellMonitor_test_persistent",true);
  hotmon_test_neighbor_           = ps.getUntrackedParameter<bool>("HotCellMonitor_test_neighbor",true); // deprecated right now
  hotmon_test_energy_             = ps.getUntrackedParameter<bool>("HotCellMonitor_test_energy",true);

  hotmon_minErrorFlag_ = ps.getUntrackedParameter<double>("HotCellMonitor_minErrorFlag",0.0);

  // rechit energy test -- cell must be above threshold value for a number of consecutive events to be considered hot
  energyThreshold_       = ps.getUntrackedParameter<double>("HotCellMonitor_energyThreshold",   1);
  HBenergyThreshold_     = ps.getUntrackedParameter<double>("HotCellMonitor_HB_energyThreshold",energyThreshold_);
  HEenergyThreshold_     = ps.getUntrackedParameter<double>("HotCellMonitor_HE_energyThreshold",energyThreshold_);
  HOenergyThreshold_     = ps.getUntrackedParameter<double>("HotCellMonitor_HO_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("HotCellMonitor_HF_energyThreshold",energyThreshold_);
  ZDCenergyThreshold_    = ps.getUntrackedParameter<double>("HotCellMonitor_HF_energyThreshold",-999);

  // rechit event-by-event energy test -- cell must be above threshold to be considered hot
  persistentThreshold_       = ps.getUntrackedParameter<double>("HotCellMonitor_persistentThreshold",   1);
  HBpersistentThreshold_     = ps.getUntrackedParameter<double>("HotCellMonitor_HB_persistentThreshold",persistentThreshold_);
  HEpersistentThreshold_     = ps.getUntrackedParameter<double>("HotCellMonitor_HE_persistentThreshold",persistentThreshold_);
  HOpersistentThreshold_     = ps.getUntrackedParameter<double>("HotCellMonitor_HO_persistentThreshold",persistentThreshold_);
  HFpersistentThreshold_     = ps.getUntrackedParameter<double>("HotCellMonitor_HF_persistentThreshold",persistentThreshold_);
  ZDCpersistentThreshold_    = ps.getUntrackedParameter<double>("HotCellMonitor_HF_persistentThreshold",-999);

  SiPMscale_                 = ps.getUntrackedParameter<double>("HotCellMonitor_HO_SiPMscalefactor",4.); // default scale factor of 4
// neighboring-cell tests
  defaultNeighborParams_.DeltaIphi = ps.getUntrackedParameter<int>("HotCellMonitor_neighbor_deltaIphi", 1);
  defaultNeighborParams_.DeltaIeta = ps.getUntrackedParameter<int>("HotCellMonitor_neighbor_deltaIeta", 1);
  defaultNeighborParams_.DeltaDepth = ps.getUntrackedParameter<int>("HotCellMonitor_neighbor_deltaDepth", 0);
  defaultNeighborParams_.minCellEnergy = ps.getUntrackedParameter<double>("HotCellMonitor_neighbor_minCellEnergy",3.);
  defaultNeighborParams_.minNeighborEnergy = ps.getUntrackedParameter<double>("HotCellMonitor_neighbor_minNeighborEnergy",0.);
  defaultNeighborParams_.maxEnergy = ps.getUntrackedParameter<double>("HotCellMonitor_neighbor_maxEnergy",50);
  defaultNeighborParams_.HotEnergyFrac = ps.getUntrackedParameter<double>("HotCellMonitor_neighbor_HotEnergyFrac",0.01);

  setupNeighborParams(ps,HBNeighborParams_ ,"HB");
  setupNeighborParams(ps,HENeighborParams_ ,"HE");
  setupNeighborParams(ps,HONeighborParams_ ,"HO");
  setupNeighborParams(ps,HFNeighborParams_ ,"HF");
  setupNeighborParams(ps,ZDCNeighborParams_,"ZDC");
  HFNeighborParams_.DeltaIphi*=2; // HF cell segmentation is 10 degrees, not 5 (mostly).  Need to multiply by 2 to convert from cell range to degree format

  zeroCounters();


  // Set up histograms
  if (m_dbe)
    {
      if (fVerbosity>1)
	std::cout <<"<HcalHotCellMonitor::setup>  Setting up histograms"<<std::endl;

      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("Hot Cell Task Event Number");
      meEVT_->Fill(ievt_);
      meTOTALEVT_ = m_dbe->bookInt("Hot Cell Task Total Events Processed");
      meTOTALEVT_->Fill(tevt_);

      // Create problem cell plots
      // Overall plot gets an initial " " in its name
      ProblemCells=m_dbe->book2D(" ProblemHotCells",
                                     " Problem Hot Cell Rate for all HCAL",
				    85,-42.5,42.5,
				    72,0.5,72.5);
      ProblemCells->setAxisTitle("i#eta",1);
      ProblemCells->setAxisTitle("i#phi",2);
      SetEtaPhiLabels(ProblemCells);

      // 1D plots count number of bad cells vs. luminoisty block
      ProblemsVsLB=m_dbe->bookProfile("TotalHotCells_HCAL_vs_LS",
				      "Total Number of Hot Hcal Cells vs lumi section", 
				      Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      ProblemsVsLB_HB=m_dbe->bookProfile("TotalHotCells_HB_vs_LS",
					 "Total Number of Hot HB Cells vs lumi section",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      ProblemsVsLB_HE=m_dbe->bookProfile("TotalHotCells_HE_vs_LS",
					 "Total Number of Hot HE Cells vs lumi section",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      ProblemsVsLB_HO=m_dbe->bookProfile("TotalHotCells_HO_vs_LS",
					 "Total Number of Hot HO Cells vs lumi section",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      ProblemsVsLB_HF=m_dbe->bookProfile("TotalHotCells_HF_vs_LS",
					 "Total Number of Hot HF Cells vs lumi section",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      ProblemsVsLB_ZDC=m_dbe->bookProfile("TotalHotCells_ZDC_vs_LS",
					  "Total Number of Hot ZDC Cells vs lumi section",
					  Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);



      // Overall Problem plot appears in main directory; plots by depth appear \in subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/problem_hotcells");
      SetupEtaPhiHists(ProblemCellsByDepth, " Problem Hot Cell Rate","");
      
      //setMinMaxHists2D(ProblemCellsByDepth,0,1.); // set minimum to hotmon_minErrorFlag_?

      // Set up plots for each failure mode of hot cells
      stringstream units; // We'll need to set the titles individually, rather than passing units to SetupEtaPhiHists (since this also would affect the name of the histograms)

      if (hotmon_test_energy_)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/hot_rechit_above_threshold");
	  SetupEtaPhiHists(AboveEnergyThresholdCellsByDepth,
			    "Hot Cells Above Energy Threshold","");
	  //setMinMaxHists2D(AboveEnergyThresholdCellsByDepth,0.,1.);
	  
	  // set more descriptive titles for plots
	  units.str("");
	  units<<"Hot Cells: Depth 1 -- HB > "<<HBenergyThreshold_<<" GeV, HE > "<<HEenergyThreshold_<<", HF > "<<HFenergyThreshold_<<" GeV";
	  AboveEnergyThresholdCellsByDepth.depth[0]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Hot Cells: Depth 2 -- HB > "<<HBenergyThreshold_<<" GeV, HE > "<<HEenergyThreshold_<<", HF > "<<HFenergyThreshold_<<" GeV";
	  AboveEnergyThresholdCellsByDepth.depth[1]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Hot Cells: Depth 3 -- HE > "<<HEenergyThreshold_<<" GeV";
	  AboveEnergyThresholdCellsByDepth.depth[2]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Hot Cells: HO > "<<HOenergyThreshold_<<" GeV";
	  AboveEnergyThresholdCellsByDepth.depth[3]->setTitle(units.str().c_str());
	  units.str("");
	}

      if (hotmon_test_persistent_)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/hot_rechit_always_above_threshold");
	  SetupEtaPhiHists(AbovePersistentThresholdCellsByDepth,
			    "Hot Cells Persistently Above Energy Threshold","");
	  //setMinMaxHists2D(AbovePersistentThresholdCellsByDepth,0.,1.);
	  
	  // set more descriptive titles for plots
	  units.str("");
	  units<<"Hot Cells: Depth 1 -- HB > "<<HBpersistentThreshold_<<" GeV, HE > "<<HEpersistentThreshold_<<", HF > "<<HFpersistentThreshold_<<" GeV for "<< hotmon_checkNevents_<<" consec. events";
	  AbovePersistentThresholdCellsByDepth.depth[0]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Hot Cells: Depth 2 -- HB > "<<HBpersistentThreshold_<<" GeV, HE > "<<HEpersistentThreshold_<<", HF > "<<HFpersistentThreshold_<<" GeV for "<<hotmon_checkNevents_<<" consec. events";
	  AbovePersistentThresholdCellsByDepth.depth[1]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Hot Cells: Depth 3 -- HE > "<<HEpersistentThreshold_<<" GeV for "<<hotmon_checkNevents_<<" consec. events";
	  AbovePersistentThresholdCellsByDepth.depth[2]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Hot Cells:  HO > "<<HOpersistentThreshold_<<" GeV for "<<hotmon_checkNevents_<<" consec. events";
	  AbovePersistentThresholdCellsByDepth.depth[3]->setTitle(units.str().c_str());
	  units.str("");
	}

      if (hotmon_test_neighbor_)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/hot_neighbortest");
	  SetupEtaPhiHists(AboveNeighborsHotCellsByDepth,"Hot Cells Failing Neighbor Test","");
	  //setMinMaxHists2D(AboveNeighborsHotCellsByDepth,0.,1.);
	}

      // The 1D energy plots are already made in the rechit monitor.  
      // Energy vs Neighbor plots might be useful for trying to set up the neighbor-identification hot cell algorithm
      if (hotmon_makeDiagnostics_)
	{
	  if (hotmon_test_energy_ || hotmon_test_persistent_)
	    {
	      m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/rechitenergy");
	      d_HBrechitenergy=m_dbe->book1D("HB_rechitenergy","HB rechit energy",1500,-10,140);
	      d_HErechitenergy=m_dbe->book1D("HE_rechitenergy","HE rechit energy",1500,-10,140);
	      d_HOrechitenergy=m_dbe->book1D("HO_rechitenergy","HO rechit energy",1500,-10,140);
	      d_HFrechitenergy=m_dbe->book1D("HF_rechitenergy","HF rechit energy",1500,-10,140);
	      SetupEtaPhiHists(d_avgrechitenergymap,
				"Rec hit energy per cell","");
	      SetupEtaPhiHists(d_avgrechitoccupancymap,"Rec hit occupancy per cell","");
	    }
	  if (hotmon_test_neighbor_)
	    {
	      m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/neighborcells");
	      d_HBenergyVsNeighbor=m_dbe->book2D("HB_energyVsNeighbor","HB  #Sigma Neighbors vs. rec hit energy",100,-5,15,100,0,25);
	      d_HEenergyVsNeighbor=m_dbe->book2D("HE_energyVsNeighbor","HE  #Sigma Neighbors vs. rec hit energy",100,-5,15,100,0,25);
	      d_HOenergyVsNeighbor=m_dbe->book2D("HO_energyVsNeighbor","HO  #Sigma Neighbors vs. rec hit energy",100,-5,15,100,0,25);
	      d_HFenergyVsNeighbor=m_dbe->book2D("HF_energyVsNeighbor","HF  #Sigma Neighbors vs. rec hit energy",100,-5,15,100,0,25);
	    }
	} // if (hotmon_makeDiagnostics_)
    } // if (m_dbe)

  return;
} //void HcalHotCellMonitor::setup(...)

/* --------------------------- */
void HcalHotCellMonitor::setupNeighborParams(const edm::ParameterSet& ps,
					      hotNeighborParams& N,
					      std::string type)
{
  // sets up parameters for neighboring-cell algorithm for each subdetector
  ostringstream myname;
  myname<<"HotCellMonitor_"<<type<<"_neighbor_deltaIphi";
  N.DeltaIphi = ps.getUntrackedParameter<int>(myname.str().c_str(),
					      defaultNeighborParams_.DeltaIphi);
  myname.str("");
  myname<<"HotCellMonitor_"<<type<<"_neighbor_deltaIeta";
  N.DeltaIeta = ps.getUntrackedParameter<int>(myname.str().c_str(),
					      defaultNeighborParams_.DeltaIeta);
  myname.str("");
  myname<<"HotCellMonitor_"<<type<<"_neighbor_deltaDepth";
  N.DeltaDepth = ps.getUntrackedParameter<int>(myname.str().c_str(),
					       defaultNeighborParams_.DeltaDepth);
  myname.str("");
  myname<<"HotCellMonitor_"<<type<<"_neighbor_minCellEnergy";
  N.minCellEnergy = ps.getUntrackedParameter<double>(myname.str().c_str(),
						     defaultNeighborParams_.minCellEnergy);
  myname.str("");
  myname<<"HotCellMonitor_"<<type<<"_neighbor_minNeighborEnergy";
  N.minNeighborEnergy = ps.getUntrackedParameter<double>(myname.str().c_str(),
							 defaultNeighborParams_.minNeighborEnergy);
  myname.str("");
  myname<<"HotCellMonitor_"<<type<<"_neighbor_maxEnergy";
  N.maxEnergy = ps.getUntrackedParameter<double>(myname.str().c_str(),
						 defaultNeighborParams_.maxEnergy);
  myname.str("");
  myname<<"HotCellMonitor_"<<type<<"_HotEnergyFrac";
  N.HotEnergyFrac = ps.getUntrackedParameter<double>(myname.str().c_str(),
						     defaultNeighborParams_.HotEnergyFrac);
  return;
} // void HcalHotCellMonitor::setupNeighborParams

/* --------------------------- */

void HcalHotCellMonitor::reset(){}  // reset function is empty for now

/* --------------------------- */

void HcalHotCellMonitor::createMaps(const HcalDbService& cond)
{

  // Creates maps for pedestals, widths, and pedestals+Nsigma*widths, using HcalDetIds as keys
  // We're no longer using this method, but keep it around for posterity's sake (and because it's useful to know how to do)

  return; 

  /*
  if (!hotmon_test_pedestal_) return; // no need to create maps if we're not running the pedestal-based hot cell finder

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  if (fVerbosity>0)
    std::cout <<"<HcalHotCellMonitor::createMaps>:  Making pedestal maps"<<std::endl;
  double ped=0;
  double width=0;
  HcalCalibrations calibs;
  const HcalQIEShape* shape = cond.getHcalShape();

  double myNsigma=0;
  double myADC=0;

  for (int ieta=-41;ieta<=41;++ieta)
    {
      for (int iphi=1;iphi<=72;++iphi)
	{
	  for (int depth=1;depth<=4;++depth)
	    {
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth))
		    continue;
		  HcalDetId hcal((HcalSubdetector)(subdet), ieta, iphi, depth);
		  
		  if (hcal.subdet()==HcalBarrel)
		    myNsigma=HBnsigma_;
		  else if (hcal.subdet()==HcalEndcap)
		    myNsigma=HEnsigma_;
		  else if (hcal.subdet()==HcalOuter)
		    myNsigma=HOnsigma_;
		  else if (hcal.subdet()==HcalForward)
		    myNsigma=HFnsigma_;
		  
		  calibs=cond.getHcalCalibrations(hcal);
		  const HcalPedestalWidth* pedw = cond.getPedestalWidth(hcal);
		   
		  ped=0.;
		  width=0.;
		  myADC=0.;
		  // loop over capids
		  for (int capid=0;capid<4;++capid)
		    {
		      // pedestals from calibs.pedestal are always in fC
		      const HcalQIECoder* channelCoder=cond.getHcalCoder(hcal);
		      
		      // Convert pedestals to ADC
		      myADC=channelCoder->adc(*shape,
					      (float)calibs.pedestal(capid),
					      capid);
		      ped+=myADC;
		      // Now the tricky part -- need to convert widths to ADC if provided in fC
		      if (doFCpeds_)
			{
			  // Form width by summing the diagonal terms of the covariance matrix (sigma_ii),
			  // and scale by ADC/fC ratio to convert from fC^2 to ADC^2
			  width+=(pedw->getSigma(capid,capid)*pow(myADC/calibs.pedestal(capid),2));
			}
		      else
			width+=pedw->getSigma(capid,capid);
		    } // for (int capid=0;capid<4;++capid)

		  ped/=4.;  // pedestal value is average over capids
		  width=pow(width,0.5)/4.;

		  pedestals_[hcal]=ped;
		  widths_[hcal]=width;
		  if (fVerbosity>1) std::cout <<"<HcalHotCellMonitor::createMaps>  Pedestal Value -- ID = "<<(HcalSubdetector)subdet<<"  ("<<ieta<<", "<<iphi<<", "<<depth<<"): "<<ped<<"; width = "<<width<<std::endl;
		  pedestal_thresholds_[hcal]=ped+myNsigma*width;
		} // for (int subdet=1,...)
	    } // for (int depth=1;...)
	} // for (int phi ...)
    } // for (int ieta...)
  
  return;
  */
} // void HcalHotCellMonitor::createMaps




/* ------------------------- */

void HcalHotCellMonitor::done(std::map<HcalDetId, unsigned int>& myqual)
{
  // moving to client; we want to be able to sum over results in offline
  return;

  if (dump2database==0) // don't do anything special unless specifically asked to dump db file
    return;

  // Dump to ascii file for database -- now handled in Channel Status objects
  /*
  char buffer [1024];
  ofstream fOutput("hcalHotCells.txt", ios::out);
  sprintf (buffer, "# %15s %15s %15s %15s %8s %10s\n", "eta", "phi", "dep", "det", "value", "DetId");
  fOutput << buffer;
  */

  int ieta=0;
  int iphi=0;
  float binval;

  int subdet;
  std::string subdetname;
  if (fVerbosity>1)
    {
      std::cout <<"<HcalHotCellMonitor>  Summary of Hot Cells in Run: "<<std::endl;
      std::cout <<"(Error rate must be >= "<<hotmon_minErrorFlag_*100.<<"% )"<<std::endl;  
    }
  int etabins=0;
  int phibins=0;
  for (unsigned int d=0;d<ProblemCellsByDepth.depth.size();++d)
    {
      etabins=ProblemCellsByDepth.depth[d]->getNbinsX();
      phibins=ProblemCellsByDepth.depth[d]->getNbinsY();
      for (int hist_eta=0;etabins;++hist_eta)
	{
	  for (int hist_phi=0;hist_phi<phibins;++hist_phi)
	    {
	      ieta=CalcIeta(hist_eta,d+1);
	      if (ieta==-9999) continue;
	      iphi=hist_phi;

	      binval=ProblemCellsByDepth.depth[d]->getBinContent(ieta,iphi)/ievt_;
	     
	        // Set subdetector labels for output
	      if (d==0)// HB/HE/HF
		{
		  // correct for HF offset 
		  if (hist_eta< 13 || hist_eta>71)
		    {
		      subdetname="HF";
		      subdet=4;
		    }
		  else if (abs(ieta)<=16) // HB extends to |ieta|=16 in depth 1, 15 in depth 2
		    {
		      subdetname="HB";
		      subdet=1;
		    }
		  else // HE at |ieta|=16 is in depth 3; don't worry about it here
		    {
		      subdetname="HE";
		      subdet=2;
		    }
		}
	      else if (d==1) // HB/HE/HF
		{
		  if (hist_eta< 13 || hist_eta>42)
		    {
		      subdetname="HF";
		      subdet=4;
		    }
		  else if (abs(ieta)<16) // HB extends to |ieta|=16 in depth 1, 15 in depth 2
		    {
		      subdetname="HB";
		      subdet=1;
		    }
		  else // HE at |ieta|=16 is in depth 3; don't worry about it here
		    {
		      subdetname="HE";
		      subdet=2;
		    }
		}
	      else if (d==2) // depth 3 is HE only
		{
		  subdetname="HE";
		  subdet=2;
		}
	      else // depth 4 is HO only
		{
		  subdetname="HO";
		  subdet=3;
		}

	      HcalDetId myid((HcalSubdetector)(subdet), ieta, iphi, d+1);
	      if (!validDetId((HcalSubdetector)(subdet), ieta, iphi, d+1))
		continue;
	      if (fVerbosity>0 && binval>hotmon_minErrorFlag_)
		std::cout <<"Hot Cell "<<subdet<<"("<<ieta<<", "<<iphi<<", "<<d+1<<"):  "<<binval*100.<<"%"<<std::endl;
	      
	      if (binval<=hotmon_minErrorFlag_)
		continue;
	      // if we've reached here, hot cell condition was met
	      int value=1;

	      if (myqual.find(myid)==myqual.end())
		{
		  myqual[myid]=(value<<BITSHIFT);  // hotcell shifted to bit 6
		}
	      else
		{
		  int mask=(1<<BITSHIFT);
		  if (value==1)
		    myqual[myid] |=mask;

		  else
		    myqual[myid] &=~mask;
		}

	    } // for (int hist_phi=1;...)
	} // for (int hist_eta=1;...)
    } // for (int d=0;d<ProblemCellsByDepth.depth.size();d++)

  return;

} // void HcalHotCellMonitor::done()



/* --------------------------------- */

void HcalHotCellMonitor::clearME()
{
  // I don't think this function gets cleared any more.  
  // And need to add code to clear out subfolders as well?
  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
    }
  return;
} // void HcalHotCellMonitor::clearME()

/* -------------------------------- */


void HcalHotCellMonitor::processEvent(const HBHERecHitCollection& hbHits,
				       const HORecHitCollection& hoHits,
				       const HFRecHitCollection& hfHits,
				       //const ZDCRecHitCollection& zdcHits,
				       const HBHEDigiCollection& hbhedigi,
				       const HODigiCollection& hodigi,
				       const HFDigiCollection& hfdigi,
				       //const ZDCDigiCollection& zdcdigi,
				       const HcalDbService& cond
				       )
{
  
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  HcalBaseMonitor::processEvent();

  if (fVerbosity>1) std::cout <<"<HcalHotCellMonitor::processEvent> Processing event..."<<std::endl;

  // Search for hot cells above a certain energy
  if (hotmon_test_energy_ || hotmon_test_persistent_)
    {
      processEvent_rechitenergy(hbHits, hoHits,hfHits);
    }

  // Search for cells that are hot compared to their neighbors -- currently disabled
  if (hotmon_test_neighbor_)
    {
      processEvent_rechitneighbors(hbHits, hoHits, hfHits);
    }

  // Fill problem cells
  if ((ievt_%hotmon_checkNevents_ ==0) && 
      (hotmon_test_persistent_ ||
       hotmon_test_neighbor_  || hotmon_test_energy_    ))
    {
      fillNevents_problemCells();
    }

  return;
} // void HcalHotCellMonitor::processEvent(...)

/* --------------------------------------- */

void HcalHotCellMonitor::fillHotHistosAtEndRun()
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

  
  if (hotmon_test_persistent_) 
    {
      for (unsigned int i=0;i<AbovePersistentThresholdCellsByDepth.depth.size();++i)
	AbovePersistentThresholdCellsByDepth.depth[i]->setBinContent(0,0,ievt_);
      if (ievt_%hotmon_checkNevents_>0)
	fillNevents_persistentenergy();
    }
  if (hotmon_test_neighbor_)
    {
      for (unsigned int i=0;i<AboveNeighborsHotCellsByDepth.depth.size();++i)
	AboveNeighborsHotCellsByDepth.depth[i]->setBinContent(0,0,ievt_);
      if (ievt_%hotmon_checkNevents_ >0) fillNevents_neighbor();
    }
  if (hotmon_test_energy_    && ievt_%hotmon_checkNevents_   >0) fillNevents_energy();
  if (hotmon_test_persistent_ || 
      hotmon_test_neighbor_  || hotmon_test_energy_)  
    fillNevents_problemCells();
}

/* --------------------------------------- */


void HcalHotCellMonitor::processEvent_rechitenergy( const HBHERecHitCollection& hbheHits,
						    const HORecHitCollection& hoHits,
						    const HFRecHitCollection& hfHits)
  
{
  // Looks at rechits of cells and compares to threshold energies.
  // Cells above thresholds get marked as hot candidates

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

 if (fVerbosity>1) std::cout <<"<HcalHotCellMonitor::processEvent_rechitenergy> Processing rechits..."<<std::endl;
 if (hotmon_test_neighbor_)   rechitEnergies_.clear();

 // loop over HBHE
 for (HBHERecHitCollection::const_iterator HBHEiter=hbheHits.begin(); HBHEiter!=hbheHits.end(); ++HBHEiter) 
   { // loop over all hits
     float en = HBHEiter->energy();
     //float ti = HBHEiter->time();

     HcalDetId id(HBHEiter->detid().rawId());
     int ieta = id.ieta();
     int iphi = id.iphi();
     int depth = id.depth();

     if (hotmon_makeDiagnostics_)
       {
	 ++rechit_occupancy_sum[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	 rechit_energy_sum[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]+=en;
       }
     if (id.subdet()==HcalBarrel)
       {
	 if (!checkHB_) continue;
	 if (hotmon_makeDiagnostics_) d_HBrechitenergy->Fill(en);
	 if (en>=HBenergyThreshold_)
	   {
	     ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	   }
	 if (en>=HBpersistentThreshold_)
	   ++abovepersistent[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
       }
     else if (id.subdet()==HcalEndcap)
       {
	 if (!checkHE_) continue;
	 if (hotmon_makeDiagnostics_) d_HErechitenergy->Fill(en);
	 if (en>=HEenergyThreshold_)
	   ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	 if (en>=HEpersistentThreshold_)
	   ++abovepersistent[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
       }
     if (hotmon_test_neighbor_) rechitEnergies_[id]=en;
   } //for (HBHERecHitCollection::const_iterator HBHEiter=...)

 // loop over HO
 if (checkHO_)
   {
     for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) 
       { // loop over all hits
	 float en = HOiter->energy();
	 
	 HcalDetId id(HOiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();
	 
	 if (hotmon_makeDiagnostics_)
	   {
	     ++rechit_occupancy_sum[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	     rechit_energy_sum[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]+=en;
	   }
	 if (hotmon_makeDiagnostics_) d_HOrechitenergy->Fill(en);
	 if (isSiPM(ieta,iphi,depth))
	   {
	    if (en>=HOenergyThreshold_*SiPMscale_)
	      ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]; 
	    if (en>=HOpersistentThreshold_*SiPMscale_)
	      ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	   }
	 else
	   {
	     if (en>=HOenergyThreshold_)
	      ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]; 
	    if (en>=HOpersistentThreshold_)
	      ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	   }

	 if (hotmon_test_neighbor_) rechitEnergies_[id]=en;
       }
   } // if (checkHO_)
 
 // loop over HF
 if (checkHF_)
   {
     for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter) 
       { // loop over all hits
	 float en = HFiter->energy();
	 
	 HcalDetId id(HFiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();

	 if (hotmon_makeDiagnostics_)
	   {
	     ++rechit_occupancy_sum[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	     rechit_energy_sum[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]=en;
	   }
	 if (hotmon_makeDiagnostics_) d_HFrechitenergy->Fill(en);
	 if (en>=HFenergyThreshold_)
	   {
	     ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	   }
	 if (en>=HFpersistentThreshold_)
	   ++abovepersistent[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	 if (hotmon_test_neighbor_) rechitEnergies_[id]=en;
       }
   } // if (checkHF_)

 // Fill Dummy histogram value each event
 for (unsigned int i=0;i<AboveEnergyThresholdCellsByDepth.depth.size();++i)
   AboveEnergyThresholdCellsByDepth.depth[i]->setBinContent(0,0,ievt_);
 for (unsigned int i=0;i<AbovePersistentThresholdCellsByDepth.depth.size();++i)
   AbovePersistentThresholdCellsByDepth.depth[i]->setBinContent(0,0,ievt_);

 
 // Fill histograms 
  if (ievt_%hotmon_checkNevents_==0 && hotmon_test_energy_)
    {
	if (fVerbosity) std::cout <<"<HcalHotCellMonitor::processEvent_digi> Filling HotCell Energy plots"<<std::endl;
	fillNevents_energy();
    }
  if (ievt_%hotmon_checkNevents_==0 && hotmon_test_persistent_)
    {
	if (fVerbosity) std::cout <<"<HcalHotCellMonitor::processEvent_digi> Filling HotCell Persistent Energy plots"<<std::endl;
	fillNevents_persistentenergy();
    }
  
 if (showTiming)
   {
     cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellMonitor PROCESSEVENT_RECHITENERGY -> "<<cpu_timer.cpuTime()<<std::endl;
   }
 
 return;
} // void HcalHotCellMonitor::processEvent_rechitenergy

/* --------------------------------------- */


void HcalHotCellMonitor::processEvent_rechitneighbors( const HBHERecHitCollection& hbheHits,
							const HORecHitCollection& hoHits,
							const HFRecHitCollection& hfHits
							)
{
  // Compares energy to energy of neighboring cells.
  // This is a slightly simplified version of D0's NADA algorithm
  // 17 June 2009 -- this needs major work.  I'm not sure I have the [eta][phi][depth] array mapping correct everywhere. 
  // Maybe even tear it apart and start again?

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

 if (fVerbosity>1) std::cout <<"<HcalHotCellMonitor::processEvent_rechitneighbors> Processing rechits..."<<std::endl;

 // if Energy tests weren't run, need to create map of Detid:rechitenergy here
 if (!hotmon_test_energy_ && !hotmon_test_persistent_)
   {
     rechitEnergies_.clear(); // clear old map
     for (HBHERecHitCollection::const_iterator HBHEiter=hbheHits.begin(); HBHEiter!=hbheHits.end(); ++HBHEiter) 
       { // loop over all hits
	 float en = HBHEiter->energy();
	 HcalDetId id(HBHEiter->detid().rawId());
	 if (!checkHB_ && id.subdet()==HcalBarrel)
	   continue;
	 if (!checkHE_ && id.subdet()==HcalEndcap)
	   continue;
	 rechitEnergies_[id]=en;
       }
     // HO
     if (checkHO_)
       {
	 for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) 
	   { // loop over all hits
	     float en = HOiter->energy();
	     HcalDetId id(HOiter->detid().rawId());
	     rechitEnergies_[id]=en;
	   }
       } // if (checkHO_)
     //HF
     if (checkHF_)
       {
	 for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter) 
	   { // loop over all hits
	     float en = HFiter->energy();
	     HcalDetId id(HFiter->detid().rawId());
	     rechitEnergies_[id]=en;
	   }
       } // if (checkHF_)

   } // if (!hotmon_test_energy_ && !hotmon_test_persistent_)   

 // Now do "real" loop, checking against each cell against its neighbors
 
 int ieta, iphi, depth;
 float en;

 int neighborsfound=0;
 float enNeighbor=0;

 // loop over HBHE
 for (HBHERecHitCollection::const_iterator HBHEiter=hbheHits.begin(); 
      HBHEiter!=hbheHits.end(); 
      ++HBHEiter) 
   { // loop over all hits
     
     en = HBHEiter->energy();
     HcalDetId id(HBHEiter->detid().rawId());
     ieta = id.ieta();
     iphi = id.iphi();
     depth = id.depth();

     if (id.subdet()==HcalBarrel)
       {
	 if (!checkHB_) continue;
	 // Case 0:  energy > max value; it's marked as hot regardless of neighbors
	 if (en>HBNeighborParams_.maxEnergy)
	   {
	     aboveneighbors[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	     continue;
	   }

	 // Search keys for neighboring cells
	 if (en<HBNeighborParams_.minCellEnergy) // cells below minCellEnergy not considered hot
	   continue;
	 neighborsfound=0;
	 enNeighbor=0;
	 for (int nD=-1*HBNeighborParams_.DeltaDepth;nD<=HBNeighborParams_.DeltaDepth;++nD)
	   {
	     for (int nP =-1*HBNeighborParams_.DeltaIphi;nP<=HBNeighborParams_.DeltaIphi;++nP)
	       {
		 for (int nE =-1*HBNeighborParams_.DeltaIeta;nE<=HBNeighborParams_.DeltaIeta;++nE)
		   {
		     if (nD==0 && nE==0 && nP==0) 
		       continue; // don't count the cell itself
		     int myphi=nP+iphi;
		     if (myphi>72) myphi-=72; // allow for wrapping of cells
		     if (myphi<=0) myphi+=72;
		     if (!validDetId((HcalSubdetector)(1),nE+ieta, myphi, nD+depth)) continue;
		     HcalDetId myid((HcalSubdetector)(1), nE+ieta, myphi, nD+depth); // HB
		     if (rechitEnergies_.find(myid)==rechitEnergies_.end())
		       continue;
		     if (rechitEnergies_[myid]<HBNeighborParams_.minNeighborEnergy)
		       continue;
		     ++neighborsfound;
		     enNeighbor+=rechitEnergies_[myid];
		   } // loop over nE (neighbor eta)
	       } // loop over nP (neighbor phi)
	   } // loop over nD depths

	 if (hotmon_makeDiagnostics_)
	   d_HBenergyVsNeighbor->Fill(en,enNeighbor);
	 
	 // Case 1:  Not enough good neighbors found
	 if (neighborsfound==0)
	   continue;
	 // Case 2:  energy/(avg. neighbor energy) too large for cell to be considered hot
	 if ((1.*enNeighbor/en)>HBNeighborParams_.HotEnergyFrac && en>0 && enNeighbor>0)
	   continue;
	 // Case 3:  Tests passed; cell marked as hot
	 aboveneighbors[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]++;
       }

     else if (id.subdet()==HcalEndcap)
       {
	 if (!checkHE_) continue;
	 // Case 0:  energy > max value; it's marked as hot regardless of neighbors
	 if (en>HENeighborParams_.maxEnergy)
	   {
	     aboveneighbors[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]++;
	     continue;
	   }
	 if (en<HENeighborParams_.minCellEnergy)
	   continue; // cells below this value can never be considered hot
	 // Search keys for neighboring cells
	 neighborsfound=0;
	 enNeighbor=0;
	 int HEDeltaIphi = HENeighborParams_.DeltaIphi;
	 // now correct for boundaries
	 if (abs(ieta)>20) HEDeltaIphi*=2; // double iphi boundary range when segmentation switches to 10 degrees
	 // This still needs to be worked on to properly deal with boundaries
	 for (int nD=-1*HENeighborParams_.DeltaDepth;nD<=HENeighborParams_.DeltaDepth;++nD)
	   {
	     for (int nP =-1*HEDeltaIphi;nP<=HEDeltaIphi;++nP)
	       {
		 for (int nE =-1*HENeighborParams_.DeltaIeta;nE<=HENeighborParams_.DeltaIeta;++nE)
		   {
		     if (nD==0 && nE==0 && nP==0) 
		       continue; // don't count the cell itself
		     
		     int myphi=nP+iphi;
                     if (myphi>72) myphi-=72; // allow for wrapping of cells
		     if (myphi<=0) myphi+=72;
		     if (!validDetId((HcalSubdetector)(2),nE+ieta, myphi, nD+depth)) continue;
                     HcalDetId myid((HcalSubdetector)(2), nE+ieta, myphi, nD+depth); // HE
		     if (rechitEnergies_.find(myid)==rechitEnergies_.end())
		       continue;
		     if (rechitEnergies_[myid]<HENeighborParams_.minNeighborEnergy)
		       continue;
		     ++neighborsfound;
		     enNeighbor+=rechitEnergies_[myid];
		   } // loop over nE (neighbor eta)
	       } // loop over nP (neighbor phi)
	   } // loop over nD depths

	 if (hotmon_makeDiagnostics_)
	   d_HEenergyVsNeighbor->Fill(en,enNeighbor);
	 
	 // Case 1:  Not enough good neighbors found
	 if (neighborsfound==0)
	   continue;
	 // Case 2:  energy/(avg. neighbor energy) too large for cell to be considered hot
	 if ((1.*enNeighbor/en)>HENeighborParams_.HotEnergyFrac && en>0 && enNeighbor>0)
	   continue;
	 // Case 3:  Tests passed; cell marked as hot
	 aboveneighbors[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]++;
       }
} //for (HBHERecHitCollection::const_iterator HBHEiter=...)

 // loop over HO
 if (checkHO_)
   {
     for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) 
       { // loop over all hits
	 float en = HOiter->energy();
	 HcalDetId id(HOiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();
	 
	 // Case 0:  energy > max value; it's marked as hot regardless of neighbors
	 if (en>HONeighborParams_.maxEnergy)
	   {
	     aboveneighbors[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]++;
	     continue;
	   }
	 if (en<HONeighborParams_.minCellEnergy)
	    continue; // cells below this value can never be considered hot

	 // Search keys for neighboring cells
	 neighborsfound=0;
	 enNeighbor=0;
	 for (int nD=-1*HONeighborParams_.DeltaDepth;nD<=HONeighborParams_.DeltaDepth;++nD)
	   {
	     for (int nP =-1*HONeighborParams_.DeltaIphi;nP<=HONeighborParams_.DeltaIphi;++nP)
	       {
		 for (int nE =-1*HONeighborParams_.DeltaIeta;nE<=HONeighborParams_.DeltaIeta;++nE)
		   {
		     if (nD==0 && nE==0 && nP==0) 
		       continue; // don't count the cell itself
		     int myphi=nP+iphi;
		     if (myphi>72) myphi-=72; // allow for wrapping of cells
		     if (myphi<=0) myphi+=72;
                     if (!validDetId((HcalSubdetector)(3),nE+ieta, myphi, nD+depth)) continue;
                     HcalDetId myid((HcalSubdetector)(3), nE+ieta, myphi, nD+depth); // HO
		     if (rechitEnergies_.find(myid)==rechitEnergies_.end())
		       continue;
		     if (rechitEnergies_[myid]<HONeighborParams_.minNeighborEnergy)
		       continue;
		     ++neighborsfound;
		     enNeighbor+=rechitEnergies_[myid];
		   } // loop over nE (neighbor eta)
	       } // loop over nP (neighbor phi)
	   } // loop over nD depths

	 if (hotmon_makeDiagnostics_)
	   d_HOenergyVsNeighbor->Fill(en,enNeighbor);

	 // We'll need to revisit this to deal with SiPMs?
	 // Case 1:  Not enough good neighbors found
	 if (neighborsfound==0)
	   continue;
	 // Case 2:  energy/(avg. neighbor energy) too large for cell to be considered hot
	 if ((1.*enNeighbor/en)>HONeighborParams_.HotEnergyFrac && en>0 && enNeighbor>0)
	   continue;
	 // Case 3:  Tests passed; cell marked as hot
	 aboveneighbors[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]++;
       } // loop over hits
   } // if (checkHO_)
 
 // loop over HF
 if (checkHF_)
   {
     for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter) 
       { // loop over all hits
	 float en = HFiter->energy();
	 HcalDetId id(HFiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();

	 // Case 0:  energy > max value; it's marked as hot regardless of neighbors
	 if (en>HFNeighborParams_.maxEnergy)
	   {
	     aboveneighbors[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]++;
	     continue;
	   }
	 if (en<HFNeighborParams_.minCellEnergy)
	   continue; // cells below this value can never be considered hot
	  // Search keys for neighboring cells
	 neighborsfound=0;
	 enNeighbor=0;
	 int HFDeltaIphi = HFNeighborParams_.DeltaIphi;
	 if (abs(ieta)>39) HFDeltaIphi*=2;  // double phi range when segmentation switches to 20 degrees
	 // Still need to create a more robust handling of boundary cases
	 for (int nD=-1*HFNeighborParams_.DeltaDepth;nD<=HFNeighborParams_.DeltaDepth;++nD)
	   {
	     for (int nP =-1*HFDeltaIphi;nP<=HFDeltaIphi;++nP)
	       {
		 for (int nE =-1*HFNeighborParams_.DeltaIeta;nE<=HFNeighborParams_.DeltaIeta;++nE)
		   {
		     if (nD==0 && nE==0 && nP==0) 
		       continue; // don't count the cell itself
		     int myphi=nP+iphi;
                     if (myphi>72) myphi-=72; // allow for wrapping of cells
                     if (myphi<=0) myphi+=72;
		     if (!validDetId((HcalSubdetector)(4),nE+ieta, myphi, nD+depth)) continue;
		     HcalDetId myid((HcalSubdetector)(4), nE+ieta, myphi, nD+depth); // HF
		     if (rechitEnergies_.find(myid)==rechitEnergies_.end())
		       continue;
		     if (rechitEnergies_[myid]<HFNeighborParams_.minNeighborEnergy)
		       continue;
		     ++neighborsfound;
		     enNeighbor+=rechitEnergies_[myid];
		   } // loop over nE (neighbor eta)
	       } // loop over nP (neighbor phi)
	   } // loop over nD depths

	 if (hotmon_makeDiagnostics_)
	   d_HFenergyVsNeighbor->Fill(en,enNeighbor);
	 
	 // Case 1:  Not enough good neighbors found
	 if (neighborsfound==0)
	   continue;
	 // Case 2:  energy/(avg. neighbor energy) too large for cell to be considered hot
	 if ((1.*enNeighbor/en)>HFNeighborParams_.HotEnergyFrac && en>0 && enNeighbor>0)
	   continue;
	 // Case 3:  Tests passed; cell marked as hot
	 aboveneighbors[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]++;
       } // loop over all hits
   } // if (checkHF_)
 
 // Fill Dummy histogram value each event
 for (unsigned int i=0;i<AboveNeighborsHotCellsByDepth.depth.size();++i)
   AboveNeighborsHotCellsByDepth.depth[i]->setBinContent(0,0,ievt_);

 // Fill histograms 
  if (ievt_%hotmon_checkNevents_==0)
    {
	if (fVerbosity) std::cout <<"<HcalHotCellMonitor::processEvent_digi> Filling HotCell Neighbor plots"<<std::endl;
	if (hotmon_test_neighbor_) fillNevents_neighbor();
    }

 if (showTiming)
   {
     cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellMonitor PROCESSEVENT_RECHITNEIGHBOR -> "<<cpu_timer.cpuTime()<<std::endl;
   }
 return;
} // void HcalHotCellMonitor::processEvent_rechitneighbor


/* --------------------------------------- */


void HcalHotCellMonitor::fillNevents_persistentenergy(void)
{
  // Fill Histograms showing rechits with energies > some threshold for N consecutive events

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalHotCellMonitor::fillNevents_persistentenergy> FILLING PERSISTENT ENERGY PLOTS"<<std::endl;
  

  for (unsigned int h=0;h<AbovePersistentThresholdCellsByDepth.depth.size();++h)
    AbovePersistentThresholdCellsByDepth.depth[h]->setBinContent(0,0,ievt_);

  int ieta=0;
  int iphi=0;
  int etabins=0;
  int phibins=0;

   for (unsigned int depth=0;depth<AbovePersistentThresholdCellsByDepth.depth.size();++depth)
     { 
       etabins=AbovePersistentThresholdCellsByDepth.depth[depth]->getNbinsX();
       phibins=AbovePersistentThresholdCellsByDepth.depth[depth]->getNbinsY();

       for (int eta=0;eta<etabins;++eta)
	 {
	   for (int phi=0;phi<phibins;++phi)
	     {
	       iphi=phi+1;
	       for (int subdet=1;subdet<=4;++subdet)
		 {
		   ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1); //converts bin to ieta
		   if (ieta==-9999) continue;
		   if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1))
		     continue;
		   if (subdet==HcalForward) // shift HcalForward ieta by 1 for filling purposes
		      ieta<0 ? ieta-- : ieta++;
	
		   if (hotmon_makeDiagnostics_ && rechit_occupancy_sum[eta][phi][depth]>0)
		     {
		       // Fill average energy plots
		       
		       d_avgrechitenergymap.depth[depth]->Fill(ieta,iphi,rechit_energy_sum[eta][phi][depth]);
		       d_avgrechitoccupancymap.depth[depth]->Fill(ieta,iphi,rechit_occupancy_sum[eta][phi][depth]);
		       rechit_energy_sum[eta][phi][depth]=0;
		       rechit_occupancy_sum[eta][phi][depth]=0;
		     }
		   
		   // MUST BE ABOVE ENERGY THRESHOLD FOR ALL N EVENTS
		   if (abovepersistent[eta][phi][depth]<hotmon_checkNevents_)
		     {
		       abovepersistent[eta][phi][depth]=0;
		       continue;  		
		     }
		   if (fVerbosity>0) std::cout <<"HOT CELL; PERSISTENT ENERGY at subdet = "<<subdet<<", eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth<<std::endl;
		   AbovePersistentThresholdCellsByDepth.depth[depth]->Fill(ieta,iphi,abovepersistent[eta][phi][depth]);
		   AbovePersistentThresholdCellsByDepth.depth[depth]->setBinContent(0,0,ievt_);
		   //ProblemCellsByDepth.depth[depth]->Fill(ieta,iphi,abovepersistent[eta][phi][depth]);
		   abovepersistent[eta][phi][depth]=0; // reset counter
		 } // for (int subdet=1; subdet<=4;++subdet)
	     } // for (int phi=0;...)
	 } // for (int eta=0;...)
     } // for (unsigned int depth=0;...)
  FillUnphysicalHEHFBins(AbovePersistentThresholdCellsByDepth);

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellMonitor FILLNEVENTS_PERSISTENTENERGY -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalHotCellMonitor::fillNevents_persistentenergy(void)



/* ----------------------------------- */

void HcalHotCellMonitor::fillNevents_energy(void)
{
  // Fill Histograms showing rec hits that are above some energy value 
  // (Fill for each instance when cell is above energy; don't require it to be hot for a number of consecutive events)

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalHotCellMonitor::fillNevents_energy> ABOVE-ENERGY-THRESHOLD PLOTS"<<std::endl;

  for (unsigned int h=0;h<AboveEnergyThresholdCellsByDepth.depth.size();++h)
    AboveEnergyThresholdCellsByDepth.depth[h]->setBinContent(0,0,ievt_);

  int ieta=0;
  int iphi=0;
  int etabins=0;
  int phibins=0;
  
  for (unsigned int depth=0;depth<AboveEnergyThresholdCellsByDepth.depth.size();++depth)
    { 
      etabins=AboveEnergyThresholdCellsByDepth.depth[depth]->getNbinsX();
      phibins=AboveEnergyThresholdCellsByDepth.depth[depth]->getNbinsY();
      
       for (int eta=0;eta<etabins;++eta)
	 {
	   for (int phi=0;phi<phibins;++phi)
	     {
	       iphi=phi+1;
	       for (int subdet=1;subdet<=4;++subdet)
		 {
		   ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1); //converts bin to ieta
		   if (ieta==-9999) continue;
		   if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1))
		     continue;
		   if (subdet==HcalForward) // shift HcalForward ieta by 1 for filling purposes
		      ieta<0 ? ieta-- : ieta++;

		   if (aboveenergy[eta][phi][depth]>0)
		     {
		       if (fVerbosity>2) 
			 std::cout <<"HOT CELL; ABOVE ENERGY THRESHOLD at subdet = "<<subdet<<", eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<std::endl;
		       AboveEnergyThresholdCellsByDepth.depth[depth]->Fill(ieta,iphi, aboveenergy[eta][phi][depth]);
		       AboveEnergyThresholdCellsByDepth.depth[depth]->setBinContent(0,0,ievt_);
		       aboveenergy[eta][phi][depth]=0;
		     } // if (aboveenergy[eta][phi][depth])
		 } // for (int subdet=0)
	     } // for (int phi=0;...)
	 } // for (int eta=0;...)
    } // for (int depth=0;...)
  FillUnphysicalHEHFBins(AboveEnergyThresholdCellsByDepth);
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellMonitor FILLNEVENTS_ENERGY -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;


} // void HcalHotCellMonitor::fillNevents_energy(void)



/* ----------------------------------- */

void HcalHotCellMonitor::fillNevents_neighbor(void)
{
  // Fill Histograms showing rec hits with energy much less than neighbors' average

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalHotCellMonitor::fillNevents_neighbor> FILLING ABOVE-NEIGHBOR-ENERGY PLOTS"<<std::endl;

  for (unsigned int h=0;h<AboveNeighborsHotCellsByDepth.depth.size();++h)
    AboveNeighborsHotCellsByDepth.depth[h]->setBinContent(0,0,ievt_);

  int ieta=0;
  int iphi=0;
  int etabins=0;
  int phibins=0;
  
  for (unsigned int depth=0;depth<AboveNeighborsHotCellsByDepth.depth.size();++depth)
    { 
      etabins=AboveNeighborsHotCellsByDepth.depth[depth]->getNbinsX();
      phibins=AboveNeighborsHotCellsByDepth.depth[depth]->getNbinsY();
      
      for (int eta=0;eta<etabins;++eta)
	{
	  for (int phi=0;phi<phibins;++phi)
	    {
	      iphi=phi+1;
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1); //converts bin to ieta
		  if (ieta==-9999) continue;
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1))
		    continue;
		  if (subdet==HcalForward) // shift HcalForward ieta by 1 for filling purposes
		    ieta<0 ? ieta-- : ieta++;
		  
		  if (aboveneighbors[eta][phi][depth]>0)
		    {
		      if (fVerbosity>2) std::cout <<"HOT CELL; ABOVE NEIGHBORS at eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<(depth>4 ? depth+1 : depth-3)<<std::endl;
		      AboveNeighborsHotCellsByDepth.depth[depth]->Fill(ieta,iphi,aboveneighbors[eta][phi][depth]);
		      AboveNeighborsHotCellsByDepth.depth[depth]->setBinContent(0,0,ievt_);
		      //ProblemCellsByDepth.depth[depth]->Fill(ieta,iphi,aboveneighbors[eta][phi][depth]);
		      //reset counter
		      aboveneighbors[eta][phi][depth]=0;
		    } // if (aboveneighbors[eta][phi][mydepth]>0)
		} // for (int subdet=1;...)
	    } // for (int phi=0;...)
	} // for (int eta=0;...)
    } // for (unsigned int depth=0;...)
  FillUnphysicalHEHFBins(AboveNeighborsHotCellsByDepth);

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellMonitor FILLNEVENTS_NEIGHBOR -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;


} // void HcalHotCellMonitor::fillNevents_neighbor(void)






void HcalHotCellMonitor::fillNevents_problemCells(void)
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalHotCellMonitor::fillNevents_problemCells> FILLING PROBLEM CELL PLOTS"<<std::endl;

  int ieta=0;
  int iphi=0;
  int etabins=0;
  int phibins=0;
  double problemvalue=0;

  // Count problem cells in each subdetector
  NumBadHB=0;
  NumBadHE=0;
  NumBadHO=0;
  NumBadHF=0;
  NumBadZDC=0;

  ProblemCells->Reset();
  ProblemCells->setBinContent(0,0,ievt_); // set underflow bin to total number of events (used for normalization)
  int zside=0;
  for (unsigned int depth=0;depth<ProblemCellsByDepth.depth.size();++depth)
    {
      ProblemCellsByDepth.depth[depth]->Reset();
      ProblemCellsByDepth.depth[depth]->setBinContent(0,0,ievt_); // set underflow bin to total number of events (used for normalization)
      etabins=ProblemCellsByDepth.depth[depth]->getNbinsX();
      phibins=ProblemCellsByDepth.depth[depth]->getNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  ieta=CalcIeta(eta,depth+1);
	  for (int phi=0;phi<phibins;++phi)
	    {
	      		  
	      // Get bad number of events from each problem type
	      problemvalue=0;
	      if (hotmon_test_persistent_)
		{
		  problemvalue+=AbovePersistentThresholdCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1);
		}
	      if (hotmon_test_neighbor_)
		{
		  problemvalue+=AboveNeighborsHotCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1);
		}
	      if (hotmon_test_energy_)
		{
		  problemvalue+=AboveEnergyThresholdCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1);
		}
	      problemvalue = min((double)ievt_, problemvalue);
	      if (problemvalue==0) continue;
	      iphi=phi+1;
	      zside=0;
	      if (ieta==-9999) continue;

	      if (depth<2)
		{
		  if (isHF(eta,depth+1))
		    ieta<0 ? zside = -1 : zside= 1;
		}
	      if (problemvalue>hotmon_minErrorFlag_*ievt_)
		{
		  if (isHB(eta,depth+1)) ++NumBadHB;
		  else if (isHE(eta,depth+1)) ++NumBadHE;
		  else if (isHO(eta,depth+1)) ++NumBadHO;
		  else if (isHF(eta,depth+1)) ++NumBadHF;
		  ProblemCellsByDepth.depth[depth]->Fill(ieta+zside,iphi,problemvalue);
		  ProblemCells->Fill(ieta+zside,iphi,problemvalue);
		}
	    } // for (int phi=0;...)
	} //for (int eta=0;...)
    } // for (int depth=0;...)
  
  // Make sure summary over depth doesn't include more than ievt_ entries per bin
  etabins=ProblemCells->getNbinsX();
  phibins=ProblemCells->getNbinsY();
  for (int eta=0;eta<etabins;++eta)
    {
      for (int phi=0;phi<phibins;++phi)
	{
	  if (ProblemCells->getBinContent(eta+1,phi+1)>ievt_)
	    ProblemCells->setBinContent(eta+1,phi+1,ievt_);
	}
    }
  
  ProblemsVsLB_HB->Fill(lumiblock,NumBadHB);
  ProblemsVsLB_HE->Fill(lumiblock,NumBadHE);
  ProblemsVsLB_HO->Fill(lumiblock,NumBadHO);
  ProblemsVsLB_HF->Fill(lumiblock,NumBadHF);
  ProblemsVsLB_ZDC->Fill(lumiblock,NumBadZDC);
  ProblemsVsLB->Fill(lumiblock,NumBadHB+NumBadHE+NumBadHO+NumBadHF+NumBadZDC);
  
  FillUnphysicalHEHFBins(ProblemCells);
  FillUnphysicalHEHFBins(ProblemCellsByDepth);
  
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalHotCellMonitor FILLNEVENTS_PROBLEMCELLS -> "<<cpu_timer.cpuTime()<<std::endl;
    }

} // void HcalHotCellMonitor::fillNevents_problemCells(void)


void HcalHotCellMonitor::zeroCounters(void)
{

  // zero all counters
  for (int i=0;i<85;++i)
    {
      for (int j=0;j<72;++j)
        {
          for (int k=0;k<4;++k)
            {
              abovepersistent[i][j][k]=0;
              aboveneighbors[i][j][k]=0;
              aboveenergy[i][j][k]=0;
	      rechit_occupancy_sum[i][j][k]=0;
	      rechit_energy_sum[i][j][k]=0.;
            }
        }
    }

  // zero diagnostic counters
  for (int i=0;i<300;++i)
    {
      diagADC_HB[i]=0;
      diagADC_HE[i]=0;
      diagADC_HO[i]=0;
      diagADC_HF[i]=0;
      diagADC_ZDC[i]=0;
    }
  // Add other diagnostic counters here later

  return;

} // void HcalHotCellMonitor::zeroCounters()

#include "DQM/HcalMonitorTasks/interface/HcalDeadCellMonitor.h"

#define OUT if(fverbosity_)cout
#define BITSHIFT 5

using namespace std;

HcalDeadCellMonitor::HcalDeadCellMonitor()
{
  ievt_=0;
  // Default initialization
  showTiming   = false;
  fVerbosity   = 0;
  deadmon_makeDiagnostics_ = false;
} //constructor

HcalDeadCellMonitor::~HcalDeadCellMonitor()
{
} //destructor


/* ------------------------------------ */ 

void HcalDeadCellMonitor::setup(const edm::ParameterSet& ps,
				DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

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

  baseFolder_ = rootFolder_+"DeadCellMonitor_Hcal";
  if (fVerbosity>0)
    std::cout <<"<HcalDeadCellMonitor::setup>  Setting up histograms"<<std::endl;

  // Assume subdetectors not present until shown otherwise
  HBpresent_ =false;
  HEpresent_ =false;
  HOpresent_ =false;
  HFpresent_ =false;
  ZDCpresent_=false;

  // Dead Cell Monitor - specific cfg variables

  if (fVerbosity>1)
    std::cout <<"<HcalDeadCellMonitor::setup>  Getting variable values from cfg files"<<std::endl;

  // deadmon_makeDiagnostics_ will take on base task value unless otherwise specified
  deadmon_makeDiagnostics_ = ps.getUntrackedParameter<bool>("DeadCellMonitor_makeDiagnosticPlots",makeDiagnostics);
  
  // Set checkNevents values
  deadmon_checkNevents_ = ps.getUntrackedParameter<int>("DeadCellMonitor_checkNevents",checkNevents_);
  // increases rate at which neverpresent tests are run
  deadmon_neverpresent_prescale_     = ps.getUntrackedParameter<int>("DeadCellMonitor_neverpresent_prescale",1);  

  // Set which dead cell checks will be performed
  /* Dead cells can be defined in three ways:
     1)  never present -- digi is never present in run
     2)  occupancy -- digi is absent for (checkNevents_) consecutive events
     3)  energy -- cell is present, but rechit energy is never above threshold value
  */
  deadmon_test_neverpresent_           = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_neverpresent",true);
  deadmon_test_occupancy_         = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_occupancy", true);
  deadmon_test_energy_            = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_energy", true);

  // rechit_occupancy duplicates digi occupancy -- ignore it
  //deadmon_test_rechit_occupancy_  = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_rechit_occupancy", true);

  // Old tests (in version 1.55 and earlier) checked pedestals, compared energies to neighbors
  // Are these ever going to be useful?  If so, we could re-enable them.
  //deadmon_test_pedestal_          = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_pedestal",  true);
  //deadmon_test_neighbor_          = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_neighbor",  true);
  
  deadmon_minErrorFlag_ = ps.getUntrackedParameter<double>("DeadCellMonitor_minErrorFlag",0.0);

  // rechit energy test -- cell must be below threshold value for a number of consecutive events to be considered dead
  energyThreshold_       = ps.getUntrackedParameter<double>("DeadCellMonitor_energyThreshold",                  0);
  HBenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HB_energyThreshold",energyThreshold_);
  HEenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HE_energyThreshold",energyThreshold_);
  HOenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HO_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HF_energyThreshold",energyThreshold_);
  ZDCenergyThreshold_    = ps.getUntrackedParameter<double>("DeadCellMonitor_ZDC_energyThreshold",           -999);

  // neighboring-cell tests -- parameters no longer used

  // Set initial event # to 0
  ievt_=0;

  zeroCounters(true);
  
  // Set up histograms
  if (m_dbe)
    {
      if (fVerbosity>1)
	std::cout <<"<HcalDeadCellMonitor::setup>  Setting up histograms"<<std::endl;

      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("Dead Cell Task Event Number");
      meEVT_->Fill(ievt_);

      // Create problem cell plots
      // Overall plot gets an initial " " in its name
      ProblemDeadCells=m_dbe->book2D(" ProblemDeadCells",
				     " Problem Dead Cell Rate for all HCAL",
				     85,-42.5,42.5,
				     72,0.5,72.5);
      ProblemDeadCells->setAxisTitle("i#eta",1);
      ProblemDeadCells->setAxisTitle("i#phi",2);
      SetEtaPhiLabels(ProblemDeadCells);

      // 1D plots count number of bad cells
      NumberOfDeadCells=m_dbe->book1D("Problem_TotalDeadCells_HCAL",
				      "Total Number of Dead Hcal Cells",
				      148, bins_cellcount);
      NumberOfDeadCellsHB=m_dbe->book1D("Problem_TotalDeadCells_HB",
					"Total Number of Dead HB Cells",
					2593,-0.5,2592.5);
      NumberOfDeadCellsHE=m_dbe->book1D("Problem_TotalDeadCells_HE",
					"Total Number of Dead HE Cells",
					2593,-0.5,2592.5);
      NumberOfDeadCellsHO=m_dbe->book1D("Problem_TotalDeadCells_HO",
					"Total Number of Dead HO Cells",
					2161,-0.5,2160.5);
      NumberOfDeadCellsHF=m_dbe->book1D("Problem_TotalDeadCells_HF",
					"Total Number of Dead HF Cells",
					1729,-0.5,1728.5);
      NumberOfDeadCellsZDC=m_dbe->book1D("Problem_TotalDeadCells_ZDC",
					"Total Number of Dead ZDC Cells",
					 19,-0.5,18.5);

      // Overall Problem plot appears in main directory; plots by depth appear \in subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/problem_deadcells");

      // Create problem cell plots
      // Overall plot gets an initial " " in its name
      SetupEtaPhiHists(ProblemDeadCellsByDepth, " Problem Dead Cell Rate","");
      
      // Set up plots for each failure mode of dead cells
      stringstream units; // We'll need to set the titles individually, rather than passing units to SetupEtaPhiHists (since this also would affect the name of the histograms)
      stringstream name;
      if (deadmon_test_neverpresent_)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/dead_digi_never_present");
	  SetupEtaPhiHists(DigisNeverPresentByDepth,
			    "Dead Cells with No Digis Ever","");
	  // 1D plots count number of bad cells
	  NumberOfNeverPresentCells=m_dbe->book1D("Problem_TotalNeverPresentCells_HCAL",
						  "Total Number of Never-Present Hcal Cells",
						  148, bins_cellcount);
	  NumberOfNeverPresentCellsHB=m_dbe->book1D("Problem_NeverPresentCells_HB",
						    "Total Number of Never-Present HB Cells",
						    2593,-0.5,2592.5);
	  NumberOfNeverPresentCellsHE=m_dbe->book1D("Problem_NeverPresentCells_HE",
						    "Total Number of Never-Present HE Cells",
						    2593,-0.5,2592.5);
	  NumberOfNeverPresentCellsHO=m_dbe->book1D("Problem_NeverPresentCells_HO",
						    "Total Number of Never-Present HO Cells",
						    2161,-0.5,2160.5);
	  NumberOfNeverPresentCellsHF=m_dbe->book1D("Problem_NeverPresentCells_HF",
						    "Total Number of Never-Present HF Cells",
						    1729,-0.5,1728.5);
	  NumberOfNeverPresentCellsZDC=m_dbe->book1D("Problem_NeverPresentCells_ZDC",
						     "Total Number of Never-Present ZDC Cells",
						     19,-0.5,18.5);
	}
      if (deadmon_test_occupancy_)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/dead_digi_often_missing");
	  //units<<"("<<deadmon_checkNevents_<<" consec. events)";
	  name<<"Dead Cells with No Digis";
	  SetupEtaPhiHists(UnoccupiedDeadCellsByDepth,
			   name.str(),
			    "");
	  name.str("");
	  name<<"HB HE HF Depth 1 Dead Cells with No Digis for "<<deadmon_checkNevents_<<" Consecutive Events";
	  UnoccupiedDeadCellsByDepth.depth[0]->setTitle(name.str().c_str());

	  name.str("");
	  name<<"HB HE HF Depth 2 Dead Cells with No Digis for "<<deadmon_checkNevents_<<" Consecutive Events";
	  UnoccupiedDeadCellsByDepth.depth[1]->setTitle(name.str().c_str());

	  name.str("");
	  name<<"HE Depth 3 Dead Cells with No Digis for "<<deadmon_checkNevents_<<" Consecutive Events";
	  UnoccupiedDeadCellsByDepth.depth[2]->setTitle(name.str().c_str());

	  name.str("");
	  name<<"HO Depth 4 Dead Cells with No Digis for "<<deadmon_checkNevents_<<" Consecutive Events";
	  UnoccupiedDeadCellsByDepth.depth[3]->setTitle(name.str().c_str());
	  name.str("");

	  // 1D plots count number of bad cells
	  name<<"Total Number of Hcal Digis Unoccupied for "<<deadmon_checkNevents_<<" Consecutive Events";
	  NumberOfUnoccupiedCells=m_dbe->book1D("Problem_TotalUnoccupiedCells_HCAL",
						name.str(),
						148, bins_cellcount);
	  name.str("");
	  name<<"Total Number of HB Digis Unoccupied for "<<deadmon_checkNevents_<<" Consecutive Events";
	  NumberOfUnoccupiedCellsHB=m_dbe->book1D("Problem_UnoccupiedCells_HB",
						  name.str(),
						  2593,-0.5,2592.5);
	  name.str("");
	  name<<"Total Number of HE Digis Unoccupied for "<<deadmon_checkNevents_<<" Consecutive Events";
	  NumberOfUnoccupiedCellsHE=m_dbe->book1D("Problem_UnoccupiedCells_HE",
						  name.str(),
						  2593,-0.5,2592.5);
	  name.str("");
	  name<<"Total Number of HO Digis Unoccupied for "<<deadmon_checkNevents_<<" Consecutive Events";
	  NumberOfUnoccupiedCellsHO=m_dbe->book1D("Problem_UnoccupiedCells_HO",
						  name.str(),
						  2161,-0.5,2160.5);
	  name.str("");
	  name<<"Total Number of HF Digis Unoccupied for "<<deadmon_checkNevents_<<" Consecutive Events";
	  NumberOfUnoccupiedCellsHF=m_dbe->book1D("Problem_UnoccupiedCells_HF",
						  name.str(),
						  1729,-0.5,1728.5);
	  name.str("");
	  name<<"Total Number of ZDC Digis Unoccupied for "<<deadmon_checkNevents_<<" Consecutive Events";
	  NumberOfUnoccupiedCellsZDC=m_dbe->book1D("Problem_UnoccupiedCells_ZDC",
						   name.str(),
						   19,-0.5,18.5);
	}
      
      if (deadmon_test_energy_)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/dead_energytest");
	  SetupEtaPhiHists(BelowEnergyThresholdCellsByDepth,"Dead Cells Failing Energy Threshold Test","");
	  // set more descriptive titles for threshold plots
	  units.str("");
	  units<<"Dead Cells with Consistent Low Energy Depth 1 -- HB <"<<HBenergyThreshold_<<" GeV, HE < "<<HEenergyThreshold_<<", HF <"<<HFenergyThreshold_<<" GeV";
	  BelowEnergyThresholdCellsByDepth.depth[0]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Dead Cells with Consistent Low Energy Depth 2 -- HB <"<<HBenergyThreshold_<<" GeV, HE < "<<HEenergyThreshold_<<", HF <"<<HFenergyThreshold_<<" GeV";
	  BelowEnergyThresholdCellsByDepth.depth[1]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Dead Cells with Consistent Low Energy Depth 3 -- HE <"<<HEenergyThreshold_<<" GeV";
	  BelowEnergyThresholdCellsByDepth.depth[2]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Dead Cells with Consistent Low Energy Depth 4 -- HO <"<<HOenergyThreshold_<<" GeV";
	  BelowEnergyThresholdCellsByDepth.depth[3]->setTitle(units.str().c_str());
	  units.str("");


	  // 1D plots count number of bad cells
	  name<<"Total Number of Hcal RecHits with Consistent Low Energy";
	  NumberOfBelowEnergyCells=m_dbe->book1D("Problem_TotalBelowEnergyCells_HCAL",
						name.str(),
						148, bins_cellcount);
	  name.str("");
	  name<<"Total Number of HB RecHits with Consistent Low Energy < "<<HBenergyThreshold_<<" GeV";
	  NumberOfBelowEnergyCellsHB=m_dbe->book1D("Problem_BelowEnergyCells_HB",
						  name.str(),
						  2593,-0.5,2592.5);
	  name.str("");
	  name<<"Total Number of HE RecHits with Consistent Low Energy < "<<HEenergyThreshold_<<" GeV";
	  NumberOfBelowEnergyCellsHE=m_dbe->book1D("Problem_BelowEnergyCells_HE",
						  name.str(),
						  2593,-0.5,2592.5);
	  name.str("");
	  name<<"Total Number of HO RecHits with Consistent Low Energy < "<<HOenergyThreshold_<<" GeV";
	  NumberOfBelowEnergyCellsHO=m_dbe->book1D("Problem_BelowEnergyCells_HO",
						  name.str(),
						  2161,-0.5,2160.5);
	  name.str("");
	  name<<"Total Number of HF RecHits with Consistent Low Energy < "<<HFenergyThreshold_<<" GeV";
	  NumberOfBelowEnergyCellsHF=m_dbe->book1D("Problem_BelowEnergyCells_HF",
						  name.str(),
						  1729,-0.5,1728.5);
	  name.str("");
	  name<<"Total Number of ZDC RecHits with Consistent Low Energy < "<<ZDCenergyThreshold_<<" GeV";
	  NumberOfBelowEnergyCellsZDC=m_dbe->book1D("Problem_BelowEnergyCells_ZDC",
						   name.str(),
						   19,-0.5,18.5);
	}

    } // if (m_dbe)

  return;
} //void HcalDeadCellMonitor::setup(...)

/* --------------------------- */
void HcalDeadCellMonitor::setupNeighborParams(const edm::ParameterSet& ps,
					      neighborParams& N,
					      std::string type)
{
  // This is no longer used -- can remove at some point, after further testing
  // sets up parameters for neighboring-cell algorithm for each subdetector
  /*
  ostringstream myname;
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_deltaIphi";
  N.DeltaIphi = ps.getUntrackedParameter<int>(myname.str().c_str(),
					      defaultNeighborParams_.DeltaIphi);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_deltaIeta";
  N.DeltaIeta = ps.getUntrackedParameter<int>(myname.str().c_str(),
					      defaultNeighborParams_.DeltaIeta);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_deltaDepth";
  N.DeltaDepth = ps.getUntrackedParameter<int>(myname.str().c_str(),
					       defaultNeighborParams_.DeltaDepth);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_maxCellEnergy";
  N.maxCellEnergy = ps.getUntrackedParameter<double>(myname.str().c_str(),
						     defaultNeighborParams_.maxCellEnergy);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_minNeighborEnergy";
  N.minNeighborEnergy = ps.getUntrackedParameter<double>(myname.str().c_str(),
							 defaultNeighborParams_.minNeighborEnergy);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_minGoodNeighborFrac";
  N.minGoodNeighborFrac = ps.getUntrackedParameter<double>(myname.str().c_str(),
							   defaultNeighborParams_.minGoodNeighborFrac);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_maxEnergyFrac";
  N.maxEnergyFrac = ps.getUntrackedParameter<double>(myname.str().c_str(),
						     defaultNeighborParams_.maxEnergyFrac);
  */
  return;
} // void HcalDeadCellMonitor::setupNeighborParams

/* --------------------------- */

void HcalDeadCellMonitor::reset(){}  // reset function is empty for now

/* --------------------------- */



void HcalDeadCellMonitor::done(std::map<HcalDetId, unsigned int>& myqual)
{
  if (dump2database==0) 
    return;

  return;  // this is now done within the client, rather than the task (so that it doesn't get done multiple times when runing offline DQM)
  // Dump to ascii file for database -- now taken care of through ChannelStatus objects
  /*
  char buffer [1024];
  
  ofstream fOutput("hcalDeadCells.txt", ios::out);
  sprintf (buffer, "# %15s %15s %15s %15s %8s %10s\n", "eta", "phi", "dep", "det", "value", "DetId");
  fOutput << buffer;
  */

  int ieta,iphi;
  float binval;

  int subdet;
  std::string subdetname;
  if (fVerbosity>1)
    { 
      std::cout <<"<HcalDeadCellMonitor>  Summary of Dead Cells in Run: "<<std::endl;
      std::cout <<"(Error rate must be >= "<<deadmon_minErrorFlag_*100.<<"% )"<<std::endl;  
    }

  int etabins=0;
  int phibins=0;
  for (int d=0;d<4;++d)
    {
      etabins=ProblemDeadCellsByDepth.depth[d]->getNbinsX();
      phibins=ProblemDeadCellsByDepth.depth[d]->getNbinsY();
      for (int hist_eta=0;hist_eta<etabins;++hist_eta)
	{
	  for (int hist_phi=0;hist_phi<phibins;++hist_phi)
	    {
	      ieta=CalcIeta(hist_eta,d+1);
	      if (ieta==-9999) continue;
	      iphi=hist_phi;

	      binval=ProblemDeadCellsByDepth.depth[d]->getBinContent(hist_eta,hist_phi);
	  
	      // Set subdetector labels for output
	      if (d==0) // HB/HE/HF
		{
		  // correct for HF offset 
		  if (hist_eta< 13) // shift negative HF ieta values by +1
		    {
		      subdetname="HF";
		      subdet=4;
		    }
		  else if (hist_eta>71) 
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
	      else if (d==1)
		{
		  // correct for HF offset 
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
	  
	      if (fVerbosity>0 && binval>deadmon_minErrorFlag_)
		std::cout <<"Dead Cell "<<subdet<<"("<<ieta<<", "<<iphi<<", "<<d+1<<"):  "<<binval*100.<<"%"<<std::endl;
	      int value = 0;
	      if (binval>deadmon_minErrorFlag_)
		value=1;
	  
	      // Case 1:  did not find any quality bit info;
	      // Make new myqual entry that contains only dead cell info
	      if (myqual.find(myid)==myqual.end())
		{
		  myqual[myid]=(value<<BITSHIFT);  // deadcell shifted to bit 5
		}
	      // Case 2: found bit; combine dead cell info with other information
	      else
		{
		  int mask=(1<<BITSHIFT);
		  // Case 1a:  cell is dead; make the "OR" of dead cell bit with other info
		  if (value==1)
		    myqual[myid] |=mask;
		  // Case 2a:  cell is not dead; make the "AND" of other info with inverse (~mask) of dead cell info
		  else
		    myqual[myid] &=~mask;
		
		  if (value==1 && fVerbosity>1) std::cout <<"myqual = "<<std::hex<<myqual[myid]<<std::dec<<"  MASK = "<<std::hex<<mask<<std::dec<<std::endl;
		}
	      /*
		sprintf(buffer, "  %15i %15i %15i %15s %8X %10X \n",ieta,iphi,d+1,subdetname,(value<<BITSHIFT),int(myid.rawId()));
		fOutput<<buffer;
	      */
	  
	    } // for (int hist_phi=1;...)
	} // for (int hist_eta=1;...)
    } // for (int d=0;d<4;++d)
  //fOutput.close();
  
  return;

} // void HcalDeadCellMonitor::done()



/* --------------------------------- */

void HcalDeadCellMonitor::clearME()
{
  // I don't think this function gets cleared any more.  
  // And need to add code to clear out subfolders as well?
  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
    }
  return;
} // void HcalDeadCellMonitor::clearME()

/* -------------------------------- */


void HcalDeadCellMonitor::processEvent(const HBHERecHitCollection& hbHits,
				       const HORecHitCollection& hoHits,
				       const HFRecHitCollection& hfHits,
				       //const ZDCRecHitCollection& zdcHits,
				       const HBHEDigiCollection& hbhedigi,
				       const HODigiCollection& hodigi,
				       const HFDigiCollection& hfdigi
				       //const ZDCDigiCollection& zdcdigi
				       )
{

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  ++ievt_;
  if (m_dbe) meEVT_->Fill(ievt_);

  // HBpresent_, HEpresent need to be determined within loop, since HBHE is a single collection
  HOpresent_ = (hodigi.size()>0||hoHits.size()>0);
  HFpresent_ = (hfdigi.size()>0||hfHits.size()>0);
  //ZDCpresent_ = (zdcdigi.size()>0 || zdcHits.size()>0);

  if (fVerbosity>1) std::cout <<"<HcalDeadCellMonitor::processEvent> Processing event..."<<std::endl;

  // Dummy fills
  NumberOfDeadCells->setBinContent(0,ievt_);
  NumberOfDeadCellsHB->setBinContent(0,ievt_);
  NumberOfDeadCellsHE->setBinContent(0,ievt_);
  NumberOfDeadCellsHO->setBinContent(0,ievt_);
  NumberOfDeadCellsHF->setBinContent(0,ievt_);
  NumberOfDeadCellsZDC->setBinContent(0,ievt_);

  // Do Digi-Based dead cell searches 

  if (deadmon_test_neverpresent_ || deadmon_test_occupancy_)
    {

      // Dummy fills
      for (unsigned int i=0;i<UnoccupiedDeadCellsByDepth.depth.size();++i)
	{
	  UnoccupiedDeadCellsByDepth.depth[i]->setBinContent(0,0,ievt_);
	  DigisNeverPresentByDepth.depth[i]->setBinContent(0,0,ievt_);
	}
      NumberOfNeverPresentCells->setBinContent(0,ievt_);
      NumberOfNeverPresentCellsHB->setBinContent(0,ievt_);
      NumberOfNeverPresentCellsHE->setBinContent(0,ievt_);
      NumberOfNeverPresentCellsHO->setBinContent(0,ievt_);
      NumberOfNeverPresentCellsHF->setBinContent(0,ievt_);
      NumberOfNeverPresentCellsZDC->setBinContent(0,ievt_);
      
      NumberOfUnoccupiedCells->setBinContent(0,ievt_);
      NumberOfUnoccupiedCellsHB->setBinContent(0,ievt_);
      NumberOfUnoccupiedCellsHE->setBinContent(0,ievt_);
      NumberOfUnoccupiedCellsHO->setBinContent(0,ievt_);
      NumberOfUnoccupiedCellsHF->setBinContent(0,ievt_);
      NumberOfUnoccupiedCellsZDC->setBinContent(0,ievt_);

      if (showTiming)
	{
	  cpu_timer.reset(); cpu_timer.start();
	}
      for (HBHEDigiCollection::const_iterator j=hbhedigi.begin();
	   j!=hbhedigi.end(); ++j)
	{
	  processEvent_HBHEdigi(j);
	}
      
      for (HODigiCollection::const_iterator j=hodigi.begin();
	   j!=hodigi.end(); ++j)
	{
	  processEvent_HOdigi(j);
	}
      
      for (HFDigiCollection::const_iterator j=hfdigi.begin();
	   j!=hfdigi.end(); ++j)
	{
	  processEvent_HFdigi(j);
	}
      /*
	for (ZDCDigiCollection::const_iterator j=zdcdigi.begin();
	j!=zdcdigi.end(); ++j)
	{
	processEvent_ZDCdigi(j);
	}
      */
      if (showTiming)
	{
	  cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor PROCESSEVENT_DIGI -> "<<cpu_timer.cpuTime()<<std::endl;
	}
    } // if (deadmon_test_neverpresent || ...)
  
  // Search for "dead" cells below a certain energy
  if (deadmon_test_energy_) 
    {
      if (showTiming)
	{
	  cpu_timer.reset(); cpu_timer.start();
	}

      // Dummy Fills
      for (unsigned int i=0;i<BelowEnergyThresholdCellsByDepth.depth.size();++i)
	{
	BelowEnergyThresholdCellsByDepth.depth[i]->setBinContent(0,0,ievt_);
	}
      NumberOfBelowEnergyCells->setBinContent(0,ievt_);
      NumberOfBelowEnergyCellsHB->setBinContent(0,ievt_);
      NumberOfBelowEnergyCellsHE->setBinContent(0,ievt_);
      NumberOfBelowEnergyCellsHO->setBinContent(0,ievt_);
      NumberOfBelowEnergyCellsHF->setBinContent(0,ievt_);
      NumberOfBelowEnergyCellsZDC->setBinContent(0,ievt_);

      for (HBHERecHitCollection::const_iterator j=hbHits.begin();
	   j!=hbHits.end(); ++j)
	{
	  processEvent_HBHERecHit(j);
	}
      
      for (HORecHitCollection::const_iterator k=hoHits.begin();
	   k!=hoHits.end(); ++k)
	{
	  processEvent_HORecHit(k);
	}
      
      for (HFRecHitCollection::const_iterator j=hfHits.begin();
	   j!=hfHits.end(); ++j)
	{
	  processEvent_HFRecHit(j);
	}
      /*
	for (ZDCRecHitCollection::const_iterator j=zdcHits.begin();
	j!=zdcHits.end(); ++j)
	{
	processEvent_ZDCRecHit(j);
	}
      */
      if (showTiming)
	{
	  cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor PROCESSEVENT_DIGI -> "<<cpu_timer.cpuTime()<<std::endl;
	}
    }
  
  // Fill problem cells every [checkNevents_]

  int scalefactor=deadmon_checkNevents_/deadmon_neverpresent_prescale_;
  if (ievt_>0 && ievt_%scalefactor==0)
    {
      if (deadmon_test_neverpresent_) fillNevents_neverpresent();
      if (deadmon_test_occupancy_) fillNevents_occupancy();
      if (deadmon_test_energy_) fillNevents_energy();
      fillNevents_problemCells();
      if (ievt_%deadmon_checkNevents_==0) zeroCounters(); // reset for next round of checks
    }

  return;
} // void HcalDeadCellMonitor::processEvent(...)

/* --------------------------------------- */

void HcalDeadCellMonitor::fillDeadHistosAtEndRun()
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

  /*
  if (deadmon_test_occupancy_ && ievt_%deadmon_checkNevents_>0) fillNevents_occupancy();
  if (deadmon_test_pedestal_  && ievt_%deadmon_checkNevents_ >0) fillNevents_pedestal();
  if (deadmon_test_neighbor_  && ievt_%deadmon_checkNevents_ >0) fillNevents_neighbor();
  if ((deadmon_test_energy_ || deadmon_test_rechit_occupancy_)    && ievt_%deadmon_checkNevents_   >0) fillNevents_energy();
  if (deadmon_test_occupancy_ || deadmon_test_pedestal_ || 
      deadmon_test_neighbor_  || deadmon_test_energy_)  
   {
     fillNevents_problemCells();
     FillUnphysicalHEHFBins(ProblemDeadCellsByDepth);
     FillUnphysicalHEHFBins(ProblemDeadCells);
   }
  */
} // fillDeadHistosAtEndOfRun()



/* --------------------------------------- */

// Digi-based dead cell checks

void HcalDeadCellMonitor::processEvent_HBHEdigi(HBHEDigiCollection::const_iterator j)
{
  // Simply check whether a digi is present.  If so, increment occupancy counter.

  const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
  int ieta=digi.id().ieta();
  int iphi=digi.id().iphi();
  int depth=digi.id().depth();
  if (!digi.id().validDetId(digi.id().subdet(),ieta,iphi,depth)) return;

  
  if ((HcalSubdetector)(digi.id().subdet())==HcalBarrel)
    {
      HBpresent_=true;
      if (!checkHB_)
	return;
    }
  else 
    {
      HEpresent_=true;
      if (!checkHE_)
	return;
    }
  ++occupancy[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1];
  present[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1]=true;
  return;
} //void HcalDeadCellMonitor::processEvent_HBHEdigi(HBHEDigiCollection::const_iterator j)


void HcalDeadCellMonitor::processEvent_HOdigi(HODigiCollection::const_iterator j)
{
  if (!checkHO_) return;
  const HODataFrame digi = (const HODataFrame)(*j);
  int ieta=digi.id().ieta();
  int iphi=digi.id().iphi();
  int depth=digi.id().depth();
  if (!digi.id().validDetId(digi.id().subdet(),ieta,iphi,depth)) return;
  ++occupancy[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1];
  //cout <<"HO "<<ieta<<", "<<iphi<<", "<<depth<<":  bin = "<<CalcEtaBin(digi.id().subdet(),ieta,depth)<<endl;
  present[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1]=true;
  return;
} //void HcalDeadCellMonitor::processEvent_HOdigi(HODigiCollection::const_iterator j)


void HcalDeadCellMonitor::processEvent_HFdigi(HFDigiCollection::const_iterator j)
{
  if (!checkHF_) return;
  const HFDataFrame digi = (const HFDataFrame)(*j);
  int ieta=digi.id().ieta();
  int iphi=digi.id().iphi();
  int depth=digi.id().depth();
  if (!digi.id().validDetId(digi.id().subdet(),ieta,iphi,depth)) return;
  ++occupancy[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1];
  present[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1]=true;
  return;
} //void HcalDeadCellMonitor::processEvent_HFdigi(HFDigiCollection::const_iterator j)

void HcalDeadCellMonitor::processEvent_ZDCdigi(ZDCDigiCollection::const_iterator j)
{
  if (!checkZDC_) return;
  return;
  // need to set up mapping of ZDC into eta/phi/depth space before counting bad entries
} //void HcalDeadCellMonitor::processEvent_ZDCdigi(ZDCDigiCollection::const_iterator j)



//RecHit-based dead cell checks

void HcalDeadCellMonitor::processEvent_HBHERecHit(HBHERecHitCollection::const_iterator HBHEiter)
{
  float en = HBHEiter->energy();
  HcalDetId id(HBHEiter->detid().rawId());
  int ieta = id.ieta();
  int iphi = id.iphi();
  int depth = id.depth();
  
  if (id.subdet()==HcalBarrel)
    {
      HBpresent_=true;
      if (!checkHB_) return;
      if (en>=HBenergyThreshold_)
	++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    }
  else
    {
      HEpresent_=true;
      if (!checkHE_)return;
      if (en>=HEenergyThreshold_)
	++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    }
  return;
} //void HcalDeadCellMonitor::processEvent_HBHERecHit(HBHERecHitCollection::const_iterator HOiter)


void HcalDeadCellMonitor::processEvent_HORecHit(HORecHitCollection::const_iterator HOiter)
{
  float en = HOiter->energy();
  HcalDetId id(HOiter->detid().rawId());
  int ieta = id.ieta();
  int iphi = id.iphi();
  int depth = id.depth();
  if (en>=HOenergyThreshold_)
    ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
  return;
} //void HcalDeadCellMonitor::processEvent_HORecHit(HORecHitCollection::const_iterator HOiter)

void HcalDeadCellMonitor::processEvent_HFRecHit(HFRecHitCollection::const_iterator HFiter)
{
  float en = HFiter->energy();
  HcalDetId id(HFiter->detid().rawId());
  int ieta = id.ieta();
  int iphi = id.iphi();
  int depth = id.depth();
  if (en>=HFenergyThreshold_)
    ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
  return;
} //void HcalDeadCellMonitor::processEvent_HFRecHit(HFRecHitCollection::const_iterator HFiter)

void HcalDeadCellMonitor::processEvent_ZDCRecHit(ZDCRecHitCollection::const_iterator ZDCiter)
{
  float en = ZDCiter->energy();
  HcalDetId id(ZDCiter->detid().rawId());
  int ieta = id.ieta();
  int iphi = id.iphi();
  int depth = id.depth();
  if (en>=ZDCenergyThreshold_)
	++aboveenergy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
  return;
} //void HcalDeadCellMonitor::processEvent_ZDCRecHit(HFRecHitCollection::const_iterator ZDCiter)



// fill histograms every N events
void HcalDeadCellMonitor::fillNevents_neverpresent(void)
{
  if (!deadmon_test_neverpresent_) return; // extra protection
   if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

   if (fVerbosity>0)
     std::cout <<"<HcalDeadCellMonitor::fillNevents_neverpresent> FILLING OCCUPANCY PLOTS"<<std::endl;

   int ieta=0;
   int iphi=0;

   int etabins=0;
   int phibins=0;

   for (unsigned int depth=0;depth<DigisNeverPresentByDepth.depth.size();++depth)
     { 
       etabins=DigisNeverPresentByDepth.depth[depth]->getNbinsX();
       phibins=DigisNeverPresentByDepth.depth[depth]->getNbinsY();

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
		   // Ignore subdetectors that weren't in run
		   if ((subdet==HcalBarrel && !HBpresent_) || 
		       (subdet==HcalEndcap &&!HEpresent_)  ||
		       (subdet==HcalOuter &&!HOpresent_)  || 
		       (subdet==HcalForward &&!HFpresent_))   continue;
		   // ignore subdetectors we explicitly mask off 
		   if ((!checkHB_ && subdet==HcalBarrel) ||
		       (!checkHE_ && subdet==HcalEndcap) ||
		       (!checkHO_ && subdet==HcalOuter) ||
		       (!checkHF_ && subdet==HcalForward))  continue;
		   if (subdet==HcalForward) // shift HcalForward ieta by 1 for filling purposes
		     ieta<0 ? ieta-- : ieta++;
		   		   
		   if (present[eta][phi][depth]==false)
		     {
		       if (fVerbosity>0) 
			 std::cout <<"DEAD CELL; NEVER PRESENT: subdet = "<<subdet<<", ieta = "<<ieta<<", iphi = "<<iphi<<" depth = "<<depth+1<<"  filling with "<<deadmon_checkNevents_/deadmon_neverpresent_prescale_<<std::endl;
		       // no digi was found for the N events; Fill cell as bad for all N events (N = deadmon_checkNevents_/prescale);
		       if (DigisNeverPresentByDepth.depth[depth])
			 {
			   DigisNeverPresentByDepth.depth[depth]->Fill(ieta,1.*iphi,deadmon_checkNevents_/deadmon_neverpresent_prescale_);
			 }
		     }
		   else  // digi found; this is no longer a dead cell -- erase it
		     if (DigisNeverPresentByDepth.depth[depth] && 
			 DigisNeverPresentByDepth.depth[depth]->getBinContent(eta+1,phi+1)!=0) 
		       {
			 DigisNeverPresentByDepth.depth[depth]->setBinContent(eta+1,phi+1,0);
		       }
		 } // subdet loop
	     } //phi loop
	 } // eta loop
     } // depth loop

   FillUnphysicalHEHFBins(DigisNeverPresentByDepth);
   return;
} // void HcalDeadCellMonitor::fillNevents_neverpresent(void)


void HcalDeadCellMonitor::fillNevents_occupancy(void)
{
  // Fill Histograms showing digi cells with no occupancy for the past checkNevents
  if (!deadmon_test_occupancy_) return; // extra protection here against calling histograms than don't exist
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalDeadCellMonitor::fillNevents_occupancy> FILLING OCCUPANCY PLOTS"<<std::endl;


  int ieta=0;
  int iphi=0;

  // Only run fills of histogram when ievt%checkNevents=0
  if ((ievt_%deadmon_checkNevents_)!=0)
    return;

  int etabins=0;
  int phibins=0;
  for (unsigned int depth=0;depth<UnoccupiedDeadCellsByDepth.depth.size();++depth)
    { 
      etabins=UnoccupiedDeadCellsByDepth.depth[depth]->getNbinsX();
      phibins=UnoccupiedDeadCellsByDepth.depth[depth]->getNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  for (int phi=0;phi<phibins;++phi)
	    {
	      iphi=phi+1;
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1);
		  if (ieta==-9999) continue;
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1))
		    continue;

		  // Ignore subdetectors that weren't in run
		  if ((subdet==HcalBarrel && !HBpresent_) || 
		      (subdet==HcalEndcap &&!HEpresent_)  ||
		      (subdet==HcalOuter &&!HOpresent_)  || 
		      (subdet==HcalForward &&!HFpresent_))   continue;
		  // ignore subdetectors we explicitly mask off 
		  if ((!checkHB_ && subdet==HcalBarrel) ||
		      (!checkHE_ && subdet==HcalEndcap) ||
		      (!checkHO_ && subdet==HcalOuter) ||
		      (!checkHF_ && subdet==HcalForward))  continue;
		  if (subdet==HcalForward) // shift HcalForward ieta
		    ieta<0 ? ieta-- : ieta++;
		  if (occupancy[eta][phi][depth]==0)
		    {
		      if (fVerbosity>0) 
			std::cout <<"DEAD CELL; NO OCCUPANCY: subdet = "<<subdet<<", eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth<<std::endl;
		      
		      // no digi was found for the N events; Fill cell as bad for all N events (N = deadmon_checkNevents_);
		      if (UnoccupiedDeadCellsByDepth.depth[depth]) UnoccupiedDeadCellsByDepth.depth[depth]->Fill(ieta,iphi,deadmon_checkNevents_);
		      
		    }
		} // for (int subdet=1;subdet<=4;++subdet)
	    } // for (int phi=0;...)
	} // for (int eta=0;...)
    } //for (int depth=1;...)
  FillUnphysicalHEHFBins(UnoccupiedDeadCellsByDepth);
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_OCCUPANCY -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;

} // void HcalDeadCellMonitor::fillNevents_occupancy(void)



/* ----------------------------------- */

void HcalDeadCellMonitor::fillNevents_energy(void)
{
  // Fill Histograms showing unoccupied rechits, or rec hits with low energy

  if (!deadmon_test_energy_) return;
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalDeadCellMonitor::fillNevents_energy> BELOW-ENERGY-THRESHOLD PLOTS"<<std::endl;

  int ieta=0;
  int iphi=0;

  for (unsigned int h=0;h<BelowEnergyThresholdCellsByDepth.depth.size();++h)
    BelowEnergyThresholdCellsByDepth.depth[h]->setBinContent(0,0,ievt_);

  // Only run fills of histogram when ievt%checkNevents=0
  if ((ievt_%deadmon_checkNevents_)!=0)
    return;

  int etabins=0;
  int phibins=0;
  for (unsigned int depth=0;depth<BelowEnergyThresholdCellsByDepth.depth.size();++depth)
    { 
      etabins=BelowEnergyThresholdCellsByDepth.depth[depth]->getNbinsX();
      phibins=BelowEnergyThresholdCellsByDepth.depth[depth]->getNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  for (int phi=0;phi<phibins;++phi)
	    {
	      iphi=phi+1;
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1);
		  if (ieta==-9999) continue;
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1))
		    continue;
		  // Ignore subdetectors that weren't in run
                  if ((subdet==HcalBarrel && !HBpresent_) || 
		      (subdet==HcalEndcap &&!HEpresent_)  ||
		      (subdet==HcalOuter &&!HOpresent_)  ||
		      (subdet==HcalForward &&!HFpresent_))   continue;

		  if ((!checkHB_ && subdet==HcalBarrel) ||
		      (!checkHE_ && subdet==HcalEndcap) ||
		      (!checkHO_ && subdet==HcalOuter) ||
		      (!checkHF_ && subdet==HcalForward))  continue;
		  
		  if (subdet==HcalForward) // shift HcalForward ieta
		    {
		      ieta<0 ? ieta-- : ieta++;
		    }
		  if (aboveenergy[eta][phi][depth]>0) continue; // cell exceeded energy at least once, so it's not dead

		  if (fVerbosity>2) 
		    std::cout <<"DEAD CELL; BELOW ENERGY THRESHOLD = "<<subdet<<" eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<std::endl;
		  // Cell is below energy for all 'checkNevents_' consecutive events; update histogram
		  
		  if (BelowEnergyThresholdCellsByDepth.depth[depth]) BelowEnergyThresholdCellsByDepth.depth[depth]->Fill(ieta,iphi,deadmon_checkNevents_);
		} // for (int subdet=1;subdet<=4;++subdet)
	    } // for (unsigned int depth=1;depth<=4;++depth)
	} // for (int phi=0;...)
    } // for (int eta=0;...)

  FillUnphysicalHEHFBins(BelowEnergyThresholdCellsByDepth);
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_ENERGY -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalDeadCellMonitor::fillNevents_energy(void)



void HcalDeadCellMonitor::fillNevents_problemCells(void)
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalDeadCellMonitor::fillNevents_problemCells> FILLING PROBLEM CELL PLOTS"<<std::endl;

  int ieta=0;
  int iphi=0;

  double problemvalue=0;

  // Count problem cells in each subdetector
  unsigned int deadHB=0;
  unsigned int deadHE=0;
  unsigned int deadHO=0;
  unsigned int deadHF=0;
  unsigned int deadZDC=0;
  
  unsigned int neverpresentHB=0;
  unsigned int neverpresentHE=0;
  unsigned int neverpresentHO=0;
  unsigned int neverpresentHF=0;
  unsigned int neverpresentZDC=0;

  unsigned int unoccupiedHB=0;
  unsigned int unoccupiedHE=0;
  unsigned int unoccupiedHO=0;
  unsigned int unoccupiedHF=0;
  unsigned int unoccupiedZDC=0;
  
  unsigned int belowenergyHB=0;
  unsigned int belowenergyHE=0;
  unsigned int belowenergyHO=0;
  unsigned int belowenergyHF=0;
  unsigned int belowenergyZDC=0;

  int etabins=0;
  int phibins=0;
  for (int depth=0;depth<4;++depth)
    {
      etabins=ProblemDeadCellsByDepth.depth[depth]->getNbinsX();
      phibins=ProblemDeadCellsByDepth.depth[depth]->getNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  for (int phi=0;phi<phibins;++phi)
	    {
	      iphi=phi+1;

	      for (int subdet=1;subdet<=4;++subdet)
		{
		  ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1);
		  if (ieta==-9999) continue;
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1))
		    continue;
		  // Ignore subdetectors that weren't in run
                  if ((subdet==HcalBarrel && !HBpresent_) || 
		      (subdet==HcalEndcap &&!HEpresent_)  ||
		      (subdet==HcalOuter &&!HOpresent_)  ||
		      (subdet==HcalForward &&!HFpresent_))   continue;

		  if ((!checkHB_ && subdet==HcalBarrel) ||
		      (!checkHE_ && subdet==HcalEndcap) ||
		      (!checkHO_ && subdet==HcalOuter) ||
		      (!checkHF_ && subdet==HcalForward))  continue;

		  // now check which dead cell tests failed; increment counter if any failed
		  if ((deadmon_test_neverpresent_ && present[eta][phi][depth]==0) ||
		      (deadmon_test_occupancy_ && occupancy[eta][phi][depth]==0 && (ievt_%deadmon_checkNevents_==0)) ||
		      (deadmon_test_energy_ && aboveenergy[eta][phi][depth]==0  && (ievt_%deadmon_checkNevents_==0)))
		    {
		      if (subdet==HcalBarrel)       ++deadHB;
		      else if (subdet==HcalEndcap)  ++deadHE;
		      else if (subdet==HcalOuter)   ++deadHO;
		      else if (subdet==HcalForward) ++deadHF;
		    }
		  if ((deadmon_test_neverpresent_ && present[eta][phi][depth]==0))
		    {
		      if (subdet==HcalBarrel) ++neverpresentHB;
		      else if (subdet==HcalEndcap) ++neverpresentHE;
		      else if (subdet==HcalOuter) ++neverpresentHO;
		      else if (subdet==HcalForward) ++neverpresentHF;
		    }
		  if (deadmon_test_occupancy_ && occupancy[eta][phi][depth]==0 && (ievt_%deadmon_checkNevents_==0))
		    {
		      if (subdet==HcalBarrel) ++unoccupiedHB;
		      else if (subdet==HcalEndcap) ++unoccupiedHE;
		      else if (subdet==HcalOuter) ++unoccupiedHO;
		      else if (subdet==HcalForward) ++unoccupiedHF;
		    }
		  if (deadmon_test_energy_ && aboveenergy[eta][phi][depth]==0 && (ievt_%deadmon_checkNevents_==0))
		    {
		      if (subdet==HcalBarrel) ++belowenergyHB;
		      else if (subdet==HcalEndcap) ++belowenergyHE;
		      else if (subdet==HcalOuter) ++belowenergyHO;
		      else if (subdet==HcalForward) ++belowenergyHF;
		      // handle ZDC elsewhere -- in its own loop?
		    }
		} // subdet loop
	    } // phi loop
	} //eta loop
    } // depth loop


 // Fill with number of problem cells found on this pass
  if (ievt_%deadmon_checkNevents_==0)
    {
      NumberOfDeadCellsHB->Fill(deadHB,deadmon_checkNevents_);
      NumberOfDeadCellsHE->Fill(deadHE,deadmon_checkNevents_);
      NumberOfDeadCellsHO->Fill(deadHO,deadmon_checkNevents_);
      NumberOfDeadCellsHF->Fill(deadHF,deadmon_checkNevents_);
      NumberOfDeadCellsZDC->Fill(deadZDC,deadmon_checkNevents_);
      NumberOfDeadCells->Fill(deadHB+deadHE+deadHO+deadHF+deadZDC,deadmon_checkNevents_);

      NumberOfUnoccupiedCellsHE->Fill(unoccupiedHB,deadmon_checkNevents_);
      NumberOfUnoccupiedCellsHE->Fill(unoccupiedHE,deadmon_checkNevents_);
      NumberOfUnoccupiedCellsHO->Fill(unoccupiedHO,deadmon_checkNevents_);
      NumberOfUnoccupiedCellsHF->Fill(unoccupiedHF,deadmon_checkNevents_);
      NumberOfUnoccupiedCellsZDC->Fill(unoccupiedZDC,deadmon_checkNevents_);
      NumberOfUnoccupiedCells->Fill(unoccupiedHB+unoccupiedHE+unoccupiedHO+unoccupiedHF+unoccupiedZDC,deadmon_checkNevents_);
      
      NumberOfBelowEnergyCellsHB->Fill(belowenergyHB,deadmon_checkNevents_);
      NumberOfBelowEnergyCellsHE->Fill(belowenergyHE,deadmon_checkNevents_);
      NumberOfBelowEnergyCellsHO->Fill(belowenergyHO,deadmon_checkNevents_);
      NumberOfBelowEnergyCellsHF->Fill(belowenergyHF,deadmon_checkNevents_);
      NumberOfBelowEnergyCellsZDC->Fill(belowenergyZDC,deadmon_checkNevents_);
      NumberOfBelowEnergyCells->Fill(belowenergyHB+belowenergyHE+belowenergyHO+belowenergyHF+belowenergyZDC,deadmon_checkNevents_);
    }

  // Neverpresent cell algorithm gets called more often; fill with smaller value
  NumberOfNeverPresentCellsHB->Fill(neverpresentHB,deadmon_checkNevents_/deadmon_neverpresent_prescale_);
  NumberOfNeverPresentCellsHE->Fill(neverpresentHE,deadmon_checkNevents_/deadmon_neverpresent_prescale_);
  NumberOfNeverPresentCellsHO->Fill(neverpresentHO,deadmon_checkNevents_/deadmon_neverpresent_prescale_);
  NumberOfNeverPresentCellsHF->Fill(neverpresentHF,deadmon_checkNevents_/deadmon_neverpresent_prescale_);
  NumberOfNeverPresentCellsZDC->Fill(neverpresentZDC,deadmon_checkNevents_/deadmon_neverpresent_prescale_);
  NumberOfNeverPresentCells->Fill(neverpresentHB+neverpresentHE+neverpresentHO+neverpresentHF+neverpresentZDC,deadmon_checkNevents_/deadmon_neverpresent_prescale_);

  ProblemDeadCells->Reset();

  etabins=0;
  phibins=0;
  int zside=0;
  for (unsigned int d=0;d<ProblemDeadCellsByDepth.depth.size();++d)
    {
      ProblemDeadCellsByDepth.depth[d]->Reset();
      ProblemDeadCellsByDepth.depth[d]->setBinContent(0,0,ievt_);
      etabins=ProblemDeadCellsByDepth.depth[d]->getNbinsX();
      phibins=ProblemDeadCellsByDepth.depth[d]->getNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  ieta=CalcIeta(eta,d+1);
	  for (int phi=0;phi<phibins;++phi)
	    {
	      problemvalue=0;
	      if (deadmon_test_neverpresent_)
		{
		  problemvalue+=DigisNeverPresentByDepth.depth[d]->getBinContent(eta+1,phi+1);
		}
	      if (deadmon_test_occupancy_)
		{
		  problemvalue+=UnoccupiedDeadCellsByDepth.depth[d]->getBinContent(eta+1,phi+1);
		}
	      if (deadmon_test_energy_)
		{
		  problemvalue+=BelowEnergyThresholdCellsByDepth.depth[d]->getBinContent(eta+1,phi+1);
		}
	      if (problemvalue==0) continue;
	      problemvalue = min((double)ievt_,problemvalue);
	      iphi=phi+1;
	      
	      zside=0;
	      
 	      if (ieta==-9999) continue;
	      if (d<2) // shift ieta down for HF in first two depths
		{
		  if (isHF(eta,d+1))
		    ieta<0 ? zside = -1 : zside = 1;
		}
	      ProblemDeadCellsByDepth.depth[d]->Fill(ieta+zside,iphi,problemvalue);
	      ProblemDeadCells->Fill(ieta+zside,iphi,problemvalue);
	    } // loop on phi
	} // loop on eta
    } // loop on depth
  
  etabins=ProblemDeadCells->getNbinsX();
  phibins=ProblemDeadCells->getNbinsY();
  for (int eta=0;eta<etabins;++eta)
    {
      for (int phi=0;phi<phibins;++phi)
	{
	  if (ProblemDeadCells->getBinContent(eta+1,phi+1)>(double)ievt_)
	    ProblemDeadCells->setBinContent(eta+1,phi+1,(double)ievt_);
	}
    }
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_PROBLEMCELLS -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalDeadCellMonitor::fillNevents_problemCells(void)


void HcalDeadCellMonitor::zeroCounters(bool resetpresent)
{

  // zero all counters

  // 2D histogram counters
  for (unsigned int i=0;i<85;++i)
    {
      for (unsigned int j=0;j<72;++j)
	{
	  for (unsigned int k=0;k<4;++k)
	    {
	      if (resetpresent) present[i][j][k]=false; // keeps track of whether digi was ever present
	      occupancy[i][j][k]=0; // counts occupancy in last (checkNevents) events
	      aboveenergy[i][j][k]=0; // counts instances of cell above threshold energy in last (checkNevents)
	    }
	}
    }

  return;
} // void HcalDeadCellMonitor::zeroCounters(bool resetpresent)

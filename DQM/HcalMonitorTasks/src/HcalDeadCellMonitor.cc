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
  
  baseFolder_ = rootFolder_+"DeadCellMonitor_Hcal";
  if (fVerbosity>0)
    std::cout <<"<HcalDeadCellMonitor::setup>  Setting up histograms"<<std::endl;

  // Assume subdetectors not present until shown otherwise
  HBpresent_ =false;
  HEpresent_ =false;
  HOpresent_ =false;
  HFpresent_ =false;

  // Dead Cell Monitor - specific cfg variables

  if (fVerbosity>1)
    std::cout <<"<HcalDeadCellMonitor::setup>  Getting variable values from cfg files"<<std::endl;

  // deadmon_makeDiagnostics_ will take on base task value unless otherwise specified
  deadmon_makeDiagnostics_ = ps.getUntrackedParameter<bool>("DeadCellMonitor_makeDiagnosticPlots",makeDiagnostics);
  
  // Set checkNevents values
  deadmon_checkNevents_ = ps.getUntrackedParameter<int>("DeadCellMonitor_checkNevents",checkNevents_);
  deadmon_prescale_     = ps.getUntrackedParameter<int>("DeadCellMonitor_prescale",10); // energy, persistent occupancy checks not performed as often as basic 'never present' check

  // Set which dead cell checks will be performed
  /* Dead cells can be defined in three ways:
     1)  never present -- digi is never present in run
     2)  occupancy -- digi is absent for (checkNevents_) consecutive events
     3)  energy -- cell is present, but rechit energy is never above threshold value
  */
  
  deadmon_test_occupancy_         = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_occupancy", false);
  deadmon_test_energy_            = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_energy", false);

  // Minimum error rate to be considered problematic
  deadmon_minErrorFlag_ = ps.getUntrackedParameter<double>("DeadCellMonitor_minErrorFlag",0.05);

  // rechit energy test -- cell must be below threshold value for a number of consecutive events to be considered dead
  energyThreshold_       = ps.getUntrackedParameter<double>("DeadCellMonitor_energyThreshold",                  0);
  HBenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HB_energyThreshold",energyThreshold_);
  HEenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HE_energyThreshold",energyThreshold_);
  HOenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HO_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HF_energyThreshold",energyThreshold_);
 
  zeroCounters(true);
  
  // Set up histograms
  if (m_dbe)
    {
      if (fVerbosity>1)
	std::cout <<"<HcalDeadCellMonitor::setup>  Setting up histograms"<<std::endl;

      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("Dead Cell Task Event Number");
      meEVT_->Fill(ievt_);
      meTOTALEVT_ = m_dbe->bookInt("Dead Cell Total Events Processed");
      meTOTALEVT_->Fill(tevt_);

      // Create problem cell plots
      // Overall plot gets an initial " " in its name
      /*
      ProblemCells=m_dbe->book2D(" ProblemDeadCells",
				     " Problem Dead Cell Rate for all HCAL;i#eta;i#phi",
				     85,-42.5,42.5,
				     72,0.5,72.5);
      SetEtaPhiLabels(ProblemCells);
      */
      // 1D plots count number of bad cells vs. luminoisty block
      ProblemsVsLB=m_dbe->bookProfile("TotalDeadCells_HCAL_vs_LS",
				      "Total Number of Dead Hcal Cells vs lumi section;Lumi Section;Dead Cells", 
				      Nlumiblocks_,0.5,Nlumiblocks_+0.5,
				      100,0,10000);
      ProblemsVsLB_HB=m_dbe->bookProfile("TotalDeadCells_HB_vs_LS",
					 "Total Number of Dead HB Cells vs lumi section;Lumi Section;Dead Cells",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,
					 100,0,10000);
      ProblemsVsLB_HE=m_dbe->bookProfile("TotalDeadCells_HE_vs_LS",
					 "Total Number of Dead HE Cells vs lumi section;Lumi Section;Dead Cells",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      ProblemsVsLB_HO=m_dbe->bookProfile("TotalDeadCells_HO_vs_LS",
					 "Total Number of Dead HO Cells vs lumi section;Lumi Section;Dead Cells",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      ProblemsVsLB_HF=m_dbe->bookProfile("TotalDeadCells_HF_vs_LS",
					 "Total Number of Dead HF Cells vs lumi section;Lumi Section;Dead Cells",
					 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      
      // Overall Problem plot appears in main directory; plots by depth appear in subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/problem_deadcells");

      // Create problem cell plots
      // Overall plot gets an initial " " in its name
      //SetupEtaPhiHists(ProblemCellsByDepth, " Problem Dead Cell Rate","");
      
      // Set up plots for each failure mode of dead cells
      stringstream units; // We'll need to set the titles individually, rather than passing units to SetupEtaPhiHists (since this also would affect the name of the histograms)
      stringstream name;

      // Never-present test will always be called, by definition of dead cell

      m_dbe->setCurrentFolder(baseFolder_+"/dead_digi_never_present");
      SetupEtaPhiHists(DigiPresentByDepth,
		       "Digi Present At Least Once","");
      // 1D plots count number of bad cells
      NumberOfNeverPresentCells=m_dbe->bookProfile("Problem_NeverPresentCells_HCAL_vs_LS",
						   "Total Number of Never-Present Hcal Cells vs LS;Lumi Section;Dead Cells",
						   Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      
      NumberOfNeverPresentCellsHB=m_dbe->bookProfile("Problem_NeverPresentCells_HB_vs_LS",
						     "Total Number of Never-Present HB Cells vs LS;Lumi Section;Dead Cells",
						     Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      
      NumberOfNeverPresentCellsHE=m_dbe->bookProfile("Problem_NeverPresentCells_HE_vs_LS",
						    "Total Number of Never-Present HE Cells vs LS;Lumi Section;Dead Cells",
						     Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      
      NumberOfNeverPresentCellsHO=m_dbe->bookProfile("Problem_NeverPresentCells_HO_vs_LS",
						     "Total Number of Never-Present HO Cells vs LS;Lumi Section;Dead Cells",
						     Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      
      NumberOfNeverPresentCellsHF=m_dbe->bookProfile("Problem_NeverPresentCells_HF_vs_LS",
						     "Total Number of Never-Present HF Cells vs LS;Lumi Section;Dead Cells",
						     Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);

      for (unsigned int depth=0;depth<DigiPresentByDepth.depth.size();++depth)
	DigiPresentByDepth.depth[depth]->Reset();
      
      FillUnphysicalHEHFBins(DigiPresentByDepth);

      if (deadmon_test_occupancy_)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/dead_digi_often_missing");
	  //units<<"("<<deadmon_checkNevents_<<" consec. events)";
	  name<<"Dead Cells with No Digis";
	  SetupEtaPhiHists(UnoccupiedDeadCellsByDepth,
			   name.str(),
			    "");
	  name.str("");
	  name<<"HB HE HF Depth 1 Dead Cells with No Digis for "<<(deadmon_checkNevents_*deadmon_prescale_)<<" Consecutive Events";
	  UnoccupiedDeadCellsByDepth.depth[0]->setTitle(name.str().c_str());

	  name.str("");
	  name<<"HB HE HF Depth 2 Dead Cells with No Digis for "<<(deadmon_checkNevents_*deadmon_prescale_)<<" Consecutive Events";
	  UnoccupiedDeadCellsByDepth.depth[1]->setTitle(name.str().c_str());

	  name.str("");
	  name<<"HE Depth 3 Dead Cells with No Digis for "<<(deadmon_checkNevents_*deadmon_prescale_)<<" Consecutive Events";
	  UnoccupiedDeadCellsByDepth.depth[2]->setTitle(name.str().c_str());

	  name.str("");
	  name<<"HO Depth 4 Dead Cells with No Digis for "<<(deadmon_checkNevents_*deadmon_prescale_)<<" Consecutive Events";
	  UnoccupiedDeadCellsByDepth.depth[3]->setTitle(name.str().c_str());
	  name.str("");

	  // 1D plots count number of bad cells
	  name<<"Total Number of Hcal Digis Unoccupied for "<<(deadmon_checkNevents_*deadmon_prescale_)<<" Consecutive Events vs LS;Lumi Section; Dead Cells";
	  NumberOfUnoccupiedCells=m_dbe->bookProfile("Problem_UnoccupiedCells_HCAL_vs_LS",
						name.str(),
						Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HB Digis Unoccupied for "<<(deadmon_checkNevents_*deadmon_prescale_)<<" Consecutive Events vs LS;Lumi Section; Dead Cells";
	  NumberOfUnoccupiedCellsHB=m_dbe->bookProfile("Problem_UnoccupiedCells_HB_vs_LS",
						  name.str(),
						  Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HE Digis Unoccupied for "<<(deadmon_checkNevents_*deadmon_prescale_)<<" Consecutive Events vs LS;Lumi Section; Dead Cells";
	  NumberOfUnoccupiedCellsHE=m_dbe->bookProfile("Problem_UnoccupiedCells_HE_vs_LS",
						  name.str(),
						  Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HO Digis Unoccupied for "<<(deadmon_checkNevents_*deadmon_prescale_)<<" Consecutive Events vs LS;Lumi Section; Dead Cells";
	  NumberOfUnoccupiedCellsHO=m_dbe->bookProfile("Problem_UnoccupiedCells_HO_vs_LS",
						  name.str(),
						  Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HF Digis Unoccupied for "<<(deadmon_checkNevents_*deadmon_prescale_)<<" Consecutive Events vs LS;Lumi Section; Dead Cells";
	  NumberOfUnoccupiedCellsHF=m_dbe->bookProfile("Problem_UnoccupiedCells_HF_vs_LS",
						  name.str(),
						  Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	}
      
      if (deadmon_test_energy_)
	{
	  // test 1:  energy never above threshold
	  m_dbe->setCurrentFolder(baseFolder_+"/dead_energy_neverpresent");
	  SetupEtaPhiHists(EnergyPresentByDepth,"RecHit Above Threshold At Least Once","");
	  // set more descriptive titles for threshold plots
	  units.str("");
	  units<<"Cells Above Energy Threshold At Least Once: Depth 1 -- HB >="<<HBenergyThreshold_<<" GeV, HE >= "<<HEenergyThreshold_<<", HF >="<<HFenergyThreshold_<<" GeV";
	  EnergyPresentByDepth.depth[0]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Cells Above Energy Threshold At Least Once: Depth 2 -- HB >="<<HBenergyThreshold_<<" GeV, HE >= "<<HEenergyThreshold_<<", HF >="<<HFenergyThreshold_<<" GeV";
	  EnergyPresentByDepth.depth[1]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Cells Above Energy Threshold At Least Once: Depth 3 -- HE >="<<HEenergyThreshold_<<" GeV";
	  EnergyPresentByDepth.depth[2]->setTitle(units.str().c_str());
	  units.str("");
	  units<<"Cells Above Energy Threshold At Least Once: Depth 4 -- HO >="<<HOenergyThreshold_<<" GeV";
	  EnergyPresentByDepth.depth[3]->setTitle(units.str().c_str());
	  units.str("");

	  // 1D plots count number of bad cells
	  name<<"Total Number of Hcal RecHits with Consistent Low Energy;Lumi Section;Dead Cells";
	  NumberOfEnergyNeverPresentCells=m_dbe->bookProfile("Problem_EnergyNeverPresentCells_HCAL_vs_LS",
						      name.str(),
						      Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HB RecHits with Energy Never >= "<<HBenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
	  NumberOfEnergyNeverPresentCellsHB=m_dbe->bookProfile("Problem_EnergyNeverPresentCells_HB_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HE RecHits with Energy Never >= "<<HEenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
	  NumberOfEnergyNeverPresentCellsHE=m_dbe->bookProfile("Problem_EnergyNeverPresentCells_HE_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HO RecHits with Energy Never >= "<<HOenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
	  NumberOfEnergyNeverPresentCellsHO=m_dbe->bookProfile("Problem_EnergyNeverPresentCells_HO_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HF RecHits with Energy Never >= "<<HFenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
	  NumberOfEnergyNeverPresentCellsHF=m_dbe->bookProfile("Problem_EnergyNeverPresentCells_HF_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);


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
	  name<<"Total Number of Hcal RecHits with Consistent Low Energy;Lumi Section;Dead Cells";
	  NumberOfBelowEnergyCells=m_dbe->bookProfile("Problem_BelowEnergyCells_HCAL_vs_LS",
						      name.str(),
						      Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HB RecHits with Consistent Low Energy < "<<HBenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
	  NumberOfBelowEnergyCellsHB=m_dbe->bookProfile("Problem_BelowEnergyCells_HB_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HE RecHits with Consistent Low Energy < "<<HEenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
	  NumberOfBelowEnergyCellsHE=m_dbe->bookProfile("Problem_BelowEnergyCells_HE_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HO RecHits with Consistent Low Energy < "<<HOenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
	  NumberOfBelowEnergyCellsHO=m_dbe->bookProfile("Problem_BelowEnergyCells_HO_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	  name.str("");
	  name<<"Total Number of HF RecHits with Consistent Low Energy < "<<HFenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
	  NumberOfBelowEnergyCellsHF=m_dbe->bookProfile("Problem_BelowEnergyCells_HF_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
	}

    } // if (m_dbe)

  return;
} //void HcalDeadCellMonitor::setup(...)


void HcalDeadCellMonitor::reset(){}  // reset function is empty for now

/* --------------------------- */


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

  // increment counters
  HcalBaseMonitor::processEvent();

  // HBpresent_, HEpresent need to be determined within loop, since HBHE is a single collection
  HOpresent_ = (hodigi.size()>0||hoHits.size()>0);
  HFpresent_ = (hfdigi.size()>0||hfHits.size()>0);
  //ZDCpresent_ = (zdcdigi.size()>0 || zdcHits.size()>0);

  if (fVerbosity>1) std::cout <<"<HcalDeadCellMonitor::processEvent> Processing event..."<<std::endl;


  // Do Digi-Based dead cell searches 


  // Dummy fills
  for (unsigned int i=0;i<DigiPresentByDepth.depth.size();++i)
    DigiPresentByDepth.depth[i]->setBinContent(0,0,ievt_); 
    
  NumberOfNeverPresentCells->update();;
  NumberOfNeverPresentCellsHB->update();
  NumberOfNeverPresentCellsHE->update();
  NumberOfNeverPresentCellsHO->update();
  NumberOfNeverPresentCellsHF->update();
  
  if (deadmon_test_occupancy_)
    {
      
      for (unsigned int i=0;i<UnoccupiedDeadCellsByDepth.depth.size();++i)
	UnoccupiedDeadCellsByDepth.depth[i]->setBinContent(0,0,ievt_);
      
      NumberOfUnoccupiedCells->update();
      NumberOfUnoccupiedCellsHB->update();
      NumberOfUnoccupiedCellsHE->update();
      NumberOfUnoccupiedCellsHO->update();
      NumberOfUnoccupiedCellsHF->update();
    }
  
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  if (checkHB_ || checkHE_)
    {
      for (HBHEDigiCollection::const_iterator j=hbhedigi.begin();
	   j!=hbhedigi.end(); ++j)
	{
	  const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
	  processEvent_HBHEdigi(digi);
	}
    }
  
  if (checkHO_)
    {
      for (HODigiCollection::const_iterator j=hodigi.begin();
	   j!=hodigi.end(); ++j)
	{
	  const HODataFrame digi = (const HODataFrame)(*j);
	  process_Digi(digi);
	}
    }
  
  if (checkHF_)
    {
      for (HFDigiCollection::const_iterator j=hfdigi.begin();
	   j!=hfdigi.end(); ++j)
	    {
	      const HFDataFrame digi = (const HFDataFrame)(*j);	 
	      process_Digi(digi);
	    }
    }
  FillUnphysicalHEHFBins(DigiPresentByDepth);
  
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor PROCESSEVENT_DIGI -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  
  // Search for "dead" cells below a certain energy
  if (deadmon_test_energy_) 
    {
      if (showTiming)
	{
	  cpu_timer.reset(); cpu_timer.start();
	}

      // Normalization Fill
      for (unsigned int i=0;i<BelowEnergyThresholdCellsByDepth.depth.size();++i)
	BelowEnergyThresholdCellsByDepth.depth[i]->setBinContent(0,0,ievt_);

      NumberOfBelowEnergyCells->update();
      NumberOfBelowEnergyCellsHB->update();
      NumberOfBelowEnergyCellsHE->update();
      NumberOfBelowEnergyCellsHO->update();
      NumberOfBelowEnergyCellsHF->update();

      if (checkHB_ || checkHE_)
	{
	  for (HBHERecHitCollection::const_iterator j=hbHits.begin();
	       j!=hbHits.end(); ++j)
	    {
	      process_RecHit(j);
	    }
	}
      if (checkHO_)
	{
	  for (HORecHitCollection::const_iterator k=hoHits.begin();
	       k!=hoHits.end(); ++k)
	    {
	      process_RecHit(k);
	    }
	}
      if (checkHF_)
	{
	  for (HFRecHitCollection::const_iterator j=hfHits.begin();
	       j!=hfHits.end(); ++j)
	    {
	      process_RecHit(j);
	    }
	}

      if (showTiming)
	{
	  cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor PROCESSEVENT_RECHIT -> "<<cpu_timer.cpuTime()<<std::endl;
	}
    }
  
  // Fill problem cells every [checkNevents_]

  if (ievt_>0 && ievt_%deadmon_checkNevents_==0)
    {
      if (deadmon_test_occupancy_) fillNevents_occupancy(deadmon_checkNevents_);
      if (deadmon_test_energy_) fillNevents_energy(deadmon_checkNevents_);
      fillNevents_problemCells(deadmon_checkNevents_);
      zeroCounters(); // reset for next round of checks
    }

  return;
} // void HcalDeadCellMonitor::processEvent(...)

/* --------------------------------------- */

void HcalDeadCellMonitor::fillDeadHistosAtEndRun()
{
  // Fill histograms one last time at endRun call
  
  /* For now, only fill the 'never present' profile histograms.  
     If the total number of events taken is less than checkNevents,
     all we can consider is the number of never present channels.
     If total # of events is < checkNevents, though, don't fill the
     problem profile histograms, though -- not enough stats to determine this.
  */
  
  unsigned int neverpresentHB=0;
  unsigned int neverpresentHE=0;
  unsigned int neverpresentHO=0;
  unsigned int neverpresentHF=0;

  unsigned int energyneverpresentHB=0;
  unsigned int energyneverpresentHE=0;
  unsigned int energyneverpresentHO=0;
  unsigned int energyneverpresentHF=0;
  
  int etabins=0;
  int phibins=0;
  int ieta=0;
  for (unsigned int depth=0;depth<DigiPresentByDepth.depth.size();++depth)
    {
      if (DigiPresentByDepth.depth[depth]==0) continue;
      etabins=DigiPresentByDepth.depth[depth]->getNbinsX();
      phibins=DigiPresentByDepth.depth[depth]->getNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  ieta=CalcIeta(eta,depth);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  if (!validDetId((HcalSubdetector)subdet, ieta, phi+1, depth+1))
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
		  if (present[eta][phi][depth]==0)
		    {
		      if (subdet==HcalBarrel) ++neverpresentHB;
		      else if (subdet==HcalEndcap) ++neverpresentHE;
		      else if (subdet==HcalOuter) ++neverpresentHO;
		      else if (subdet==HcalForward) ++neverpresentHF;
		    }
		} // subdet loop
	    } // phi loop
	} //eta loop
    } // depth loop

  NumberOfNeverPresentCellsHB->Fill(lumiblock,neverpresentHB);
  NumberOfNeverPresentCellsHE->Fill(lumiblock,neverpresentHE);
  NumberOfNeverPresentCellsHO->Fill(lumiblock,neverpresentHO);
  NumberOfNeverPresentCellsHF->Fill(lumiblock,neverpresentHF);
  NumberOfNeverPresentCells->Fill(lumiblock,neverpresentHB+neverpresentHE+neverpresentHO+neverpresentHF);

  if (!deadmon_test_energy_) return;
  // Now look for rechits always below threshold energy
  for (unsigned int depth=0;depth<EnergyPresentByDepth.depth.size();++depth)
    {
      if (EnergyPresentByDepth.depth[depth]==0) continue;
      etabins=EnergyPresentByDepth.depth[depth]->getNbinsX();
      phibins=EnergyPresentByDepth.depth[depth]->getNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  ieta=CalcIeta(eta,depth);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  if (!validDetId((HcalSubdetector)subdet, ieta, phi+1, depth+1))
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
		  if (present_energy[eta][phi][depth]==0)
		    {
		      if (subdet==HcalBarrel) ++energyneverpresentHB;
		      else if (subdet==HcalEndcap) ++energyneverpresentHE;
		      else if (subdet==HcalOuter) ++energyneverpresentHO;
		      else if (subdet==HcalForward) ++energyneverpresentHF;
		    }
		} // subdet loop
	    } // phi loop
	} //eta loop
    } // depth loop

  NumberOfEnergyNeverPresentCellsHB->Fill(lumiblock,neverpresentHB);
  NumberOfEnergyNeverPresentCellsHE->Fill(lumiblock,neverpresentHE);
  NumberOfEnergyNeverPresentCellsHO->Fill(lumiblock,neverpresentHO);
  NumberOfEnergyNeverPresentCellsHF->Fill(lumiblock,neverpresentHF);
  NumberOfEnergyNeverPresentCells->Fill(lumiblock,neverpresentHB+neverpresentHE+neverpresentHO+neverpresentHF);

  return;

} // fillDeadHistosAtEndOfRun()



/* --------------------------------------- */

// Digi-based dead cell checks

void HcalDeadCellMonitor::processEvent_HBHEdigi(const HBHEDataFrame digi)
{
  // Simply check whether a digi is present.  If so, increment occupancy counter.
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
  process_Digi(digi);
  return;
} //void HcalDeadCellMonitor::processEvent_HBHEdigi(HBHEDigiCollection::const_iterator j)

template<class DIGI> 
void HcalDeadCellMonitor::process_Digi(DIGI& digi)
{
  // Remove the validate check as when we figure out how to access bad digis in digi monitor
  //if (!digi.validate()) return; // digi must be good to be counted
  int ieta=digi.id().ieta();
  int iphi=digi.id().iphi();
  int depth=digi.id().depth();

  // Fill occupancy counter
  ++occupancy[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1];

  // If previously-missing digi found, change boolean status and fill histogram
  if (present[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1]==false)
    {
      //cout <<"Found new digi at "<<ieta<<"  "<<iphi<<"  "<<depth<<"  subdet = "<<digi.id().subdet()<<endl;
      if (DigiPresentByDepth.depth[depth-1])
	DigiPresentByDepth.depth[depth-1]->setBinContent(CalcEtaBin(digi.id().subdet(),ieta,depth)+1,iphi,1);
      present[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1]=true;
    }
  return;
}

//RecHit-based dead cell checks

template<class RECHIT>
void HcalDeadCellMonitor::process_RecHit(RECHIT& rechit)
{
  float en = rechit->energy();
  HcalDetId id(rechit->detid().rawId());
  int ieta = id.ieta();
  int iphi = id.iphi();
  int depth = id.depth();
  
  if (id.subdet()==HcalBarrel)
    {
      HBpresent_=true;
      if (!checkHB_) return;
      if (en>=HBenergyThreshold_)
	{
	  ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	  present_energy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]=true;
	  if (EnergyPresentByDepth.depth[depth-1])
	    EnergyPresentByDepth.depth[depth-1]->setBinContent(CalcEtaBin(id.subdet(),ieta,depth)+1,iphi,1);
	}
    }
  else if (id.subdet()==HcalEndcap)
    {
      HEpresent_=true;
      if (!checkHE_) return;
      if (en>=HEenergyThreshold_)
	{
	++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	present_energy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]=true;
	if (EnergyPresentByDepth.depth[depth-1])
	  EnergyPresentByDepth.depth[depth-1]->setBinContent(CalcEtaBin(id.subdet(),ieta,depth)+1,iphi,1);
	}
    }
  else if (id.subdet()==HcalForward)
    {
      HFpresent_=true;
      if (en>=HFenergyThreshold_)
	{
	  ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	  present_energy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]=true;
	  if (EnergyPresentByDepth.depth[depth-1])
	    EnergyPresentByDepth.depth[depth-1]->setBinContent(CalcEtaBin(id.subdet(),ieta,depth)+1,iphi,1);
	}
	++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    }
  else if (id.subdet()==HcalOuter)
    {
      HOpresent_=true;
      if (en>=HOenergyThreshold_)
	{
	  ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	  present_energy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]=true;
	  if (EnergyPresentByDepth.depth[depth-1])
	    EnergyPresentByDepth.depth[depth-1]->setBinContent(CalcEtaBin(id.subdet(),ieta,depth)+1,iphi,1); 
	}
    }
}

void HcalDeadCellMonitor::fillNevents_occupancy(int checkN)
{
  // Fill Histograms showing digi cells with no occupancy for the past checkNevents
  if (!deadmon_test_occupancy_) return; // extra protection here against calling histograms than don't exist
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalDeadCellMonitor::fillNevents_occupancy> FILLING OCCUPANCY PLOTS"<<std::endl;

  // Only run fills of histogram when ievt%(checkN%deadmon_prescale_)=0
  if (ievt_%(checkN*deadmon_prescale_)!=0)
    return;

  int ieta=0;
  int iphi=0;

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
		  // Skip cells that are already marked as dead by the "neverpresent" test?
		  // Nope!  Gotta keep track of the overall fraction of unoccupied events, in case digi suddenly appears
		  //if (present[eta][phi][depth]==false) continue;
		  
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
		      // no digi was found for the N events; Fill cell as bad for all N events (N = checkN);
		      if (UnoccupiedDeadCellsByDepth.depth[depth]) UnoccupiedDeadCellsByDepth.depth[depth]->Fill(ieta,iphi,checkN*deadmon_prescale_);
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

} // void HcalDeadCellMonitor::fillNevents_occupancy(int checkN)



/* ----------------------------------- */

void HcalDeadCellMonitor::fillNevents_energy(int checkN)
{
  // Fill Histograms showing unoccupied rechits, or rec hits with low energy

  // This test is a bit pointless, unless the energy threshold is greater than the ZS threshold.
  // If we require that cells are always < thresh to be flagged by this test, and if 
  // thresh < ZS, then we will never catch any cells, since they'll show up as dead in the
  // neverpresent/occupancy test plots first.
  // Only exception is if something strange is going on between ZS ADC value an RecHit energy?

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

  // Only run fills of histogram when ievt%(checkN%deadmon_prescale_)=0
  if (ievt_%(checkN*deadmon_prescale_)!=0)
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
		  if (aboveenergy[eta][phi][depth]>0) continue; // cell exceeded energy at least once, so it's not dead

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
		  
		  // Don't think we want to do this -- need to keep track of overall energy occupancy
		  //if (deadmon_test_occupancy_ && occupancy[eta][phi][depth]==0) continue;  // don't mark cells as dead that are already caught by digi occupancy check
		  if (fVerbosity>2) 
		    std::cout <<"DEAD CELL; BELOW ENERGY THRESHOLD = "<<subdet<<" eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<std::endl;
		  // Cell is below energy for all 'checkNevents_' consecutive events; update histogram
		  
		  if (BelowEnergyThresholdCellsByDepth.depth[depth]) BelowEnergyThresholdCellsByDepth.depth[depth]->Fill(ieta,iphi,checkN);
		} // for (int subdet=1;subdet<=4;++subdet)
	    } // for (unsigned int depth=1;depth<=4;++depth)
	} // for (int phi=0;...)
    } // for (int eta=0;...)

  FillUnphysicalHEHFBins(EnergyPresentByDepth);
  FillUnphysicalHEHFBins(BelowEnergyThresholdCellsByDepth);
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_ENERGY -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalDeadCellMonitor::fillNevents_energy(void)



void HcalDeadCellMonitor::fillNevents_problemCells(int checkN)
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalDeadCellMonitor::fillNevents_problemCells> FILLING PROBLEM CELL PLOTS"<<std::endl;

  int ieta=0;
  int iphi=0;

  // Count problem cells in each subdetector

  NumBadHB=0;
  NumBadHE=0;
  NumBadHO=0;
  NumBadHF=0;
  
  unsigned int neverpresentHB=0;
  unsigned int neverpresentHE=0;
  unsigned int neverpresentHO=0;
  unsigned int neverpresentHF=0;
  
  unsigned int unoccupiedHB=0;
  unsigned int unoccupiedHE=0;
  unsigned int unoccupiedHO=0;
  unsigned int unoccupiedHF=0;
    
  unsigned int belowenergyHB=0;
  unsigned int belowenergyHE=0;
  unsigned int belowenergyHO=0;
  unsigned int belowenergyHF=0;
  
  unsigned int energyneverpresentHB=0;
  unsigned int energyneverpresentHE=0;
  unsigned int energyneverpresentHO=0;
  unsigned int energyneverpresentHF=0;
  

  int etabins=0;
  int phibins=0;
  for (unsigned int depth=0;depth<DigiPresentByDepth.depth.size();++depth)
    {

      etabins=DigiPresentByDepth.depth[depth]->getNbinsX();
      phibins=DigiPresentByDepth.depth[depth]->getNbinsY();
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
		  if ((present[eta][phi][depth]==0) ||
		      (deadmon_test_occupancy_ && occupancy[eta][phi][depth]==0 && (ievt_%(checkN*deadmon_prescale_)==0)) ||
		      (deadmon_test_energy_ && aboveenergy[eta][phi][depth]==0  && (ievt_%(checkN*deadmon_prescale_)==0)))
		    {
		      if (subdet==HcalBarrel)       
			  ++NumBadHB;
		      else if (subdet==HcalEndcap)  ++NumBadHE;
		      else if (subdet==HcalOuter)   ++NumBadHO;
		      else if (subdet==HcalForward) ++NumBadHF;
		    }
		  if (present[eta][phi][depth]==0)
		    {
		      if (subdet==HcalBarrel) ++neverpresentHB;
		      else if (subdet==HcalEndcap) ++neverpresentHE;
		      else if (subdet==HcalOuter) ++neverpresentHO;
		      else if (subdet==HcalForward) ++neverpresentHF;
		    }
		  if (deadmon_test_occupancy_ && occupancy[eta][phi][depth]==0 && (ievt_%(checkN*deadmon_prescale_)==0))
		    {
		      if (subdet==HcalBarrel) ++unoccupiedHB;
		      else if (subdet==HcalEndcap) ++unoccupiedHE;
		      else if (subdet==HcalOuter) ++unoccupiedHO;
		      else if (subdet==HcalForward) ++unoccupiedHF;
		    }
		  if (deadmon_test_energy_)
		    {
		      if (present_energy[eta][phi][depth]==0)
			{
			  if (subdet==HcalBarrel) ++energyneverpresentHB;
			  else if (subdet==HcalEndcap) ++energyneverpresentHE;
			  else if (subdet==HcalOuter) ++energyneverpresentHO;
			  else if (subdet==HcalForward) ++energyneverpresentHF;
			}
		      if (aboveenergy[eta][phi][depth]==0 && (ievt_%(checkN*deadmon_prescale_)==0))
			{
			  if (subdet==HcalBarrel) ++belowenergyHB;
			  else if (subdet==HcalEndcap) ++belowenergyHE;
			  else if (subdet==HcalOuter) ++belowenergyHO;
			  else if (subdet==HcalForward) ++belowenergyHF;
			}
		    }
		} // subdet loop
	    } // phi loop
	} //eta loop
    } // depth loop

  // Fill with number of problem cells found on this pass

  if (ievt_%(checkN)==0)
    {
      ProblemsVsLB_HB->Fill(lumiblock,NumBadHB);
      ProblemsVsLB_HE->Fill(lumiblock,NumBadHE);
      ProblemsVsLB_HO->Fill(lumiblock,NumBadHO);
      ProblemsVsLB_HF->Fill(lumiblock,NumBadHF);
      ProblemsVsLB->Fill(lumiblock,NumBadHB+NumBadHE+NumBadHO+NumBadHF);
      /*
	//Don't want to include this behavior yet
      if (Online_ && oldlumiblock<lumiblock)
	{
	  for (int i=oldlumiblock+1;i<lumiblock;++i)
	    {
	      if (ProblemsVsLB)
		ProblemsVsLB->Fill(i,NumBadHB+NumBadHE+NumBadHO+NumBadHF);
	      if (ProblemsVsLB_HB)
		ProblemsVsLB_HB->Fill(i,NumBadHB);
	      if (ProblemsVsLB_HE)
		ProblemsVsLB_HE->Fill(i,NumBadHE);
	      if (ProblemsVsLB_HO)
		ProblemsVsLB_HO->Fill(i,NumBadHO);
	      if (ProblemsVsLB_HF)
		ProblemsVsLB_HF->Fill(i,NumBadHF);
	    }
	}
      */
      oldlumiblock=lumiblock; // oldlumiblock keeps track of last block in which plot filled

      if (deadmon_test_occupancy_)
	{
	  NumberOfUnoccupiedCellsHE->Fill(lumiblock,unoccupiedHB);
	  NumberOfUnoccupiedCellsHE->Fill(lumiblock,unoccupiedHE);
	  NumberOfUnoccupiedCellsHO->Fill(lumiblock,unoccupiedHO);
	  NumberOfUnoccupiedCellsHF->Fill(lumiblock,unoccupiedHF);
	  NumberOfUnoccupiedCells->Fill(lumiblock,unoccupiedHB+unoccupiedHE+unoccupiedHO+unoccupiedHF);
	}

      if (deadmon_test_energy_)
	{
	  NumberOfEnergyNeverPresentCellsHB->Fill(lumiblock,energyneverpresentHB);
	  NumberOfEnergyNeverPresentCellsHE->Fill(lumiblock,energyneverpresentHE);
	  NumberOfEnergyNeverPresentCellsHO->Fill(lumiblock,energyneverpresentHO);
	  NumberOfEnergyNeverPresentCellsHF->Fill(lumiblock,energyneverpresentHF);
	  NumberOfEnergyNeverPresentCells->Fill(lumiblock,energyneverpresentHB+energyneverpresentHE+energyneverpresentHO+energyneverpresentHF);

	  NumberOfBelowEnergyCellsHB->Fill(lumiblock,belowenergyHB);
	  NumberOfBelowEnergyCellsHE->Fill(lumiblock,belowenergyHE);
	  NumberOfBelowEnergyCellsHO->Fill(lumiblock,belowenergyHO);
	  NumberOfBelowEnergyCellsHF->Fill(lumiblock,belowenergyHF);
	  NumberOfBelowEnergyCells->Fill(lumiblock,belowenergyHB+belowenergyHE+belowenergyHO+belowenergyHF);
	}

      NumberOfNeverPresentCellsHB->Fill(lumiblock,neverpresentHB);
      NumberOfNeverPresentCellsHE->Fill(lumiblock,neverpresentHE);
      NumberOfNeverPresentCellsHO->Fill(lumiblock,neverpresentHO);
      NumberOfNeverPresentCellsHF->Fill(lumiblock,neverpresentHF);
      NumberOfNeverPresentCells->Fill(lumiblock,neverpresentHB+neverpresentHE+neverpresentHO+neverpresentHF);
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
	      if (resetpresent) present_energy[i][j][k]=false;
	      occupancy[i][j][k]=0; // counts occupancy in last (checkNevents) events
	      aboveenergy[i][j][k]=0; // counts instances of cell above threshold energy in last (checkNevents)
	    }
	}
    }

  return;
} // void HcalDeadCellMonitor::zeroCounters(bool resetpresent)



void HcalDeadCellMonitor::periodicReset()
{

  // first reset base class objects
  HcalBaseMonitor::periodicReset();

  // then reset the temporary histograms
  HcalDeadCellMonitor::zeroCounters(true);

  // now reset all the MonitorElements
  if(NumberOfNeverPresentCells)     NumberOfNeverPresentCells->Reset();
  if(NumberOfNeverPresentCellsHB)   NumberOfNeverPresentCellsHB->Reset();
  if(NumberOfNeverPresentCellsHE)   NumberOfNeverPresentCellsHE->Reset();
  if(NumberOfNeverPresentCellsHO)   NumberOfNeverPresentCellsHO->Reset();
  if(NumberOfNeverPresentCellsHF)   NumberOfNeverPresentCellsHF->Reset();
  
  if(NumberOfUnoccupiedCells)     NumberOfUnoccupiedCells->Reset();
  if(NumberOfUnoccupiedCellsHB)   NumberOfUnoccupiedCellsHB->Reset();
  if(NumberOfUnoccupiedCellsHE)   NumberOfUnoccupiedCellsHE->Reset();
  if(NumberOfUnoccupiedCellsHO)   NumberOfUnoccupiedCellsHO->Reset();
  if(NumberOfUnoccupiedCellsHF)   NumberOfUnoccupiedCellsHF->Reset();
  
  if(NumberOfBelowEnergyCells)     NumberOfBelowEnergyCells->Reset();
  if(NumberOfBelowEnergyCellsHB)   NumberOfBelowEnergyCellsHB->Reset();
  if(NumberOfBelowEnergyCellsHE)   NumberOfBelowEnergyCellsHE->Reset();
  if(NumberOfBelowEnergyCellsHO)   NumberOfBelowEnergyCellsHO->Reset();
  if(NumberOfBelowEnergyCellsHF)   NumberOfBelowEnergyCellsHF->Reset();
  
  //if (ProblemCells) ProblemCells->Reset();

  // now reset the display histograms
  UnoccupiedDeadCellsByDepth.Reset();
  DigiPresentByDepth.Reset();
  BelowEnergyThresholdCellsByDepth.Reset();

  // NeverPresent Histograms start with a value of 1 for all valid bins
  for (unsigned int depth=0;depth<DigiPresentByDepth.depth.size();++depth)
    DigiPresentByDepth.depth[depth]->Reset();
  FillUnphysicalHEHFBins(DigiPresentByDepth);

  // okay, we are out of here.
  return;
}




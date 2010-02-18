#include "DQM/HcalMonitorTasks/interface/HcalDeadCellMonitor.h"

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
  deadmon_minEvents_    = ps.getUntrackedParameter<int>("DeadCellMonitor_minEvents",500); // minimum number of events that must be present in a luminosity block in order to check for dead cells just in that block

  deadmon_lumiblockcount_=0;
  deadmon_prescale_      = ps.getUntrackedParameter<int>("DeadCellMonitor_LBprescale",1); // prescale to require multiple lumi blocks be processed

  // Set which dead cell checks will be performed
  /* Dead cells can be defined in the following ways:
     1)  never present digi -- digi is never present in run
     2)  digis -- digi is absent for a number of consecutive events (or an entire LB)
     3)  never present rechit -- digi / rechit occasionally present, but never above a threshold energy
     4)  rechits -- cell is present, but rechit energy below threshold for a number of consecutive events (or an entire LB)

     Of these tests, never-present digis are always checked.
     Occasional digis are checked only if deadmon_test_digis_ is true,
     and both rechit tests are made only if deadmon_test_rechits_ is true
  */
  
  deadmon_test_digis_         = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_digis", false);
  deadmon_test_rechits_            = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_rechits", false);

  // Minimum error rate to be considered problematic
  deadmon_minErrorFlag_ = ps.getUntrackedParameter<double>("DeadCellMonitor_minErrorFlag",0.05);

  // rechit energy test -- cell must be below threshold value for a number of consecutive events to be considered dead
  energyThreshold_       = ps.getUntrackedParameter<double>("DeadCellMonitor_energyThreshold",                  0);
  HBenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HB_energyThreshold",energyThreshold_);
  HEenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HE_energyThreshold",energyThreshold_);
  HOenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HO_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HF_energyThreshold",energyThreshold_);
 
  // Set allowed types of events for running through rechitmon
  AllowedCalibTypes_ = ps.getUntrackedParameter<vector<int> >("DeadCellMonitor_AllowedCalibTypes",AllowedCalibTypes_);

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor SETUP -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  
  return;
} // void HcalDeadCellMonitor::setup(...)

void HcalDeadCellMonitor::beginRun()
{
  HcalBaseMonitor::beginRun();
  zeroCounters(true); // zero all counters, including never-present counters
  // Set up histograms
  if (!m_dbe) return;
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (fVerbosity>1)
    std::cout <<"<HcalDeadCellMonitor::beginRun>  Setting up histograms"<<std::endl;

  m_dbe->setCurrentFolder(baseFolder_);
  //clearME();
  meEVT_ = m_dbe->bookInt("Dead Cell Task Event Number");
  meEVT_->Fill(ievt_);
  meTOTALEVT_ = m_dbe->bookInt("Dead Cell Total Events Processed");
  meTOTALEVT_->Fill(tevt_);
  
  Nevents = m_dbe->book1D("NumberOfDeadCellEvents","# of Events Seen by DeadCellMonitor",2,0,2);
  Nevents->setBinLabel(1,"allEvents");
  Nevents->setBinLabel(2,"lumiCheck");
 // 1D plots count number of bad cells vs. luminosity block
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
  (ProblemsVsLB->getTProfile())->SetMarkerStyle(20);
  (ProblemsVsLB_HB->getTProfile())->SetMarkerStyle(20);
  (ProblemsVsLB_HE->getTProfile())->SetMarkerStyle(20);
  (ProblemsVsLB_HO->getTProfile())->SetMarkerStyle(20);
  (ProblemsVsLB_HF->getTProfile())->SetMarkerStyle(20);

  // ProblemCells plots are in HcalDeadCellClient!
      
  // Set up plots for each failure mode of dead cells
  stringstream units; // We'll need to set the titles individually, rather than passing units to SetupEtaPhiHists (since this also would affect the name of the histograms)
  stringstream name;

  // Never-present test will always be called, by definition of dead cell

  m_dbe->setCurrentFolder(baseFolder_+"/dead_digi_never_present");
  SetupEtaPhiHists(DigiPresentByDepth,
		   "Digi Present At Least Once","");
  // 1D plots count number of bad cells
  NumberOfNeverPresentDigis=m_dbe->bookProfile("Problem_NeverPresentDigis_HCAL_vs_LS",
					       "Total Number of Never-Present Hcal Cells vs LS;Lumi Section;Dead Cells",
					       Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      
  NumberOfNeverPresentDigisHB=m_dbe->bookProfile("Problem_NeverPresentDigis_HB_vs_LS",
						 "Total Number of Never-Present HB Cells vs LS;Lumi Section;Dead Cells",
						 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      
  NumberOfNeverPresentDigisHE=m_dbe->bookProfile("Problem_NeverPresentDigis_HE_vs_LS",
						 "Total Number of Never-Present HE Cells vs LS;Lumi Section;Dead Cells",
						 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      
  NumberOfNeverPresentDigisHO=m_dbe->bookProfile("Problem_NeverPresentDigis_HO_vs_LS",
						 "Total Number of Never-Present HO Cells vs LS;Lumi Section;Dead Cells",
						 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      
  NumberOfNeverPresentDigisHF=m_dbe->bookProfile("Problem_NeverPresentDigis_HF_vs_LS",
						 "Total Number of Never-Present HF Cells vs LS;Lumi Section;Dead Cells",
						 Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
  (NumberOfNeverPresentDigis->getTProfile())->SetMarkerStyle(20);
  (NumberOfNeverPresentDigisHB->getTProfile())->SetMarkerStyle(20);
  (NumberOfNeverPresentDigisHE->getTProfile())->SetMarkerStyle(20);
  (NumberOfNeverPresentDigisHO->getTProfile())->SetMarkerStyle(20);
  (NumberOfNeverPresentDigisHF->getTProfile())->SetMarkerStyle(20);


  for (unsigned int depth=0;depth<DigiPresentByDepth.depth.size();++depth)
    DigiPresentByDepth.depth[depth]->Reset();
      
  FillUnphysicalHEHFBins(DigiPresentByDepth);

  if (deadmon_test_digis_)
    {
      m_dbe->setCurrentFolder(baseFolder_+"/dead_digi_often_missing");
      //units<<"("<<deadmon_checkNevents_<<" consec. events)";
      name<<"Dead Cells with No Digis";
      SetupEtaPhiHists(RecentMissingDigisByDepth,
		       name.str(),
		       "");
      name.str("");
      name<<"HB HE HF Depth 1 Dead Cells with No Digis for 1 Full Luminosity Block"; 
      RecentMissingDigisByDepth.depth[0]->setTitle(name.str().c_str());

      name.str("");
      name<<"HB HE HF Depth 2 Dead Cells with No Digis for 1 Full Luminosity Block";
      RecentMissingDigisByDepth.depth[1]->setTitle(name.str().c_str());

      name.str("");
      name<<"HE Depth 3 Dead Cells with No Digis for 1 Full Luminosity Block";
      RecentMissingDigisByDepth.depth[2]->setTitle(name.str().c_str());

      name.str("");
      name<<"HO Depth 4 Dead Cells with No Digis for 1 Full Luminosity Block";
      RecentMissingDigisByDepth.depth[3]->setTitle(name.str().c_str());
      name.str("");

      // 1D plots count number of bad cells
      name<<"Total Number of Hcal Digis Unoccupied for 1 Full Luminosity Block"; 
      NumberOfRecentMissingDigis=m_dbe->bookProfile("Problem_RecentMissingDigis_HCAL_vs_LS",
						    name.str(),
						    Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HB Digis Unoccupied for 1 Full LS vs LS;Lumi Section; Dead Cells";
      NumberOfRecentMissingDigisHB=m_dbe->bookProfile("Problem_RecentMissingDigis_HB_vs_LS",
						      name.str(),
						      Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HE Digis Unoccupied for 1 Full LS vs LS;Lumi Section; Dead Cells";
      NumberOfRecentMissingDigisHE=m_dbe->bookProfile("Problem_RecentMissingDigis_HE_vs_LS",
						      name.str(),
						      Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HO Digis Unoccupied for 1 Full LS vs LS;Lumi Section; Dead Cells";
      NumberOfRecentMissingDigisHO=m_dbe->bookProfile("Problem_RecentMissingDigis_HO_vs_LS",
						      name.str(),
						      Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HF Digis Unoccupied for 1 Full LS vs LS;Lumi Section; Dead Cells";
      NumberOfRecentMissingDigisHF=m_dbe->bookProfile("Problem_RecentMissingDigis_HF_vs_LS",
						      name.str(),
						      Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      (NumberOfRecentMissingDigis->getTProfile())->SetMarkerStyle(20);
      (NumberOfRecentMissingDigisHB->getTProfile())->SetMarkerStyle(20);
      (NumberOfRecentMissingDigisHE->getTProfile())->SetMarkerStyle(20);
      (NumberOfRecentMissingDigisHO->getTProfile())->SetMarkerStyle(20);
      (NumberOfRecentMissingDigisHF->getTProfile())->SetMarkerStyle(20);

    }
      
  if (deadmon_test_rechits_)
    {
      // test 1:  energy never above threshold
      m_dbe->setCurrentFolder(baseFolder_+"/dead_rechit_neverpresent");
      SetupEtaPhiHists(RecHitPresentByDepth,"RecHit Above Threshold At Least Once","");
      // set more descriptive titles for threshold plots
      units.str("");
      units<<"Cells Above Energy Threshold At Least Once: Depth 1 -- HB >="<<HBenergyThreshold_<<" GeV, HE >= "<<HEenergyThreshold_<<", HF >="<<HFenergyThreshold_<<" GeV";
      RecHitPresentByDepth.depth[0]->setTitle(units.str().c_str());
      units.str("");
      units<<"Cells Above Energy Threshold At Least Once: Depth 2 -- HB >="<<HBenergyThreshold_<<" GeV, HE >= "<<HEenergyThreshold_<<", HF >="<<HFenergyThreshold_<<" GeV";
      RecHitPresentByDepth.depth[1]->setTitle(units.str().c_str());
      units.str("");
      units<<"Cells Above Energy Threshold At Least Once: Depth 3 -- HE >="<<HEenergyThreshold_<<" GeV";
      RecHitPresentByDepth.depth[2]->setTitle(units.str().c_str());
      units.str("");
      units<<"Cells Above Energy Threshold At Least Once: Depth 4 -- HO >="<<HOenergyThreshold_<<" GeV";
      RecHitPresentByDepth.depth[3]->setTitle(units.str().c_str());
      units.str("");

      // 1D plots count number of bad cells
      NumberOfNeverPresentRecHits=m_dbe->bookProfile("Problem_RecHitsNeverPresent_HCAL_vs_LS",
						     "Total Number of Hcal Rechits with Low Energy;Lumi Section;Dead Cells",
						     Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HB RecHits with Energy Never >= "<<HBenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
      NumberOfNeverPresentRecHitsHB=m_dbe->bookProfile("Problem_RecHitsNeverPresent_HB_vs_LS",
						       name.str(),
						       Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HE RecHits with Energy Never >= "<<HEenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
      NumberOfNeverPresentRecHitsHE=m_dbe->bookProfile("Problem_RecHitsNeverPresent_HE_vs_LS",
						       name.str(),
						       Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HO RecHits with Energy Never >= "<<HOenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
      NumberOfNeverPresentRecHitsHO=m_dbe->bookProfile("Problem_RecHitsNeverPresent_HO_vs_LS",
						       name.str(),
						       Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HF RecHits with Energy Never >= "<<HFenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
      NumberOfNeverPresentRecHitsHF=m_dbe->bookProfile("Problem_RecHitsNeverPresent_HF_vs_LS",
						       name.str(),
						       Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      (NumberOfNeverPresentRecHits->getTProfile())->SetMarkerStyle(20);
      (NumberOfNeverPresentRecHitsHB->getTProfile())->SetMarkerStyle(20);
      (NumberOfNeverPresentRecHitsHE->getTProfile())->SetMarkerStyle(20);
      (NumberOfNeverPresentRecHitsHO->getTProfile())->SetMarkerStyle(20);
      (NumberOfNeverPresentRecHitsHF->getTProfile())->SetMarkerStyle(20);
 
      m_dbe->setCurrentFolder(baseFolder_+"/dead_rechit_often_missing");
      SetupEtaPhiHists(RecentMissingRecHitsByDepth,"RecHits Failing Energy Threshold Test","");
      // set more descriptive titles for threshold plots
      units.str("");
      units<<"RecHits with Consistent Low Energy Depth 1 -- HB <"<<HBenergyThreshold_<<" GeV, HE < "<<HEenergyThreshold_<<", HF <"<<HFenergyThreshold_<<" GeV";
      RecentMissingRecHitsByDepth.depth[0]->setTitle(units.str().c_str());
      units.str("");
      units<<"RecHits with Consistent Low Energy Depth 2 -- HB <"<<HBenergyThreshold_<<" GeV, HE < "<<HEenergyThreshold_<<", HF <"<<HFenergyThreshold_<<" GeV";
      RecentMissingRecHitsByDepth.depth[1]->setTitle(units.str().c_str());
      units.str("");
      units<<"RecHits with Consistent Low Energy Depth 3 -- HE <"<<HEenergyThreshold_<<" GeV";
      RecentMissingRecHitsByDepth.depth[2]->setTitle(units.str().c_str());
      units.str("");
      units<<"RecHits with Consistent Low Energy Depth 4 -- HO <"<<HOenergyThreshold_<<" GeV";
      RecentMissingRecHitsByDepth.depth[3]->setTitle(units.str().c_str());
      units.str("");


      // 1D plots count number of bad cells
      name<<"Total Number of Hcal RecHits with Consistent Low Energy;Lumi Section;Dead Cells";
      NumberOfRecentMissingRecHits=m_dbe->bookProfile("Problem_BelowEnergyRecHits_HCAL_vs_LS",
						      name.str(),
						      Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HB RecHits with Consistent Low Energy < "<<HBenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
      NumberOfRecentMissingRecHitsHB=m_dbe->bookProfile("Problem_BelowEnergyRecHits_HB_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HE RecHits with Consistent Low Energy < "<<HEenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
      NumberOfRecentMissingRecHitsHE=m_dbe->bookProfile("Problem_BelowEnergyRecHits_HE_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HO RecHits with Consistent Low Energy < "<<HOenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
      NumberOfRecentMissingRecHitsHO=m_dbe->bookProfile("Problem_BelowEnergyRecHits_HO_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      name.str("");
      name<<"Total Number of HF RecHits with Consistent Low Energy < "<<HFenergyThreshold_<<" GeV;Lumi Section;Dead Cells";
      NumberOfRecentMissingRecHitsHF=m_dbe->bookProfile("Problem_BelowEnergyRecHits_HF_vs_LS",
							name.str(),
							Nlumiblocks_,0.5,Nlumiblocks_+0.5,100,0,10000);
      (NumberOfRecentMissingRecHits->getTProfile())->SetMarkerStyle(20);
      (NumberOfRecentMissingRecHitsHB->getTProfile())->SetMarkerStyle(20);
      (NumberOfRecentMissingRecHitsHE->getTProfile())->SetMarkerStyle(20);
      (NumberOfRecentMissingRecHitsHO->getTProfile())->SetMarkerStyle(20);
      (NumberOfRecentMissingRecHitsHF->getTProfile())->SetMarkerStyle(20);

    } // if (deadmon_test_rechits)

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor BEGINRUN -> "<<cpu_timer.cpuTime()<<std::endl;
    }
  
  return;
} //void HcalDeadCellMonitor::beginRun()


void HcalDeadCellMonitor::reset(){}  // reset function is empty for now

/* ------------------------------------------------------------------------- */


void HcalDeadCellMonitor::clearME()
{
  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
      if(m_dbe->dirExists("Collector"))
	m_dbe->rmdir("");
    }
  return;
} // void HcalDeadCellMonitor::clearME()

/* ------------------------------------------------------------------------- */


void HcalDeadCellMonitor::processEvent(const HBHERecHitCollection& hbHits,
				       const HORecHitCollection& hoHits,
				       const HFRecHitCollection& hfHits,
				       const HBHEDigiCollection& hbhedigi,
				       const HODigiCollection& hodigi,
				       const HFDigiCollection& hfdigi,
				       int CalibType
				       )
{

  // Check that event is of proper calibration type
  bool processevent=false;
  if (AllowedCalibTypes_.size()==0)
    processevent=true;
  else
    {
      for (unsigned int i=0;i<AllowedCalibTypes_.size();++i)
	{
	  if (AllowedCalibTypes_[i]==CalibType)
	    {
	      processevent=true;
	      break;
	    }
	}
    }
  if (fVerbosity>1) std::cout <<"<HcalDeadCellMonitor::processEvent>  calibType = "<<CalibType<<"  processing event? "<<processevent<<endl;
  if (!processevent)
    return;
  
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  // increment counters
  HcalBaseMonitor::processEvent();

  // HBpresent_, HEpresent need to be determined within loop, since HBHE is a single collection
  HOpresent_ = (hodigi.size()>0||hoHits.size()>0);
  HFpresent_ = (hfdigi.size()>0||hfHits.size()>0);

  if (fVerbosity>1) std::cout <<"<HcalDeadCellMonitor::processEvent> Processing event..."<<std::endl;

  // Do Digi-Based dead cell searches 

  // Dummy fills needed for client normalization of problems
  // (though not necessarily here; we could do this in endluminosityblock)
  for (unsigned int i=0;i<DigiPresentByDepth.depth.size();++i)
    DigiPresentByDepth.depth[i]->setBinContent(0,0,ievt_); 
    
  NumberOfNeverPresentDigis->update();;
  NumberOfNeverPresentDigisHB->update();
  NumberOfNeverPresentDigisHE->update();
  NumberOfNeverPresentDigisHO->update();
  NumberOfNeverPresentDigisHF->update();
  
  if (deadmon_test_digis_)
    {
      
      for (unsigned int i=0;i<RecentMissingDigisByDepth.depth.size();++i)
	RecentMissingDigisByDepth.depth[i]->setBinContent(0,0,ievt_);
      
      NumberOfRecentMissingDigis->update();
      NumberOfRecentMissingDigisHB->update();
      NumberOfRecentMissingDigisHE->update();
      NumberOfRecentMissingDigisHO->update();
      NumberOfRecentMissingDigisHF->update();
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
  if (deadmon_test_rechits_) 
    {
      if (showTiming)
	{
	  cpu_timer.reset(); cpu_timer.start();
	}

      // Normalization Fill
      for (unsigned int i=0;i<RecentMissingRecHitsByDepth.depth.size();++i)
	RecentMissingRecHitsByDepth.depth[i]->setBinContent(0,0,ievt_);

      NumberOfRecentMissingRecHits->update();
      NumberOfRecentMissingRecHitsHB->update();
      NumberOfRecentMissingRecHitsHE->update();
      NumberOfRecentMissingRecHitsHO->update();
      NumberOfRecentMissingRecHitsHF->update();

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

  Nevents->Fill(0,1);
  return;
} // void HcalDeadCellMonitor::processEvent(...)


void HcalDeadCellMonitor::beginLuminosityBlock(int lb)
{
  int tmpevt=levt_;
  HcalBaseMonitor::beginLuminosityBlock(lb);
  // Don't reset counters unless lumi block is integer multiple of prescale
  if (deadmon_lumiblockcount_%deadmon_prescale_==0)
    {
      zeroCounters();
      LBprocessed_=false;
    }
  else
    levt_=tmpevt;
  return;
} // beginLuminosityBlock(int lb)

void HcalDeadCellMonitor::endLuminosityBlock()
{
  if (LBprocessed_==true)
    return; // histograms already filled for this LB
  ++deadmon_lumiblockcount_;
  if (deadmon_lumiblockcount_%deadmon_prescale_!=0)
    {
      LBprocessed_=true;
      return;
    }
  // fillNevents_problemCells checks for never-present cells
  fillNevents_problemCells();
  fillNevents_recentdigis();
  fillNevents_recentrechits();
  LBprocessed_=true;
  return;
} //endLuminosityBlock()

void HcalDeadCellMonitor::endRun()
{
  // Always carry out tests at endRun, regardless of lumiblock prescaling?
  fillNevents_problemCells(); // always check for never-present cells

  // Only check for recent missing cells if they haven't been checked already
  if (LBprocessed_==true)
    return; // histograms already filled for this LB
  ++deadmon_lumiblockcount_;
  fillNevents_recentdigis();
  fillNevents_recentrechits();
  LBprocessed_=true;
  return;
}


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
  ++recentoccupancy_digi[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1];

  // If previously-missing digi found, change boolean status and fill histogram
  if (present_digi[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1]==false)
    {
      if (DigiPresentByDepth.depth[depth-1])
	{
	  DigiPresentByDepth.depth[depth-1]->setBinContent(CalcEtaBin(digi.id().subdet(),ieta,depth)+1,iphi,1);
	}
      present_digi[CalcEtaBin(digi.id().subdet(),ieta,depth)][iphi-1][depth-1]=true;
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
	  ++recentoccupancy_rechit[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	  present_rechit[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]=true;
	  if (RecHitPresentByDepth.depth[depth-1])
	    RecHitPresentByDepth.depth[depth-1]->setBinContent(CalcEtaBin(id.subdet(),ieta,depth)+1,iphi,1);
	}
    }
  else if (id.subdet()==HcalEndcap)
    {
      HEpresent_=true;
      if (!checkHE_) return;
      if (en>=HEenergyThreshold_)
	{
	++recentoccupancy_rechit[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	present_rechit[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]=true;
	if (RecHitPresentByDepth.depth[depth-1])
	  RecHitPresentByDepth.depth[depth-1]->setBinContent(CalcEtaBin(id.subdet(),ieta,depth)+1,iphi,1);
	}
    }
  else if (id.subdet()==HcalForward)
    {
      HFpresent_=true;

      if (en>=HFenergyThreshold_)
	{
	  ++recentoccupancy_rechit[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	
	  present_rechit[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]=true;
	  if (RecHitPresentByDepth.depth[depth-1])
	    RecHitPresentByDepth.depth[depth-1]->setBinContent(CalcEtaBin(id.subdet(),ieta,depth)+1,iphi,1);
	}
    }
  else if (id.subdet()==HcalOuter)
    {
      HOpresent_=true;
      if (en>=HOenergyThreshold_)
	{
	  ++recentoccupancy_rechit[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
	  present_rechit[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]=true;
	  if (RecHitPresentByDepth.depth[depth-1])
	    RecHitPresentByDepth.depth[depth-1]->setBinContent(CalcEtaBin(id.subdet(),ieta,depth)+1,iphi,1); 
	}
    }
}

void HcalDeadCellMonitor::fillNevents_recentdigis()
{
  // Fill Histograms showing digi cells with no occupancy for the past checkNevents
  if (!deadmon_test_digis_) return; // extra protection here against calling histograms than don't exist

  if (levt_ < deadmon_minEvents_) return; // not enough entries to make a determination for this LS

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalDeadCellMonitor::fillNevents_recentdigis> CHECKING FOR RECENT MISSING DIGIS   evtcount = "<<levt_<<std::endl;

  int ieta=0;
  int iphi=0;

  int etabins=0;
  int phibins=0;
  for (unsigned int depth=0;depth<RecentMissingDigisByDepth.depth.size();++depth)
    { 
      etabins=RecentMissingDigisByDepth.depth[depth]->getNbinsX();
      phibins=RecentMissingDigisByDepth.depth[depth]->getNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  for (int subdet=1;subdet<=4;++subdet)
	    {
	      ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1);
	      if (ieta==-9999) continue;
	      for (int phi=0;phi<phibins;++phi)
		{
		  iphi=phi+1;
		  
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1))
		    continue;
		  
		  // Ignore subdetectors that weren't in run?
		  if ((subdet==HcalBarrel && !HBpresent_) || 
		      (subdet==HcalEndcap &&!HEpresent_)  ||
		      (subdet==HcalOuter &&!HOpresent_)  || 
		      (subdet==HcalForward &&!HFpresent_))   continue;
		  // ignore subdetectors we explicitly mask off 
		  if ((!checkHB_ && subdet==HcalBarrel) ||
		      (!checkHE_ && subdet==HcalEndcap) ||
		      (!checkHO_ && subdet==HcalOuter) ||
		      (!checkHF_ && subdet==HcalForward))  continue;
		  int zside=0;
		  if (subdet==HcalForward) // shift HcalForward ieta
		    ieta<0 ? zside=-1 : zside=+1;
		  
		  if (recentoccupancy_digi[eta][phi][depth]==0)
		    {
		      if (fVerbosity>0)
			{
			  std::cout <<"DEAD CELL; NO RECENT OCCUPANCY: subdet = "<<subdet<<", ieta = "<<ieta<<", iphi = "<<iphi<<" depth = "<<depth+1<<std::endl;
			  std::cout <<"\t RAW COORDINATES:  eta = "<<eta<< " phi = "<<phi<<" depth = "<<depth<<std::endl;
 			  std::cout <<"\t Present? "<<present_digi[eta][phi][depth]<<std::endl;
			}
			     // no digi was found for the N events; Fill cell as bad for all N events (N = checkN);
		      if (RecentMissingDigisByDepth.depth[depth]) RecentMissingDigisByDepth.depth[depth]->Fill(ieta+zside,iphi,levt_);
		    }
		} // for (int subdet=1;subdet<=4;++subdet)
	    } // for (int phi=0;...)
	} // for (int eta=0;...)
    } //for (int depth=1;...)
  FillUnphysicalHEHFBins(RecentMissingDigisByDepth);
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_OCCUPANCY -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;

} // void HcalDeadCellMonitor::fillNevents_recentdigis()



/* ----------------------------------- */

void HcalDeadCellMonitor::fillNevents_recentrechits()
{
  // Fill Histograms showing unoccupied rechits, or rec hits with low energy

  // This test is a bit pointless, unless the energy threshold is greater than the ZS threshold.
  // If we require that cells are always < thresh to be flagged by this test, and if 
  // thresh < ZS, then we will never catch any cells, since they'll show up as dead in the
  // neverpresent/occupancy test plots first.
  // Only exception is if something strange is going on between ZS ADC value an RecHit energy?

  if (!deadmon_test_rechits_) return;
  if (levt_ < deadmon_minEvents_) return; // not enough entries to make a determination for this LS

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    std::cout <<"<HcalDeadCellMonitor::fillNevents_energy> BELOW-ENERGY-THRESHOLD PLOTS"<<std::endl;

  int ieta=0;
  int iphi=0;

  int etabins=0;
  int phibins=0;
  for (unsigned int depth=0;depth<RecentMissingRecHitsByDepth.depth.size();++depth)
    { 
      etabins=RecentMissingRecHitsByDepth.depth[depth]->getNbinsX();
      phibins=RecentMissingRecHitsByDepth.depth[depth]->getNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  for (int subdet=1;subdet<=4;++subdet)
	    {
	      ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1);
	      if (ieta==-9999) continue;
	      for (int phi=0;phi<phibins;++phi)
		{
		  iphi=phi+1;
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1))
		    continue;
		  if (recentoccupancy_rechit[eta][phi][depth]>0) continue; // cell exceeded energy at least once, so it's not dead

		  // Ignore subdetectors that weren't in run?
                  if ((subdet==HcalBarrel && !HBpresent_) || 
		      (subdet==HcalEndcap &&!HEpresent_)  ||
		      (subdet==HcalOuter &&!HOpresent_)  ||
		      (subdet==HcalForward &&!HFpresent_))   continue;

		  if ((!checkHB_ && subdet==HcalBarrel) ||
		      (!checkHE_ && subdet==HcalEndcap) ||
		      (!checkHO_ && subdet==HcalOuter) ||
		      (!checkHF_ && subdet==HcalForward))  continue;
		  
		  int zside=0;
		  if (subdet==HcalForward) // shift HcalForward ieta
		    {
		      ieta<0 ? zside=-1 : zside=+1;
		    }
		  
		  if (fVerbosity>2) 
		    std::cout <<"DEAD CELL; BELOW ENERGY THRESHOLD; subdet = "<<subdet<<" ieta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<std::endl;
			  
		  if (RecentMissingRecHitsByDepth.depth[depth]) RecentMissingRecHitsByDepth.depth[depth]->Fill(ieta+zside,iphi,levt_);
		} // loop on phi bins
	    } // for (unsigned int depth=1;depth<=4;++depth)
	} // // loop on subdetectors
    } // for (int eta=0;...)

  FillUnphysicalHEHFBins(RecHitPresentByDepth);
  FillUnphysicalHEHFBins(RecentMissingRecHitsByDepth);
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_ENERGY -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} // void HcalDeadCellMonitor::fillNevents_recentrechits()



void HcalDeadCellMonitor::fillNevents_problemCells()
{
  //fillNevents_problemCells now only performs checks of never-present cells

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

  if (levt_>=deadmon_minEvents_)
    Nevents->Fill(1,levt_);

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
		  if ((present_digi[eta][phi][depth]==0) ||
		      (deadmon_test_digis_ && recentoccupancy_digi[eta][phi][depth]==0 && (levt_>=deadmon_minEvents_)) ||
		      (deadmon_test_rechits_ && recentoccupancy_rechit[eta][phi][depth]==0  && (levt_>=deadmon_minEvents_))
		      )
		    {
		      if (subdet==HcalBarrel)       ++NumBadHB;
		      else if (subdet==HcalEndcap)  ++NumBadHE;
		      else if (subdet==HcalOuter)   ++NumBadHO;
		      else if (subdet==HcalForward) ++NumBadHF;
		    }
		  if (present_digi[eta][phi][depth]==0)
		    {
		      if (subdet==HcalBarrel) ++neverpresentHB;
		      else if (subdet==HcalEndcap) ++neverpresentHE;
		      else if (subdet==HcalOuter) ++neverpresentHO;
		      else if (subdet==HcalForward) ++neverpresentHF;
		    }
		  // Count recent unoccupied digis if the total events in this lumi section is > minEvents_
		  if (deadmon_test_digis_ && recentoccupancy_digi[eta][phi][depth]==0 && (levt_>=deadmon_minEvents_))
		    {
		      if (subdet==HcalBarrel) ++unoccupiedHB;
		      else if (subdet==HcalEndcap) ++unoccupiedHE;
		      else if (subdet==HcalOuter) ++unoccupiedHO;
		      else if (subdet==HcalForward) ++unoccupiedHF;
		    }
		  // Look at rechit checks
		  if (deadmon_test_rechits_)
		    {
		      if (present_rechit[eta][phi][depth]==0)
			{
			  if (subdet==HcalBarrel) ++energyneverpresentHB;
			  else if (subdet==HcalEndcap) ++energyneverpresentHE;
			  else if (subdet==HcalOuter) ++energyneverpresentHO;
			  else if (subdet==HcalForward) ++energyneverpresentHF;
			}
		      if (recentoccupancy_rechit[eta][phi][depth]==0 && (levt_>=deadmon_minEvents_))
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


  NumberOfNeverPresentDigisHB->Fill(lumiblock,neverpresentHB);
  NumberOfNeverPresentDigisHE->Fill(lumiblock,neverpresentHE);
  NumberOfNeverPresentDigisHO->Fill(lumiblock,neverpresentHO);
  NumberOfNeverPresentDigisHF->Fill(lumiblock,neverpresentHF);
  NumberOfNeverPresentDigis->Fill(lumiblock,neverpresentHB+neverpresentHE+neverpresentHO+neverpresentHF);
  
  ProblemsVsLB_HB->Fill(lumiblock,NumBadHB);
  ProblemsVsLB_HE->Fill(lumiblock,NumBadHE);
  ProblemsVsLB_HO->Fill(lumiblock,NumBadHO);
  ProblemsVsLB_HF->Fill(lumiblock,NumBadHF);
  ProblemsVsLB->Fill(lumiblock,NumBadHB+NumBadHE+NumBadHO+NumBadHF);
  

  if (levt_<deadmon_minEvents_)
    return;
  
  if (deadmon_test_digis_)
    {
      NumberOfRecentMissingDigisHE->Fill(lumiblock,unoccupiedHB);
      NumberOfRecentMissingDigisHE->Fill(lumiblock,unoccupiedHE);
      NumberOfRecentMissingDigisHO->Fill(lumiblock,unoccupiedHO);
      NumberOfRecentMissingDigisHF->Fill(lumiblock,unoccupiedHF);
      NumberOfRecentMissingDigis->Fill(lumiblock,unoccupiedHB+unoccupiedHE+unoccupiedHO+unoccupiedHF);
    }
  
  if (deadmon_test_rechits_)
    {
      NumberOfNeverPresentRecHitsHB->Fill(lumiblock,energyneverpresentHB);
      NumberOfNeverPresentRecHitsHE->Fill(lumiblock,energyneverpresentHE);
      NumberOfNeverPresentRecHitsHO->Fill(lumiblock,energyneverpresentHO);
      NumberOfNeverPresentRecHitsHF->Fill(lumiblock,energyneverpresentHF);
      NumberOfNeverPresentRecHits->Fill(lumiblock,energyneverpresentHB+energyneverpresentHE+energyneverpresentHO+energyneverpresentHF);
      
      NumberOfRecentMissingRecHitsHB->Fill(lumiblock,belowenergyHB);
      NumberOfRecentMissingRecHitsHE->Fill(lumiblock,belowenergyHE);
      NumberOfRecentMissingRecHitsHO->Fill(lumiblock,belowenergyHO);
      NumberOfRecentMissingRecHitsHF->Fill(lumiblock,belowenergyHF);
      NumberOfRecentMissingRecHits->Fill(lumiblock,belowenergyHB+belowenergyHE+belowenergyHO+belowenergyHF);
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
	      if (resetpresent) present_digi[i][j][k]=false; // keeps track of whether digi was ever present
	      if (resetpresent) present_rechit[i][j][k]=false;
	      recentoccupancy_digi[i][j][k]=0; // counts occupancy in last (checkNevents) events
	      recentoccupancy_rechit[i][j][k]=0; // counts instances of cell above threshold energy in last (checkNevents)
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
  if(NumberOfNeverPresentDigis)     NumberOfNeverPresentDigis->Reset();
  if(NumberOfNeverPresentDigisHB)   NumberOfNeverPresentDigisHB->Reset();
  if(NumberOfNeverPresentDigisHE)   NumberOfNeverPresentDigisHE->Reset();
  if(NumberOfNeverPresentDigisHO)   NumberOfNeverPresentDigisHO->Reset();
  if(NumberOfNeverPresentDigisHF)   NumberOfNeverPresentDigisHF->Reset();
  
  if(NumberOfRecentMissingDigis)     NumberOfRecentMissingDigis->Reset();
  if(NumberOfRecentMissingDigisHB)   NumberOfRecentMissingDigisHB->Reset();
  if(NumberOfRecentMissingDigisHE)   NumberOfRecentMissingDigisHE->Reset();
  if(NumberOfRecentMissingDigisHO)   NumberOfRecentMissingDigisHO->Reset();
  if(NumberOfRecentMissingDigisHF)   NumberOfRecentMissingDigisHF->Reset();
  
  if(NumberOfRecentMissingRecHits)     NumberOfRecentMissingRecHits->Reset();
  if(NumberOfRecentMissingRecHitsHB)   NumberOfRecentMissingRecHitsHB->Reset();
  if(NumberOfRecentMissingRecHitsHE)   NumberOfRecentMissingRecHitsHE->Reset();
  if(NumberOfRecentMissingRecHitsHO)   NumberOfRecentMissingRecHitsHO->Reset();
  if(NumberOfRecentMissingRecHitsHF)   NumberOfRecentMissingRecHitsHF->Reset();
  
  // now reset the display histograms
  RecentMissingDigisByDepth.Reset();
  DigiPresentByDepth.Reset();
  RecentMissingRecHitsByDepth.Reset();

  // NeverPresent Histograms start with a value of 1 for all valid bins
  for (unsigned int depth=0;depth<DigiPresentByDepth.depth.size();++depth)
    DigiPresentByDepth.depth[depth]->Reset();
  FillUnphysicalHEHFBins(DigiPresentByDepth);

  // okay, we are out of here.
  return;
}




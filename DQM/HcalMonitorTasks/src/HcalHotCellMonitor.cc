#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "DQM/HcalMonitorTasks/interface/HcalHotCellMonitor.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

HcalHotCellMonitor::HcalHotCellMonitor(const edm::ParameterSet& ps):HcalBaseDQMonitor(ps) {
  // Standard information, inherited from base class
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","HotCellMonitor_Hcal/"); // HotCellMonitor_Hcal
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",true);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);

  // Collection type info
  hbheRechitLabel_       = ps.getUntrackedParameter<edm::InputTag>("hbheRechitLabel");
  hoRechitLabel_         = ps.getUntrackedParameter<edm::InputTag>("hoRechitLabel");
  hfRechitLabel_         = ps.getUntrackedParameter<edm::InputTag>("hfRechitLabel");

  // register for data access
  tok_hbhe_ = consumes<HBHERecHitCollection>(hbheRechitLabel_);
  tok_ho_ = consumes<HORecHitCollection>(hoRechitLabel_);
  tok_hf_ = consumes<HFRecHitCollection>(hfRechitLabel_);

  // Hot Cell-specific tests
  minEvents_      = ps.getUntrackedParameter<int>("minEvents");
  minErrorFlag_   = ps.getUntrackedParameter<double>("minErrorFlag",1);

  // exclude HO ring 2
  excludeHORing2_       = ps.getUntrackedParameter<bool>("excludeHORing2",false);


  // Set which hot cell checks will be performed
  test_persistent_         = ps.getUntrackedParameter<bool>("test_persistent"); // true by default
  test_neighbor_           = ps.getUntrackedParameter<bool>("test_neighbor"); // false by default; test disabled
  test_energy_             = ps.getUntrackedParameter<bool>("test_energy"); // true by default
  test_et_                 = ps.getUntrackedParameter<bool>("test_et"); // true by default


  // rechit energy test -- cell must be above threshold value for a number of consecutive events to be considered hot
  energyThreshold_                = ps.getUntrackedParameter<double>("energyThreshold");
  ETThreshold_                    = ps.getUntrackedParameter<double>("ETThreshold");

  HBenergyThreshold_              = ps.getUntrackedParameter<double>("energyThreshold_HB",energyThreshold_);
  HEenergyThreshold_              = ps.getUntrackedParameter<double>("energyThreshold_HE",energyThreshold_);
  HOenergyThreshold_              = ps.getUntrackedParameter<double>("energyThreshold_HO",energyThreshold_);
  HFenergyThreshold_              = ps.getUntrackedParameter<double>("energyThreshold_HF",energyThreshold_);

  HBETThreshold_                  = ps.getUntrackedParameter<double>("ETThreshold_HB",ETThreshold_);
  HEETThreshold_                  = ps.getUntrackedParameter<double>("ETThreshold_HE",ETThreshold_);
  HOETThreshold_                  = ps.getUntrackedParameter<double>("ETThreshold_HO",ETThreshold_);
  HFETThreshold_                  = ps.getUntrackedParameter<double>("ETThreshold_HF",ETThreshold_);

  // rechit event-by-event energy test -- cell must be above threshold to be considered hot
  persistentThreshold_           = ps.getUntrackedParameter<double>("persistentThreshold");

  HBpersistentThreshold_         = ps.getUntrackedParameter<double>("persistentThreshold_HB",persistentThreshold_);
  HEpersistentThreshold_         = ps.getUntrackedParameter<double>("persistentThreshold_HE",persistentThreshold_);
  HOpersistentThreshold_         = ps.getUntrackedParameter<double>("persistentThreshold_HO",persistentThreshold_);
  HFpersistentThreshold_         = ps.getUntrackedParameter<double>("persistentThreshold_HF",persistentThreshold_);

  persistentETThreshold_           = ps.getUntrackedParameter<double>("persistentETThreshold");

  HBpersistentETThreshold_         = ps.getUntrackedParameter<double>("persistentETThreshold_HB",persistentETThreshold_);
  HEpersistentETThreshold_         = ps.getUntrackedParameter<double>("persistentETThreshold_HE",persistentETThreshold_);
  HOpersistentETThreshold_         = ps.getUntrackedParameter<double>("persistentETThreshold_HO",persistentETThreshold_);
  HFpersistentETThreshold_         = ps.getUntrackedParameter<double>("persistentETThreshold_HF",persistentETThreshold_);

  HFfarfwdScale_                 = ps.getUntrackedParameter<double>("HFfwdScale",2.);
  SiPMscale_                     = ps.getUntrackedParameter<double>("HO_SiPMscalefactor",1.); // default scale factor of 4?
  
  // neighboring-cell tests
  HBHENeighborParams_.DeltaIphi            = ps.getUntrackedParameter<int>("HBHE_neighbor_deltaIphi", 1);
  HBHENeighborParams_.DeltaIeta            = ps.getUntrackedParameter<int>("HBHE_neighbor_deltaIeta", 1);
  HBHENeighborParams_.DeltaDepth           = ps.getUntrackedParameter<int>("HBHE_neighbor_deltaDepth", 2);
  HBHENeighborParams_.minCellEnergy        = ps.getUntrackedParameter<double>("HBHE_neighbor_minCellEnergy",3.);
  HBHENeighborParams_.minNeighborEnergy    = ps.getUntrackedParameter<double>("HBHE_neighbor_minNeighborEnergy",0.);
  HBHENeighborParams_.maxEnergy            = ps.getUntrackedParameter<double>("HBHE_neighbor_maxEnergy",100);
  HBHENeighborParams_.HotEnergyFrac        = ps.getUntrackedParameter<double>("HBHE_neighbor_HotEnergyFrac",0.05);

  HONeighborParams_.DeltaIphi            = ps.getUntrackedParameter<int>("HO_neighbor_deltaIphi", 1);
  HONeighborParams_.DeltaIeta            = ps.getUntrackedParameter<int>("HO_neighbor_deltaIeta", 1);
  HONeighborParams_.DeltaDepth           = ps.getUntrackedParameter<int>("HO_neighbor_deltaDepth", 0);
  HONeighborParams_.minCellEnergy        = ps.getUntrackedParameter<double>("HO_neighbor_minCellEnergy",10.);
  HONeighborParams_.minNeighborEnergy    = ps.getUntrackedParameter<double>("HO_neighbor_minNeighborEnergy",0.);
  HONeighborParams_.maxEnergy            = ps.getUntrackedParameter<double>("HO_neighbor_maxEnergy",100);
  HONeighborParams_.HotEnergyFrac        = ps.getUntrackedParameter<double>("HO_neighbor_HotEnergyFrac",0.01);

  HFNeighborParams_.DeltaIphi            = ps.getUntrackedParameter<int>("HF_neighbor_deltaIphi", 1);
  HFNeighborParams_.DeltaIeta            = ps.getUntrackedParameter<int>("HF_neighbor_deltaIeta", 1);
  HFNeighborParams_.DeltaDepth           = ps.getUntrackedParameter<int>("HF_neighbor_deltaDepth", 1);
  HFNeighborParams_.minCellEnergy        = ps.getUntrackedParameter<double>("HF_neighbor_minCellEnergy",10.);
  HFNeighborParams_.minNeighborEnergy    = ps.getUntrackedParameter<double>("HF_neighbor_minNeighborEnergy",0.);
  HFNeighborParams_.maxEnergy            = ps.getUntrackedParameter<double>("HF_neighbor_maxEnergy",100);
  HFNeighborParams_.HotEnergyFrac        = ps.getUntrackedParameter<double>("HF_neighbor_HotEnergyFrac",0.01);
  setupDone_=false;
} //constructor

HcalHotCellMonitor::~HcalHotCellMonitor() { } //destructor


/* ------------------------------------ */ 

void HcalHotCellMonitor::setup(DQMStore::IBooker &ib)
{
  if (setupDone_)
    return;
  setupDone_ = true;
  // Call base class setup
  HcalBaseDQMonitor::setup(ib);

  if (debug_>1)
    std::cout <<"<HcalHotCellMonitor::setup>  Setting up histograms"<<std::endl;

  ib.setCurrentFolder(subdir_);

  MonitorElement* me;
  me=ib.bookFloat("minErrorFractionPerLumiSection");
  me->Fill(minErrorFlag_);
  // Create plots of problems vs LB

  // 1D plots count number of bad cells vs. luminosity block
  ProblemsVsLB=ib.bookProfile("TotalHotCells_HCAL_vs_LS",
				 "Total Number of Hot Hcal Cells vs lumi section", 
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000);

  ProblemsVsLB_HB=ib.bookProfile("TotalHotCells_HB_vs_LS",
				    "Total Number of Hot HB Cells vs lumi section",
				    NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,3000);
  ProblemsVsLB_HE=ib.bookProfile("TotalHotCells_HE_vs_LS",
				    "Total Number of Hot HE Cells vs lumi section",
				    NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,3000);
  ProblemsVsLB_HO=ib.bookProfile("TotalHotCells_HO_vs_LS",
				    "Total Number of Hot HO Cells vs lumi section",
				    NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,3000);
  ProblemsVsLB_HF=ib.bookProfile("TotalHotCells_HF_vs_LS",
				    "Total Number of Hot HF Cells vs lumi section",
				    NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,2000);
  ProblemsVsLB_HBHEHF=ib.bookProfile("TotalHotCells_HBHEHF_vs_LS",
				    "Total Number of Hot HBHEHF Cells vs lumi section",
				    NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,2000);
 
  ProblemsVsLB->getTProfile()->SetMarkerStyle(20);
  ProblemsVsLB_HB->getTProfile()->SetMarkerStyle(20);
  ProblemsVsLB_HE->getTProfile()->SetMarkerStyle(20);
  ProblemsVsLB_HO->getTProfile()->SetMarkerStyle(20);
  ProblemsVsLB_HF->getTProfile()->SetMarkerStyle(20);
  ProblemsVsLB_HBHEHF->getTProfile()->SetMarkerStyle(20);

  // Set up plots for each failure mode of hot cells
  std::stringstream units; // We'll need to set the titles individually, rather than passing units to SetupEtaPhiHists (since this also would affect the name of the histograms)

  ib.setCurrentFolder(subdir_+"hot_rechit_above_threshold");
  me=ib.bookInt("HotCellAboveThresholdTestEnabled");
  me->Fill(0);
  
  if (test_energy_)
    {
      me->Fill(1);
      SetupEtaPhiHists(ib,AboveEnergyThresholdCellsByDepth,
		       "Hot Cells Above Energy Threshold","");
      //setMinMaxHists2D(AboveEnergyThresholdCellsByDepth,0.,1.);
      
      // set more descriptive titles for plots
      units.str("");
      units<<"Hot Cells: Depth 1 -- HB > "<<HBenergyThreshold_<<" GeV, HE > "<<HEenergyThreshold_<<" GeV, HF > "<<HFenergyThreshold_<<" GeV";
      AboveEnergyThresholdCellsByDepth.depth[0]->setTitle(units.str().c_str());
      units.str("");
      units<<"Hot Cells: Depth 2 -- HB > "<<HBenergyThreshold_<<" GeV, HE > "<<HEenergyThreshold_<<" GeV, HF > "<<HFenergyThreshold_<<" GeV";
      AboveEnergyThresholdCellsByDepth.depth[1]->setTitle(units.str().c_str());
      units.str("");
      units<<"Hot Cells: Depth 3 -- HE > "<<HEenergyThreshold_<<" GeV";
      AboveEnergyThresholdCellsByDepth.depth[2]->setTitle(units.str().c_str());
      units.str("");
      units<<"Hot Cells: HO > "<<HOenergyThreshold_<<" GeV";
      AboveEnergyThresholdCellsByDepth.depth[3]->setTitle(units.str().c_str());
      units.str("");
    }
  if (test_et_)
    {
      me->Fill(1);
      SetupEtaPhiHists(ib,AboveETThresholdCellsByDepth,
		       "Hot Cells Above ET Threshold","");
      //setMinMaxHists2D(AboveETThresholdCellsByDepth,0.,1.);
      
      // set more descriptive titles for plots
      units.str("");
      units<<"Hot Cells: Depth 1 -- HB > "<<HBETThreshold_<<" GeV (ET), HE > "<<HEETThreshold_<<" GeV (ET), HF > "<<HFETThreshold_<<" GeV (ET)";
      AboveETThresholdCellsByDepth.depth[0]->setTitle(units.str().c_str());
      units.str("");
      units<<"Hot Cells: Depth 2 -- HB > "<<HBETThreshold_<<" GeV (ET), HE > "<<HEETThreshold_<<" GeV (ET), HF > "<<HFETThreshold_<<" GeV (ET)";
      AboveETThresholdCellsByDepth.depth[1]->setTitle(units.str().c_str());
      units.str("");
      units<<"Hot Cells: Depth 3 -- HE > "<<HEETThreshold_<<" GeV (ET)";
      AboveETThresholdCellsByDepth.depth[2]->setTitle(units.str().c_str());
      units.str("");
      units<<"Hot Cells: HO > "<<HOETThreshold_<<" GeV (ET)";
      AboveETThresholdCellsByDepth.depth[3]->setTitle(units.str().c_str());
      units.str("");
    }
  
  ib.setCurrentFolder(subdir_+"hot_rechit_always_above_threshold");
  me=ib.bookInt("PersistentHotCellTestEnabled");
  me->Fill(0);
  if (test_persistent_)
    {
      me->Fill(1);
      me=ib.bookInt("minEventsPerLS");
      me->Fill(minEvents_);

      if (test_energy_) {
	SetupEtaPhiHists(ib,AbovePersistentThresholdCellsByDepth,
			 "Hot Cells Persistently Above Energy Threshold","");
	//setMinMaxHists2D(AbovePersistentThresholdCellsByDepth,0.,1.);
	
	// set more descriptive titles for plots
	units.str("");
	units<<"Hot Cells: Depth 1 -- HB > "<<HBpersistentThreshold_<<" GeV, HE > "<<HEpersistentThreshold_<<", HF > "<<HFpersistentThreshold_<<" GeV for 1 full Lumi Block";
	AbovePersistentThresholdCellsByDepth.depth[0]->setTitle(units.str().c_str());
	units.str("");
	units<<"Hot Cells: Depth 2 -- HB > "<<HBpersistentThreshold_<<" GeV, HE > "<<HEpersistentThreshold_<<", HF > "<<HFpersistentThreshold_<<" GeV for 1 full Lumi Block";
	AbovePersistentThresholdCellsByDepth.depth[1]->setTitle(units.str().c_str());
	units.str("");
	units<<"Hot Cells: Depth 3 -- HE > "<<HEpersistentThreshold_<<" GeV for 1 full Lumi Block";
	AbovePersistentThresholdCellsByDepth.depth[2]->setTitle(units.str().c_str());
	units.str("");
	units<<"Hot Cells:  HO > "<<HOpersistentThreshold_<<" GeV for 1 full Lumi Block";
	AbovePersistentThresholdCellsByDepth.depth[3]->setTitle(units.str().c_str());
	units.str("");
      }
  
      if (test_et_) {
	SetupEtaPhiHists(ib,AbovePersistentETThresholdCellsByDepth,
			 "Hot Cells Persistently Above ET Threshold","");
	//setMinMaxHists2D(AbovePersistentThresholdCellsByDepth,0.,1.);
	
	// set more descriptive titles for plots
	units.str("");
	units<<"Hot Cells: Depth 1 -- HB > "<<HBpersistentETThreshold_<<" GeV (ET), HE > "<<HEpersistentETThreshold_<<" GeV (ET), HF > "<<HFpersistentETThreshold_<<" GeV (ET) for 1 full Lumi Block";
	AbovePersistentETThresholdCellsByDepth.depth[0]->setTitle(units.str().c_str());
	units.str("");
	units<<"Hot Cells: Depth 2 -- HB > "<<HBpersistentETThreshold_<<" GeV (ET), HE > "<<HEpersistentETThreshold_<<" GeV (ET), HF > "<<HFpersistentETThreshold_<<" GeV (ET) for 1 full Lumi Block";
	AbovePersistentETThresholdCellsByDepth.depth[1]->setTitle(units.str().c_str());
	units.str("");
	units<<"Hot Cells: Depth 3 -- HE > "<<HEpersistentETThreshold_<<" GeV (ET) for 1 full Lumi Block";
	AbovePersistentETThresholdCellsByDepth.depth[2]->setTitle(units.str().c_str());
	units.str("");
	units<<"Hot Cells:  HO > "<<HOpersistentETThreshold_<<" GeV (ET) for 1 full Lumi Block";
	AbovePersistentETThresholdCellsByDepth.depth[3]->setTitle(units.str().c_str());
	units.str("");
      }
    }
  
  ib.setCurrentFolder(subdir_+"hot_neighbortest");
  me=ib.bookInt("NeighborTestEnabled");
  me->Fill(0);
  if (test_neighbor_)
    me->Fill(1);
  if (test_neighbor_ || makeDiagnostics_)
    {
      SetupEtaPhiHists(ib,AboveNeighborsHotCellsByDepth,"Hot Cells Failing Neighbor Test","");
      if (makeDiagnostics_)
	{
	  d_HBenergyVsNeighbor=ib.book1D("NeighborSumOverEnergyHB","HB Neighbor Sum Energy/Cell Energy;sum(neighbors)/E_cell",500,0,10);
	  d_HEenergyVsNeighbor=ib.book1D("NeighborSumOverEnergyHE","HE Neighbor Sum Energy/Cell Energy;sum(neighbors)/E_cell",500,0,10);
	  d_HOenergyVsNeighbor=ib.book1D("NeighborSumOverEnergyHO","HO Neighbor Sum Energy/Cell Energy;sum(neighbors)/E_cell",500,0,10);
	  d_HFenergyVsNeighbor=ib.book1D("NeighborSumOverEnergyHF","HF Neighbor Sum Energy/Cell Energy;sum(neighbors)/E_cell",500,0,10);
	}
    } // if (test_neighbor_ || makeDiagnostics_)
  
  this->reset();
} // void HcalHotCellMonitor::setup(...)

void HcalHotCellMonitor::bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>1) std::cout <<"HcalHotCellMonitor::bookHistograms"<<std::endl;
  HcalBaseDQMonitor::bookHistograms(ib,run,c);

  if (tevt_==0) this->setup(ib); // set up histograms if they have not been created before
  if (mergeRuns_==false)
    this->reset();

  return;
} //void HcalHotCellMonitor::bookHistograms(...)


/* --------------------------- */

void HcalHotCellMonitor::reset()
{
  HcalBaseDQMonitor::reset();
  zeroCounters();

  // now reset all the MonitorElements

  // resetting eta-phi histograms
  if (test_neighbor_ || makeDiagnostics_)
    AboveNeighborsHotCellsByDepth.Reset();

  if (test_energy_ ) 
    AboveEnergyThresholdCellsByDepth.Reset();

  if ( test_et_ ) 
    AboveETThresholdCellsByDepth.Reset();

  if (test_persistent_)
    {
      if (test_energy_) AbovePersistentThresholdCellsByDepth.Reset();
      if (test_et_)     AbovePersistentETThresholdCellsByDepth.Reset();
    }
  if (makeDiagnostics_)
    {
      d_HBenergyVsNeighbor->Reset();
      d_HEenergyVsNeighbor->Reset();
      d_HOenergyVsNeighbor->Reset();
      d_HFenergyVsNeighbor->Reset();
    }
}  

void HcalHotCellMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					      const edm::EventSetup& c)
{
  if (LumiInOrder(lumiSeg.luminosityBlock())==false) return;
  HcalBaseDQMonitor::beginLuminosityBlock(lumiSeg,c);
  zeroCounters(); // zero hot cell counters at the start of each luminosity block
  ProblemsCurrentLB->Reset();
  return;
} // beginLuminosityBlock(...)


void HcalHotCellMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					    const edm::EventSetup& c)
{
  //FIX with check on whether LB already processed
  
  if (LumiInOrder(lumiSeg.luminosityBlock())==false) return;

  edm::ESHandle<HcalTopology> topo;
  c.get<HcalRecNumberingRecord>().get(topo);

  if (test_neighbor_ || makeDiagnostics_)
    fillNevents_neighbor(*topo);

  if (test_energy_ || test_et_)
    fillNevents_energy(*topo);

  if (test_persistent_)
    fillNevents_persistentenergy(*topo);

  fillNevents_problemCells(*topo);
  return;
} //endLuminosityBlock(...)



/* ------------------------- */

void HcalHotCellMonitor::done()
{
  // moved database dumps to client; we want to be able to sum over results in offline
  return;

} // void HcalHotCellMonitor::done()


void HcalHotCellMonitor::analyze(edm::Event const&e, edm::EventSetup const&s)
{
  HcalBaseDQMonitor::analyze(e,s);
  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(e.luminosityBlock())==false) return;

  // try to get rechits
  edm::Handle<HBHERecHitCollection> hbhe_rechit;
  edm::Handle<HORecHitCollection> ho_rechit;
  edm::Handle<HFRecHitCollection> hf_rechit;

  if (!(e.getByToken(tok_hbhe_,hbhe_rechit)))
    {
      edm::LogWarning("HcalHotCellMonitor")<< hbheRechitLabel_<<" hbhe_rechit not available";
      return;
    }

  if (!(e.getByToken(tok_hf_,hf_rechit)))
    {
      edm::LogWarning("HcalHotCellMonitor")<< hfRechitLabel_<<" hf_rechit not available";
      return;
    }
  if (!(e.getByToken(tok_ho_,ho_rechit)))
    {
      edm::LogWarning("HcalHotCellMonitor")<< hoRechitLabel_<<" ho_rechit not available";
      return;
    }

  // Good event found; increment counter (via base class analyze method)
  edm::ESHandle<HcalTopology> topo;
  s.get<HcalRecNumberingRecord>().get(topo);

  //  HcalBaseDQMonitor::analyze(e,s);
  if (debug_>1) std::cout <<"\t<HcalHotCellMonitor::analyze>  Processing good event! event # = "<<ievt_<<std::endl;

  processEvent(*hbhe_rechit, *ho_rechit, *hf_rechit, *topo);

} // void HcalHotCellMonitor::analyze(...)




/* -------------------------------- */


void HcalHotCellMonitor::processEvent(const HBHERecHitCollection& hbHits,
				      const HORecHitCollection& hoHits,
				      const HFRecHitCollection& hfHits,
				      const HcalTopology& topology) {
  
  if (debug_>1) std::cout <<"<HcalHotCellMonitor::processEvent> Processing event..."<<std::endl;

  // Search for hot cells above a certain energy
  if (test_energy_ || test_et_ || test_persistent_) {
    processEvent_rechitenergy(hbHits, hoHits,hfHits, topology);
  }

  return;
} // void HcalHotCellMonitor::processEvent(...)


/* --------------------------------------- */


void HcalHotCellMonitor::processEvent_rechitenergy( const HBHERecHitCollection& hbheHits,
						    const HORecHitCollection& hoHits,
						    const HFRecHitCollection& hfHits,
						    const HcalTopology& topology) {

  // Looks at rechits of cells and compares to threshold energies.
  // Cells above thresholds get marked as hot candidates

  if (debug_>1) std::cout <<"<HcalHotCellMonitor::processEvent_rechitenergy> Processing rechits..."<<std::endl;

  // loop over HBHE
  for (HBHERecHitCollection::const_iterator HBHEiter=hbheHits.begin(); HBHEiter!=hbheHits.end(); ++HBHEiter) { // loop over all hits
    float en = HBHEiter->energy();
    //float ti = HBHEiter->time();

    HcalDetId id(HBHEiter->detid().rawId());
    int ieta = id.ieta();
    int iphi = id.iphi();
    int depth = id.depth();
    std::pair<double,double> etas = topology.etaRange(id.subdet(),abs(ieta));
    double fEta=fabs(0.5*(etas.first+etas.second));
    float et = en/cosh(fEta);

    if (test_neighbor_ || makeDiagnostics_) {
      processHit_rechitNeighbors(HBHEiter, hbheHits, HBHENeighborParams_, topology);
    }
    if (id.subdet()==HcalBarrel) {
      if (en>=HBenergyThreshold_)
	++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
      if (et>=HBETThreshold_)
	++aboveet[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
      if (test_energy_ && en>=HBpersistentThreshold_)
	++abovepersistent[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
      if (test_et_ && et>=HBpersistentETThreshold_)
	++abovepersistentET[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    } else if (id.subdet()==HcalEndcap)	{
      if (en>=HEenergyThreshold_)
	++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
      if (et>=HEETThreshold_)
	++aboveet[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
      if (test_energy_) 
	if (en>=HEpersistentThreshold_)
	  ++abovepersistent[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
      if (test_et_) 
	if (et>=HEpersistentETThreshold_)
	  ++abovepersistentET[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    }
  } //for (HBHERecHitCollection::const_iterator HBHEiter=...)

  // loop over HO
  for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) { // loop over all hits
    float en = HOiter->energy();
     
    HcalDetId id(HOiter->detid().rawId());
    int ieta = id.ieta();
    int iphi = id.iphi();
    int depth = id.depth();
    std::pair<double,double> etas = topology.etaRange(id.subdet(),abs(ieta));
    double fEta=fabs(0.5*(etas.first+etas.second));
    float et = en/cosh(fEta);

    if (test_neighbor_ || makeDiagnostics_)
      processHit_rechitNeighbors(HOiter, hoHits, HONeighborParams_, topology);

    if (isSiPM(ieta,iphi,depth)) {
      if (en>=HOenergyThreshold_*SiPMscale_)
	++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]; 
      if (et>=HOETThreshold_*SiPMscale_)
	++aboveet[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]; 
      if (test_energy_) 
	if (en>=HOpersistentThreshold_*SiPMscale_)
	  ++abovepersistent[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
      if (test_et_) 
	if (et>=HOpersistentETThreshold_*SiPMscale_)
	  ++abovepersistentET[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    } else {
      // Skip HO ring 2 when required
      if (abs(ieta)>10 && excludeHORing2_==true)
	continue;

      if (en>=HOenergyThreshold_)
	++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]; 
      if (et>=HOETThreshold_)
	++aboveet[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]; 
      if (test_energy_) 
	if (en>=HOpersistentThreshold_)
	  ++abovepersistent[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
      if (test_et_) 
	if (en>=HOpersistentETThreshold_)
	  ++abovepersistentET[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    }
  }
    
  // loop over HF
  for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter)  { // loop over all hits
    float en = HFiter->energy();
    float threshold=HFenergyThreshold_;
    float threshold_pers = HFpersistentThreshold_; 
    float etthreshold=HFETThreshold_;
    HcalDetId id(HFiter->detid().rawId());
    int ieta = id.ieta();
    int iphi = id.iphi();
    int depth = id.depth();
    std::pair<double,double> etas = topology.etaRange(id.subdet(),abs(ieta));
    double fEta=fabs(0.5*(etas.first+etas.second));
    float et = en/cosh(fEta);

    if (test_neighbor_ || makeDiagnostics_)
      processHit_rechitNeighbors(HFiter, hfHits, HFNeighborParams_, topology);

    if (abs(ieta)>39) { // increase the thresholds in far-forward part of HF
      threshold*=HFfarfwdScale_;
      threshold_pers*=HFfarfwdScale_;
    }
      
    if (en>=threshold)
      ++aboveenergy[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    if (et>=etthreshold)
      ++aboveet[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    if (test_energy_) {
      if (en>=threshold_pers)
	++abovepersistent[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    }
    if (test_et_) {
      if (et>=HFpersistentETThreshold_)
	++abovepersistentET[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1];
    }
  }

  // call update every event -- still necessary?
 
  for (unsigned int i=0;i<AbovePersistentThresholdCellsByDepth.depth.size();++i)
    AbovePersistentThresholdCellsByDepth.depth[i]->update();
  for (unsigned int i=0;i<AboveEnergyThresholdCellsByDepth.depth.size();++i)
    AboveEnergyThresholdCellsByDepth.depth[i]->update();
  for (unsigned int i=0;i<AboveETThresholdCellsByDepth.depth.size();++i)
    AboveETThresholdCellsByDepth.depth[i]->update();
  for (unsigned int i=0;i<AboveNeighborsHotCellsByDepth.depth.size();++i)
    AboveNeighborsHotCellsByDepth.depth[i]->update();

  return;
} // void HcalHotCellMonitor::processEvent_rechitenergy

/* --------------------------------------- */

 
template <class RECHIT, class RECHITCOLLECTION>
void HcalHotCellMonitor::processHit_rechitNeighbors(RECHIT& rechit,
						    RECHITCOLLECTION& coll,
						    hotNeighborParams& params,
						    const HcalTopology& topology) {
  // Compares energy to energy of neighboring cells.
  // This is a slightly simplified version of D0's NADA algorithm
  // 17 June 2009 -- this needs major work.  I'm not sure I have the [eta][phi][depth] array mapping correct everywhere. 
  // Maybe even tear it apart and start again?
 
  int ieta, iphi, depth;
  float en;
  
  int neighborsfound=0;
  float enNeighbor=0;

  en = rechit->energy();
  HcalDetId id(rechit->detid().rawId());
  ieta = id.ieta();
  iphi = id.iphi();
  depth = id.depth();
 
  std::pair<double,double> etas = topology.etaRange(id.subdet(),abs(ieta));
  double fEta=fabs(0.5*(etas.first+etas.second));

  float et = en/cosh(fEta);
  
  // Case 0:  ET too low to trigger hot cell check
  if (et<=params.minCellEnergy) return;
  
  // Case 1:  above threshold energy; always count as hot
  if (et>params.maxEnergy) {
    if (makeDiagnostics_) {
      // fill overflow bin when energy > max threshold
      if       (id.subdet()==HcalBarrel)  d_HBenergyVsNeighbor->Fill(1000);
      else if  (id.subdet()==HcalEndcap)  d_HEenergyVsNeighbor->Fill(1000);
      else if  (id.subdet()==HcalOuter)   d_HOenergyVsNeighbor->Fill(1000);
      else if  (id.subdet()==HcalForward) d_HFenergyVsNeighbor->Fill(1000);
    }
    return;
  }
     
  // Case 2:  Search keys for neighboring cells

  neighborsfound=0;
  enNeighbor=0;

  int mydeltaphi=params.DeltaIphi;
  // scale appropriately for larger cells at higher eta values
  if (abs(ieta)>39) mydeltaphi*=4;
  else if (abs(ieta)>20) mydeltaphi*=2;

  for (int nD=-1*params.DeltaDepth;nD<=params.DeltaDepth;++nD) {
    for (int nP =-1*mydeltaphi;nP<=mydeltaphi;++nP) {
      for (int nE =-1*params.DeltaIeta;nE<=params.DeltaIeta;++nE) {
	if (nD==0 && nE==0 && nP==0) 
	  continue; // don't count the cell itself
	int myphi=(nP+iphi)%72;
	HcalDetId myid((HcalSubdetector)(1), nE+ieta, myphi, nD+depth); // HB
	RECHIT part=coll.find(myid);
	if (part==coll.end())
	  continue;
	if (part->energy()<params.minNeighborEnergy)
	  continue;
	++neighborsfound;
	enNeighbor+=part->energy();
      } // loop over nE (neighbor eta)
    } // loop over nP (neighbor phi)
  } // loop over nD depths
 
  // Case 2a:  Not enough good neighbors found -- do we want to implement this?
  //if (neighborsfound==0)
  //  return;

  // Case 2b: (avg. neighbor energy)/energy too large for cell to be considered hot
  if (makeDiagnostics_) {
    int myval=(int)(enNeighbor/en*50);
    if (myval<0) myval=0;
    if (myval>499) myval=499;
    if (enNeighbor/en<0 || enNeighbor/en>=10) return;
    if       (id.subdet()==HcalBarrel)  ++hbVsNeighbor[myval];
    else if  (id.subdet()==HcalEndcap)  ++heVsNeighbor[myval];
    else if  (id.subdet()==HcalOuter)   ++hoVsNeighbor[myval];
    else if  (id.subdet()==HcalForward) ++hfVsNeighbor[myval];
  }
  if ((1.*enNeighbor/en)>params.HotEnergyFrac && en>0 && enNeighbor>0)
    return;
  
  // Case 2c:  Tests passed; cell marked as hot
  aboveneighbors[CalcEtaBin(id.subdet(),ieta,depth)][iphi-1][depth-1]++;

  return;
} // void HcalHotCellMonitor::processEvent_rechitneighbor


/* --------------------------------------- */


void HcalHotCellMonitor::fillNevents_persistentenergy(const HcalTopology& topology) {
  // Fill Histograms showing rechits with energies > some threshold for N consecutive events

  if (levt_<minEvents_) return;

  if (debug_>0)
    std::cout <<"<HcalHotCellMonitor::fillNevents_persistentenergy> FILLING PERSISTENT ENERGY PLOTS"<<std::endl;
  
  if (test_energy_) {
    for (unsigned int h=0;h<AbovePersistentThresholdCellsByDepth.depth.size();++h)
      AbovePersistentThresholdCellsByDepth.depth[h]->setBinContent(0,0,ievt_);

    int ieta=0;
    int iphi=0;
    int etabins=0;
    int phibins=0;

    for (unsigned int depth=0;depth<AbovePersistentThresholdCellsByDepth.depth.size();++depth) { 
      etabins=AbovePersistentThresholdCellsByDepth.depth[depth]->getNbinsX();
      phibins=AbovePersistentThresholdCellsByDepth.depth[depth]->getNbinsY();

      for (int eta=0;eta<etabins;++eta)	{
	for (int phi=0;phi<phibins;++phi) {
	  iphi=phi+1;
	  for (int subdet=1;subdet<=4;++subdet) {
	    ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1); //converts bin to ieta
	    if (ieta==-9999) continue;
	    if (!(topology.validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1)))
	      continue;
	    if (subdet==HcalForward) // shift HcalForward ieta by 1 for filling purposes
	      ieta<0 ? ieta-- : ieta++;
		   
	    // MUST BE ABOVE ENERGY THRESHOLD FOR ALL N EVENTS in a luminosity block
	    if (abovepersistent[eta][phi][depth]<levt_) {
	      abovepersistent[eta][phi][depth]=0;
	      continue;  		
	    }
	    if (debug_>0) std::cout <<"HOT CELL; PERSISTENT ENERGY at subdet = "<<subdet<<", eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth<<std::endl;
	    AbovePersistentThresholdCellsByDepth.depth[depth]->Fill(ieta,iphi,abovepersistent[eta][phi][depth]);
	    AbovePersistentThresholdCellsByDepth.depth[depth]->setBinContent(0,0,ievt_);
	    abovepersistent[eta][phi][depth]=0; // reset counter
	  } // for (int subdet=1; subdet<=4;++subdet)
	} // for (int phi=0;...)
      } // for (int eta=0;...)
    } // for (unsigned int depth=0;...)
    FillUnphysicalHEHFBins(AbovePersistentThresholdCellsByDepth);
  } // if (test_energy_)

  if (test_et_) {
    for (unsigned int h=0;h<AbovePersistentETThresholdCellsByDepth.depth.size();++h)
      AbovePersistentETThresholdCellsByDepth.depth[h]->setBinContent(0,0,ievt_);
      
    int ieta=0;
    int iphi=0;
    int etabins=0;
    int phibins=0;
    
    for (unsigned int depth=0;depth<AbovePersistentETThresholdCellsByDepth.depth.size();++depth) { 
      etabins=AbovePersistentETThresholdCellsByDepth.depth[depth]->getNbinsX();
      phibins=AbovePersistentETThresholdCellsByDepth.depth[depth]->getNbinsY();
	  
      for (int eta=0;eta<etabins;++eta) {
	for (int phi=0;phi<phibins;++phi) {
	  iphi=phi+1;
	  for (int subdet=1;subdet<=4;++subdet) {
	    ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1); //converts bin to ieta
	    if (ieta==-9999) continue;
	    if (!(topology.validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1)))
	      continue;
	    if (subdet==HcalForward) // shift HcalForward ieta by 1 for filling purposes
	      ieta<0 ? ieta-- : ieta++;
		      
	    // MUST BE ABOVE ET THRESHOLD FOR ALL N EVENTS in a luminosity block
	    if (abovepersistentET[eta][phi][depth]<levt_) {
	      abovepersistentET[eta][phi][depth]=0;
	      continue;  		
	    }
	    if (debug_>0) std::cout <<"HOT CELL; PERSISTENT ENERGY at subdet = "<<subdet<<", eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth<<std::endl;
	    AbovePersistentETThresholdCellsByDepth.depth[depth]->Fill(ieta,iphi,abovepersistentET[eta][phi][depth]);
	    AbovePersistentETThresholdCellsByDepth.depth[depth]->setBinContent(0,0,ievt_);
	    abovepersistentET[eta][phi][depth]=0; // reset counter
	  } // for (int subdet=1; subdet<=4;++subdet)
	} // for (int phi=0;...)
      } // for (int eta=0;...)
    } // for (unsigned int depth=0;...)
    FillUnphysicalHEHFBins(AbovePersistentETThresholdCellsByDepth);
    
  } // if (test_et_)
  // Add test_ET
  return;
} // void HcalHotCellMonitor::fillNevents_persistentenergy(const HcalTopology&)



/* ----------------------------------- */

void HcalHotCellMonitor::fillNevents_energy(const HcalTopology& topology) {
  // Fill Histograms showing rec hits that are above some energy value 
  // (Fill for each instance when cell is above energy; don't require it to be hot for a number of consecutive events)

  if (debug_>0)
    std::cout <<"<HcalHotCellMonitor::fillNevents_energy> ABOVE-ENERGY-THRESHOLD PLOTS"<<std::endl;
  
  if (test_energy_) {
    for (unsigned int h=0;h<AboveEnergyThresholdCellsByDepth.depth.size();++h)
      AboveEnergyThresholdCellsByDepth.depth[h]->setBinContent(0,0,ievt_);
  }
  if (test_et_) {
    for (unsigned int h=0;h<AboveETThresholdCellsByDepth.depth.size();++h)
      AboveETThresholdCellsByDepth.depth[h]->setBinContent(0,0,ievt_);
  }

  int ieta=0;
  int iphi=0;
  int etabins=0;
  int phibins=0;
  unsigned int maxdepth=0;
  
  if (test_energy_)
    maxdepth = AboveEnergyThresholdCellsByDepth.depth.size();
  if (maxdepth==0 && test_et_)
    maxdepth = AboveETThresholdCellsByDepth.depth.size();
  for (unsigned int depth=0;depth<maxdepth;++depth)  { 
    if (test_energy_) {
      etabins=AboveEnergyThresholdCellsByDepth.depth[depth]->getNbinsX();
      phibins=AboveEnergyThresholdCellsByDepth.depth[depth]->getNbinsY();
    }
    if (test_et_) {
      etabins=AboveETThresholdCellsByDepth.depth[depth]->getNbinsX();
      phibins=AboveETThresholdCellsByDepth.depth[depth]->getNbinsY();
    }
    for (int eta=0;eta<etabins;++eta) {
      for (int phi=0;phi<phibins;++phi) {
	iphi=phi+1;
	for (int subdet=1;subdet<=4;++subdet) {
	  ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1); //converts bin to ieta
	  if (ieta==-9999) continue;
	  if (!(topology.validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1)))
	    continue;
	  if (subdet==HcalForward) // shift HcalForward ieta by 1 for filling purposes
	    ieta<0 ? ieta-- : ieta++;
		  
	  if (test_energy_) {
	    if (aboveenergy[eta][phi][depth]>0)	{
	      if (debug_>2) 
		std::cout <<"HOT CELL; ABOVE ENERGY THRESHOLD at subdet = "<<subdet<<", eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<"  ABOVE THRESHOLD IN "<<aboveenergy[eta][phi][depth]<<"  EVENTS"<<std::endl;
	      AboveEnergyThresholdCellsByDepth.depth[depth]->Fill(ieta,iphi, aboveenergy[eta][phi][depth]);
	      aboveenergy[eta][phi][depth]=0;
	    } // if (aboveenergy[eta][phi][depth])
	  } // if (test_energy_)
	  if (test_et_)  {
	    if (aboveet[eta][phi][depth]>0) {
	      if (debug_>2) 
		std::cout <<"HOT CELL; ABOVE ET THRESHOLD at subdet = "<<subdet<<", eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<"  ABOVE THRESHOLD IN "<<aboveet[eta][phi][depth]<<"  EVENTS"<<std::endl;
	      AboveETThresholdCellsByDepth.depth[depth]->Fill(ieta,iphi, aboveet[eta][phi][depth]);
	      aboveet[eta][phi][depth]=0;
	    } // if (aboveet[eta][phi][depth])
	  } // if (test_et_)
	} // for (int subdet=0)
      } // for (int phi=0;...)
    } // for (int eta=0;...)
  } // for (int depth=0;...)

  if (test_energy_) 
    FillUnphysicalHEHFBins(AboveEnergyThresholdCellsByDepth);

  if (test_et_)
    FillUnphysicalHEHFBins(AboveETThresholdCellsByDepth);

  return;


} // void HcalHotCellMonitor::fillNevents_energy(const HcalTopology&)



/* ----------------------------------- */

void HcalHotCellMonitor::fillNevents_neighbor(const HcalTopology& topology) {
  // Fill Histograms showing rec hits with energy much less than neighbors' average

  if (debug_>0)
    std::cout <<"<HcalHotCellMonitor::fillNevents_neighbor> FILLING ABOVE-NEIGHBOR-ENERGY PLOTS"<<std::endl;

  for (unsigned int h=0;h<AboveNeighborsHotCellsByDepth.depth.size();++h)
    AboveNeighborsHotCellsByDepth.depth[h]->setBinContent(0,0,ievt_);

  int ieta=0;
  int iphi=0;
  int etabins=0;
  int phibins=0;
  
  for (unsigned int depth=0;depth<AboveNeighborsHotCellsByDepth.depth.size();++depth) { 
    etabins=AboveNeighborsHotCellsByDepth.depth[depth]->getNbinsX();
    phibins=AboveNeighborsHotCellsByDepth.depth[depth]->getNbinsY();
      
    for (int eta=0;eta<etabins;++eta) {
      for (int phi=0;phi<phibins;++phi) {
	iphi=phi+1;
	for (int subdet=1;subdet<=4;++subdet) {
	  ieta=CalcIeta((HcalSubdetector)subdet,eta,depth+1); //converts bin to ieta
	  if (ieta==-9999) continue;
	  if (!(topology.validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1)))
	    continue;
	  if (subdet==HcalForward) // shift HcalForward ieta by 1 for filling purposes
	    ieta<0 ? ieta-- : ieta++;
		  
	  if (aboveneighbors[eta][phi][depth]>0) {
	    if (debug_>2) std::cout <<"HOT CELL; ABOVE NEIGHBORS at eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<(depth>4 ? depth+1 : depth-3)<<std::endl;
	    AboveNeighborsHotCellsByDepth.depth[depth]->Fill(ieta,iphi,aboveneighbors[eta][phi][depth]);
	    //reset counter
	    aboveneighbors[eta][phi][depth]=0;
	  } // if (aboveneighbors[eta][phi][mydepth]>0)
	} // for (int subdet=1;...)
      } // for (int phi=0;...)
    } // for (int eta=0;...)
  } // for (unsigned int depth=0;...)
  FillUnphysicalHEHFBins(AboveNeighborsHotCellsByDepth);

  if (!makeDiagnostics_) return;
  for (int i=0;i<500;++i) {
    d_HBenergyVsNeighbor->Fill(i/50.,hbVsNeighbor[i]);
    hbVsNeighbor[i]=0;
    d_HEenergyVsNeighbor->Fill(i/50.,heVsNeighbor[i]);
    heVsNeighbor[i]=0;
    d_HOenergyVsNeighbor->Fill(i/50.,hoVsNeighbor[i]);
    hoVsNeighbor[i]=0;
    d_HFenergyVsNeighbor->Fill(i/50.,hfVsNeighbor[i]);
    hfVsNeighbor[i]=0;
  }

  return;

} // void HcalHotCellMonitor::fillNevents_neighbor(const HcalTopology&)






void HcalHotCellMonitor::fillNevents_problemCells(const HcalTopology& topology){
  if (debug_>0)
    std::cout <<"<HcalHotCellMonitor::fillNevents_problemCells> FILLING PROBLEM CELL PLOTS"<<std::endl;

  if (ievt_==0) return;  // no events; no need to bother with this 

  int ieta=0;
  int etabins=0;
  int phibins=0;
  bool problemvalue=false;

  // Count problem cells in each subdetector
  int NumBadHB=0;
  int NumBadHE=0;
  int NumBadHO=0;
  int NumBadHF=0;
  int NumBadHO0=0;
  int NumBadHO12=0;
  int NumBadHFLUMI=0;

  unsigned int DEPTH = 0;

  if (test_persistent_)  
    {
      if (test_energy_)
	DEPTH = AbovePersistentThresholdCellsByDepth.depth.size();
      else if (test_et_)
	DEPTH = AbovePersistentETThresholdCellsByDepth.depth.size();
    }
  else if (test_energy_ && DEPTH==0)    DEPTH = AboveEnergyThresholdCellsByDepth.depth.size();
  else if (test_et_ && DEPTH==0)        DEPTH = AboveETThresholdCellsByDepth.depth.size();
  else if (test_neighbor_ && DEPTH==0)  DEPTH = AboveNeighborsHotCellsByDepth.depth.size();
  
  if (DEPTH==0) return;

  for (unsigned int depth=0;depth<DEPTH;++depth)
    {
      if (test_persistent_) 
	{
	  if (test_energy_)
	    {
	      etabins=AbovePersistentThresholdCellsByDepth.depth[depth]->getNbinsX();
	      phibins=AbovePersistentThresholdCellsByDepth.depth[depth]->getNbinsY();
	    }
	  else if (test_et_)
	    {
              etabins=AbovePersistentETThresholdCellsByDepth.depth[depth]->getNbinsX();
              phibins=AbovePersistentETThresholdCellsByDepth.depth[depth]->getNbinsY();
            }
	}

      if (test_neighbor_ && (etabins==0 || phibins==0))
	{
	  etabins=AboveNeighborsHotCellsByDepth.depth[depth]->getNbinsX();
	  phibins=AboveNeighborsHotCellsByDepth.depth[depth]->getNbinsY();
	}

      if (test_energy_ && (etabins==0 || phibins==0))
	{
	  etabins=AboveEnergyThresholdCellsByDepth.depth[depth]->getNbinsX();
	  phibins=AboveEnergyThresholdCellsByDepth.depth[depth]->getNbinsY();
	}

      if (test_et_ && (etabins==0 || phibins==0))
	{
	  etabins=AboveETThresholdCellsByDepth.depth[depth]->getNbinsX();
	  phibins=AboveETThresholdCellsByDepth.depth[depth]->getNbinsY();
	}

      for (int eta=0;eta<etabins;++eta)
	{
	  ieta=CalcIeta(eta,depth+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      if (abs(ieta)>20 && phi%2==1) continue; //skip non-physical cells
	      else if (abs(ieta)>39 && (phi+1)%4!=3) continue;
	      // find problem rate for particular cell
	      problemvalue=false;
	      if (test_energy_ && test_persistent_ && AbovePersistentThresholdCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1)>minErrorFlag_*ievt_)
		problemvalue=true;
	      if (test_neighbor_ && AboveNeighborsHotCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1)>minErrorFlag_*ievt_)
		problemvalue=true;
	      if (test_energy_  && AboveEnergyThresholdCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1)>minErrorFlag_*ievt_)
		problemvalue=true;
	      if (test_et_      && AboveETThresholdCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1)>minErrorFlag_*ievt_)
		problemvalue=true;
	      if (test_et_ && test_persistent_ && AbovePersistentETThresholdCellsByDepth.depth[depth]->getBinContent(eta+1,phi+1)>minErrorFlag_*ievt_)
		problemvalue=true;
	      if (problemvalue==false) continue;
	      if (isHB(eta,depth+1)) ++NumBadHB;
	      else if (isHE(eta,depth+1)) 
		++NumBadHE;
	      else if (isHO(eta,depth+1))
		{
		  ++NumBadHO;
		  if (abs(ieta)<5) ++NumBadHO0;
		  else ++NumBadHO12;
		}
	      else if (isHF(eta,depth+1)) 
		{
		  ++NumBadHF;
		  if (depth+1==1 && (abs(ieta)==33 || abs(ieta)==34)) ++NumBadHFLUMI;
		  else if (depth+1==2 && (abs(ieta)==35 || abs(ieta)==36)) ++NumBadHFLUMI;
		}
	    } // for (int phi=0;...)
	} //for (int eta=0;...)
    } // for (int depth=0;...)
  
  if (debug_>2) std::cout <<"<HcalHotCellMonitor::fillNevents_problemCells>  Num Bad HB = "<<NumBadHB<<"  Num Bad HE = "<<NumBadHE<<"  Num Bad HO = "<<NumBadHO<<"  Num Bad HF = "<<NumBadHF<<"  CURRENT LS = "<<currentLS<<std::endl;
  // Fill number of problem cells
  ProblemsVsLB_HB->Fill(currentLS,NumBadHB);
  ProblemsVsLB_HE->Fill(currentLS,NumBadHE);
  ProblemsVsLB_HO->Fill(currentLS,NumBadHO);
  ProblemsVsLB_HF->Fill(currentLS,NumBadHF);
  ProblemsVsLB_HBHEHF->Fill(currentLS,NumBadHB+NumBadHE+NumBadHF);
  ProblemsVsLB->Fill(currentLS,NumBadHB+NumBadHE+NumBadHO+NumBadHF);

  ProblemsCurrentLB->Fill(-1,-1,levt_);
  ProblemsCurrentLB->Fill(0,0,NumBadHB);
  ProblemsCurrentLB->Fill(1,0,NumBadHE);
  ProblemsCurrentLB->Fill(2,0,NumBadHO);
  ProblemsCurrentLB->Fill(3,0,NumBadHF);
  ProblemsCurrentLB->Fill(4,0,NumBadHO0);
  ProblemsCurrentLB->Fill(5,0,NumBadHO12);
  ProblemsCurrentLB->Fill(6,0,NumBadHFLUMI);

} // void HcalHotCellMonitor::fillNevents_problemCells(const HcalTopology&)


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
	      abovepersistentET[i][j][k]=0;
              aboveneighbors[i][j][k]=0;
              aboveenergy[i][j][k]=0;
              aboveet[i][j][k]=0;
	      rechit_occupancy_sum[i][j][k]=0;
            }
        }
    }

  for (int i=0;i<500;++i)
    {
      hbVsNeighbor[i]=0;
      heVsNeighbor[i]=0;
      hoVsNeighbor[i]=0;
      hfVsNeighbor[i]=0;
    }
  return;

} // void HcalHotCellMonitor::zeroCounters()

void HcalHotCellMonitor::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  // Anything to do here?
}

void HcalHotCellMonitor::endJob()
{
  if (debug_>0) std::cout <<"HcalHotCellMonitor::endJob()"<<std::endl;
  if (enableCleanup_) cleanup(); // when do we force cleanup?
}

/*void HcalHotCellMonitor::cleanup()
{
  if (debug_>0) std::cout <<"HcalHotCellMonitor::cleanup()"<<std::endl;
  if (!enableCleanup_) return;
  if (dbe_)
    {
      // removeContents doesn't remove subdirectories
      dbe_->setCurrentFolder(subdir_);
      dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"hot_rechit_above_threshold");
      dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"hot_rechit_always_above_threshold");
      dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"hot_neighbortest");
      dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"LSvalues");
      dbe_->removeContents();
    }
}*/ // cleanup

void HcalHotCellMonitor::periodicReset()
{

  // first reset base class objects
  //FIX HcalBaseMonitor::periodicReset();

  // then reset the temporary histograms
  zeroCounters();

  // now reset all the MonitorElements

  // resetting eta-phi histograms
  if (test_neighbor_)
    AboveNeighborsHotCellsByDepth.Reset();
  if (test_energy_ || makeDiagnostics_)
    AboveEnergyThresholdCellsByDepth.Reset();
  if (test_et_ || makeDiagnostics_)
    AboveETThresholdCellsByDepth.Reset();
  if (test_persistent_)
    {
      AbovePersistentThresholdCellsByDepth.Reset();
      if (test_et_)
	AbovePersistentETThresholdCellsByDepth.Reset();
    }
  return;
}
DEFINE_FWK_MODULE(HcalHotCellMonitor);

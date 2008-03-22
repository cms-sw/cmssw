#include "DQM/HcalMonitorTasks/interface/HcalHotCellMonitor.h"
#include <map>

// Use for stringstream
#include <iostream>
#include <iomanip>

HcalHotCellMonitor::HcalHotCellMonitor() {
  ievt_=0;
}

HcalHotCellMonitor::~HcalHotCellMonitor() {
}
void HcalHotCellMonitor::reset(){}

void HcalHotCellMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  baseFolder_ = rootFolder_+"HotCellMonitor";

  // Set input parameters from .cfi file
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  if (fVerbosity) cout << "HotCell eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  if (fVerbosity) cout << "HotCell phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;

  checkHB_=ps.getUntrackedParameter<bool>("checkHB","true");
  checkHE_=ps.getUntrackedParameter<bool>("checkHE","true");
  checkHO_=ps.getUntrackedParameter<bool>("checkHO","true");
  checkHF_=ps.getUntrackedParameter<bool>("checkHF","true");

  // Energy thresholds for 
  thresholds_ = ps.getUntrackedParameter<vector <double> >("thresholds");
  HEthresholds_ = ps.getUntrackedParameter<vector <double> >("HEthresholds",thresholds_);
  HBthresholds_ = ps.getUntrackedParameter<vector <double> >("HBthresholds",thresholds_);
  HFthresholds_ = ps.getUntrackedParameter<vector <double> >("HFthresholds",thresholds_);
  HOthresholds_ = ps.getUntrackedParameter<vector <double> >("HOthresholds",thresholds_);

  NADA_Ecand_cut0_ = ps.getUntrackedParameter<double>("NADA_Ecand_cut0",1.);
  NADA_Ecand_cut1_ = ps.getUntrackedParameter<double>("NADA_Ecand_cut1",5.);
  NADA_Ecand_cut2_ = ps.getUntrackedParameter<double>("NADA_Ecand_cut2",500.);
  NADA_Ecube_cut_ = ps.getUntrackedParameter<double>("NADA_Ecube_cut",.1);
  NADA_Ecell_cut_ = ps.getUntrackedParameter<double>("NADA_Ecell_cut",.1);
  // Changed negative cut from D0 default of -1 GeV to -1.5 GeV based on CMS run 24934
  NADA_NegCand_cut_ = ps.getUntrackedParameter<double>("NADA_NegCand_cut",-1.5);
  NADA_Ecube_frac_ = ps.getUntrackedParameter<double>("NADA_Ecube_frac",0.02);
  NADA_Ecell_frac_ = ps.getUntrackedParameter<double>("NADA_Ecell_frac",0.02);
  NADA_maxdepth_ = ps.getUntrackedParameter<int>("NADA_maxdepth",1);
  NADA_maxeta_ = ps.getUntrackedParameter<int>("NADA_maxeta",1);
  NADA_maxphi_ = ps.getUntrackedParameter<int>("NADA_maxphi",1);


  HB_NADA_Ecand_cut0_ = ps.getUntrackedParameter<double>("HB_NADA_Ecand_cut0",NADA_Ecand_cut0_);
  HB_NADA_Ecand_cut1_ = ps.getUntrackedParameter<double>("HB_NADA_Ecand_cut1",NADA_Ecand_cut1_);
  HB_NADA_Ecand_cut2_ = ps.getUntrackedParameter<double>("HB_NADA_Ecand_cut2",NADA_Ecand_cut2_);
  HB_NADA_Ecube_cut_ = ps.getUntrackedParameter<double>("HB_NADA_Ecube_cut",NADA_Ecube_cut_);
  HB_NADA_Ecell_cut_ = ps.getUntrackedParameter<double>("HB_NADA_Ecell_cut",NADA_Ecell_cut_);
  HB_NADA_NegCand_cut_ = ps.getUntrackedParameter<double>("HB_NADA_NegCand_cut",NADA_NegCand_cut_);
  HB_NADA_Ecube_frac_ = ps.getUntrackedParameter<double>("HB_NADA_Ecube_frac",NADA_Ecube_frac_);
  HB_NADA_Ecell_frac_ = ps.getUntrackedParameter<double>("HB_NADA_Ecell_frac",NADA_Ecell_frac_);
  HB_NADA_maxdepth_ = ps.getUntrackedParameter<int>("HB_NADA_maxdepth",NADA_maxdepth_);
  HB_NADA_maxeta_ = ps.getUntrackedParameter<int>("HB_NADA_maxeta",NADA_maxeta_);
  HB_NADA_maxphi_ = ps.getUntrackedParameter<int>("HB_NADA_maxphi",NADA_maxeta_);


  HE_NADA_Ecand_cut0_ = ps.getUntrackedParameter<double>("HE_NADA_Ecand_cut0",NADA_Ecand_cut0_);
  HE_NADA_Ecand_cut1_ = ps.getUntrackedParameter<double>("HE_NADA_Ecand_cut1",NADA_Ecand_cut1_);
  HE_NADA_Ecand_cut2_ = ps.getUntrackedParameter<double>("HE_NADA_Ecand_cut2",NADA_Ecand_cut2_);
  HE_NADA_Ecube_cut_ = ps.getUntrackedParameter<double>("HE_NADA_Ecube_cut",NADA_Ecube_cut_);
  HE_NADA_Ecell_cut_ = ps.getUntrackedParameter<double>("HE_NADA_Ecell_cut",NADA_Ecell_cut_);
  HE_NADA_NegCand_cut_ = ps.getUntrackedParameter<double>("HE_NADA_NegCand_cut",NADA_NegCand_cut_);
  HE_NADA_Ecube_frac_ = ps.getUntrackedParameter<double>("HE_NADA_Ecube_frac",NADA_Ecube_frac_);
  HE_NADA_Ecell_frac_ = ps.getUntrackedParameter<double>("HE_NADA_Ecell_frac",NADA_Ecell_frac_);
  HE_NADA_maxdepth_ = ps.getUntrackedParameter<int>("HE_NADA_maxdepth",NADA_maxdepth_);
  HE_NADA_maxeta_ = ps.getUntrackedParameter<int>("HE_NADA_maxeta",NADA_maxeta_);
  HE_NADA_maxphi_ = ps.getUntrackedParameter<int>("HE_NADA_maxphi",NADA_maxeta_);

  HO_NADA_Ecand_cut0_ = ps.getUntrackedParameter<double>("HO_NADA_Ecand_cut0",NADA_Ecand_cut0_);
  HO_NADA_Ecand_cut1_ = ps.getUntrackedParameter<double>("HO_NADA_Ecand_cut1",NADA_Ecand_cut1_);
  HO_NADA_Ecand_cut2_ = ps.getUntrackedParameter<double>("HO_NADA_Ecand_cut2",NADA_Ecand_cut2_);
  HO_NADA_Ecube_cut_ = ps.getUntrackedParameter<double>("HO_NADA_Ecube_cut",NADA_Ecube_cut_);
  HO_NADA_Ecell_cut_ = ps.getUntrackedParameter<double>("HO_NADA_Ecell_cut",NADA_Ecell_cut_);
  HO_NADA_NegCand_cut_ = ps.getUntrackedParameter<double>("HO_NADA_NegCand_cut",NADA_NegCand_cut_);
  HO_NADA_Ecube_frac_ = ps.getUntrackedParameter<double>("HO_NADA_Ecube_frac",NADA_Ecube_frac_);
  HO_NADA_Ecell_frac_ = ps.getUntrackedParameter<double>("HO_NADA_Ecell_frac",NADA_Ecell_frac_);
  HO_NADA_maxdepth_ = ps.getUntrackedParameter<int>("HO_NADA_maxdepth",NADA_maxdepth_);
  HO_NADA_maxeta_ = ps.getUntrackedParameter<int>("HO_NADA_maxeta",NADA_maxeta_);
  HO_NADA_maxphi_ = ps.getUntrackedParameter<int>("HO_NADA_maxphi",NADA_maxeta_);

  HF_NADA_Ecand_cut0_ = ps.getUntrackedParameter<double>("HF_NADA_Ecand_cut0",NADA_Ecand_cut0_);
  HF_NADA_Ecand_cut1_ = ps.getUntrackedParameter<double>("HF_NADA_Ecand_cut1",NADA_Ecand_cut1_);
  HF_NADA_Ecand_cut2_ = ps.getUntrackedParameter<double>("HF_NADA_Ecand_cut2",NADA_Ecand_cut2_);
  HF_NADA_Ecube_cut_ = ps.getUntrackedParameter<double>("HF_NADA_Ecube_cut",NADA_Ecube_cut_);
  HF_NADA_Ecell_cut_ = ps.getUntrackedParameter<double>("HF_NADA_Ecell_cut",NADA_Ecell_cut_);
  HF_NADA_NegCand_cut_ = ps.getUntrackedParameter<double>("HF_NADA_NegCand_cut",NADA_NegCand_cut_);
  HF_NADA_Ecube_frac_ = ps.getUntrackedParameter<double>("HF_NADA_Ecube_frac",NADA_Ecube_frac_);
  HF_NADA_Ecell_frac_ = ps.getUntrackedParameter<double>("HF_NADA_Ecell_frac",NADA_Ecell_frac_);
  HF_NADA_maxdepth_ = ps.getUntrackedParameter<int>("HF_NADA_maxdepth",NADA_maxdepth_);
  HF_NADA_maxeta_ = ps.getUntrackedParameter<int>("HF_NADA_maxeta",NADA_maxeta_);
  HF_NADA_maxphi_ = ps.getUntrackedParameter<int>("HF_NADA_maxphi",NADA_maxeta_);

  ievt_=0;
  
  if ( m_dbe !=NULL ) {    

    m_dbe->setCurrentFolder(baseFolder_);

    meEVT_ = m_dbe->bookInt("HotCell Task Event Number");    
    meEVT_->Fill(ievt_);


    meMAX_E_all =  m_dbe->book1D("HotCellEnergy","HotCell Energy",2000,0,200);
    meMAX_T_all =  m_dbe->book1D("HotCellTime","HotCell Time",175,-50,300);
    
    meOCC_MAP_L1= m_dbe->book2D("HotCellDepth1OccupancyMap","HotCell Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L1= m_dbe->book2D("HotCellDepth1EnergyMap","HotCell Depth 1 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_L2= m_dbe->book2D("HotCellDepth2OccupancyMap","HotCell Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L2= m_dbe->book2D("HotCellDepth2EnergyMap","HotCell Depth 2 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_L3= m_dbe->book2D("HotCellDepth3OccupancyMap","HotCell Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L3= m_dbe->book2D("HotCellDepth3EnergyMap","HotCell Depth 3 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_L4= m_dbe->book2D("HotCellDepth4OccupancyMap","HotCell Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L4= m_dbe->book2D("HotCellDepth4EnergyMap","HotCell Depth 4 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_all = m_dbe->book2D("HotCellOccupancyMap","HotCell Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_all  = m_dbe->book2D("HotCellEnergyMap","HotCell Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    NADA_NumHotCells= m_dbe->book1D("NADA_NumHotCells","# of NADA Hot Cells/Event",1000,0,1000);
    NADA_NumNegCells= m_dbe->book1D("NADA_NumNegCells","# of NADA Negative-Energy Cells/Event",1000,0,1000);


    // Book HB histograms
    m_dbe->setCurrentFolder(baseFolder_+"/HB");
    hbHists.meMAX_E =  m_dbe->book1D("HBHotCellEnergy","HB HotCell Energy",2000,0,20);
    hbHists.meMAX_T =  m_dbe->book1D("HBHotCellTime","HB HotCell Time",200,-50,300);
    hbHists.meMAX_ID =  m_dbe->book1D("HBHotCellID","HB HotCell ID",36000,-18000,18000);
    
    hbHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HBHotCellGeoOccupancyMap_MaxCell","HB HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HBHotCellGeoEnergyMap_MaxCell","HB HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    /*
    // The SetOption command doesn't work -- MonitorElement doesn't inherit all standard root histogram methods? 
    hbHists.meOCC_MAP_GEO_Max->SetOption("box");
    hbHists.meEN_MAP_GEO_Max->SetOption("box");
    */

    
    for (int k=0;k<int(HBthresholds_.size());k++)
      {
	std::stringstream myoccname;
	myoccname<<"HBHotCellOCCmap_Thresh"<<k;
	const char *occname=myoccname.str().c_str();
	std::stringstream myocctitle;
	myocctitle<<"HB Hot Cell Occupancy, Cells > "<<HBthresholds_[k]<<" GeV";
	const char *occtitle=myocctitle.str().c_str();
	hbHists.OCCmap.push_back(m_dbe->book2D(occname,occtitle,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));

	std::stringstream myenergyname;
	myenergyname<<"HBHotCellENERGYmap_Thresh"<<k;
	const char *energyname=myenergyname.str().c_str();
	std::stringstream myenergytitle;
	myenergytitle<<"HB Hot Cell Energy, Cells > "<<HBthresholds_[k]<<" GeV";
	const char *energytitle=myenergytitle.str().c_str();
	hbHists.ENERGYmap.push_back(m_dbe->book2D(energyname,energytitle,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      }



    NADA_hbHists.NADA_OCC_MAP = m_dbe->book2D("NADA_HB_OCC_MAP","NADA HB Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hbHists.NADA_EN_MAP = m_dbe->book2D("NADA_HB_EN_MAP","NADA HB Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hbHists.NADA_NumHotCells = m_dbe->book1D("NADA_HB_NumHotCells","# of NADA HB Hot Cells/Event",1000,0,1000);
    NADA_hbHists.NADA_testcell = m_dbe->book1D("NADA_HB_testcell","Energy for test cell",1000,-10,90);
    NADA_hbHists.NADA_Energy = m_dbe->book1D("NADA_HB_Energy","Energy for all cells",1000,-10,90);
    NADA_hbHists.NADA_NumNegCells = m_dbe->book1D("NADA_HB_NumNegCells","# of NADA HB Negative-Energy Cells/Event",1000,0,1000);
    NADA_hbHists.NADA_NEG_OCC_MAP = m_dbe->book2D("NADA_HB_NEG_OCC_MAP","NADA HB Negative Energy Cell Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hbHists.NADA_NEG_EN_MAP = m_dbe->book2D("NADA_HB_NEG_EN_MAP","NADA HB Negative Cell Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


    // Book HE histograms
    m_dbe->setCurrentFolder(baseFolder_+"/HE");
    heHists.meMAX_E =  m_dbe->book1D("HEHotCellEnergy","HE HotCell Energy",2000,0,20);
    heHists.meMAX_T =  m_dbe->book1D("HEHotCellTime","HE HotCell Time",200,-50,300);
    heHists.meMAX_ID =  m_dbe->book1D("HEHotCellID","HE HotCell ID",36000,-18000,18000);
    

    heHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HEHotCellGeoOccupancyMap_MaxCell","HE HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HEHotCellGeoEnergyMap_MaxCell","HE HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    /*
    heHists.meOCC_MAP_GEO_Max->SetOption("box");
    heHists.meEN_MAP_GEO_Max->SetOption("box");
    */
    
    for (int k=0;k<int(HEthresholds_.size());k++)
      {
	std::stringstream myoccname;
	myoccname<<"HEHotCellOCCmap_Thresh"<<k;
	const char *occname=myoccname.str().c_str();
	std::stringstream myocctitle;
	myocctitle<<"HE Hot Cell Occupancy, Cells > "<<HEthresholds_[k]<<" GeV";
	const char *occtitle=myocctitle.str().c_str();
	heHists.OCCmap.push_back(m_dbe->book2D(occname,occtitle,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));

	std::stringstream myenergyname;
	myenergyname<<"HEHotCellENERGYmap_Thresh"<<k;
	const char *energyname=myenergyname.str().c_str();
	std::stringstream myenergytitle;
	myenergytitle<<"HE Hot Cell Energy, Cells > "<<HEthresholds_[k]<<" GeV";
	const char *energytitle=myenergytitle.str().c_str();
	heHists.ENERGYmap.push_back(m_dbe->book2D(energyname,energytitle,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      }



    NADA_heHists.NADA_OCC_MAP = m_dbe->book2D("NADA_HE_OCC_MAP","NADA HE Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_heHists.NADA_EN_MAP = m_dbe->book2D("NADA_HE_EN_MAP","NADA HE Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_heHists.NADA_NumHotCells = m_dbe->book1D("NADA_HE_NumHotCells","# of NADA HE Hot Cells/Event",1000,0,1000);
    NADA_heHists.NADA_testcell = m_dbe->book1D("NADA_HE_testcell","Energy for test cell",1000,-10,90);
    NADA_heHists.NADA_Energy = m_dbe->book1D("NADA_HE_Energy","Energy for all cells",1000,-10,90);
    NADA_heHists.NADA_NumNegCells = m_dbe->book1D("NADA_HE_NumNegCells","# of NADA HE Negative-Energy Cells/Event",1000,0,1000);
    NADA_heHists.NADA_NEG_OCC_MAP = m_dbe->book2D("NADA_HE_NEG_OCC_MAP","NADA HE Negative Energy Cell Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_heHists.NADA_NEG_EN_MAP = m_dbe->book2D("NADA_HE_NEG_EN_MAP","NADA HE Negative Cell Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


    // Book HF histograms
    m_dbe->setCurrentFolder(baseFolder_+"/HF");
    hfHists.meMAX_E =  m_dbe->book1D("HFHotCellEnergy","HF HotCell Energy",2000,0,20);
    hfHists.meMAX_T =  m_dbe->book1D("HFHotCellTime","HF HotCell Time",200,-50,300);
    hfHists.meMAX_ID =  m_dbe->book1D("HFHotCellID","HF HotCell ID",36000,-18000,18000);



    hfHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HFHotCellGeoOccupancyMap_MaxCell","HF HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HFHotCellGeoEnergyMap_MaxCell","HF HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    /*
    hfHists.meOCC_MAP_GEO_Max->SetOption("box");
    hfHists.meEN_MAP_GEO_Max->SetOption("box");
    */

    for (int k=0;k<int(HFthresholds_.size());k++)
      {
	std::stringstream myoccname;
	myoccname<<"HFHotCellOCCmap_Thresh"<<k;
	const char *occname=myoccname.str().c_str();
	std::stringstream myocctitle;
	myocctitle<<"HF Hot Cell Occupancy, Cells > "<<HFthresholds_[k]<<" GeV";
	const char *occtitle=myocctitle.str().c_str();
	hfHists.OCCmap.push_back(m_dbe->book2D(occname,occtitle,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));

	std::stringstream myenergyname;
	myenergyname<<"HFHotCellENERGYmap_Thresh"<<k;
	const char *energyname=myenergyname.str().c_str();
	std::stringstream myenergytitle;
	myenergytitle<<"HF Hot Cell Energy, Cells > "<<HFthresholds_[k]<<" GeV";
	const char *energytitle=myenergytitle.str().c_str();
	hfHists.ENERGYmap.push_back(m_dbe->book2D(energyname,energytitle,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      }



    NADA_hfHists.NADA_OCC_MAP = m_dbe->book2D("NADA_HF_OCC_MAP","NADA HF Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hfHists.NADA_EN_MAP = m_dbe->book2D("NADA_HF_EN_MAP","NADA HF Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hfHists.NADA_NumHotCells = m_dbe->book1D("NADA_HF_NumHotCells","# of NADA HF Hot Cells/Event",1000,0,1000);
    NADA_hfHists.NADA_testcell = m_dbe->book1D("NADA_HF_testcell","Energy for test cell",1000,-10,90);
    NADA_hfHists.NADA_Energy = m_dbe->book1D("NADA_HF_Energy","Energy for all cells",1000,-10,90);
    NADA_hfHists.NADA_NumNegCells = m_dbe->book1D("NADA_HF_NumNegCells","# of NADA HF Negative-Energy Cells/Event",1000,0,1000);
    NADA_hfHists.NADA_NEG_OCC_MAP = m_dbe->book2D("NADA_HF_NEG_OCC_MAP","NADA HF Negative Energy Cell Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hfHists.NADA_NEG_EN_MAP = m_dbe->book2D("NADA_HF_NEG_EN_MAP","NADA HF Negative Cell Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    // Book HO histograms
    m_dbe->setCurrentFolder(baseFolder_+"/HO");
    hoHists.meMAX_E =  m_dbe->book1D("HOHotCellEnergy","HO HotCell Energy",2000,0,20);
    hoHists.meMAX_T =  m_dbe->book1D("HOHotCellTime","HO HotCell Time",200,-50,300);
    hoHists.meMAX_ID =  m_dbe->book1D("HOHotCellID","HO HotCell ID",36000,-18000,18000);
    
    hoHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HOHotCellGeoOccupancyMap_MaxCell","HO HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HOHotCellGeoEnergyMap_MaxCell","HO HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    /*
    hoHists.meOCC_MAP_GEO_Max->SetOption("box");
    hoHists.meEN_MAP_GEO_Max->SetOption("box");
    */

    for (int k=0;k<int(HOthresholds_.size());k++)
      {
	std::stringstream myoccname;
	myoccname<<"HOHotCellOCCmap_Thresh"<<k;
	const char *occname=myoccname.str().c_str();
	std::stringstream myocctitle;
	myocctitle<<"HO Hot Cell Occupancy, Cells > "<<HOthresholds_[k]<<" GeV";
	const char *occtitle=myocctitle.str().c_str();
	hoHists.OCCmap.push_back(m_dbe->book2D(occname,occtitle,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));

	std::stringstream myenergyname;
	myenergyname<<"HOHotCellENERGYmap_Thresh"<<k;
	const char *energyname=myenergyname.str().c_str();
	std::stringstream myenergytitle;
	myenergytitle<<"HO Hot Cell Energy, Cells > "<<HOthresholds_[k]<<" GeV";
	const char *energytitle=myenergytitle.str().c_str();
	hoHists.ENERGYmap.push_back(m_dbe->book2D(energyname,energytitle,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      }


    NADA_hoHists.NADA_OCC_MAP = m_dbe->book2D("NADA_HO_OCC_MAP","NADA HO Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hoHists.NADA_EN_MAP = m_dbe->book2D("NADA_HO_EN_MAP","NADA HO Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hoHists.NADA_NumHotCells = m_dbe->book1D("NADA_HO_NumHotCells","# of NADA HO Hot Cells/Event",1000,0,1000);
    NADA_hoHists.NADA_testcell = m_dbe->book1D("NADA_HO_testcell","Energy for test cell",1000,-10,90);
    NADA_hoHists.NADA_Energy = m_dbe->book1D("NADA_HO_Energy","Energy for all cells",1000,-10,90);
    NADA_hoHists.NADA_NumNegCells = m_dbe->book1D("NADA_HO_NumNegCells","# of NADA HO Negative-Energy Cells/Event",1000,0,1000);
    NADA_hoHists.NADA_NEG_OCC_MAP = m_dbe->book2D("NADA_HO_NEG_OCC_MAP","NADA HO Negative Energy Cell Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hoHists.NADA_NEG_EN_MAP = m_dbe->book2D("NADA_HO_NEG_EN_MAP","NADA HO Negative Cell Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


    // Book global histograms
    m_dbe->setCurrentFolder(baseFolder_+"/HCAL");
    hcalHists.meMAX_E =  m_dbe->book1D("HCALHotCellEnergy","HCAL HotCell Energy",2000,0,20);
    hcalHists.meMAX_T =  m_dbe->book1D("HCALHotCellTime","HCAL HotCell Time",200,-50,300);
    hcalHists.meMAX_ID =  m_dbe->book1D("HCALHotCellID","HCAL HotCell ID",36000,-18000,18000);
    
    hcalHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HCALHotCellGeoOccupancyMap_MaxCell","HCAL HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hcalHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HCALHotCellGeoEnergyMap_MaxCell","HCAL HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    /*
    hoHists.meOCC_MAP_GEO_Max->SetOption("box");
    hoHists.meEN_MAP_GEO_Max->SetOption("box");
    */

    for (int k=0;k<int(thresholds_.size());k++)
      {
	std::stringstream myoccname;
	myoccname<<"HCALHotCellOCCmap_Thresh"<<k;
	const char *occname=myoccname.str().c_str();
	std::stringstream myocctitle;
	myocctitle<<"HCAL Hot Cell Occupancy, Cells > "<<thresholds_[k]<<" GeV";
	const char *occtitle=myocctitle.str().c_str();
	hcalHists.OCCmap.push_back(m_dbe->book2D(occname,occtitle,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));

	std::stringstream myenergyname;
	myenergyname<<"HCALHotCellENERGYmap_Thresh"<<k;
	const char *energyname=myenergyname.str().c_str();
	std::stringstream myenergytitle;
	myenergytitle<<"HCAL Hot Cell Energy, Cells > "<<thresholds_[k]<<" GeV";
	const char *energytitle=myenergytitle.str().c_str();
	hcalHists.ENERGYmap.push_back(m_dbe->book2D(energyname,energytitle,etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_));
      }


    NADA_hcalHists.NADA_OCC_MAP = m_dbe->book2D("NADA_HCAL_OCC_MAP","NADA HCAL Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hcalHists.NADA_EN_MAP = m_dbe->book2D("NADA_HCAL_EN_MAP","NADA HCAL Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hcalHists.NADA_NumHotCells = m_dbe->book1D("NADA_HCAL_NumHotCells","# of NADA HCAL Hot Cells/Event",1000,0,1000);
    NADA_hcalHists.NADA_testcell = m_dbe->book1D("NADA_HCAL_testcell","Energy for test cell",1000,-10,90);
    NADA_hcalHists.NADA_Energy = m_dbe->book1D("NADA_HCAL_Energy","Energy for all cells",1000,-10,90);
    NADA_hcalHists.NADA_NumNegCells = m_dbe->book1D("NADA_HCAL_NumNegCells","# of NADA HCAL Negative-Energy Cells/Event",1000,0,1000);
    NADA_hcalHists.NADA_NEG_OCC_MAP = m_dbe->book2D("NADA_HCAL_NEG_OCC_MAP","NADA HCAL Negative Energy Cell Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hcalHists.NADA_NEG_EN_MAP = m_dbe->book2D("NADA_HCAL_NEG_EN_MAP","NADA HCAL Negative Cell Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


  } // if (m_dbe !=NULL)

  // Code to correct for wrong pedestal offset in early data
  // At this point (23 Oct 2007), the code could be removed
  // Keep it around, though, just in case it's needed in the future.
  /*
  if (fVerbosity) cout <<"INITIALIZING OFFSETS"<<endl;
  for (int i=0;i<13;i++)
    {
      for (int j =0;j<36;j++)
	{ 
	  for (int k=0;k<2;k++)
	    {
	      //HF_offsets[i][j][k]=-11.39;
	      HF_offsets[i][j][k]=0.;
	    }
	}
    }
  */
  return;

}

void HcalHotCellMonitor::processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits){

  if(!m_dbe) { printf("HcalHotCellMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }

  ievt_++;
  meEVT_->Fill(ievt_);

 // Fill HF thresholds on first event

  if (fVerbosity) cout <<"HcalHotCellMonitor::processEvent   Starting process"<<endl;
  /*
    // Unnecessary code, now that offset problem has been corrected
  if (ievt_==1)
    {
      
      // Replace initial offset with offset from original event.
      // This is still not a perfect solution (or even a good one), but it's a start.
      if (hfHits.size()>0)
	{
	  
	  for (HFRecHitCollection::const_iterator _if=hfHits.begin(); _if!=hfHits.end(); _if++)
	    {
	      if (_if->energy()<0)
		HF_offsets[_if->id().ieta()-29][(_if->id().iphi()-1)/2][_if->id().depth()]=_if->energy();
	      else
		HF_offsets[_if->id().ieta()-29][(_if->id().iphi()-1)/2][_if->id().depth()]=0;
	    }

	}

    } // if (ievt_==1)
  */


  // Set coordinates of highest-energy cells in each subdetector
  enS=0; tS=0; etaS=0; phiS=0; idS=-1;
  enA=0; tA=0; etaA=0; phiA=0;
  depth=0;

  hotcells=0;
  negcells=0;

  if (checkHB_) // check Barrel
    {
      FindHBHEHotCells(hbHits,hbHists,1);
      HBHE_NADAFinder(hbHits,NADA_hbHists,1);
    }
  if (checkHE_) // endcap gets same hit collections as HB
    {
      FindHBHEHotCells(hbHits,heHists,0);
      HBHE_NADAFinder(hbHits,NADA_heHists,0);
    }
  if (checkHO_) // check out calorimeter
    {
      FindHOHotCells(hoHits,hoHists);
      HO_NADAFinder(hoHits,NADA_hoHists);
    }
  if (checkHF_) // check forward calorimeter
    {
      FindHFHotCells(hfHits,hfHists);
      HF_NADAFinder(hfHits,NADA_hfHists);
    }


  // ///////////////////////////////////////////////////////

  if (fVerbosity) cout <<"Checking enA > threshold"<<endl;

  if(enA>thresholds_[0]){
    if (fVerbosity) cout <<"Filling MAX histograms"<<endl;
    meMAX_E_all->Fill(enA);
    meMAX_T_all->Fill(tA);
    meOCC_MAP_all->Fill(etaA,phiA);
    meEN_MAP_all->Fill(etaA,phiA,enA);
    // hcalHists now contains duplicates of meMAX histograms
    // remove meMAX histos in the future?
    hcalHists.meMAX_E->Fill(enA);
    hcalHists.meMAX_T->Fill(enA);
    hcalHists.meOCC_MAP_GEO_Max->Fill(etaA,phiA);
    hcalHists.meEN_MAP_GEO_Max->Fill(etaA,phiA,enA);
    
    if(depth==1)
      {
	if (fVerbosity) cout <<"\tFilling Depth1 histos"<<endl;
	meOCC_MAP_L1->Fill(etaA,phiA);
	meEN_MAP_L1->Fill(etaA,phiA,enA);
      }
    else if(depth==2)
      {
	if (fVerbosity) cout <<"\tFilling Depth2 histos"<<endl;
	meOCC_MAP_L2->Fill(etaA,phiA);
	meEN_MAP_L2->Fill(etaA,phiA,enA);
      }
    else if(depth==3)
      {
	if (fVerbosity) cout <<"\tFilling Depth3 histos"<<endl;
	meOCC_MAP_L3->Fill(etaA,phiA);
	meEN_MAP_L3->Fill(etaA,phiA,enA);
      }
    else if(depth==4)
      {
	if (fVerbosity) cout <<"\tFilling Depth4 histos"<<endl;
	
	meOCC_MAP_L4->Fill(etaA,phiA);
	meEN_MAP_L4->Fill(etaA,phiA,enA);
      }
  }

  // These histograms are now duplicates of NADA_hcalHists -- remove in future
  NADA_NumHotCells->Fill(hotcells);
  NADA_NumNegCells->Fill(negcells);

  NADA_hcalHists.NADA_NumHotCells->Fill(hotcells);
  NADA_hcalHists.NADA_NumNegCells->Fill(negcells);

  return;
}


void HcalHotCellMonitor::FindHBHEHotCells(const HBHERecHitCollection& hbHits, HistList& hist, bool HB=true)

{  
  enS=0, tS=0, etaS=0, phiS=0;
  int depthS=0;

  HBHERecHitCollection::const_iterator _ib;
  if (fVerbosity)
    {
      if (HB) cout << "looping over HB"<<endl;
      else cout <<"looping over HE"<<endl;
    }

  if(hbHits.size()>0)
    {
      for (_ib=hbHits.begin(); _ib!=hbHits.end(); _ib++) 
	{ // loop over all hits  

	  // Check that subdetector region is correct
	  if (HB && (HcalSubdetector)(_ib->id().subdet())!=HcalBarrel) continue;
	  else if (!HB && (HcalSubdetector)(_ib->id().subdet())!=HcalEndcap) continue;
	
	  if(vetoCell(_ib->id()))
	    {
	      if (fVerbosity) cout <<"Vetoed cell with id = "<<_ib->id()<<endl;
	      continue;
	    }

	  double cellenergy=_ib->energy();

	  if (HB==true)
	    {
	      for (int k=0;k<int(HBthresholds_.size());k++)
		{
		  if (cellenergy>HBthresholds_[k])
		    {
		      if (hbHists.OCCmap[k]!=NULL)
			{
			  hbHists.OCCmap[k]->Fill(_ib->id().ieta(),_ib->id().iphi());
			  hcalHists.OCCmap[k]->Fill(_ib->id().ieta(),_ib->id().iphi());
			}
		      if (hbHists.ENERGYmap[k]!=NULL)
			{
			  hbHists.ENERGYmap[k]->Fill(_ib->id().ieta(),_ib->id().iphi(),cellenergy);
			  hcalHists.ENERGYmap[k]->Fill(_ib->id().ieta(),_ib->id().iphi(),cellenergy);
			}
		    }
		}
	    }
	  else
	    {
	      for (unsigned int k=0;k<HEthresholds_.size();k++)
		{
		  if (cellenergy>HEthresholds_[k])
		    {
		      if (heHists.OCCmap[k]!=NULL)
			{
			  heHists.OCCmap[k]->Fill(_ib->id().ieta(),_ib->id().iphi());
			  hcalHists.OCCmap[k]->Fill(_ib->id().ieta(),_ib->id().iphi());
			}
		      if (heHists.ENERGYmap[k]!=NULL)
			{
			  heHists.ENERGYmap[k]->Fill(_ib->id().ieta(),_ib->id().iphi(),cellenergy);
			  hcalHists.ENERGYmap[k]->Fill(_ib->id().ieta(),_ib->id().iphi(),cellenergy);
			}
		    }
		}
	    }
	  
	  if(cellenergy>enS)
	    {
	      enS = cellenergy;
	      tS = _ib->time();
	      etaS = _ib->id().ieta();
	      phiS = _ib->id().iphi();
	      idS = 1000*etaS;
	      depthS = _ib->id().depth(); // change depth before altering idS?
	      if(idS<0) idS -= (10*phiS+depthS);
	      else idS += (10*phiS+depthS);
	      //depth = _ib->id().depth();
	    }

	} // loop over all hits
 
      if(HB==true) // && enS>HBthresholds_[0] ??
	{
	  hbHists.meMAX_E->Fill(enS);
	  hbHists.meMAX_T->Fill(tS);
	  hbHists.meOCC_MAP_GEO_Max->Fill(etaS,phiS);
	  hbHists.meEN_MAP_GEO_Max->Fill(etaS,phiS,enS);
	  hbHists.meMAX_ID->Fill(idS);
	}
      else if (HB==false) // && enS>HEthresholds_[0])
	{
	  heHists.meMAX_E->Fill(enS);
	  heHists.meMAX_T->Fill(tS);
	  heHists.meOCC_MAP_GEO_Max->Fill(etaS,phiS);
	  heHists.meEN_MAP_GEO_Max->Fill(etaS,phiS,enS);
	  heHists.meMAX_ID->Fill(idS);
	}

      if(enS>enA)
	{
	  enA = enS;
	  tA = tS;
	  etaA = etaS;
	  phiA = phiS;
	  depth=depthS;
	}
    } // if (hbHits.size()>0)
  return;
  
}


void HcalHotCellMonitor::FindHOHotCells(const HORecHitCollection& hoHits, HistList& hist)

{  
  enS=0, tS=0, etaS=0, phiS=0;
  int depthS=0;
  
  HORecHitCollection::const_iterator _io;

  if(hoHits.size()>0)
    {
      for (_io=hoHits.begin(); _io!=hoHits.end(); _io++) 
	{ // loop over all hits
	  if (vetoCell(_io->id()))
	    {
	      if (fVerbosity) cout <<"Vetoing HO cell with ID = "<<_io->id()<<endl;
	      continue;
	    }

	  double cellenergy = _io->energy();
	  for (unsigned int k=0;k<HOthresholds_.size();k++)
	    {
	      if (cellenergy>HOthresholds_[k])
		{
		  if (hoHists.OCCmap[k]!=NULL)
		    {
		      hoHists.OCCmap[k]->Fill(_io->id().ieta(),_io->id().iphi());
		      hcalHists.OCCmap[k]->Fill(_io->id().ieta(),_io->id().iphi());
		    }
		  if (hoHists.ENERGYmap[k]!=NULL)
		    {
		      hoHists.ENERGYmap[k]->Fill(_io->id().ieta(),_io->id().iphi(),cellenergy);
		      hcalHists.ENERGYmap[k]->Fill(_io->id().ieta(),_io->id().iphi(),cellenergy);
		    }
		}
	    }
	  
	  if(cellenergy>enS)
	      {
		enS = cellenergy;
		tS = _io->time();
		etaS = _io->id().ieta();
		phiS = _io->id().iphi();
		idS = 1000*etaS;
		depthS = _io->id().depth();
		if(idS<0) idS -= (10*phiS+depthS);
		else idS += (10*phiS+depthS);
		//depth = _io->id().depth();
	      }
	} // for (_io=hoHits.begin...)  loop over all hits

      if(enS>0)
	{
	  hoHists.meMAX_E->Fill(enS);
	  hoHists.meMAX_T->Fill(tS);
	  hoHists.meOCC_MAP_GEO_Max->Fill(etaS,phiS);
	  hoHists.meEN_MAP_GEO_Max->Fill(etaS,phiS,enS);
	  hoHists.meMAX_ID->Fill(idS);
	}
    } // if (hoHits.size()>0)
  if(enS>enA)
    {
      enA = enS;
      tA = tS;
      etaA = etaS;
      phiA = phiS;
      depth = depthS;
    }
  return;
}


void HcalHotCellMonitor::FindHFHotCells(const HFRecHitCollection& hfHits, HistList& hist)
{
  
  enS=0, tS=0, etaS=0, phiS=0;
  int depthS=0;
  
  HFRecHitCollection::const_iterator _if;
  
  if(hfHits.size()>0)
    {
      for (_if=hfHits.begin(); _if!=hfHits.end(); _if++) 
	{ // loop over all HF hits
	  if (vetoCell(_if->id()))
	    {
	      if (fVerbosity) cout <<"Vetoing HF cell with ID = "<<_if->id()<<endl;
	      continue;
	    }

	  double cellenergy = _if->energy();
	  for (unsigned int k=0;k<HFthresholds_.size();k++)
	    {
	      if (cellenergy>HFthresholds_[k])
		{
		  if (hfHists.OCCmap[k]!=NULL)
		    {
		      hfHists.OCCmap[k]->Fill(_if->id().ieta(),_if->id().iphi());
		      hcalHists.OCCmap[k]->Fill(_if->id().ieta(),_if->id().iphi());
		    }
		  if (hfHists.ENERGYmap[k]!=NULL)
		    {
		      hfHists.ENERGYmap[k]->Fill(_if->id().ieta(),_if->id().iphi(),cellenergy);
		      hcalHists.ENERGYmap[k]->Fill(_if->id().ieta(),_if->id().iphi(),cellenergy);
		    }
		}
	    }
	  
	  if(cellenergy>enS)
	    {
	      enS = cellenergy;
	      tS = _if->time();
	      etaS = _if->id().ieta();
	      phiS = _if->id().iphi();
	      idS = 1000*etaS;
	      depthS = _if->id().depth();
	      if(idS<0) idS -= (10*phiS+depthS);
	      else idS += (10*phiS+depthS);
	      //depth = _if->id().depth();
	    }
	} // loop over all HF hits
      if(enS>0)
	{
	  hfHists.meMAX_E->Fill(enS);
	  hfHists.meMAX_T->Fill(tS);
	  hfHists.meOCC_MAP_GEO_Max->Fill(etaS,phiS);
	  hfHists.meEN_MAP_GEO_Max->Fill(etaS,phiS,enS);
	  hfHists.meMAX_ID->Fill(idS);
	}
    
    } // if (hfHits.size()>0)
 
  if(enS>enA)
    {
      enA = enS;
      tA = tS;
      etaA = etaS;
      phiA = phiS;
      depth=depthS;
    }
  return;
}


void HcalHotCellMonitor::HBHE_NADAFinder(const HBHERecHitCollection& c, NADAHistList& h, bool HB=true)
{
  if (c.size()>0)
    {
      
      HBHERecHitCollection::const_iterator _cell;
      
      // Copying NADA algorithm from D0 Note 4057.
      // The implementation needs to be optimized -- double looping over iterators is not an efficient approach.
      float  Ecube=0;
      int numhotcells=0;
      int numnegcells=0;
      int CellPhi=-1, CellEta=-1, CellDepth=-1;
      
      float HBHE_Ecube_cut=0.;
      float HBHE_Ecell_cut=0.;
      
      float NADA_neg_cut, NADA_Ecell_frac, NADA_Ecube_frac;
      float NADA_cube_cut, NADA_cell_cut;
      float NADA_cand_cut0, NADA_cand_cut1, NADA_cand_cut2; 
      
      int NADA_maxdepth, NADA_maxeta, NADA_maxphi;

      if (HB==true)
	{
	  NADA_neg_cut = HB_NADA_NegCand_cut_;
	  NADA_cand_cut0 = HB_NADA_Ecand_cut0_;
	  NADA_cand_cut1 = HB_NADA_Ecand_cut1_;
	  NADA_cand_cut2 = HB_NADA_Ecand_cut2_;
	  NADA_cube_cut = HB_NADA_Ecube_cut_;
	  NADA_cell_cut = HB_NADA_Ecell_cut_;
	  NADA_Ecell_frac = HB_NADA_Ecell_frac_;
	  NADA_Ecube_frac = HB_NADA_Ecube_frac_;
	  NADA_maxdepth = HB_NADA_maxdepth_;
	  NADA_maxeta = HB_NADA_maxeta_;
	  NADA_maxphi = HB_NADA_maxphi_;
	}
      else
	{
	  NADA_neg_cut = HE_NADA_NegCand_cut_;
	  NADA_cand_cut0 = HE_NADA_Ecand_cut0_;
	  NADA_cand_cut1 = HE_NADA_Ecand_cut1_;
	  NADA_cand_cut2 = HE_NADA_Ecand_cut2_;
	  NADA_cube_cut = HE_NADA_Ecube_cut_;
	  NADA_cell_cut = HE_NADA_Ecell_cut_;
	  NADA_Ecell_frac = HE_NADA_Ecell_frac_;
	  NADA_Ecube_frac = HE_NADA_Ecube_frac_;
	  NADA_maxdepth = HE_NADA_maxdepth_;
	  NADA_maxeta = HE_NADA_maxeta_;
	  NADA_maxphi = HE_NADA_maxphi_;
	}
      
      
      float cellenergy=0;
      
      for (_cell=c.begin(); _cell!=c.end(); _cell++)
	{
	  // HB==true:  loop over barrels only
	  if (HB==true && (HcalSubdetector)(_cell->id().subdet())!=HcalBarrel) continue;
	  // HB==false:  loop over endcap only
	  else if (HB==false && (HcalSubdetector)(_cell->id().subdet())!=HcalEndcap) continue;
	  
	  // Make histogram that stores vetoCell energies?
	  if (vetoCell(_cell->id())) continue;
	  cellenergy=_cell->energy();
	  
	  h.NADA_Energy->Fill(cellenergy);
	  NADA_hcalHists.NADA_Energy->Fill(cellenergy);

	  if (fVerbosity && cellenergy<0) cout <<"WARNING:  NEGATIVE CELL ENERGY FOUND IN HBHE NADA:  "<<cellenergy<<endl;
	  
	  // _cell points to the current hot cell candidate
	  Ecube=0; // reset Ecube energy counter
	  
	  // Case 1:  E< -1 GeV or E>500 GeV:  Each counts as hot cell
	  if (cellenergy<NADA_neg_cut || cellenergy>NADA_cand_cut2)
	    {
	      // Case 1a:  E< negative cutoff
	      if (cellenergy<NADA_neg_cut) 
		{ 
		  if (fVerbosity) cout <<"<NEGATIVE HBHE CELL ENERGY>  Energy = "<<cellenergy<<" at position ("<<_cell->id().ieta()<<", "<<_cell->id().iphi()<<")  HB = "<<HB<<endl;
		  numnegcells++;
		  h.NADA_NEG_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  NADA_hcalHists.NADA_NEG_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  // Fill with -1*E to make plotting easier (large negative values appear as peaks rather than troughs, etc.)
		  h.NADA_NEG_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),-1*cellenergy);
		  NADA_hcalHists.NADA_NEG_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),-1*cellenergy);
		}
	      // Case 1b:  E>maximum
	      else
		{
		  numhotcells++;
		  h.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  
		  h.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
		  NADA_hcalHists.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  
		  NADA_hcalHists.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
		}
	      // Cells marked as hot; no need to complete remaining code
	      continue;
	      
	    }
	  // Case 2:  Energy is < negative cutoff, but less than minimum threshold -- skip the cell
	  
	  else if (cellenergy<=NADA_cand_cut0)
	    continue;
	  
	  // Case 3:  Set thresholds according to input variables
	  else if (cellenergy>NADA_cand_cut0 && cellenergy<NADA_cand_cut1) 
	    {
	      HBHE_Ecube_cut=NADA_cube_cut;
	      HBHE_Ecell_cut=NADA_cell_cut;
	    }
	  
	  // Case 3A: IF E>5 and <500, set thresholds based on fraction of energy in candidate cell
	  else if (cellenergy>=NADA_cand_cut1 && cellenergy<=NADA_cand_cut2)
	    {
	      HBHE_Ecube_cut=NADA_Ecube_frac*cellenergy;
	      HBHE_Ecell_cut=NADA_Ecell_frac*cellenergy;
	    }
	  
	  CellPhi = _cell->id().iphi();
	  CellEta = _cell->id().ieta();
	  CellDepth = _cell->id().depth();
	  
	  // Form cube of nearest neighbor cells around _cell

	  if (fVerbosity) cout <<"****** Candidate Cell Energy: "<<cellenergy<<endl;
	  for (HBHERecHitCollection::const_iterator _neighbor = c.begin();_neighbor!=c.end();_neighbor++)
	    // Form cube centered on _cell.  This needs to be looked at more carefully to deal with boundary conditions.  Should Ecube constraints change at the boundaries?
	    {
	      // An HE cell can be a neighbor of an HB cell, right?
	      // Do we want to include both?  Probably not; different subsystems
	      
	      if (vetoCell(_neighbor->id())) continue; // unnecessary?  Vetoed earlier?
	      // Get only cells in cube around candiate cell
	      if  ((HcalSubdetector)(_neighbor->id().subdet())!=(HcalSubdetector)(_cell->id().subdet())) continue;
	      if (abs(_neighbor->id().depth()-CellDepth)>NADA_maxdepth) continue;
	      if (abs(_neighbor->id().ieta()-CellEta)>NADA_maxeta) continue;
	      if ((abs(_neighbor->id().iphi()-CellPhi)%72)>NADA_maxphi) continue;

		  
	      if (_neighbor->energy()>HBHE_Ecell_cut)
		{
		  if (fVerbosity) cout <<"\t Neighbor energy = "<<_neighbor->energy()<<endl;
		  Ecube+=_neighbor->energy();
		  if (fVerbosity) cout <<"\t\t Cube energy = "<<Ecube<<endl;
		}
	    } // for (cell_iter _neighbor = c.begin()...)
	  
	  //Remove energy due to _cell
	  Ecube -=cellenergy;
	  if (fVerbosity) cout <<"\t\t\t\t Final Cube energy = "<<Ecube<<endl;

	  if (fVerbosity && Ecube <=HBHE_Ecube_cut)
	    {
	      cout <<"NADA Hot Cell found!"<<endl;
	      cout <<"\t NADA Ecube energy: "<<Ecube<<endl;
	      cout <<"\t NADA Ecell energy: "<<cellenergy<<endl;
	      cout <<"\t NADA Cell position: "<<_cell->id().ieta()<<", "<<_cell->id().iphi()<<endl;
	    }
	  
	  // Identify hot cells by value of Ecube
	  if (Ecube <= HBHE_Ecube_cut)
	    {   
	      if (fVerbosity) cout <<"Found NADA hot cell in HBHE:  Ecube energy = "<<Ecube<<endl;
	      numhotcells++;
	      h.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
	      
	      h.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
	      NADA_hcalHists.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
	      
	      NADA_hcalHists.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
	    }
	} //for (_cell=c.begin(); _cell!=c.end(); _cell++)
      if (fVerbosity) cout <<"Filling HBHE NADA NumHotCell histo"<<endl;
      h.NADA_NumHotCells->Fill(numhotcells);
      h.NADA_NumNegCells->Fill(numnegcells);
      hotcells+=numhotcells;
      negcells+=numnegcells;
      
    } // if c.size()>0
  return;
}


void HcalHotCellMonitor::HO_NADAFinder(const HORecHitCollection& c, NADAHistList& h)
{
  if (c.size()>0)
    {
      
      HORecHitCollection::const_iterator _cell;
      
      // Copying NADA algorithm from D0 Note 4057.
      // The implementation needs to be optimized -- double looping over iterators is not an efficient approach.
      float  Ecube=0;
      int numhotcells=0;
      int numnegcells=0;
      int CellPhi=-1, CellEta=-1, CellDepth=-1;
      
      float HO_Ecube_cut=0.;
      float HO_Ecell_cut=0.;
      
      float cellenergy=0.;
      int NADA_maxdepth, NADA_maxeta, NADA_maxphi;
      NADA_maxdepth = HO_NADA_maxdepth_;
      NADA_maxeta = HO_NADA_maxeta_;
      NADA_maxphi = HO_NADA_maxphi_;

      for (_cell=c.begin(); _cell!=c.end(); _cell++)
	{
	  if (vetoCell(_cell->id()))continue; // veto cell before or after filling energy histogram?  Make separate vetocell histos?
	  cellenergy=_cell->energy();
	  h.NADA_Energy->Fill(cellenergy);
	  NADA_hcalHists.NADA_Energy->Fill(cellenergy);
	   		
	  // _cell points to the current hot cell candidate
	  Ecube=0; // reset Ecube energy counter
	    
	  // Case 1:  E< -1 GeV or E>500 GeV
	  if (cellenergy<HO_NADA_NegCand_cut_ || cellenergy>HO_NADA_Ecand_cut2_)
	    {
	      // Case 1a:   E< negative threshold
	      if (cellenergy<HO_NADA_NegCand_cut_)
		{
		  if (fVerbosity) cout <<"<NEGATIVE HO CELL ENERGY>  Energy = "<<cellenergy<<endl;
		  numnegcells++;
		  h.NADA_NEG_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  h.NADA_NEG_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),-1*cellenergy);
		  NADA_hcalHists.NADA_NEG_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  NADA_hcalHists.NADA_NEG_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),-1*cellenergy);
		}
	      else
		{
		  numhotcells++;
		  h.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  h.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
		  NADA_hcalHists.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  NADA_hcalHists.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
		}
	      continue;
	    } // end of Case 1

	    
	  // Case 2:  Energy is < negative cutoff, but less than minimum threshold -- skip the cell
	    
	  else if (cellenergy<=HO_NADA_Ecand_cut0_)
	    continue;

	  // Case 3:  Set thresholds according to input variables
	  else if (cellenergy>HO_NADA_Ecand_cut0_ && cellenergy<HO_NADA_Ecand_cut1_)
	    {
	      HO_Ecube_cut=HO_NADA_Ecube_cut_;
	      HO_Ecell_cut=HO_NADA_Ecell_cut_;
	    }
	  // Case 4:  Set thresholds to 2% of cell energy (or whatever fraction is given by user)
	  else
	    {
	      HO_Ecube_cut=HO_NADA_Ecube_frac_*cellenergy;
	      HO_Ecell_cut=HO_NADA_Ecell_frac_*cellenergy;
	    }
	    
	  CellPhi = _cell->id().iphi();
	  CellEta = _cell->id().ieta();
	  CellDepth = _cell->id().depth();

	  // Form cube of nearest neighbor cells around _cell
	    
	  for (HORecHitCollection::const_iterator _neighbor = c.begin();_neighbor!=c.end();_neighbor++)
	    // Form cube centered on _cell.  This needs to be looked at more carefully to deal with boundary conditions.  Should Ecube constraints change at the boundaries?
	    {

	      if (vetoCell(_neighbor->id())) continue; // don't count vetoed cells in cube
	      // Ignore cells outside user-defined cube
	      if (abs(_neighbor->id().depth()-CellDepth)>NADA_maxdepth) continue;
	      if (abs(_neighbor->id().ieta()-CellEta)>NADA_maxeta) continue;
	      if ((abs(_neighbor->id().iphi()-CellPhi)%72)>NADA_maxphi) continue;
	      if (_neighbor->energy()>HO_Ecell_cut) // energy must be above minimum
		Ecube+=_neighbor->energy();
	    
	    } // for (cell_iter _neighbor = c.begin()...)
		
	  //Remove energy due to _cell
	  Ecube -=cellenergy;

	  if (Ecube<=HO_Ecube_cut)
	    {   
	      numhotcells++;
	      h.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
	      h.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
	      NADA_hcalHists.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
	      NADA_hcalHists.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
	    }

	} //for (_cell=c.begin(); _cell!=c.end(); _cell++)
	
      h.NADA_NumHotCells->Fill(numhotcells);
      h.NADA_NumNegCells->Fill(numnegcells);
      hotcells+=numhotcells;
      negcells+=numnegcells;
    } // if c.size()>0
  return;
}


void HcalHotCellMonitor::HF_NADAFinder(const HFRecHitCollection& c, NADAHistList& h)
{
  if (c.size()>0)
    {
      
      HFRecHitCollection::const_iterator _cell;

      // Copying NADA algorithm from D0 Note 4057.
      // The implementation needs to be optimized -- double looping over iterators is not an efficient approach.
      float  Ecube=0;
      int numhotcells=0;
      int numnegcells=0;
      int CellPhi=-1, CellEta=-1, CellDepth=-1;

      float HF_Ecube_cut=0.;
      float HF_Ecell_cut=0.;

      float cellenergy=0.;
      int NADA_maxdepth, NADA_maxeta, NADA_maxphi;
      NADA_maxdepth = HF_NADA_maxdepth_;
      NADA_maxeta = HF_NADA_maxeta_;
      NADA_maxphi = HF_NADA_maxphi_;

      for (_cell=c.begin(); _cell!=c.end(); _cell++)
	{
	  if (vetoCell(_cell->id()))continue; // veto cell before or after filling energy histogram?  Make separate vetocell histos?
	  cellenergy=_cell->energy();
	  h.NADA_Energy->Fill(cellenergy);
	    		
	  // _cell points to the current hot cell candidate
	  Ecube=0; // reset Ecube energy counter
	    
	  // Case 1:  E< -1 GeV or E>500 GeV
	  if (cellenergy<HF_NADA_NegCand_cut_ || cellenergy>HF_NADA_Ecand_cut2_)
	    {
	      // Case 1a:   E< negative threshold
	      if (cellenergy<HF_NADA_NegCand_cut_)
		{
		  if (fVerbosity) cout <<"<NEGATIVE HF CELL ENERGY>  Energy = "<<cellenergy<<endl;
		  numnegcells++;
		  h.NADA_NEG_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  h.NADA_NEG_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),-1*cellenergy);
		  NADA_hcalHists.NADA_NEG_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  NADA_hcalHists.NADA_NEG_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),-1*cellenergy);
		  
		}
	      else
		{
		  numhotcells++;
		  h.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  h.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
		  NADA_hcalHists.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
		  NADA_hcalHists.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);	}
	      continue;
	    } // end of Case 1

	    
	  // Case 2:  Energy is < negative cutoff, but less than minimum threshold -- skip the cell
	    
	  else if (cellenergy<=HF_NADA_Ecand_cut0_)
	    continue;

	  // Case 3:  Set thresholds according to input variables
	  else if (cellenergy>HF_NADA_Ecand_cut0_ && cellenergy<HF_NADA_Ecand_cut1_)
	    {
	      HF_Ecube_cut=HF_NADA_Ecube_cut_;
	      HF_Ecell_cut=HF_NADA_Ecell_cut_;
	    }
	  // Case 4:  Set thresholds to 2% of cell energy (or whatever fraction is given by user)
	  else
	    {
	      HF_Ecube_cut=HF_NADA_Ecube_frac_*cellenergy;
	      HF_Ecell_cut=HF_NADA_Ecell_frac_*cellenergy;
	    }
	    
	  CellPhi = _cell->id().iphi();
	  CellEta = _cell->id().ieta();
	  CellDepth = _cell->id().depth();
	    
	  // Form cube of nearest neighbor cells around _cell
	    
	  for (HFRecHitCollection::const_iterator _neighbor = c.begin();_neighbor!=c.end();_neighbor++)
	    // Form cube centered on _cell.  This needs to be looked at more carefully to deal with boundary conditions.  Should Ecube constraints change at the boundaries?
	    {
	      if (vetoCell(_neighbor->id())) continue; // don't count vetoed cells in cube
	      // Ignore cells outside user-defined cube
	      if (abs(_neighbor->id().depth()-CellDepth)>NADA_maxdepth) continue;
	      if (abs(_neighbor->id().ieta()-CellEta)>NADA_maxeta) continue;
	      if ((abs(_neighbor->id().iphi()-CellPhi)%72)>NADA_maxphi) continue;
	      if (_neighbor->energy()>HF_Ecell_cut) // energy must be above minimum
		Ecube+=_neighbor->energy();

	    } // for (cell_iter _neighbor = c.begin()...)
		
	  //Remove energy due to _cell
	  Ecube -=cellenergy;

	  if (Ecube<=HF_Ecube_cut)
	    {   
	      numhotcells++;
	      h.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
	      h.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
	      NADA_hcalHists.NADA_OCC_MAP->Fill(_cell->id().ieta(),_cell->id().iphi());
	      NADA_hcalHists.NADA_EN_MAP->Fill(_cell->id().ieta(),_cell->id().iphi(),cellenergy);
	    }

	} //for (_cell=c.begin(); _cell!=c.end(); _cell++)
	
      h.NADA_NumHotCells->Fill(numhotcells);
      h.NADA_NumNegCells->Fill(numnegcells);
      hotcells+=numhotcells;
      negcells+=numnegcells;
    } // if c.size()>0
  return;
}

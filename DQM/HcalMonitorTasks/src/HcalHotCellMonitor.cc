#include "DQM/HcalMonitorTasks/interface/HcalHotCellMonitor.h"
#include <map>

HcalHotCellMonitor::HcalHotCellMonitor() {
  ievt_=0;
}

HcalHotCellMonitor::~HcalHotCellMonitor() {
}

void HcalHotCellMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HB");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HE");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HF");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HO");
    m_dbe->removeContents();
    

    hbHists.meOCC_MAP_GEO_Thr0 = 0;
    hbHists.meEN_MAP_GEO_Thr0 = 0;
    hbHists.meOCC_MAP_GEO_Thr1 = 0;
    hbHists.meEN_MAP_GEO_Thr1 = 0;
    hbHists.meOCC_MAP_GEO_Max = 0;
    hbHists.meEN_MAP_GEO_Max = 0;
    hbHists.meMAX_E = 0;
    hbHists.meMAX_T = 0;

    heHists.meOCC_MAP_GEO_Thr0 = 0;
    heHists.meEN_MAP_GEO_Thr0 = 0;
    heHists.meOCC_MAP_GEO_Thr1 = 0;
    heHists.meEN_MAP_GEO_Thr1 = 0;
    heHists.meOCC_MAP_GEO_Max = 0;
    heHists.meEN_MAP_GEO_Max = 0;
    heHists.meMAX_E = 0;
    heHists.meMAX_T = 0;

    hfHists.meOCC_MAP_GEO_Thr0 = 0;
    hfHists.meEN_MAP_GEO_Thr0 = 0;
    hfHists.meOCC_MAP_GEO_Thr1 = 0;
    hfHists.meEN_MAP_GEO_Thr1 = 0;
    hfHists.meOCC_MAP_GEO_Max = 0;
    hfHists.meEN_MAP_GEO_Max = 0;
    hfHists.meMAX_E = 0;
    hfHists.meMAX_T = 0;

    hoHists.meOCC_MAP_GEO_Thr0 = 0;
    hoHists.meEN_MAP_GEO_Thr0 = 0;
    hoHists.meOCC_MAP_GEO_Thr1 = 0;
    hoHists.meEN_MAP_GEO_Thr1 = 0;
    hoHists.meOCC_MAP_GEO_Max = 0;
    hoHists.meEN_MAP_GEO_Max = 0;
    hoHists.meMAX_E = 0;
    hoHists.meMAX_T = 0;

    meMAX_E_all= 0;
    meMAX_T_all= 0;
    meEVT_= 0;

    // Clear NADA Histograms
    NADA_hbHists.NADA_OCC_MAP=0;
    NADA_hbHists.NADA_EN_MAP=0;
    NADA_hbHists.NADA_NumHotCells=0;
    NADA_hbHists.NADA_testcell=0;
    NADA_hbHists.NADA_Energy=0;
    
    NADA_heHists.NADA_OCC_MAP=0;
    NADA_heHists.NADA_EN_MAP=0;
    NADA_heHists.NADA_NumHotCells=0;
    NADA_heHists.NADA_testcell=0;
    NADA_heHists.NADA_Energy=0;
    
    NADA_hoHists.NADA_OCC_MAP=0;
    NADA_hoHists.NADA_EN_MAP=0;
    NADA_hoHists.NADA_NumHotCells=0;
    NADA_hoHists.NADA_testcell=0;
    NADA_hoHists.NADA_Energy=0;
    
    NADA_hfHists.NADA_OCC_MAP=0;
    NADA_hfHists.NADA_EN_MAP=0;
    NADA_hfHists.NADA_NumHotCells=0;
    NADA_hfHists.NADA_testcell=0;
    NADA_hfHists.NADA_Energy=0;
    
    NADA_NumHotCells=0;
  }

}


void HcalHotCellMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

  //cout <<"SETTING UP HOT CELL MONITOR"<<endl;

  HcalBaseMonitor::setup(ps,dbe);

  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  //cout << "HotCell eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  //cout << "HotCell phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;

  occThresh0_ = ps.getUntrackedParameter<double>("HotCellThresh0", 0);
  occThresh1_ = ps.getUntrackedParameter<double>("HotCellThresh1", 5);
  //cout << "Hot Cell thresholds set to " << occThresh0_ << " "<< occThresh1_ <<endl;

  ievt_=0;
  
  if ( m_dbe !=NULL ) {    

    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor");

    meEVT_ = m_dbe->bookInt("HotCell Task Event Number");    
    meEVT_->Fill(ievt_);

    meMAX_E_all =  m_dbe->book1D("HotCell Energy","HotCell Energy",200,0,1000);
    meMAX_T_all =  m_dbe->book1D("HotCell Time","HotCell Time",200,-50,300);
    
    meOCC_MAP_L1= m_dbe->book2D("HotCell Depth 1 Occupancy Map","HotCell Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L1= m_dbe->book2D("HotCell Depth 1 Energy Map","HotCell Depth 1 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_L2= m_dbe->book2D("HotCell Depth 2 Occupancy Map","HotCell Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L2= m_dbe->book2D("HotCell Depth 2 Energy Map","HotCell Depth 2 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_L3= m_dbe->book2D("HotCell Depth 3 Occupancy Map","HotCell Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L3= m_dbe->book2D("HotCell Depth 3 Energy Map","HotCell Depth 3 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_L4= m_dbe->book2D("HotCell Depth 4 Occupancy Map","HotCell Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L4= m_dbe->book2D("HotCell Depth 4 Energy Map","HotCell Depth 4 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_all = m_dbe->book2D("HotCell Occupancy Map","HotCell Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_all  = m_dbe->book2D("HotCell Energy Map","HotCell Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    NADA_NumHotCells= m_dbe->book1D("NADA_NumHotCells","# of NADA Hot Cells/Event",1000,0,1000);
    
    

    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HB");
    hbHists.meMAX_E =  m_dbe->book1D("HB HotCell Energy","HB HotCell Energy",200,0,1000);
    hbHists.meMAX_T =  m_dbe->book1D("HB HotCell Time","HB HotCell Time",200,-50,300);
    hbHists.meMAX_ID =  m_dbe->book1D("HB HotCell ID","HB HotCell ID",10000,1000,12000);
    
    hbHists.meOCC_MAP_GEO_Thr0  = m_dbe->book2D("HB HotCell Geo Occupancy Map, Threshold 0","HB HotCell Geo Occupancy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meEN_MAP_GEO_Thr0  = m_dbe->book2D("HB HotCell Geo Energy Map, Threshold 0","HB HotCell Geo Energy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meOCC_MAP_GEO_Thr1  = m_dbe->book2D("HB HotCell Geo Occupancy Map, Threshold 1","HB HotCell Geo Occupancy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meEN_MAP_GEO_Thr1  = m_dbe->book2D("HB HotCell Geo Energy Map, Threshold 1","HB HotCell Geo Energy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HB HotCell Geo Occupancy Map, Max Cell","HB HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HB HotCell Geo Energy Map, Max Cell","HB HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    NADA_hbHists.NADA_OCC_MAP = m_dbe->book2D("NADA_HB_OCC_MAP","NADA HB Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hbHists.NADA_EN_MAP = m_dbe->book2D("NADA_HB_EN_MAP","NADA HB Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hbHists.NADA_NumHotCells = m_dbe->book1D("NADA_HB_NumHotCells","# of NADA HB Hot Cells/Event",1000,0,1000);
    NADA_hbHists.NADA_testcell = m_dbe->book1D("NADA_HB_testcell","Energy for test cell",1000,-100,900);
    NADA_hbHists.NADA_Energy = m_dbe->book1D("NADA_HB_Energy","Energy for all cells",1000,-100,900);
    



    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HE");
    heHists.meMAX_E =  m_dbe->book1D("HE HotCell Energy","HE HotCell Energy",200,0,1000);
    heHists.meMAX_T =  m_dbe->book1D("HE HotCell Time","HE HotCell Time",200,-50,300);
    heHists.meMAX_ID =  m_dbe->book1D("HE HotCell ID","HE HotCell ID",4000,1000,5000);
    heHists.meOCC_MAP_GEO_Thr0  = m_dbe->book2D("HE HotCell Geo Occupancy Map, Threshold 0","HE HotCell Geo Occupancy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meEN_MAP_GEO_Thr0  = m_dbe->book2D("HE HotCell Geo Energy Map, Threshold 0","HE HotCell Geo Energy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meOCC_MAP_GEO_Thr1  = m_dbe->book2D("HE HotCell Geo Occupancy Map, Threshold 1","HE HotCell Geo Occupancy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meEN_MAP_GEO_Thr1  = m_dbe->book2D("HE HotCell Geo Energy Map, Threshold 1","HE HotCell Geo Energy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HE HotCell Geo Occupancy Map, Max Cell","HE HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HE HotCell Geo Energy Map, Max Cell","HE HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    
    NADA_heHists.NADA_OCC_MAP = m_dbe->book2D("NADA_HE_OCC_MAP","NADA HE Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_heHists.NADA_EN_MAP = m_dbe->book2D("NADA_HE_EN_MAP","NADA HE Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_heHists.NADA_NumHotCells = m_dbe->book1D("NADA_HE_NumHotCells","# of NADA HE Hot Cells/Event",1000,0,1000);
    NADA_heHists.NADA_testcell = m_dbe->book1D("NADA_HE_testcell","Energy for test cell",1000,-100,900);
    NADA_heHists.NADA_Energy = m_dbe->book1D("NADA_HE_Energy","Energy for all cells",1000,-100,900);


    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HF");
    hfHists.meMAX_E =  m_dbe->book1D("HF HotCell Energy","HF HotCell Energy",200,0,1000);
    hfHists.meMAX_T =  m_dbe->book1D("HF HotCell Time","HF HotCell Time",200,-50,300);
    hfHists.meMAX_ID =  m_dbe->book1D("HF HotCell ID","HF HotCell ID",10000,0,10000);
    hfHists.meOCC_MAP_GEO_Thr0  = m_dbe->book2D("HF HotCell Geo Occupancy Map, Threshold 0","HF HotCell Geo Occupancy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meEN_MAP_GEO_Thr0  = m_dbe->book2D("HF HotCell Geo Energy Map, Threshold 0","HF HotCell Geo Energy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meOCC_MAP_GEO_Thr1  = m_dbe->book2D("HF HotCell Geo Occupancy Map, Threshold 1","HF HotCell Geo Occupancy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meEN_MAP_GEO_Thr1  = m_dbe->book2D("HF HotCell Geo Energy Map, Threshold 1","HF HotCell Geo Energy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HF HotCell Geo Occupancy Map, Max Cell","HF HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HF HotCell Geo Energy Map, Max Cell","HF HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
     NADA_hfHists.NADA_OCC_MAP = m_dbe->book2D("NADA_HF_OCC_MAP","NADA HF Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hfHists.NADA_EN_MAP = m_dbe->book2D("NADA_HF_EN_MAP","NADA HF Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hfHists.NADA_NumHotCells = m_dbe->book1D("NADA_HF_NumHotCells","# of NADA HF Hot Cells/Event",1000,0,1000);
    NADA_hfHists.NADA_testcell = m_dbe->book1D("NADA_HF_testcell","Energy for test cell",1000,-100,900);
    NADA_hfHists.NADA_Energy = m_dbe->book1D("NADA_HF_Energy","Energy for all cells",1000,-100,900);

     // Jeff's addition
    /*
    NADA_OCC_MAP_all = m_dbe->book2D("NADA_HotCell_Occupancy_Map","HotCell Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_EN_MAP_all  = m_dbe->book2D("NADA_HotCell_Energy_Map","HotCell Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_numhotcells = m_dbe->book1D("NADA_numhotcells","# of hot cells per event",50,0,50);
    NADA_HF_energy = m_dbe->book1D("NADA_hfenergy","Energy deposited in each cell",10000,-100,900);
    */
    // Is depth numbered from 0-1?  1-2?  How many layers in HF? Just 2?
    /*
    NADA_OCC_MAP_depth1 = m_dbe->book2D("NADA_HotCell_Occupancy_Map_depth1","HotCell Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_EN_MAP_depth1  = m_dbe->book2D("NADA_HotCell_Energy_Map_depth1","HotCell Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_OCC_MAP_depth2 = m_dbe->book2D("NADA_HotCell_Occupancy_Map_depth2","HotCell Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_EN_MAP_depth2  = m_dbe->book2D("NADA_HotCell_Energy_Map_depth2","HotCell Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    NADA_testcell = m_dbe->book1D("NADA_testcell","Energy distribution at (#eta,#phi,depth)=(33,17,1)",1000,-10,10);
    NADA_goodcell = m_dbe->book1D("NADA_goodcell","Energy distribution at (#eta,#phi,depth)=(29,21,1)",1000,-10,10);
    */

    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HO");
    hoHists.meMAX_E =  m_dbe->book1D("HO HotCell Energy","HO HotCell Energy",200,0,1000);
    hoHists.meMAX_T =  m_dbe->book1D("HO HotCell Time","HO HotCell Time",200,-50,300);
    hoHists.meMAX_ID =  m_dbe->book1D("HO HotCell ID","HO HotCell ID",1000,4000,5000);
    hoHists.meOCC_MAP_GEO_Thr0  = m_dbe->book2D("HO HotCell Geo Occupancy Map, Threshold 0","HO HotCell Geo Occupancy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meEN_MAP_GEO_Thr0  = m_dbe->book2D("HO HotCell Geo Energy Map, Threshold 0","HO HotCell Geo Energy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meOCC_MAP_GEO_Thr1  = m_dbe->book2D("HO HotCell Geo Occupancy Map, Threshold 1","HO HotCell Geo Occupancy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meEN_MAP_GEO_Thr1  = m_dbe->book2D("HO HotCell Geo Energy Map, Threshold 1","HO HotCell Geo Energy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HO HotCell Geo Occupancy Map, Max Cell","HO HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HO HotCell Geo Energy Map, Max Cell","HO HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
 
    NADA_hoHists.NADA_OCC_MAP = m_dbe->book2D("NADA_HO_OCC_MAP","NADA HO Occupancy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hoHists.NADA_EN_MAP = m_dbe->book2D("NADA_HO_EN_MAP","NADA HO Energy",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    NADA_hoHists.NADA_NumHotCells = m_dbe->book1D("NADA_HO_NumHotCells","# of NADA HO Hot Cells/Event",1000,0,1000);
    NADA_hoHists.NADA_testcell = m_dbe->book1D("NADA_HO_testcell","Energy for test cell",1000,-100,900);
    NADA_hoHists.NADA_Energy = m_dbe->book1D("NADA_HO_Energy","Energy for all cells",1000,-100,900);

  } // if (m_dbe !=NULL)


  // Remove this at some point, once data has been corrected
  //cout <<"INITIALIZING OFFSETS"<<endl;
  for (int i=0;i<13;i++)
    {
    for (int j =0;j<36;j++)
      { 
	for (int k=0;k<2;k++)
	  HF_offsets[i][j][k]=-11.39;
      }
    }


  return;
}

void HcalHotCellMonitor::processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits){

  if(!m_dbe) { printf("HcalHotCellMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }



  ievt_++;
  meEVT_->Fill(ievt_);
  
  hotcells_=0;

  
  // Fill HF thresholds on first event

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
    }

  // Variables for storing the largest energy found over all subsystems
  enA=0; tA=0;
  etaA=0; phiA=0;
  depth=0;

  // Apply NADA algorithm to find cells with anomalously large energy

  NADAFinder(hbHits, NADA_hbHists,1);
  NADAFinder(hbHits, NADA_heHists,2);
  NADAFinder(hoHits, NADA_hoHists);
  NADAFinder(hfHits, NADA_hfHists,3);

  NADA_NumHotCells->Fill(hotcells_);

  // Find hottest cell in event, cells above certain thresholds
  // Perform check for each subdetector in turn

  // Both Hb and He use same set of hits (hbHits)
  // Use modifiers to distinguish between the two
  // (1 = hb, 2 = he).
  FindHotCell(hbHits, hbHists,1);
  FindHotCell(hbHits, heHists,2);
  FindHotCell(hoHits, hoHists);
  FindHotCell(hfHits, hfHists);

  // Plot cell with largest overall energy
  if(enA>occThresh0_)
    {
      meMAX_E_all->Fill(enA);
      meMAX_T_all->Fill(tA);
      meOCC_MAP_all->Fill(etaA,phiA);
      meEN_MAP_all->Fill(etaA,phiA,enA);
      // Perhaps make these a vector of histograms?
      if(depth==1)
	{
	  meOCC_MAP_L1->Fill(etaA,phiA);
	  meEN_MAP_L1->Fill(etaA,phiA,enA);
	}
      else if(depth==2)
	{
	  meOCC_MAP_L2->Fill(etaA,phiA);
	  meEN_MAP_L2->Fill(etaA,phiA,enA);
	}
      else if(depth==3)
	{
	  meOCC_MAP_L3->Fill(etaA,phiA);
	  meEN_MAP_L3->Fill(etaA,phiA,enA);
	}
      else if(depth==4)
	{
	  meOCC_MAP_L4->Fill(etaA,phiA);
	  meEN_MAP_L4->Fill(etaA,phiA,enA);
	}
    }
  
  
  return;
}


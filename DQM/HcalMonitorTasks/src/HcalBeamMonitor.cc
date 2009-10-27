#include "DQM/HcalMonitorTasks/interface/HcalBeamMonitor.h"

// define sizes of ieta arrays for each subdetector

#define PI        3.1415926535897932
#define HBETASIZE 34  // one more bin than needed, I think
#define HEETASIZE 60  // ""
#define HOETASIZE 32  // ""
#define HFETASIZE 84  // ""

using namespace std;
/*  Task calculates various moments of Hcal recHits 

    v1.0
    16 August 2008
    by Jeff Temple

*/

const float HcalBeamMonitor::etaBounds[] = { 2.853, 2.964, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889};
const float HcalBeamMonitor::area[]={0.111,0.175,0.175,0.175,0.175,0.175,0.174,0.178,0.172,0.175,0.178,0.346,0.604};
const float HcalBeamMonitor::radius[]={1300,1162,975,818,686,576,483,406,340,286,240,201,169};

// constructor
HcalBeamMonitor::HcalBeamMonitor():
  ETA_OFFSET_HB(16),
  ETA_OFFSET_HE(29),
  ETA_BOUND_HE(17),
  ETA_OFFSET_HO(15),
  ETA_OFFSET_HF(41),
  ETA_BOUND_HF(29)
{occThresh_ = 0.0625;}

HcalBeamMonitor::~HcalBeamMonitor() {}

void HcalBeamMonitor::reset() {}

void HcalBeamMonitor::clearME()
{
  if (m_dbe) 
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
    } // if (m_dbe)
  meEVT_=0;
} // void HcalBeamMonitor::clearME()


void HcalBeamMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);  // perform setups of base class

  ievt_=0; // event counter
  baseFolder_ = rootFolder_ + "BeamMonitor_Hcal";
  if (fVerbosity) std::cout <<"<HcalBeamMonitor::setup> Setup in progress"<<std::endl;

  beammon_makeDiagnostics_ = ps.getUntrackedParameter<bool>("BeamMonitor_makeDiagnosticPlots",makeDiagnostics);
  // These two variables aren't yet in use
  beammon_checkNevents_    = ps.getUntrackedParameter<int>("BeamMonitor_checkNevents",checkNevents_);
  beammon_minErrorFlag_    = ps.getUntrackedParameter<double>("BeamMonitor_minErrorFlag",0.);
  beammon_lumiprescale_   = ps.getUntrackedParameter<int>("BeamMonitor_lumiprescale",1);

  if (m_dbe)
    {

      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("BeamMonitor Event Number");

      //jason's
      m_dbe->setCurrentFolder(baseFolder_);
      CenterOfEnergyRadius = m_dbe->book1D("CenterOfEnergyRadius",
					   "Center Of Energy radius",
					   200,0,1);
      
      CenterOfEnergyRadius->setAxisTitle("(normalized) radius",1);
      
      CenterOfEnergy = m_dbe->book2D("CenterOfEnergy",
				     "Center of Energy",
				     40,-1,1,
				     40,-1,1);
      CenterOfEnergy->setAxisTitle("normalized x coordinate",1);
      CenterOfEnergy->setAxisTitle("normalized y coordinate",2);

      COEradiusVSeta = m_dbe->bookProfile("COEradiusVSeta",
					  "Center of Energy radius vs i#eta",
					  172,-43,43,
					  20,0,1);
      COEradiusVSeta->setAxisTitle("i#eta",1);
      COEradiusVSeta->setAxisTitle("(normalized) radius",2);
      
      std::stringstream histname;
      std::stringstream histtitle;
      m_dbe->setCurrentFolder(baseFolder_+"/HB");
      HBCenterOfEnergyRadius = m_dbe->book1D("HBCenterOfEnergyRadius",
					     "HB Center Of Energy radius",
					     200,0,1);
      HBCenterOfEnergy = m_dbe->book2D("HBCenterOfEnergy",
				       "HB Center of Energy",
				       40,-1,1,
				       40,-1,1);
      if (beammon_makeDiagnostics_)
	{
	  for (int i=-16;i<=16;++i)
	    {
	      if (i==0) continue;
	      histname.str("");
	      histtitle.str("");
	      histname<<"HB_CenterOfEnergyRadius_ieta"<<i;
	      histtitle<<"HB Center Of Energy ieta = "<<i;
	      HB_CenterOfEnergyRadius[i+ETA_OFFSET_HB]=m_dbe->book1D(histname.str().c_str(),
							  histtitle.str().c_str(),
							  200,0,1);
	    } // end of HB loop
	}
      m_dbe->setCurrentFolder(baseFolder_+"/HE");
      HECenterOfEnergyRadius = m_dbe->book1D("HECenterOfEnergyRadius",
					     "HE Center Of Energy radius",
					     200,0,1);
      HECenterOfEnergy = m_dbe->book2D("HECenterOfEnergy",
				       "HE Center of Energy",
				       40,-1,1,
				       40,-1,1);

      if (beammon_makeDiagnostics_)
	{
	  for (int i=-29;i<=29;++i)
	    {
	      if (abs(i)<ETA_BOUND_HE) continue;
	      histname.str("");
	      histtitle.str("");
	      histname<<"HE_CenterOfEnergyRadius_ieta"<<i;
	      histtitle<<"HE Center Of Energy ieta = "<<i;
	      HE_CenterOfEnergyRadius[i+ETA_OFFSET_HE]=m_dbe->book1D(histname.str().c_str(),
							  histtitle.str().c_str(),
							  200,0,1);
	    } // end of HE loop
	}
      m_dbe->setCurrentFolder(baseFolder_+"/HO");
      HOCenterOfEnergyRadius = m_dbe->book1D("HOCenterOfEnergyRadius",
					     "HO Center Of Energy radius",
					     200,0,1);
      HOCenterOfEnergy = m_dbe->book2D("HOCenterOfEnergy",
				       "HO Center of Energy",
				       40,-1,1,
				       40,-1,1);
      if (beammon_makeDiagnostics_)
	{
	  for (int i=-15;i<=15;++i)
	    {
	      if (i==0) continue;
	      histname.str("");
	      histtitle.str("");
	      histname<<"HO_CenterOfEnergyRadius_ieta"<<i;
	      histtitle<<"HO Center Of Energy radius ieta = "<<i;
	      HO_CenterOfEnergyRadius[i+ETA_OFFSET_HO]=m_dbe->book1D(histname.str().c_str(),
								     histtitle.str().c_str(),
								     200,0,1);
	    } // end of HO loop
	}
      m_dbe->setCurrentFolder(baseFolder_+"/HF");
      HFCenterOfEnergyRadius = m_dbe->book1D("HFCenterOfEnergyRadius",
					     "HF Center Of Energy radius",
					     200,0,1);
      HFCenterOfEnergy = m_dbe->book2D("HFCenterOfEnergy",
				       "HF Center of Energy",
				       40,-1,1,
				       40,-1,1);
      if (beammon_makeDiagnostics_)
	{
	  for (int i=-41;i<=41;++i)
	    {
	      if (abs(i)<ETA_BOUND_HF) continue;
	      histname.str("");
	      histtitle.str("");
	      histname<<"HF_CenterOfEnergyRadius_ieta"<<i;
	      histtitle<<"HF Center Of Energy radius ieta = "<<i;
	      HF_CenterOfEnergyRadius[i+ETA_OFFSET_HF]=m_dbe->book1D(histname.str().c_str(),
								     histtitle.str().c_str(),
								     200,0,1);
	    } // end of HF loop
	}
      
      m_dbe->setCurrentFolder(baseFolder_+"/Lumi");
      // Wenhan's 
      // reducing bins from ",200,0,2000" to ",40,0,800"
      
      float radiusbins[13]={169,201,240,286,340,406,483,576,686,818,975,1162,1300};
      float phibins[71]={-3.5,-3.4,-3.3,-3.2,-3.1,
			 -3.0,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,
			 -2.0,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,
			 -1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,
			 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
			 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
			 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
			 3.0, 3.1, 3.2, 3.3, 3.4, 3.5};
      Etsum_eta_L=m_dbe->bookProfile("Et Sum vs Eta Long Fiber","Et Sum per Area vs Eta Long Fiber",83,-41.5,41.5,100,0,100);
      Etsum_eta_S=m_dbe->bookProfile("Et Sum vs Eta Short Fiber","Et Sum per Area vs Eta Short Fiber",83,-41.5,41.5,100,0,100);
      Etsum_phi_L=m_dbe->bookProfile("Et Sum vs Phi Long Fiber","Et Sum per Area vs Phi Long Fiber",36,0.5,72.5,100,0,100);
      Etsum_phi_S=m_dbe->bookProfile("Et Sum vs Phi Short Fiber","Et Sum per Area crossing vs Phi Short Fiber",36,0.5,72.5,100,0,100);
      Etsum_ratio_p=m_dbe->book1D("Occ vs fm HF+","Energy difference of Long and Short Fiber HF+",105,0.,1.05);
      Energy_Occ=m_dbe->book1D("Occ vs Energy","Occupancy vs Energy",200,0,2000);
      Etsum_ratio_m=m_dbe->book1D("Occ vs fm HF-","Energy difference of Long and Short Fiber HF-",105,0.,1.05);
      Etsum_map_L=m_dbe->book2D("EtSum 2D phi and eta Long Fiber","Et Sum 2D phi and eta Long Fiber",83,-41.5,41.5,36,0.5,72.5);
      Etsum_map_S=m_dbe->book2D("EtSum 2D phi and eta Short Fiber","Et Sum 2D phi and eta Short Fiber",83,-41.5,41.5,36,0.5,72.5);
      Etsum_rphi_S=m_dbe->book2D("EtSum 2D phi and radius Short Fiber","Et Sum 2D phi and radius Short Fiber",12, radiusbins, 70, phibins);
      Etsum_rphi_L=m_dbe->book2D("EtSum 2D phi and radius Long Fiber","Et Sum 2D phi and radius Long Fiber",12, radiusbins, 70, phibins);

      Etsum_ratio_map=m_dbe->book2D("Abnormal fm","Abnormal fm",
				    8,0,8,36, 0.5,72.5);
      SetEtaLabels(Etsum_ratio_map);

      HFlumi_occ_LS = m_dbe->book2D("HFlumi_occ_LS","HFlumi occupancy for current LS",
				      8,0,8,36, 0.5,72.5);
      SetEtaLabels(HFlumi_occ_LS);
      
      HFlumi_total_deadcells = m_dbe->book2D("HFlumi_total_deadcells","# of times each HFlumi cell was dead for 1 full LS",
					     8,0,8,36,0.5,72.5);
      SetEtaLabels(HFlumi_total_deadcells);
      HFlumi_total_hotcells = m_dbe->book2D("HFlumi_total_hotcells","# of times each HFlumi cell was hot for 1 full LS",
					     8,0,8,36,0.5,72.5);
      SetEtaLabels(HFlumi_total_hotcells);


      Occ_rphi_S=m_dbe->book2D("Occ 2D phi and radius Short Fiber","Occupancy 2D phi and radius Short Fiber",12, radiusbins, 70, phibins);
      Occ_rphi_L=m_dbe->book2D("Occ 2D phi and radius Long Fiber","Occupancy 2D phi and radius Long Fiber",12, radiusbins, 70, phibins);
      Occ_eta_S=m_dbe->bookProfile("Occ vs iEta Short Fiber","Occ per Bunch crossing vs iEta Short Fiber",83,-41.5,41.5,40,0,800);
      Occ_eta_L=m_dbe->bookProfile("Occ vs iEta Long Fiber","Occ per Bunch crossing vs iEta Long Fiber",83,-41.5,41.5,40,0,800);
      
      Occ_phi_L=m_dbe->bookProfile("Occ vs iPhi Long Fiber","Occ per Bunch crossing vs iPhi Long Fiber",36,0.5,72.5,40,0,800);
      
      Occ_phi_S=m_dbe->bookProfile("Occ vs iPhi Short Fiber","Occ per Bunch crossing vs iPhi Short Fiber",36,0.5,72.5,40,0,800);
      
      Occ_map_L=m_dbe->book2D("Occ_map Long Fiber","Occ Map long Fiber (above threshold)",83,-41.5,41.5,36,0.5,72.5);
      Occ_map_S=m_dbe->book2D("Occ_map Short Fiber","Occ Map Short Fiber (above threshold)",83,-41.5,41.5,36,0.5,72.5);
      
      //HFlumi plots
      HFlumi_ETsum_perwedge =  m_dbe->book1D("HF lumi ET-sum per wedge","HF lumi ET-sum per wedge",36,1,37);
      
      HFlumi_Occupancy_above_thr_r1 =  m_dbe->book1D("HF lumi Occupancy above threshold ring1","HF lumi Occupancy above threshold ring1",36,1,37);
      HFlumi_Occupancy_between_thrs_r1 = m_dbe->book1D("HF lumi Occupancy between thresholds ring1","HF lumi Occupancy between thresholds ring1",36,1,37);
      HFlumi_Occupancy_below_thr_r1 = m_dbe->book1D("HF lumi Occupancy below threshold ring1","HF lumi Occupancy below threshold ring1",36,1,37);
      HFlumi_Occupancy_above_thr_r2 = m_dbe->book1D("HF lumi Occupancy above threshold ring2","HF lumi Occupancy above threshold ring2",36,1,37);
      HFlumi_Occupancy_between_thrs_r2 = m_dbe->book1D("HF lumi Occupancy between thresholds ring2","HF lumi Occupancy between thresholds ring2",36,1,37);
      HFlumi_Occupancy_below_thr_r2 = m_dbe->book1D("HF lumi Occupancy below threshold ring2","HF lumi Occupancy below threshold ring2",36,1,37);
      
      HFlumi_Occupancy_per_channel_vs_lumiblock_RING1 = m_dbe->bookProfile("HFlumi Occupancy per channel vs lumi-block (RING 1)","HFlumi Occupancy per channel vs lumi-block (RING 1);LS; -ln(empty fraction)",Nlumiblocks_/beammon_lumiprescale_,0.5,Nlumiblocks_+0.5,100,0,10000);
      HFlumi_Occupancy_per_channel_vs_lumiblock_RING2 = m_dbe->bookProfile("HFlumi Occupancy per channel vs lumi-block (RING 2)","HFlumi Occupancy per channel vs lumi-block (RING 2);LS; -ln(empty fraction)",Nlumiblocks_/beammon_lumiprescale_,0.5,Nlumiblocks_+0.5,100,0,10000);

      HFlumi_Et_per_channel_vs_lumiblock = m_dbe->bookProfile("HFlumi Et per channel vs lumi-block","HFlumi Et per channel vs lumi-block",Nlumiblocks_/beammon_lumiprescale_,0.5,Nlumiblocks_+0.5,100,0,10000);

      HFlumi_Occupancy_per_channel_vs_lumiblock_RING1->getTProfile()->SetMarkerStyle(20);
      HFlumi_Occupancy_per_channel_vs_lumiblock_RING2->getTProfile()->SetMarkerStyle(20);
      HFlumi_Et_per_channel_vs_lumiblock->getTProfile()->SetMarkerStyle(20);
    } // if (m_dbe)

  return;

} // void HcalBeamMonitor::setup()

void HcalBeamMonitor::processEvent(const HBHERecHitCollection& hbheHits,
				   const HORecHitCollection& hoHits,
				   const HFRecHitCollection& hfHits,
                                   const HFDigiCollection& hf
				     // const ZDCRecHitCollection & zdcHits // include this once we see ZDC rec hits read out
				   )
  
{ //processEvent loop
  if (!m_dbe)
    {
      if (fVerbosity) std::cout <<"HcalBeamMonitor::processEvent   DQMStore not instantiated!!!"<<std::endl;
      return;
    }

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  ievt_++;
  meEVT_->Fill(ievt_);

  HBHERecHitCollection::const_iterator HBHEiter;
  HORecHitCollection::const_iterator HOiter;
  HFRecHitCollection::const_iterator HFiter;

  double totalX=0;
  double totalY=0;
  double totalE=0;

  double HBtotalX=0;
  double HBtotalY=0;
  double HBtotalE=0;
  double HEtotalX=0;
  double HEtotalY=0;
  double HEtotalE=0;
  double HOtotalX=0;
  double HOtotalY=0;
  double HOtotalE=0;
  double HFtotalX=0;
  double HFtotalY=0;
  double HFtotalE=0;
     
  float hitsp[13][36][2];
  float hitsm[13][36][2];
  float hitsp_Et[13][36][2];
  float hitsm_Et[13][36][2];
  
  for(int m=0;m<13;m++){
    for(int n=0;n<36;n++){
      hitsp[m][n][0]=0;
      hitsp[m][n][1]=0; 
      hitsm[m][n][0]=0;
      hitsm[m][n][1]=0;

      hitsp_Et[m][n][0]=0;
      hitsp_Et[m][n][1]=0; 
      hitsm_Et[m][n][0]=0;
      hitsm_Et[m][n][1]=0;
    }
  }
  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalBeamMonitor BEAMMON analyze pre-process-> " << cpu_timer.cpuTime() << std::endl;
      cpu_timer.reset(); cpu_timer.start();
    } // if (showTiming)


  if(hbheHits.size()>0)
    {
      double HB_weightedX[HBETASIZE]={0.};
      double HB_weightedY[HBETASIZE]={0.};
      double HB_energy[HBETASIZE]={0.};
      
      double HE_weightedX[HEETASIZE]={0.};
      double HE_weightedY[HEETASIZE]={0.};
      double HE_energy[HEETASIZE]={0.};
      
      int ieta, iphi;
      
      for (HBHEiter=hbheHits.begin(); 
	   HBHEiter!=hbheHits.end(); 
	   ++HBHEiter) 
	{ 
	  
	  // loop over all hits
	  if (HBHEiter->energy()<0) continue; // don't consider negative-energy cells
	  HcalDetId id(HBHEiter->detid().rawId());
	  ieta=id.ieta();
	  iphi=id.iphi();
	  
	  unsigned int index;
	  if ((HcalSubdetector)(id.subdet())==HcalBarrel)
	    {
	      HBtotalX+=HBHEiter->energy()*cos(2*PI*iphi/72);
	      HBtotalY+=HBHEiter->energy()*sin(2*PI*iphi/72);
	      HBtotalE+=HBHEiter->energy();
	      
	      index=ieta+ETA_OFFSET_HB;
	      if (index<0 || index> HBETASIZE) continue;
	      HB_weightedX[index]+=HBHEiter->energy()*cos(2.*PI*iphi/72);
	      HB_weightedY[index]+=HBHEiter->energy()*sin(2.*PI*iphi/72);
	      HB_energy[index]+=HBHEiter->energy();
	    } // if id.subdet()==HcalBarrel
	  
	  else
	    {
	      HEtotalX+=HBHEiter->energy()*cos(2*PI*iphi/72);
	      HEtotalY+=HBHEiter->energy()*sin(2*PI*iphi/72);
	      HEtotalE+=HBHEiter->energy();
	      
	      index=ieta+ETA_OFFSET_HE;
	      if (index<0 || index> HEETASIZE) continue;
	      HE_weightedX[index]+=HBHEiter->energy()*cos(2.*PI*iphi/72);
	      HE_weightedY[index]+=HBHEiter->energy()*sin(2.*PI*iphi/72);
	      HE_energy[index]+=HBHEiter->energy();
	    }
	} // for (HBHEiter=hbheHits.begin()...
	  // Fill each histogram
      
      int hbeta=ETA_OFFSET_HB;
      for (int i=-1*hbeta;i<=hbeta;++i)
	{
	  if (i==0) continue;
	  int index = i+ETA_OFFSET_HB;
	  if (index<0 || index> HBETASIZE) continue;
	  if (HB_energy[index]==0) continue;
	  double moment=pow(HB_weightedX[index],2)+pow(HB_weightedY[index],2);
	  //cout <<"index = "<<i<<"  X = "<<HB_weightedX[index]<<"  Y = "<<HB_weightedY[index]<<" Energy = "<<HB_energy[index]<<endl;
	  moment=pow(moment,0.5);
	  moment/=HB_energy[index];
	  //cout <<"\tMOMENT = "<<moment<<endl;
	  if (moment!=0)
	    {
	      if (beammon_makeDiagnostics_) HB_CenterOfEnergyRadius[index]->Fill(moment);
	      COEradiusVSeta->Fill(i,moment);
	    }
	} // for (int i=-1*hbeta;i<=hbeta;++i)

      int heeta=ETA_OFFSET_HE;
      for (int i=-1*heeta;i<=heeta;++i)
	{
	  if (i==0) continue;
	  if (i>-1*ETA_BOUND_HE && i <ETA_BOUND_HE) continue;
	  int index = i + ETA_OFFSET_HE;
	  if (index<0 || index> HEETASIZE) continue;
	  if (HE_energy[index]==0) continue;
	  double moment=pow(HE_weightedX[index],2)+pow(HE_weightedY[index],2);
	  moment=pow(moment,0.5);
	  moment/=HE_energy[index];
	  if (moment!=0)
	    {
	      if (beammon_makeDiagnostics_) HE_CenterOfEnergyRadius[index]->Fill(moment);
	      COEradiusVSeta->Fill(i,moment);
	    }
	} // for (int i=-1*heeta;i<=heeta;++i)

    } // if (hbheHits.size()>0)


  
  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalBeamMonitor BEAMMON HBHE-> " << cpu_timer.cpuTime() << std::endl;
      cpu_timer.reset(); cpu_timer.start();
    } // if (showTiming)
  
  // HO loop
  if(hoHits.size()>0)
    {
      double HO_weightedX[HOETASIZE]={0.};
      double HO_weightedY[HOETASIZE]={0.};
      double HO_energy[HOETASIZE]={0.};
      double offset;

      int ieta, iphi;
      for (HOiter=hoHits.begin(); 
	   HOiter!=hoHits.end(); 
	   ++HOiter) 
	{ 
	  // loop over all cells
	  if (HOiter->energy()<0) continue;  // don't include negative-energy cells?
	  HcalDetId id(HOiter->detid().rawId());
	  ieta=id.ieta();
	  iphi=id.iphi();

	  HOtotalX+=HOiter->energy()*cos(2.*PI*iphi/72);
	  HOtotalY+=HOiter->energy()*sin(2.*PI*iphi/72);
	  HOtotalE+=HOiter->energy();

	  unsigned int index;
	  index=ieta+ETA_OFFSET_HO;
	  if (index<0 || index>HOETASIZE) continue;
	  HO_weightedX[index]+=HOiter->energy()*cos(2.*PI*iphi/72);
	  HO_weightedY[index]+=HOiter->energy()*sin(2.*PI*iphi/72);
	  HO_energy[index]+=HOiter->energy();
	} // for (HOiter=hoHits.begin();...)
	  
      for (int i=-1*ETA_OFFSET_HO;i<=ETA_OFFSET_HO;++i)
	{
	  if (i==0) continue;
	  int index = i + ETA_OFFSET_HO;
	  if (index < 0 || index> HOETASIZE) continue;
	  if (HO_energy[index]==0) continue;
	  double moment=pow(HO_weightedX[index],2)+pow(HO_weightedY[index],2);
	  moment=pow(moment,0.5);
	  moment/=HO_energy[index];
	  // Shift HO values by 0.5 units in eta relative to HB
	  offset = (i>0 ? 0.5: -0.5);
	  if (moment!=0)
	    {
	      if (beammon_makeDiagnostics_) HO_CenterOfEnergyRadius[index]->Fill(moment);
	      COEradiusVSeta->Fill(i+offset,moment);
	    }
	} // for (int i=-1*hoeta;i<=hoeta;++i)
    } // if (hoHits.size()>0)
    
  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalBeamMonitor BEAMMON HO-> " << cpu_timer.cpuTime() << std::endl;
      cpu_timer.reset(); cpu_timer.start();
    } // if (showTiming)

  ///////////////////////////////////
  // HF loop

  {
    if(hfHits.size()>0)
      {
	double HF_weightedX[HFETASIZE]={0.};
	double HF_weightedY[HFETASIZE]={0.};
	double HF_energy[HFETASIZE]={0.};
	double offset;
	
	// Assume ZS until shown otherwise
	double emptytowersRing1 = 144;
	double emptytowersRing2 = 144;
	double ZStowersRing1 = 144;
	double ZStowersRing2 = 144;
	
	int ieta, iphi;
	float et,eta,phi,r;

	HFlumi_occ_LS->Fill(-1,-1,1 ); // event counter in occupancy histogram underflow bin
	// set maximum to HFlumi_occ_LS->getBinContent(0,0)?  
	// that won't work -- offline will add multiple histograms, and maximum will get screwed up?
	// No, we can add it here, but we also need a call to setMaximum in the client as well.
	HFlumi_occ_LS->getTH2F()->SetMaximum(HFlumi_occ_LS->getBinContent(0,0));

	for (HFiter=hfHits.begin(); 
	     HFiter!=hfHits.end(); 
	     ++HFiter) 
	  {  // loop on hfHits
	    // If hit present, don't count it as ZS any more
	    ieta = HFiter->id().ieta();
	    iphi = HFiter->id().iphi();

	    // Count that hit was found in one of the rings used for luminosity calculation.
	    // If so, decrease the number of empty channels per ring by 1
	    if (abs(ieta)>=33 && abs(ieta)<=36) // luminosity ring check
	      (HFiter->id().depth()==1) ? --ZStowersRing1 : --ZStowersRing2;

	    if (HFiter->energy()<0) continue;  // don't include negative-energy cells?

	    eta=etaBounds[abs(ieta)-29];
	    et=HFiter->energy()/cosh(eta)/area[abs(ieta)-29];
	    if (abs(ieta)>=33 && abs(ieta)<=36) // Luminosity ring check
	      {
		if (et>=0.0625) // minimum ET threshold
		  (HFiter->id().depth()==1) ? --emptytowersRing1 : --emptytowersRing2;
	      }
	    r=radius[abs(ieta)-29];
	    if(HFiter->id().iphi()<37)
	      phi=HFiter->id().iphi()*0.087266;
	    else phi=(HFiter->id().iphi()-72)*0.087266;
           
	    if (HFiter->id().depth()==1)
	      {
		Etsum_eta_L->Fill(ieta,et);
		Etsum_phi_L->Fill(iphi,et);
		Etsum_map_L->Fill(ieta,iphi,et);
		Etsum_rphi_L->Fill(r,phi,et);
	      
		if(ieta>0) {
		  hitsp[ieta-29][(HFiter->id().iphi()-1)/2][0]=HFiter->energy();
		  hitsp_Et[ieta-29][(HFiter->id().iphi()-1)/2][0]=et;
		}
		else if(ieta<0) {
		  hitsm[-ieta-29][(HFiter->id().iphi()-1)/2][0]=HFiter->energy(); 
		  hitsm_Et[-ieta-29][(HFiter->id().iphi()-1)/2][0]=et; 
		}
	      } // if (HFiter->id().depth()==1)
         
	    //Fill 3 histos for Short Fibers :
	    if (HFiter->id().depth()==2)
	      {
		Etsum_eta_S->Fill(ieta,et);
		Etsum_phi_S->Fill(iphi,et);
		Etsum_rphi_S->Fill(r,phi,et); 
		Etsum_map_S->Fill(ieta,iphi,et);
		if(ieta>0)  
		  {
		    hitsp[ieta-29][(HFiter->id().iphi()-1)/2][1]=HFiter->energy();
		    hitsp_Et[ieta-29][(HFiter->id().iphi()-1)/2][1]=et;
		  }
		else if(ieta<0)  { 
		  hitsm[-ieta-29][(HFiter->id().iphi()-1)/2][1]=HFiter->energy();
		  hitsm_Et[-ieta-29][(HFiter->id().iphi()-1)/2][1]=et;
		}
          
	      } // depth()==2
	    Energy_Occ->Fill(HFiter->energy()); 
            
	    //HF: no non-threshold occupancy map is filled?

	    if ((abs(ieta) == 33 || abs(ieta) == 34) && HFiter->id().depth() == 1)
	      { 
		HFlumi_Et_per_channel_vs_lumiblock->Fill(lumiblock,et);
		if (et>occThresh_)
		  {
		    int etabin=0;
		    if (ieta<0)
		      etabin=36+ieta; // bins 0-3 correspond to ieta = -36, -35, -34, -33
		    else
		      etabin=ieta-29; // bins 4-7 correspond to ieta = 33, 34, 35, 36
		    HFlumi_occ_LS->Fill(etabin,HFiter->id().iphi());
		  }
	      }

	    else if ((abs(ieta) == 35 || abs(ieta) == 36) && HFiter->id().depth() == 2)
	      { 
		HFlumi_Et_per_channel_vs_lumiblock->Fill(lumiblock,et);
		if (et>occThresh_)
		  {
		    int etabin=0;
		    if (ieta<0)
		      etabin=36+ieta; // bins 0-3 correspond to ieta = -36, -35, -34, -33
		    else
		      etabin=ieta-29; // bins 4-7 correspond to ieta = 33, 34, 35, 36
		    HFlumi_occ_LS->Fill(etabin,HFiter->id().iphi());
		  }
	      }

	    // Fill occupancy plots.
	    
	    int value=0;
	    if(et>occThresh_) value=1;
	    if (HFiter->id().depth()==1)
	      {
		Occ_eta_L->Fill(ieta,value);
		Occ_phi_L->Fill(iphi,value);
		Occ_map_L->Fill(ieta,iphi,value);
		Occ_rphi_L->Fill(r,phi,value);
	      }
	      
	    else if (HFiter->id().depth()==2)
	      {
		Occ_eta_S->Fill(ieta,value);
		Occ_phi_S->Fill(iphi,value);
		Occ_map_S->Fill(ieta,iphi,value);
		Occ_rphi_S->Fill(r,phi,value);
	      }  
	    HcalDetId id(HFiter->detid().rawId());

	    HFtotalX+=HFiter->energy()*cos(2.*PI*iphi/72);
	    HFtotalY+=HFiter->energy()*sin(2.*PI*iphi/72);
	    HFtotalE+=HFiter->energy();

	    unsigned int index;
	    index=ieta+ETA_OFFSET_HF;
	    if (index<0 || index>HFETASIZE) continue;
	    HF_weightedX[index]+=HFiter->energy()*cos(2.*PI*iphi/72);
	    HF_weightedY[index]+=HFiter->energy()*sin(2.*PI*iphi/72);
	    HF_energy[index]+=HFiter->energy();
	    
	  } // for (HFiter=hfHits.begin();...)
	
	// looped on all HF hits; calculate empty fraction
	//  empty towers  = # of cells with ET < 0.0625 GeV, or cells missing because of ZS
	//  Calculated as :  144 - (# of cells with ET >= 0.0625 GeV)
	//  At some point, allow for calculations when channels are masked (and less than 144 channels expected)

	// Check Ring 1
	double logvalue=0;
	int ring1totalchannels=144;
	int ring2totalchannels=144;
	if (emptytowersRing1>0)
	  logvalue=-1.*log(emptytowersRing1/ring1totalchannels);
	HFlumi_Occupancy_per_channel_vs_lumiblock_RING1->Fill(lumiblock,logvalue);
	
	// Check Ring 2
	emptytowersRing2>0 ? logvalue=-1.*log(emptytowersRing2/ring2totalchannels) : logvalue = 0;
	HFlumi_Occupancy_per_channel_vs_lumiblock_RING2->Fill(lumiblock,logvalue);

	int hfeta=ETA_OFFSET_HF;
	for (int i=-1*hfeta;i<=hfeta;++i)
	  {
	    if (i==0) continue;
	    if (i>-1*ETA_BOUND_HF && i <ETA_BOUND_HF) continue;
	    int index = i + ETA_OFFSET_HF;
	    if (index<0 || index>HFETASIZE) continue;
	    if (HF_energy[index]==0) continue;
	    double moment=pow(HF_weightedX[index],2)+pow(HF_weightedY[index],2);
	    moment=pow(moment,0.5);
	    moment/=HF_energy[index];
	    offset = (i>0 ? 0.5: -0.5);
	    if (moment!=0)
	      {
		if (beammon_makeDiagnostics_) HF_CenterOfEnergyRadius[index]->Fill(moment);
		COEradiusVSeta->Fill(i+offset,moment);
	      }
	  } // for (int i=-1*hfeta;i<=hfeta;++i)
	float ratiom,ratiop;
	  
	for(int i=0;i<13;i++){
	  for(int j=0;j<36;j++){
	      
	    if(hitsp[i][j][0]==hitsp[i][j][1]) continue;
	      
	    if (hitsp[i][j][0] < 1.2 && hitsp[i][j][1] < 1.8) continue;
	    //use only lumi rings
	    if (((i+29) < 33) || ((i+29) > 36)) continue;
	    ratiop=fabs((fabs(hitsp[i][j][0])-fabs(hitsp[i][j][1]))/(fabs(hitsp[i][j][0])+fabs(hitsp[i][j][1])));
	    //cout<<ratiop<<endl;
	    if ((hitsp_Et[i][j][0] > 5. && hitsp[i][j][1] < 1.8) || (hitsp_Et[i][j][1] > 5. &&  hitsp[i][j][0] < 1.2)){
	      Etsum_ratio_p->Fill(ratiop);
	      if(abs(ratiop>0.95)) Etsum_ratio_map->Fill(i,2*j+1); // i=4,5,6,7 for HFlumi rings 
	    }
	  }
	}
	  
	for(int p=0;p<13;p++){
	  for(int q=0;q<36;q++){
	      
	    if(hitsm[p][q][0]==hitsm[p][q][1]) continue;

	    if (hitsm[p][q][0] < 1.2 && hitsm[p][q][1] < 1.8) continue;
	    //use only lumi rings
	    if (((p+29) < 33) || ((p+29) > 36)) continue;
	    ratiom=fabs((fabs(hitsm[p][q][0])-fabs(hitsm[p][q][1]))/(fabs(hitsm[p][q][0])+fabs(hitsm[p][q][1])));         
	    if ((hitsm_Et[p][q][0] > 5. && hitsm[p][q][1] < 1.8) || (hitsm_Et[p][q][1] > 5. && hitsm[p][q][0] < 1.2)){
	      Etsum_ratio_m->Fill(ratiom);
	      if(abs(ratiom>0.95)) Etsum_ratio_map->Fill(7-p,2*q+1); // p=4,5,6,7 for HFlumi rings
	      //p=7:  ieta=-36; p=4:  ieta=-33
	    }
	  }
	} 
      } // if (hfHits.size()>0)
  
    if (showTiming)
      {
	cpu_timer.stop(); std::cout << " TIMER::HcalBeamMonitor BEAMMON HF-> " << cpu_timer.cpuTime() << std::endl;
      } // if (showTiming)

    totalX=HBtotalX+HEtotalX+HOtotalX+HFtotalX;
    totalY=HBtotalY+HEtotalY+HOtotalY+HFtotalY;
    totalE=HBtotalE+HEtotalE+HOtotalE+HFtotalE;

    double moment;
    if (HBtotalE>0)
      {
	moment=pow(HBtotalX*HBtotalX+HBtotalY*HBtotalY,0.5)/HBtotalE;
	HBCenterOfEnergyRadius->Fill(moment);
	HBCenterOfEnergy->Fill(HBtotalX/HBtotalE, HBtotalY/HBtotalE);
      }
    if (HEtotalE>0)
      {
	moment=pow(HEtotalX*HEtotalX+HEtotalY*HEtotalY,0.5)/HEtotalE;
	HECenterOfEnergyRadius->Fill(moment);
	HECenterOfEnergy->Fill(HEtotalX/HEtotalE, HEtotalY/HEtotalE);
      }
    if (HOtotalE>0)
      {
	moment=pow(HOtotalX*HOtotalX+HOtotalY*HOtotalY,0.5)/HOtotalE;
	HOCenterOfEnergyRadius->Fill(moment);
	HOCenterOfEnergy->Fill(HOtotalX/HOtotalE, HOtotalY/HOtotalE);
      }
    if (HFtotalE>0)
      {
	moment=pow(HFtotalX*HFtotalX+HFtotalY*HFtotalY,0.5)/HFtotalE;
	HFCenterOfEnergyRadius->Fill(moment);
	HFCenterOfEnergy->Fill(HFtotalX/HFtotalE, HFtotalY/HFtotalE);
      }
    if (totalE>0)
      {
	moment = pow(totalX*totalX+totalY*totalY,0.5)/totalE;
	CenterOfEnergyRadius->Fill(moment);
	CenterOfEnergy->Fill(totalX/totalE, totalY/totalE);
      }


    
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++){
      const HFDataFrame digi = (const HFDataFrame)(*j);
      //  calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private. 
      //       float en=0;
      //       float ts =0; float bs=0;
      //       int maxi=0; float maxa=0;
      //       for(int i=sigS0_; i<=sigS1_; i++){
      // 	if(digi.sample(i).adc()>maxa){maxa=digi.sample(i).adc(); maxi=i;}
      //       }
      //       for(int i=sigS0_; i<=sigS1_; i++){	  
      // 	float tmp1 =0;   
      //         int j1=digi.sample(i).adc();
      //         tmp1 = (LedMonAdc2fc[j1]+0.5);   	  
      // 	en += tmp1-calibs_.pedestal(digi.sample(i).capid());
      // 	if(i>=(maxi-1) && i<=maxi+1){
      // 	  ts += i*(tmp1-calibs_.pedestal(digi.sample(i).capid()));
      // 	  bs += tmp1-calibs_.pedestal(digi.sample(i).capid());
      // 	}
      //       }

      //---HFlumiplots
      int theTStobeused = 6;
      // will have masking later:
      int mask=1; 
      if(mask!=1) continue;
      //if we want to sum the 10 TS instead of just taking one:
      for (int i=0; i<digi.size(); i++) {
	if (i==theTStobeused) {
	  float tmpET =0;
	  int jadc=digi.sample(i).adc();
	  //NOW LUT used in HLX are only identy LUTs, so Et filled
	  //with unlinearised adc, ie tmpET = jadc
	  //	  tmpET = (adc2fc[jadc]+0.5);
	  tmpET = jadc;

	  //-find which wedge we are in
	  //  ETsum and Occupancy will be summed for both L and S
	  if(digi.id().ieta()>28){
	    if((digi.id().iphi()==1)||(digi.id().iphi()==71)){
	      HFlumi_ETsum_perwedge->Fill(1,tmpET);
              if((digi.id().ieta()==33)||(digi.id().ieta()==34)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(1,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(1,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(1,1);
	      }
	      else if((digi.id().ieta()==35)||(digi.id().ieta()==36)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(1,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(1,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(1,1);
	      }
	    }
	    else {
	      for (int iwedge=2; iwedge<19; iwedge++) {
		int itmp=4*(iwedge-1);
		if( (digi.id().iphi()==(itmp+1)) || (digi.id().iphi()==(itmp-1))) {
                  HFlumi_ETsum_perwedge->Fill(iwedge,tmpET);
		  if((digi.id().ieta()==33)||(digi.id().ieta()==34)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(iwedge,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(iwedge,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(iwedge,1);
		  }
		  else if((digi.id().ieta()==35)||(digi.id().ieta()==36)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(iwedge,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(iwedge,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(iwedge,1);
		  }
                  iwedge=99;
		}
	      }
	    }
	  }  //--endif ieta in HF+
	  else if(digi.id().ieta()<-28){
	    if((digi.id().iphi()==1)||(digi.id().iphi()==71)){
	      HFlumi_ETsum_perwedge->Fill(19,tmpET);
              if((digi.id().ieta()==-33)||(digi.id().ieta()==-34)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(19,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(19,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(19,1);
	      }
	      else if((digi.id().ieta()==-35)||(digi.id().ieta()==-36)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(19,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(19,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(19,1);
	      }
	    }
	    else {
	      for (int iw=2; iw<19; iw++) {
		int itemp=4*(iw-1);
		if( (digi.id().iphi()==(itemp+1)) || (digi.id().iphi()==(itemp-1))) {
                  HFlumi_ETsum_perwedge->Fill(iw+18,tmpET);
		  if((digi.id().ieta()==-33)||(digi.id().ieta()==-34)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(iw+18,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(iw+18,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(iw+18,1);
		  }
		  else if((digi.id().ieta()==-35)||(digi.id().ieta()==-36)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(iw+18,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(iw+18,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(iw+18,1);
		  }
                  iw=99;
		}
	      }
	    }
	  }//---endif ieta inHF-
	}//---endif TS=nr6
      } 
    }//------end loop over TS for lumi
    return;
  }
}
 // void HcalBeamMonitor::processEvent(const HBHERecHit Collection&hbheHits; ...)


void HcalBeamMonitor::beginLuminosityBlock()
{
  // reset histograms that get updated each luminosity section
  HFlumi_occ_LS->Reset();
  stringstream title;
  title <<"HFlumi occupancy for LS # " <<lumiblock;
  HFlumi_occ_LS->getTH2F()->SetTitle(title.str().c_str());
  return;
} // void HcalBeamMonitor::beginLuminosityBlock()

void HcalBeamMonitor::endLuminosityBlock()
{
  float Nentries=HFlumi_occ_LS->getBinContent(-1,-1);
  if (Nentries==0) return;
  for (int x=1;x<=HFlumi_occ_LS->getTH2F()->GetNbinsX();++x)
    {
      for (int y=1;y<=HFlumi_occ_LS->getTH2F()->GetNbinsY();++y)
	{
	  if (HFlumi_occ_LS->getBinContent(x,y)==0)
	    {
	      // One new luminosity section found with no entries for the cell in question
	      // Add protection requiring a minimum number of entries before counting as dead?
	      HFlumi_total_deadcells->Fill(x-1,2*y-1,1);
	    } // dead cell check

	  if (HFlumi_occ_LS->getBinContent(x,y)>0.5*Nentries)
	    {
	      HFlumi_total_hotcells->Fill(x-1,2*y-1,1);
	      //HFlumi_total_hotcells->setBinContent(x,y,HFlumi_total_hotcells->getBinContent(x,y)+1);
	    } // hot cell check
	} // loop over y
    } // loop over x
  return;
}


void HcalBeamMonitor::SetEtaLabels(MonitorElement * h)
{
  h->getTH2F()->GetXaxis()->SetBinLabel(1,"-36S");
  h->getTH2F()->GetXaxis()->SetBinLabel(2,"-35S");
  h->getTH2F()->GetXaxis()->SetBinLabel(3,"-34L");
  h->getTH2F()->GetXaxis()->SetBinLabel(4,"-33L");
  h->getTH2F()->GetXaxis()->SetBinLabel(5,"33L");
  h->getTH2F()->GetXaxis()->SetBinLabel(6,"34L");
  h->getTH2F()->GetXaxis()->SetBinLabel(7,"35S");
  h->getTH2F()->GetXaxis()->SetBinLabel(8,"36S");
  return;
}

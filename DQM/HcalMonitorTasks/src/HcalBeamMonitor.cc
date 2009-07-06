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
{occThresh_ = 1;}

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
  if (fVerbosity) cout <<"<HcalBeamMonitor::setup> Setup in progress"<<endl;

  beammon_makeDiagnostics_ = ps.getUntrackedParameter<bool>("BeamMonitor_makeDiagnosticPlots",makeDiagnostics);
  // These two variables aren't yet in use
  beammon_checkNevents_    = ps.getUntrackedParameter<int>("BeamMonitor_checkNevents",checkNevents_);
  beammon_minErrorFlag_    = ps.getUntrackedParameter<double>("BeamMonitor_minErrorFlag",0.);

  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("BeamMonitor Event Number");
    
      // Basic Problem Cells
      ProblemBeamCells=m_dbe->book2D(" ProblemBeamCells",
                                     " Problem Beam Cell Rate for all HCAL",
                                     etaBins_,etaMin_,etaMax_,
                                     phiBins_,phiMin_,phiMax_);
      ProblemBeamCells->setAxisTitle("i#eta",1);
      ProblemBeamCells->setAxisTitle("i#phi",2);
      // Only show problem cells that are above problem threshold
      //(ProblemBeamCells->getTH2F())->SetMinimum(beammon_minErrorFlag_);
      (ProblemBeamCells->getTH2F())->SetMinimum(0);
      (ProblemBeamCells->getTH2F())->SetMaximum(1.);
      
      // Overall Problem plot appears in main directory; plots by depth appear \in subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/problem_beammonitor");
      setupDepthHists2D(ProblemBeamCellsByDepth, " Problem BeamMonitor Rate","");

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
      Etsum_eta_L=m_dbe->bookProfile("Et Sum vs Eta Long Fiber","Et Sum per Area vs Eta Long Fiber",120,-6,6,40,0,800);
      Etsum_eta_S=m_dbe->bookProfile("Et Sum vs Eta Short Fiber","Et Sum per Area vs Eta Short Fiber",120,-6,6,40,0,800);
      Etsum_phi_L=m_dbe->bookProfile("Et Sum vs Phi Long Fiber","Et Sum per Area vs Phi Long Fiber",70,-3.5,3.5,40,0,800);
      Etsum_phi_S=m_dbe->bookProfile("Et Sum vs Phi Short Fiber","Et Sum per Area crossing vs Phi Short Fiber",70,-3.5,3.5,40,0,800);
      Etsum_ratio_p=m_dbe->book1D("Occ vs fm HF+","Energy difference of Long and Short Fiber HF+",105,-1.05,1.05);
      Energy_Occ=m_dbe->book1D("Occ vs Energy","Occupancy vs Energy",200,0,2000);
      Etsum_ratio_m=m_dbe->book1D("Occ vs fm HF-","Energy difference of Long and Short Fiber HF-",105,-1.05,1.05);
      Etsum_map_L=m_dbe->book2D("EtSum 2D phi and eta Long Fiber","Et Sum 2D phi and eta Long Fiber",120,-6,6,80,-4,4);
      Etsum_map_S=m_dbe->book2D("EtSum 2D phi and eta Short Fiber","Et Sum 2D phi and eta Short Fiber",120,-6,6,80,-4,4);
      Etsum_rphi_S=m_dbe->book2D("EtSum 2D phi and radius Short Fiber","Et Sum 2D phi and radius Short Fiber",12, radiusbins, 70, phibins);
      Etsum_rphi_L=m_dbe->book2D("EtSum 2D phi and radius Long Fiber","Et Sum 2D phi and radius Long Fiber",12, radiusbins, 70, phibins);
      Etsum_ratio_map=m_dbe->book2D("Abnormal fm","Abnormal fm",84,-42,42,72,0,72);
      
      Occ_rphi_S=m_dbe->book2D("Occ 2D phi and radius Short Fiber","Occupancy 2D phi and radius Short Fiber",12, radiusbins, 70, phibins);
      Occ_rphi_L=m_dbe->book2D("Occ 2D phi and radius Long Fiber","Occupancy 2D phi and radius Long Fiber",12, radiusbins, 70, phibins);
      Occ_eta_S=m_dbe->bookProfile("Occ vs Eta Short Fiber","Occ per Bunch crossing vs Eta Short Fiber",120,-6,6,40,0,800);
      Occ_eta_L=m_dbe->bookProfile("Occ vs Eta Long Fiber","Occ per Bunch crossing vs Eta Long Fiber",120,-6,6,40,0,800);
      
      Occ_phi_L=m_dbe->bookProfile("Occ vs Phi Long Fiber","Occ per Bunch crossing vs Phi Long Fiber",70,-3.5,3.5,40,0,800);
      
      Occ_phi_S=m_dbe->bookProfile("Occ vs Phi Short Fiber","Occ per Bunch crossing vs Phi Short Fiber",70,-3.5,3.5,40,0,800);
      
      Occ_map_L=m_dbe->book2D("Occ_map Long Fiber","Occ Map long Fiber",120,-6,6,70,-3.5,3.5);
      Occ_map_S=m_dbe->book2D("Occ_map Short Fiber","Occ Map Short Fiber",120,-6,6,70,-3.5,3.5);
      
      //HFlumi plots
      HFlumi_ETsum_perwedge =  m_dbe->book1D("HF lumi ET-sum per wedge","HF lumi ET-sum per wedge",36,1,37);
      
      HFlumi_Occupancy_above_thr_r1 =  m_dbe->book1D("HF lumi Occupancy above threshold ring1","HF lumi Occupancy above threshold ring1",36,1,37);
      HFlumi_Occupancy_between_thrs_r1 = m_dbe->book1D("HF lumi Occupancy between thresholds ring1","HF lumi Occupancy between thresholds ring1",36,1,37);
      HFlumi_Occupancy_below_thr_r1 = m_dbe->book1D("HF lumi Occupancy below threshold ring1","HF lumi Occupancy below threshold ring1",36,1,37);
      HFlumi_Occupancy_above_thr_r2 = m_dbe->book1D("HF lumi Occupancy above threshold ring2","HF lumi Occupancy above threshold ring2",36,1,37);
      HFlumi_Occupancy_between_thrs_r2 = m_dbe->book1D("HF lumi Occupancy between thresholds ring2","HF lumi Occupancy between thresholds ring2",36,1,37);
      HFlumi_Occupancy_below_thr_r2 = m_dbe->book1D("HF lumi Occupancy below threshold ring2","HF lumi Occupancy below threshold ring2",36,1,37);
      
      
    } // if (m_dbe)
  return;

} // void HcalBeamMonitor::setup()

void HcalBeamMonitor::processEvent(const HBHERecHitCollection& hbheHits,
				   const HORecHitCollection& hoHits,
				   const HFRecHitCollection& hfHits,
                                   const HFDigiCollection& hf
				     // const ZDCRecHitCollection & zdcHits // include this once we see ZDC rec hits read out
				   )
  
{
  if (!m_dbe)
    {
      if (fVerbosity) cout <<"HcalBeamMonitor::processEvent   DQMStore not instantiated!!!"<<endl;
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
  
  for(int m=0;m<13;m++){
    for(int n=0;n<36;n++){
      hitsp[m][n][0]=0;
      hitsp[m][n][1]=0; 
      hitsm[m][n][0]=0;
      hitsm[m][n][1]=0;
    }
  }
  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalBeamMonitor BEAMMON analyze pre-process-> " << cpu_timer.cpuTime() << std::endl;
      cpu_timer.reset(); cpu_timer.start();
    } // if (showTiming)

  try
    {
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
    } // try
  catch (...)
    {
      if (fVerbosity) cout <<"HcalBeamMonitor::processEvent   Error in HBHE RecHit loop"<<endl;
    } // catch
  
  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalBeamMonitor BEAMMON HBHE-> " << cpu_timer.cpuTime() << std::endl;
      cpu_timer.reset(); cpu_timer.start();
    } // if (showTiming)
  
   // HO loop
  try
    {
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
	      HO_weightedX[index]+=HOiter->energy()*cos(2.*PI*iphi/72);
	      HO_weightedY[index]+=HOiter->energy()*sin(2.*PI*iphi/72);
	      HO_energy[index]+=HOiter->energy();
	    } // for (HOiter=hoHits.begin();...)
	  
	  for (int i=-1*ETA_OFFSET_HO;i<=ETA_OFFSET_HO;++i)
	    {
	      if (i==0) continue;
	      int index = i + ETA_OFFSET_HO;
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
    } // try (HO loop)
  catch (...)
    {
      if (fVerbosity) cout <<"HcalBeamMonitor::processEvent   Error in HO RecHit loop"<<endl;
    } // catch
  
  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalBeamMonitor BEAMMON HO-> " << cpu_timer.cpuTime() << std::endl;
      cpu_timer.reset(); cpu_timer.start();
    } // if (showTiming)

  ///////////////////////////////////
  // HF loop
  try
    {
      if(hfHits.size()>0)
	{
	  double HF_weightedX[HFETASIZE]={0.};
	  double HF_weightedY[HFETASIZE]={0.};
	  double HF_energy[HFETASIZE]={0.};
	  double offset;

	  int ieta, iphi;
          float et,eta,phi,r;
	  for (HFiter=hfHits.begin(); 
	       HFiter!=hfHits.end(); 
	       ++HFiter) 
	    { 
       	      if (HFiter->energy()<0) continue;  // don't include negative-energy cells?

           eta=etaBounds[abs(HFiter->id().ieta())-29];
           et=HFiter->energy()/cosh(eta)/area[abs(HFiter->id().ieta())-29];
           r=radius[abs(HFiter->id().ieta())-29];
           if(HFiter->id().iphi()<37)
           phi=HFiter->id().iphi()*0.087266;
           else phi=(HFiter->id().iphi()-72)*0.087266;
           
	  if (HFiter->id().depth()==1){
            
            
            if(HFiter->id().ieta()>0) {
            
	    Etsum_eta_L->Fill(eta,et);
            Etsum_phi_L->Fill(phi,et);
            Etsum_map_L->Fill(eta,phi,et);
            Etsum_rphi_L->Fill(r,phi,et);
            hitsp[HFiter->id().ieta()-29][(HFiter->id().iphi()-1)/2][0]=HFiter->energy();}
            if(HFiter->id().ieta()<0) {
            Etsum_eta_L->Fill(-eta,et);
            Etsum_phi_L->Fill(phi,et);
            Etsum_rphi_L->Fill(r,phi,et);
            Etsum_map_L->Fill(-eta,phi,et);
            hitsm[-HFiter->id().ieta()-29][(HFiter->id().iphi()-1)/2][0]=HFiter->energy(); 	    
	    }
	  }
         
          //Fill 3 histos for Short Fibers :
	  if (HFiter->id().depth()==2){
	    if(HFiter->id().ieta()>0)  {
              Etsum_eta_S->Fill(eta,et);
              Etsum_phi_S->Fill(phi,et);
              Etsum_rphi_S->Fill(r,phi,et); 
              Etsum_map_S->Fill(eta,phi,et);
              hitsp[HFiter->id().ieta()-29][(HFiter->id().iphi()-1)/2][1]=HFiter->energy();
	    }
	    if(HFiter->id().ieta()<0)  {  Etsum_eta_S->Fill(-eta,et);
              Etsum_map_S->Fill(-eta,phi,et);
              Etsum_phi_S->Fill(phi,et);
              Etsum_rphi_S->Fill(r,phi,et); 
	      hitsm[-HFiter->id().ieta()-29][(HFiter->id().iphi()-1)/2][1]=HFiter->energy();}
          
	  }
	  Energy_Occ->Fill(HFiter->energy()); 
            
	  //HF: no non-threshold occupancy map is filled?
	           
	  if(HFiter->energy()>occThresh_){
	    
            if (HFiter->id().depth()==1){
	      if(HFiter->id().ieta()>0)  
		{ Occ_eta_L->Fill(eta,1);
                  Occ_phi_L->Fill(phi,1);
                  Occ_map_L->Fill(eta,phi,1);
                  Occ_rphi_L->Fill(r,phi,1);
		}

	      if(HFiter->id().ieta()<0)   { 
                  Occ_eta_L->Fill(-eta,1);
                  Occ_phi_L->Fill(phi,1);
                  Occ_map_L->Fill(-eta,phi,1);
	         Occ_rphi_L->Fill(r,phi,1);
	      }}

            if (HFiter->id().depth()==2){
	    if(HFiter->id().ieta()>0) { 
                  Occ_eta_S->Fill(eta,1);
                  Occ_phi_S->Fill(phi,1);
                  Occ_map_S->Fill(eta,phi,1);
                  Occ_rphi_S->Fill(r,phi,1);
            }  
            
            if(HFiter->id().ieta()<0) { 
                  Occ_eta_S->Fill(-eta,1);
                  Occ_map_S->Fill(-eta,phi,1);
                  Occ_phi_S->Fill(phi,1);
                  Occ_rphi_S->Fill(r,phi,1);
	    }  
	    }
   
	  }
           
          else { if (HFiter->id().depth()==1){ 
	      if(HFiter->id().ieta()>0)  
		{ Occ_eta_L->Fill(eta,0);
                  Occ_map_L->Fill(eta,phi,0);
                  Occ_phi_L->Fill(phi,0); 
                  Occ_rphi_L->Fill(r,phi,0);}
	      if(HFiter->id().ieta()<0)   { 
                  Occ_eta_L->Fill(-eta,0);
                  Occ_map_L->Fill(-eta,phi,0);
                  Occ_phi_L->Fill(phi,0);
                  Occ_rphi_L->Fill(r,phi,0);}
	    }

            if (HFiter->id().depth()==2){
	    if(HFiter->id().ieta()>0) { 
                  Occ_eta_S->Fill(eta,0);
                  Occ_map_S->Fill(eta,phi,0);
                  Occ_phi_S->Fill(phi,0);
                  Occ_rphi_S->Fill(r,phi,0);}  
            
            if(HFiter->id().ieta()<0) { 
                  Occ_eta_S->Fill(-eta,0);
                  Occ_map_S->Fill(-eta,phi,0);
                  Occ_phi_S->Fill(phi,0);
                  Occ_rphi_S->Fill(r,phi,0);}  
	    }
	  }//else
	      HcalDetId id(HFiter->detid().rawId());
	      ieta=id.ieta();
	      iphi=id.iphi();

	      HFtotalX+=HFiter->energy()*cos(2.*PI*iphi/72);
	      HFtotalY+=HFiter->energy()*sin(2.*PI*iphi/72);
	      HFtotalE+=HFiter->energy();

	      unsigned int index;
	      index=ieta+ETA_OFFSET_HF;
	      HF_weightedX[index]+=HFiter->energy()*cos(2.*PI*iphi/72);
	      HF_weightedY[index]+=HFiter->energy()*sin(2.*PI*iphi/72);
	      HF_energy[index]+=HFiter->energy();
	    } // for (HFiter=hfHits.begin();...)
	  
	  int hfeta=ETA_OFFSET_HF;
	  for (int i=-1*hfeta;i<=hfeta;++i)
	    {
	      if (i==0) continue;
	      if (i>-1*ETA_BOUND_HF && i <ETA_BOUND_HF) continue;
	      int index = i + ETA_OFFSET_HF;
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
	      
	      ratiop=(hitsp[i][j][0]-hitsp[i][j][1])/(hitsp[i][j][0]+hitsp[i][j][1]);
          //cout<<ratiop<<endl;
	      Etsum_ratio_p->Fill(ratiop);
	      if(abs(ratiop>0.85))
		Etsum_ratio_map->Fill(i+29,2*j+1); 
	    }
	  }
	  
	  for(int p=0;p<13;p++){
	    for(int q=0;q<36;q++){
	      
	      if(hitsm[p][q][0]==hitsm[p][q][1]) continue;
	      ratiom=(hitsm[p][q][0]-hitsm[p][q][1])/(hitsm[p][q][0]+hitsm[p][q][1]);         
	      Etsum_ratio_m->Fill(ratiom);
	      if(abs(ratiom>0.85))
		Etsum_ratio_map->Fill(-p-29,2*q+1);
	    }
	  } 
	} // if (hfHits.size()>0)
    } // try (HF loop)
  catch (...)
    {
      if (fVerbosity) cout <<"HcalBeamMonitor::processEvent   Error in HF RecHit loop"<<endl;
    } // catch
  
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
      // cout <<"MOMENT = "<<moment<<endl;
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

 // void HcalBeamMonitor::processEvent(const HBHERecHit Collection&hbheHits; ...)


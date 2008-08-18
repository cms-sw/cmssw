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


// constructor
HcalBeamMonitor::HcalBeamMonitor():
  ETA_OFFSET_HB(16),
  ETA_OFFSET_HE(29),
  ETA_BOUND_HE(17),
  ETA_OFFSET_HO(15),
  ETA_OFFSET_HF(41),
  ETA_BOUND_HF(29)
{}

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
  baseFolder_ = rootFolder_ + "BeamMonitor";
  if (fVerbosity) cout <<"<HcalBeamMonitor::setup> Setup in progress"<<endl;

  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      char* type;
      type = "BeamMonitor Event Number";
      meEVT_ = m_dbe->bookInt(type);

      CenterOfEnergyRadius = m_dbe->book1D("CenterOfEnergyRadius",
				   "Center Of Energy radius",
				   200,0,1);
      // Grr... MonitorElements don't have GetXaxis(), GetYaxis() methods
      //CenterOfEnergyRadius->GetXaxis()->SetTitle("(normalized) radius");

      CenterOfEnergy = m_dbe->book2D("CenterOfEnergy",
				     "Center of Energy",
				     200,-1,1,
				     200,-1,1);
      //CenterOfEnergy->GetXaxis()->SetTitle("normalized x coordinate");
      //CenterOfEnergy->GetYaxis()->SetTitle("normalized y coordinate");

      COEradiusVSeta = m_dbe->bookProfile("COEradiusVSeta",
					  "Center of Energy radius vs #ieta",
					  172,-43,43,
					  200,0,1);
      //COEradiusVSeta->GetXaxis()->SetTitle("i#eta");
      //COEradiusVSeta->GetYaxis()->SetTitle("(normalized) radius");
      
      std::stringstream histname;
      std::stringstream histtitle;
      m_dbe->setCurrentFolder(baseFolder_+"/HB");
      HBCenterOfEnergyRadius = m_dbe->book1D("HBCenterOfEnergyRadius",
					     "HB Center Of Energy radius",
					     200,0,1);
      HBCenterOfEnergy = m_dbe->book2D("HBCenterOfEnergy",
				       "HB Center of Energy",
				       200,-1,1,
				       200,-1,1);
      for (int i=-16;i<=16;++i)
	{
	  if (i==0) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HB_CenterOfEnergyRadius_ieta"<<i;
	  histtitle<<"HB Center Of Energy i#eta = "<<i;
	  HB_CenterOfEnergyRadius[i]=m_dbe->book1D(histname.str().c_str(),
					    histtitle.str().c_str(),
					    200,0,1);
	} // end of HB loop
      m_dbe->setCurrentFolder(baseFolder_+"/HE");
      HECenterOfEnergyRadius = m_dbe->book1D("HECenterOfEnergyRadius",
					     "HE Center Of Energy radius",
					     200,0,1);
      HECenterOfEnergy = m_dbe->book2D("HECenterOfEnergy",
				       "HE Center of Energy",
				       200,-1,1,
				       200,-1,1);
      for (int i=-29;i<=29;++i)
	{
	  if (i>-17 && i<17) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HE_CenterOfEnergyRadius_ieta"<<i;
	  histtitle<<"HE Center Of Energy i#eta = "<<i;
	  HE_CenterOfEnergyRadius[i]=m_dbe->book1D(histname.str().c_str(),
					    histtitle.str().c_str(),
					    200,0,1);
	} // end of HE loop

      m_dbe->setCurrentFolder(baseFolder_+"/HO");
      HOCenterOfEnergyRadius = m_dbe->book1D("HOCenterOfEnergyRadius",
					     "HO Center Of Energy radius",
					     200,0,1);
      HOCenterOfEnergy = m_dbe->book2D("HOCenterOfEnergy",
				       "HO Center of Energy",
				       200,-1,1,
				       200,-1,1);
      for (int i=-15;i<=15;++i)
	{
	  if (i==0) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HO_CenterOfEnergyRadius_ieta"<<i;
	  histtitle<<"HO Center Of Energy radius i#eta = "<<i;
	  HO_CenterOfEnergyRadius[i]=m_dbe->book1D(histname.str().c_str(),
					    histtitle.str().c_str(),
					    200,0,1);
	} // end of HO loop
      m_dbe->setCurrentFolder(baseFolder_+"/HF");
      HFCenterOfEnergyRadius = m_dbe->book1D("HFCenterOfEnergyRadius",
					     "HF Center Of Energy radius",
					     200,0,1);
      HFCenterOfEnergy = m_dbe->book2D("HFCenterOfEnergy",
				       "HF Center of Energy",
				       200,-1,1,
				       200,-1,1);
      for (int i=-41;i<=41;++i)
	{
	  if (i>-29 && i<29) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HF_CenterOfEnergyRadius_ieta"<<i;
	  histtitle<<"HF Center Of Energy radius i#eta = "<<i;
	  HF_CenterOfEnergyRadius[i]=m_dbe->book1D(histname.str().c_str(),
					    histtitle.str().c_str(),
					    200,0,1);
	} // end of HO loop

    }
} // void HcalBeamMonitor::setup()

void HcalBeamMonitor::processEvent(const HBHERecHitCollection& hbheHits,
				   const HORecHitCollection& hoHits,
				   const HFRecHitCollection& hfHits
				     // const ZDCRecHitCollection & zdcHits // include this once we see ZDC rec hits read out
				   )
  
{
  if (!m_dbe)
    {
      if (fVerbosity) cout <<"HcalBeamMonitor::processEvent   DQMStore not instantiated!!!"<<endl;
      return;
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
  

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  //  try
  if (1>0)
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
		  HB_CenterOfEnergyRadius[i]->Fill(moment);
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
		  HE_CenterOfEnergyRadius[i]->Fill(moment);
		  COEradiusVSeta->Fill(i,moment);
		}
	    } // for (int i=-1*heeta;i<=heeta;++i)

	} // if (hbheHits.size()>0)
    } // try
  //catch (...)
  else
    {
      if (fVerbosity) cout <<"HcalBeamMonitor::processEvent   Error in HBHE RecHit loop"<<endl;
    } // catch
  
  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalRecHit RECHIT HBHE-> " << cpu_timer.cpuTime() << std::endl;
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
		  HO_CenterOfEnergyRadius[i]->Fill(moment);
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
      cpu_timer.stop(); std::cout << " TIMER::HcalRecHit RECHIT HO-> " << cpu_timer.cpuTime() << std::endl;
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
	  for (HFiter=hfHits.begin(); 
	       HFiter!=hfHits.end(); 
	       ++HFiter) 
	    { 
	      // loop over all cells
	      if (HFiter->energy()<0) continue;  // don't include negative-energy cells?
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
		  HF_CenterOfEnergyRadius[i]->Fill(moment);
		  COEradiusVSeta->Fill(i+offset,moment);
		}
	    } // for (int i=-1*hfeta;i<=hfeta;++i)
	} // if (hfHits.size()>0)
    } // try (HF loop)
  catch (...)
    {
      if (fVerbosity) cout <<"HcalBeamMonitor::processEvent   Error in HF RecHit loop"<<endl;
    } // catch
  
  if (showTiming)
    {
      cpu_timer.stop(); std::cout << " TIMER::HcalRecHit RECHIT HF-> " << cpu_timer.cpuTime() << std::endl;
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
  return;
} // void HcalBeamMonitor::processEvent(const HBHERecHit Collection&hbheHits; ...)

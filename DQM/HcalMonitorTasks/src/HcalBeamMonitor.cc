#include "DQM/HcalMonitorTasks/interface/HcalBeamMonitor.h"
// define sizes of ieta arrays for each subdetector

#define PI        3.1415926535897932
#define HBETASIZE 34
#define HEETASIZE 59
#define HFETASIZE 83

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
      type = "TrigPrim Event Number";
      meEVT_ = m_dbe->bookInt(type);

      SecondMoment = m_dbe->book1D("SecondMoment",
				   "Second Moment",
				   1000,-72,72);

      std::stringstream histname;
      std::stringstream histtitle;
      m_dbe->setCurrentFolder(baseFolder_+"/HB");
      for (int i=-16;i<=16;++i)
	{
	  if (i==0) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HB_SecondMoment_eta"<<i;
	  histtitle<<"HB Second Moment #eta = "<<i;
	  HB_SecondMoments[i]=m_dbe->book1D(histname.str().c_str(),
					    histtitle.str().c_str(),
					    1000,-72,72);
	} // end of HB loop
      m_dbe->setCurrentFolder(baseFolder_+"/HE");
      for (int i=-29;i<=29;++i)
	{
	  if (i>-17 && i<17) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HE_SecondMoment_eta"<<i;
	  histtitle<<"HE Second Moment #eta = "<<i;
	  HE_SecondMoments[i]=m_dbe->book1D(histname.str().c_str(),
					    histtitle.str().c_str(),
					    1000,-72,72);
	} // end of HE loop
      m_dbe->setCurrentFolder(baseFolder_+"/HO");
      for (int i=-15;i<=15;++i)
	{
	  if (i==0) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HO_SecondMoment_eta"<<i;
	  histtitle<<"HO Second Moment #eta = "<<i;
	  HO_SecondMoments[i]=m_dbe->book1D(histname.str().c_str(),
					    histtitle.str().c_str(),
					    1000,-72,72);
	} // end of HO loop
      m_dbe->setCurrentFolder(baseFolder_+"/HF");
      for (int i=-41;i<=41;++i)
	{
	  if (i>-29 && i<29) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HF_SecondMoment_eta"<<i;
	  histtitle<<"HF Second Moment #eta = "<<i;
	  HF_SecondMoments[i]=m_dbe->book1D(histname.str().c_str(),
					    histtitle.str().c_str(),
					    1000,-72,72);
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

	      totalX+=HBHEiter->energy()*cos(2*PI*iphi/72);
	      totalY+=HBHEiter->energy()*sin(2*PI*iphi/72);
	      totalE+=HBHEiter->energy();

	      unsigned int index;
	      if ((HcalSubdetector)(id.subdet())==HcalBarrel)
		{
		  index=ieta+ETA_OFFSET_HB;
		  //cout <<"------ "<<ieta<<"  ENERGY = "<<HBHEiter->energy()<<"   PHI = "<<iphi<<endl;
		  HB_weightedX[index]+=HBHEiter->energy()*cos(2.*PI*iphi/72);
		  //cout <<"\t\t X + : "<<HBHEiter->energy()*cos(2.*PI*iphi/72);
		  //cout <<"\t\t\t newX = "<<HB_weightedX[index]<<endl;
		  HB_weightedY[index]+=HBHEiter->energy()*sin(2.*PI*iphi/72);
		  HB_energy[index]+=HBHEiter->energy();
		} // if id.subdet()==HcalBarrel

	      else
		{
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
		HB_SecondMoments[i]->Fill(moment);
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
		HE_SecondMoments[i]->Fill(moment);
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
  

  // HF loop
  try
    {
      if(hfHits.size()>0)
	{
	  double HF_weightedX[HFETASIZE]={0.};
	  double HF_weightedY[HFETASIZE]={0.};
	  double HF_energy[HFETASIZE]={0.};

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

	      totalX+=HFiter->energy()*cos(2.*PI*iphi/72);
	      totalY+=HFiter->energy()*cos(2.*PI*iphi/72);
	      totalE+=HFiter->energy();

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
	      if (moment!=0)
	      HF_SecondMoments[i]->Fill(moment);
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
      cpu_timer.reset(); cpu_timer.start();
    } // if (showTiming)

  
  if (totalE>0)
    {
      double moment = pow(totalX*totalX+totalY*totalY,0.5)/totalE;
      cout <<"MOMENT = "<<moment<<endl;
      SecondMoment->Fill(moment);
    }
  return;
} // void HcalBeamMonitor::processEvent(const HBHERecHit Collection&hbheHits; ...)

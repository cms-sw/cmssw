#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"
#include <iostream>
#include <fstream>
//to exclude bits 2 to 5
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"

#define TIME_MIN -250
#define TIME_MAX 250

using namespace std;

HcalRecHitMonitor::HcalRecHitMonitor()
{
} //constructor

HcalRecHitMonitor::~HcalRecHitMonitor()
{
} //destructor


/* ------------------------------------ */ 

void HcalRecHitMonitor::setup(const edm::ParameterSet& ps,
			      DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (fVerbosity>0)
    std::cout <<"<HcalRecHitMonitor::setup>  Setting up histograms"<<endl;

  baseFolder_ = rootFolder_+"RecHitMonitor_Hcal";

  // Assume subdetectors not present until shown otherwise
  HBpresent_=false;
  HEpresent_=false;
  HOpresent_=false;
  HFpresent_=false;

  // RecHit Monitor - specific cfg variables

  if (fVerbosity>1)
    std::cout <<"<HcalRecHitMonitor::setup>  Getting variable values from cfg files"<<endl;
  
  // rechit_makeDiagnostics_ will take on base task value unless otherwise specified
  rechit_makeDiagnostics_ = ps.getUntrackedParameter<bool>("RecHitMonitor_makeDiagnosticPlots",makeDiagnostics);
  
  // Set checkNevents values
  rechit_checkNevents_ = ps.getUntrackedParameter<int>("RecHitMonitor_checkNevents",checkNevents_);
  rechit_minErrorFlag_ = ps.getUntrackedParameter<double>("RecHitMonitor_minErrorFlag",0.0);

  energyThreshold_       = ps.getUntrackedParameter<double>("RecHitMonitor_energyThreshold",                  0);
  HBenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HB_energyThreshold",energyThreshold_);
  HEenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HE_energyThreshold",energyThreshold_);
  HOenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HO_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HF_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HF_energyThreshold",            -999);
  ZDCenergyThreshold_    = ps.getUntrackedParameter<double>("RecHitMonitor_ZDC_energyThreshold",           -999);

  // zero all counters

  zeroCounters();

  // Set up histograms
  if (m_dbe)
    {
      if (fVerbosity>1)
	std::cout <<"<HcalRecHitMonitor::setup>  Setting up histograms"<<endl;

      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("RecHit Task Event Number");
      meTOTALEVT_ = m_dbe->bookInt("RecHit Task Total Events Processed");

      m_dbe->setCurrentFolder(baseFolder_+"/rechit_info");
      SetupEtaPhiHists(OccupancyByDepth,"RecHit Occupancy","");;

      m_dbe->setCurrentFolder(baseFolder_+"/rechit_info/sumplots");
      SetupEtaPhiHists(SumEnergyByDepth,"RecHit Summed Energy","GeV");
      SetupEtaPhiHists(SqrtSumEnergy2ByDepth,"RecHit Sqrt Summed Energy2","GeV");
      SetupEtaPhiHists(SumTimeByDepth,"RecHit Summed Time","nS");
      
      m_dbe->setCurrentFolder(baseFolder_+"/rechit_info_threshold");
      SetupEtaPhiHists(OccupancyThreshByDepth,"Above Threshold RecHit Occupancy","");

      m_dbe->setCurrentFolder(baseFolder_+"/rechit_info_threshold/sumplots");
      SetupEtaPhiHists(SumEnergyThreshByDepth,"Above Threshold RecHit Summed Energy","GeV");
      SetupEtaPhiHists(SumTimeThreshByDepth,"Above Threshold RecHit Summed Time","nS");

      TH1F* tempflag;
      m_dbe->setCurrentFolder(baseFolder_+"/AnomalousCellFlags");// HB Flag Histograms
      h_HBflagcounter=m_dbe->book1D("HBflags","HB flags",32,-0.5,31.5);
      h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHEHpdHitMultiplicity, "HpdHitMult",1);
      h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHEPulseShape, "PulseShape",1);
      h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_R1R2, "HSCP R1R2",1);
      h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_FracLeader, "HSCP FracLeader",1);
      h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_OuterEnergy, "HSCP OuterEnergy",1);
      h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_ExpFit, "HSCP ExpFit",1);
      h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingSubtractedBit, "Subtracted",1);
      h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingAddedBit, "Added",1);
      h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingErrorBit, "TimingError",1);
      h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::ADCSaturationBit, "Saturation",1);

      // HE Flag Histograms
      h_HEflagcounter=m_dbe->book1D("HEflags","HE flags",32,-0.5,31.5);
      h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHEHpdHitMultiplicity, "HpdHitMult",1);
      h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHEPulseShape, "PulseShape",1);
      h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_R1R2, "HSCP R1R2",1);
      h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_FracLeader, "HSCP FracLeader",1);
      h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_OuterEnergy, "HSCP OuterEnergy",1);
      h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_ExpFit, "HSCP ExpFit",1);
      h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingSubtractedBit, "Subtracted",1);
      h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingAddedBit, "Added",1);
      h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingErrorBit, "TimingError",1);
      h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::ADCSaturationBit, "Saturation",1);

      // HO Flag Histograms
      h_HOflagcounter=m_dbe->book1D("HOflags","HO flags",32,-0.5,31.5);
      h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingSubtractedBit, "Subtracted",1);
      h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingAddedBit, "Added",1);
      h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingErrorBit, "TimingError",1);
      h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::ADCSaturationBit, "Saturation",1);
  
      // HF Flag Histograms
      h_HFflagcounter=m_dbe->book1D("HFflags","HF flags",32,-0.5,31.5);
      h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::HFLongShort, "LongShort",1);
      h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::HFDigiTime, "DigiTime",1);
      h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingSubtractedBit, "Subtracted",1);
      h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingAddedBit, "Added",1);
      h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingErrorBit, "TimingError",1);
      h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::ADCSaturationBit, "Saturation",1);

      tempflag=h_HBflagcounter->getTH1F();
      tempflag->LabelsOption("v");
      tempflag=h_HEflagcounter->getTH1F();
      tempflag->LabelsOption("v");
      tempflag=h_HOflagcounter->getTH1F();
      tempflag->LabelsOption("v");
      tempflag=h_HFflagcounter->getTH1F();
      tempflag->LabelsOption("v");
      
      if (rechit_makeDiagnostics_)
	{
          // hb
	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/hb");
	  h_HBEnergy=m_dbe->book1D("HB_energy","HB RecHit Energy",200,-5,5);
	  h_HBThreshEnergy=m_dbe->book1D("HB_energy_thresh", "HB RecHit Energy Above Threshold",200,-5,5);
	  h_HBTotalEnergy=m_dbe->book1D("HB_total_energy","HB RecHit Total Energy",200,-200,200);
	  h_HBThreshTotalEnergy=m_dbe->book1D("HB_total_energy_thresh", "HB RecHit Total Energy Above Threshold",200,-200,200);
	  h_HBTime=m_dbe->book1D("HB_time","HB RecHit Time",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HBThreshTime=m_dbe->book1D("HB_time_thresh", "HB RecHit Time Above Threshold",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HBOccupancy=m_dbe->book1D("HB_occupancy","HB RecHit Occupancy",2593,-0.5,2592.5);
	  h_HBThreshOccupancy=m_dbe->book1D("HB_occupancy_thresh","HB RecHit Occupancy Above Threshold",2593,-0.5,2592.5);
          
          //he
	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/he");	
	  h_HEEnergy=m_dbe->book1D("HE_energy","HE RecHit Energy",200,-5,5);
	  h_HEThreshEnergy=m_dbe->book1D("HE_energy_thresh", "HE RecHit Energy Above Threshold",200,-5,5);
	  h_HETotalEnergy=m_dbe->book1D("HE_total_energy","HE RecHit Total Energy",200,-200,200);
	  h_HEThreshTotalEnergy=m_dbe->book1D("HE_total_energy_thresh", "HE RecHit Total Energy Above Threshold",200,-200,200);
	  h_HETime=m_dbe->book1D("HE_time","HE RecHit Time",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HEThreshTime=m_dbe->book1D("HE_time_thresh", "HE RecHit Time Above Threshold",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HEOccupancy=m_dbe->book1D("HE_occupancy","HE RecHit Occupancy",2593,-0.5,2592.5);
	  h_HEThreshOccupancy=m_dbe->book1D("HE_occupancy_thresh","HE RecHit Occupancy Above Threshold",2593,-0.5,2592.5);

          // ho
	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/ho");	
	  h_HOEnergy=m_dbe->book1D("HO_energy","HO RecHit Energy",200,-5,5);
	  h_HOThreshEnergy=m_dbe->book1D("HO_energy_thresh", "HO RecHit Energy Above Threshold",200,-5,5);
	  h_HOTotalEnergy=m_dbe->book1D("HO_total_energy","HO RecHit Total Energy",200,-200,200);
	  h_HOThreshTotalEnergy=m_dbe->book1D("HO_total_energy_thresh", "HO RecHit Total Energy Above Threshold",200,-200,200);
	  h_HOTime=m_dbe->book1D("HO_time","HO RecHit Time",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HOThreshTime=m_dbe->book1D("HO_time_thresh", "HO RecHit Time Above Threshold",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HOOccupancy=m_dbe->book1D("HO_occupancy","HO RecHit Occupancy",2161,-0.5,2160.5);
	  h_HOThreshOccupancy=m_dbe->book1D("HO_occupancy_thresh","HO RecHit Occupancy Above Threshold",2161,-0.5,2160.5);

          // hf
	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/hf");	
	  h_HFEnergy=m_dbe->book1D("HF_energy","HF RecHit Energy",200,-5,5);
	  h_HFThreshEnergy=m_dbe->book1D("HF_energy_thresh", "HF RecHit Energy Above Threshold",200,-5,5);
	  h_HFTotalEnergy=m_dbe->book1D("HF_total_energy","HF RecHit Total Energy",200,-200,200);
	  h_HFThreshTotalEnergy=m_dbe->book1D("HF_total_energy_thresh", "HF RecHit Total Energy Above Threshold",200,-200,200);
	  h_HFTime=m_dbe->book1D("HF_time","HF RecHit Time",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HFThreshTime=m_dbe->book1D("HF_time_thresh", "HF RecHit Time Above Threshold",int(TIME_MAX-TIME_MIN),TIME_MIN,TIME_MAX);
	  h_HFOccupancy=m_dbe->book1D("HF_occupancy","HF RecHit Occupancy",1729,-0.5,1728.5);
	  h_HFThreshOccupancy=m_dbe->book1D("HF_occupancy_thresh","HF RecHit Occupancy Above Threshold",1729,-0.5,1728.5);
	} // if (rechit_Diagnostics_)
    } // if (m_dbe)

  return;
  
} //void HcalRecHitMonitor::setup(...)


/* --------------------------- */

void HcalRecHitMonitor::reset(){}  // reset function is empty for now


/* ------------------------- */

void HcalRecHitMonitor::done()
{
  // Can eventually dump bad rec hit info here, when we decide on a definition for bad rec hits
  return;
  
} // void HcalRecHitMonitor::done()



/* --------------------------------- */

void HcalRecHitMonitor::clearME()
{
  // I don't think this function gets cleared any more.  
  // And need to add code to clear out subfolders as well?
  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
    }
  return;
} // void HcalRecHitMonitor::clearME()

/* -------------------------------- */


void HcalRecHitMonitor::processEvent(const HBHERecHitCollection& hbHits,
				     const HORecHitCollection& hoHits,
				     const HFRecHitCollection& hfHits
				     //const ZDCRecHitCollection& zdcHits,
				     )
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
    
  // increment counters  
  HcalBaseMonitor::processEvent();
  
  if (hoHits.size()>0) HOpresent_=true;
  if (hfHits.size()>0) HFpresent_=true;

  if (fVerbosity>1) std::cout <<"<HcalRecHitMonitor::processEvent> Processing event..."<<endl;

  processEvent_rechit(hbHits, hoHits, hfHits);
  
  // Fill problem cells
  if (ievt_%rechit_checkNevents_ ==0)
    {
      fillNevents();
    }

  return;
} // void HcalRecHitMonitor::processEvent(...)


/* --------------------------------------- */


void HcalRecHitMonitor::fillRecHitHistosAtEndRun()
{
  // Fill histograms one last time at endRun call

  fillNevents();
}

/* --------------------------------------- */


void HcalRecHitMonitor::processEvent_rechit( const HBHERecHitCollection& hbheHits,
					     const HORecHitCollection& hoHits,
					     const HFRecHitCollection& hfHits)
  
{
  // Gather rechit info
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>1) std::cout <<"<HcalRecHitMonitor::processEvent_rechitenergy> Processing rechits..."<<endl;
  
  // loop over HBHE
  
  int     hbocc=0;
  int     heocc=0;
  int     hboccthresh=0;
  int     heoccthresh=0;
  double  hbenergy=0;
  double  heenergy=0;
  double  hbenergythresh=0;
  double  heenergythresh=0;

  for (unsigned int i=0;i<4;++i)
    {
      OccupancyByDepth.depth[i]->update();
      OccupancyThreshByDepth.depth[i]->update();
      SumEnergyByDepth.depth[i]->update();
      SqrtSumEnergy2ByDepth.depth[i]->update();
      SumTimeByDepth.depth[i]->update();
    }
    
  h_HBflagcounter->update();
  h_HEflagcounter->update();
  h_HFflagcounter->update();
  h_HOflagcounter->update();
  
  
  for (HBHERecHitCollection::const_iterator HBHEiter=hbheHits.begin(); HBHEiter!=hbheHits.end(); ++HBHEiter) 
    { // loop over all hits
      float en = HBHEiter->energy();
      float ti = HBHEiter->time();
      HcalDetId id(HBHEiter->detid().rawId());
      int ieta = id.ieta();
      int iphi = id.iphi();
      int depth = id.depth();
      HcalSubdetector subdet = id.subdet();
      int calcEta = CalcEtaBin(subdet,ieta,depth);
     
      if (subdet==HcalBarrel)
	{
	  HBpresent_=true;
	  if (!checkHB_) continue;
	  
	  
	  //Looping over HB searching for flags --- cris
	  for (int f=0;f<32;f++)
	    {
	      // Let's display HSCP just to see if tese bits are set
	      /*
	       if(f == HcalCaloFlagLabels::HSCP_R1R2)
		continue;
              if(f == HcalCaloFlagLabels::HSCP_FracLeader)
                continue;
              if(f == HcalCaloFlagLabels::HSCP_OuterEnergy)
                continue;
              if(f == HcalCaloFlagLabels::HSCP_ExpFit)
                continue;
	      */
	      if (HBHEiter->flagField(f))
		HBflagcounter_[f]++;
	      
	    }
	  

	  ++occupancy_[calcEta][iphi-1][depth-1];
	  energy_[calcEta][iphi-1][depth-1]+=en;
          energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
	  time_[calcEta][iphi-1][depth-1]+=ti;
	  if (en>=HBenergyThreshold_)
	    {
	      ++occupancy_thresh_[calcEta][iphi-1][depth-1];
	      energy_thresh_[calcEta][iphi-1][depth-1]+=en;
	      time_thresh_[calcEta][iphi-1][depth-1]+=ti;
	    }
	  if (rechit_makeDiagnostics_)
	    {
	      ++hbocc;
	      hbenergy+=en;
	      if (ti<TIME_MIN || ti>TIME_MAX)
		h_HBTime->Fill(ti);
	      else
		++HBtime_[int(ti-TIME_MIN)];
	      if (en<5 || en>-5)
		h_HBEnergy->Fill(en);
	      else
		++HBenergy_[20*int(en+5)];
	      if (en>=HBenergyThreshold_)
		{
		  ++hboccthresh;
		  hbenergythresh+=en;
		  if (ti<TIME_MIN || ti>TIME_MAX)
		    h_HBThreshTime->Fill(ti);
		  else
		    ++HBtime_thresh_[int(ti-TIME_MIN)];
		  if (en<5 || en>-5)
		    h_HBThreshEnergy->Fill(en);
		  else
		    ++HBenergy_thresh_[20*int(en+5)];
		} // if (en>=HBenergyThreshold_)
	    } // if (rechit_makeDiagnostics_)
	} // if (id.subdet()==HcalBarrel)

      else if (subdet==HcalEndcap)
	{
	  HEpresent_=true;
	  if (!checkHE_) continue;
	  

	  //Looping over HE searching for flags --- cris
	  for (int f=0;f<32;f++)
            {
              if (HBHEiter->flagField(f))
                HEflagcounter_[f]++;
            }


	  ++occupancy_[calcEta][iphi-1][depth-1];
	  energy_[calcEta][iphi-1][depth-1]+=en;
          energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
	  time_[calcEta][iphi-1][depth-1]+=ti;
	  if (en>=HEenergyThreshold_)
	    {
	      ++occupancy_thresh_[calcEta][iphi-1][depth-1];
	      energy_thresh_[calcEta][iphi-1][depth-1]+=en;
	      time_thresh_[calcEta][iphi-1][depth-1]+=ti;
	    }
	  if (rechit_makeDiagnostics_)
	    {
	      ++heocc;
	      heenergy+=en;
	      if (ti<-100 || ti>200)
		h_HETime->Fill(ti);
	      else
		++HEtime_[int(ti+100)];
	      if (en<5 || en>-5)
		h_HEEnergy->Fill(en);
	      else
		++HEenergy_[20*int(en+5)];
	      if (en>=HEenergyThreshold_)
		{
		  ++heoccthresh;
		  heenergythresh+=en;
		  if (ti<-100 || ti>200)
		    h_HEThreshTime->Fill(ti);
		  else
		    ++HEtime_thresh_[int(ti+100)];
		  if (en<5 || en>-5)
		    h_HEThreshEnergy->Fill(en);
		  else
		    ++HEenergy_thresh_[20*int(en+5)];
		} // if (en>=HEenergyThreshold_)
	    } // if (rechit_makeDiagnostics_)


	} // else if (id.subdet()==HcalEndcap)
     
    } //for (HBHERecHitCollection::const_iterator HBHEiter=...)

  if (rechit_makeDiagnostics_)
    {
      ++HB_occupancy_[hbocc];
      ++HE_occupancy_[heocc];
      ++HB_occupancy_thresh_[hboccthresh];
      ++HE_occupancy_thresh_[heoccthresh];
      h_HBTotalEnergy->Fill(hbenergy);
      h_HETotalEnergy->Fill(heenergy);
      h_HBThreshTotalEnergy->Fill(hbenergythresh);
      h_HEThreshTotalEnergy->Fill(heenergythresh);
    }

  // loop over HO

  if (checkHO_)
   {
     int hoocc=0;
     int hooccthresh=0;
     double hoenergy=0;
     double hoenergythresh=0;
     for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) 
       { // loop over all hits
	 float en = HOiter->energy();
	 float ti = HOiter->time();

	 HcalDetId id(HOiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();
         int calcEta = CalcEtaBin(HcalOuter,ieta,depth);


	 //Looping over HO searching for flags --- cris
	 for (int f=0;f<32;f++)
	   {
	     if (HOiter->flagField(f))
	       HOflagcounter_[f]++;
	   }


	 ++occupancy_[calcEta][iphi-1][depth-1];
	 energy_[calcEta][iphi-1][depth-1]+=en;
         energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
	 time_[calcEta][iphi-1][depth-1]+=ti;

	 if (en>=HOenergyThreshold_)
	   {
	     ++occupancy_thresh_[calcEta][iphi-1][depth-1];
	     energy_thresh_[calcEta][iphi-1][depth-1]+=en;
	     time_thresh_[calcEta][iphi-1][depth-1]+=ti;
	   }
	 if (rechit_makeDiagnostics_)
	   {
	     ++hoocc;
	     hoenergy+=en;
	     if (ti<-100 || ti>200)
	       h_HOTime->Fill(ti);
	     else
	       ++HOtime_[int(ti+100)];
	     if (en<5 && en>-5)
	       h_HOEnergy->Fill(en);
	     else
	       ++HOenergy_[20*int(en+5)];
	     if (en>=HOenergyThreshold_)
	       {
		 ++hooccthresh;
		 hoenergythresh+=en;
		 if (ti<-100 || ti>200)
		   h_HOThreshTime->Fill(ti);
		 else
		   ++HOtime_thresh_[int(ti+100)];
		 if (en<5 && en>-5)
		   h_HOThreshEnergy->Fill(en);
		 else
		   ++HOenergy_thresh_[20*int(en+5)];
	       } // if (en>=HOenergyThreshold_)
	   } // if (rechit_makeDiagnostics_)
       } // loop over all HO hits
     if (rechit_makeDiagnostics_)
       {
	 ++HO_occupancy_[hoocc];
	 ++HO_occupancy_thresh_[hooccthresh];
	 h_HOTotalEnergy->Fill(hoenergy);
	 h_HOThreshTotalEnergy->Fill(hoenergythresh);
       }

   } // if (checkHO_)
 
  // loop over HF
  if (checkHF_)
   {
     int hfocc=0;
     int hfoccthresh=0;
     double hfenergy=0;
     double hfenergythresh=0;
     for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter) 
       { // loop over all hits
	 float en = HFiter->energy();
	 float ti = HFiter->time();

	 HcalDetId id(HFiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();
         int calcEta = CalcEtaBin(HcalForward,ieta,depth);


	 //Looping over HF searching for flags --- cris
	 for (int f=0;f<32;f++)
	   {
	     if (HFiter->flagField(f))
	       HFflagcounter_[f]++;
	   }


	 ++occupancy_[calcEta][iphi-1][depth-1];
	 energy_[calcEta][iphi-1][depth-1]+=en;
         energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
	 time_[calcEta][iphi-1][depth-1]+=ti;

	 if (en>=HFenergyThreshold_)
	   {
	     ++occupancy_thresh_[calcEta][iphi-1][depth-1];
	     energy_thresh_[calcEta][iphi-1][depth-1]+=en;
	     time_thresh_[calcEta][iphi-1][depth-1]+=ti;
	   }
	 if (rechit_makeDiagnostics_)
	   {
	     ++hfocc;
	     hfenergy+=en;
	     if (ti<-100 || ti>200)
	       h_HFTime->Fill(ti);
	     else
	       ++HFtime_[int(ti+100)];
	     if (en<5 && en>-5)
	       h_HFEnergy->Fill(en);
	     else
	       ++HFenergy_[20*int(en+5)];
	     if (en>=HFenergyThreshold_)
	       {
		 ++hfoccthresh;
		 hfenergythresh+=en;
		 if (ti<-100 || ti>200)
		   h_HFThreshTime->Fill(ti);
		 else
		   ++HFtime_thresh_[int(ti+100)];
		 if (en<5 && en>-5)
		   h_HFThreshEnergy->Fill(en);
		 else
		   ++HFenergy_thresh_[20*int(en+5)];
	       } // if (en>=HFenergyThreshold_)
	   } // if (rechit_makeDiagnostics_)
       } // loop over all HF hits
     if (rechit_makeDiagnostics_)
       {
	 ++HF_occupancy_[hfocc];
	 ++HF_occupancy_thresh_[hfoccthresh];
	 h_HFTotalEnergy->Fill(hfenergy);
	 h_HFThreshTotalEnergy->Fill(hfenergythresh);
       }
   } // if (checkHF_)
 
 if (showTiming)
   {
     cpu_timer.stop();  std::cout <<"TIMER:: HcalRecHitMonitor PROCESSEVENT_RECHITENERGY -> "<<cpu_timer.cpuTime()<<endl;
   }
 return;
} // void HcalRecHitMonitor::processEvent_rechitenergy

/* --------------------------------------- */




void HcalRecHitMonitor::fillNevents(void)
  //void HcalRecHitMonitor::fillNevents(const HBHERecHitCollection& hbheHits)

{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    } 

  // looking at the contents of HbFlagcounters
  if (fVerbosity>0)
    {
      for (int k = 0; k <= 32; k++){
	std::cout << "<HcalRecHitMonitor::fillNevents>  HF Flag counter:  Bin #" << k+1 << " = "<< HFflagcounter_[k] << endl;
      }
    }

  for (int i=0;i<32;i++)
    {
      h_HBflagcounter->Fill(i,HBflagcounter_[i]);
      h_HEflagcounter->Fill(i,HEflagcounter_[i]);
      h_HOflagcounter->Fill(i,HOflagcounter_[i]);
      h_HFflagcounter->Fill(i,HFflagcounter_[i]);
      HBflagcounter_[i]=0;
      HEflagcounter_[i]=0;
      HOflagcounter_[i]=0;
      HFflagcounter_[i]=0;
    }

  // Fill Occupancy & Sum Energy, Time plots
  if (ievt_>0)
    {
      for (int mydepth=0;mydepth<4;++mydepth)
	{
	  for (int eta=0;eta<OccupancyByDepth.depth[mydepth]->getNbinsX();++eta)
	    {
	      for (int phi=0;phi<72;++phi)
		{
		  OccupancyByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,occupancy_[eta][phi][mydepth]);
		  OccupancyThreshByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,occupancy_thresh_[eta][phi][mydepth]);
		  SumEnergyByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,energy_[eta][phi][mydepth]);
                  SqrtSumEnergy2ByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,sqrt(energy2_[eta][phi][mydepth]));
		  SumEnergyThreshByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,energy_thresh_[eta][phi][mydepth]);
		  SumTimeByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,time_[eta][phi][mydepth]);
		  SumTimeThreshByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,time_thresh_[eta][phi][mydepth]);
		} // for (int phi=0;phi<72;++phi)
	    } // for (int eta=0;eta<OccupancyByDepth...;++eta)
	} // for (int mydepth=0;...)

      FillUnphysicalHEHFBins(OccupancyByDepth);
      FillUnphysicalHEHFBins(OccupancyThreshByDepth);
      FillUnphysicalHEHFBins(SumEnergyByDepth);
      FillUnphysicalHEHFBins(SqrtSumEnergy2ByDepth);
      FillUnphysicalHEHFBins(SumEnergyThreshByDepth);
      FillUnphysicalHEHFBins(SumTimeByDepth);
      FillUnphysicalHEHFBins(SumTimeThreshByDepth);

    } // if (ievt_>0)

  // Fill subdet plots

  for (int i=0;i<200;++i)
    {
      if (HBenergy_[i]!=0) 
	{
	  h_HBEnergy->setBinContent(i+1,HBenergy_[i]);
	}
      if (HBenergy_thresh_[i]!=0) 
	{
	  h_HBThreshEnergy->setBinContent(i+1,HBenergy_thresh_[i]);
	}
      if (HEenergy_[i]!=0) 
	{
	  h_HEEnergy->setBinContent(i+1,HEenergy_[i]);
	}
      if (HEenergy_thresh_[i]!=0) 
	{
	  h_HEThreshEnergy->setBinContent(i+1,HEenergy_thresh_[i]);
	}
      if (HOenergy_[i]!=0) 
	{
	  h_HOEnergy->setBinContent(i+1,HOenergy_[i]);
	}
      if (HOenergy_thresh_[i]!=0) 
	{
	  h_HOThreshEnergy->setBinContent(i+1,HOenergy_thresh_[i]);
	}
      if (HFenergy_[i]!=0) 
	{
	  h_HFEnergy->setBinContent(i+1,HFenergy_[i]);
	}
      if (HFenergy_thresh_[i]!=0) 
	{
	  h_HFThreshEnergy->setBinContent(i+1,HFenergy_thresh_[i]);
	}
    }// for (int i=0;i<200;++i) // Jeff

  for (int i=0;i<(TIME_MAX-TIME_MIN);++i)
    {
      if (HBtime_[i]!=0)
	{
	  h_HBTime->setBinContent(i+1,HBtime_[i]);
	}
      if (HBtime_thresh_[i]!=0)
	{
	  h_HBThreshTime->setBinContent(i+1,HBtime_thresh_[i]);
	}
      if (HEtime_[i]!=0)
	{

	  h_HETime->setBinContent(i+1,HEtime_[i]);
	}
      if (HEtime_thresh_[i]!=0)
	{
	  h_HEThreshTime->setBinContent(i+1,HEtime_thresh_[i]);
	}
      if (HOtime_[i]!=0)
	{
	  h_HOTime->setBinContent(i+1,HOtime_[i]);
	}
      if (HOtime_thresh_[i]!=0)
	{
	  h_HOThreshTime->setBinContent(i+1,HOtime_thresh_[i]);
	}
      if (HFtime_[i]!=0)
	{
	  h_HFTime->setBinContent(i+1,HFtime_[i]);
	}
      if (HFtime_thresh_[i]!=0)
	{
	  h_HFThreshTime->setBinContent(i+1,HFtime_thresh_[i]);
	}
    } // for (int  i=0;i<(TIME_MAX-TIME_MIN);++i)

  for (int i=0;i<2593;++i)
    {
      if (HB_occupancy_[i]>0)
	{
	  h_HBOccupancy->setBinContent(i+1,HB_occupancy_[i]);
	}
      if (HB_occupancy_thresh_[i]>0)
	{
	  h_HBThreshOccupancy->setBinContent(i+1,HB_occupancy_thresh_[i]);
	}
      if (HE_occupancy_[i]>0)
	{
	  h_HEOccupancy->setBinContent(i+1,HE_occupancy_[i]);
	}
      if (HE_occupancy_thresh_[i]>0)
	{
	  h_HEThreshOccupancy->setBinContent(i+1,HE_occupancy_thresh_[i]);
	}
    }//for (int i=0;i<2593;++i)

  for (int i=0;i<2161;++i)
    {
      if (HO_occupancy_[i]>0)
	{
	  h_HOOccupancy->setBinContent(i+1,HO_occupancy_[i]);
	}
      if (HO_occupancy_thresh_[i]>0)
	{
	  h_HOThreshOccupancy->setBinContent(i+1,HO_occupancy_thresh_[i]);
	}
    }//  for (int i=0;i<2161;++i)

  for (int i=0;i<1729;++i)
    {
      if (HF_occupancy_[i]>0)
	{
	  h_HFOccupancy->setBinContent(i+1,HF_occupancy_[i]);
	}
      if (HF_occupancy_thresh_[i]>0)
	{
	  h_HFThreshOccupancy->setBinContent(i+1,HF_occupancy_thresh_[i]);
	}
    }//  for (int i=0;i<2161;++i)

  //zeroCounters();

  if (fVerbosity>0)
    std::cout <<"<HcalRecHitMonitor::fillNevents_problemCells> FILLED REC HIT CELL PLOTS"<<endl;
    
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalRecHitMonitor FILLNEVENTS -> "<<cpu_timer.cpuTime()<<endl;
    }

} // void HcalRecHitMonitor::fillNevents(void)


void HcalRecHitMonitor::zeroCounters(void)
{
  // Set all histogram counters back to zero

  for (int i=0;i<32;++i)
    {
      HBflagcounter_[i]=0;
      HEflagcounter_[i]=0;
      HOflagcounter_[i]=0;
      HFflagcounter_[i]=0;

    }
  // TH2F counters
  for (int i=0;i<ETABINS;++i)
    {
      for (int j=0;j<PHIBINS;++j)
	{
	  for (int k=0;k<6;++k)
	    {
	      occupancy_[i][j][k]=0;
	      occupancy_thresh_[i][j][k]=0;
	      energy_[i][j][k]=0;
              energy2_[i][j][k]=0;
	      energy_thresh_[i][j][k]=0;
	      time_[i][j][k]=0;
	      time_thresh_[i][j][k]=0;
	    }
	} // for (int j=0;j<PHIBINS;++j)
    } // for (int i=0;i<ETABINS;++i)

  // TH1F counters
  
  // energy
  for (int i=0;i<200;++i)
    {
      HBenergy_[i]=0;
      HBenergy_thresh_[i]=0;
      HEenergy_[i]=0;
      HEenergy_thresh_[i]=0;
      HOenergy_[i]=0;
      HOenergy_thresh_[i]=0;
      HFenergy_[i]=0;
      HFenergy_thresh_[i]=0;
      HFenergyLong_[i]=0;
      HFenergyLong_thresh_[i]=0;
      HFenergyShort_[i]=0;
      HFenergyShort_thresh_[i]=0;
    }

  // time
  for (int i=0;i<(TIME_MAX-TIME_MIN);++i)
    {
      HBtime_[i]=0;
      HBtime_thresh_[i]=0;
      HEtime_[i]=0;
      HEtime_thresh_[i]=0;
      HOtime_[i]=0;
      HOtime_thresh_[i]=0;
      HFtime_[i]=0;
      HFtime_thresh_[i]=0;
      HFtimeLong_[i]=0;
      HFtimeLong_thresh_[i]=0;
      HFtimeShort_[i]=0;
      HFtimeShort_thresh_[i]=0;
    }

  // occupancy
  for (int i=0;i<2593;++i)
    {
      HB_occupancy_[i]=0;
      HE_occupancy_[i]=0;
      HB_occupancy_thresh_[i]=0;
      HE_occupancy_thresh_[i]=0;
      if (i<=2160)
	{
	  HO_occupancy_[i]=0;
	  HO_occupancy_thresh_[i]=0;
	}
      if (i<=1728)
	{
	  HF_occupancy_[i]=0;
	  HF_occupancy_thresh_[i]=0;
	}
      if (i<=864)
	{
	  HFlong_occupancy_[i] =0;
	  HFshort_occupancy_[i]=0;
	  HFlong_occupancy_thresh_[i] =0;
	  HFshort_occupancy_thresh_[i]=0;
	}
    } // for (int i=0;i<2592;++i)

  return;
} //void HcalRecHitMonitor::zeroCounters(void)

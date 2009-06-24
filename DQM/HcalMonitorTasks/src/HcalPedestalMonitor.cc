#include "DQM/HcalMonitorTasks/interface/HcalPedestalMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalPedestalMonitor::HcalPedestalMonitor() 
{ 
  shape_=NULL; 
} // constructor


HcalPedestalMonitor::~HcalPedestalMonitor() 
{
  // Do we need to delete all pointers here?  If not, will he have a memory leak?  Does this even get explicitly called?  std::cout statements placed here didn't seem to work
  
} // destructor


void HcalPedestalMonitor::reset(){}


void HcalPedestalMonitor::clearME()
{
  // remove monitor elements.  Is this necessary?
  if(m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
    }
  return;
} // void HcalPedestalMonitor::clearME();


void HcalPedestalMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity)
    std::cout <<"<HcalPedestalMonitor::setup>  Setting up histograms"<<endl;

  stringstream name;
  baseFolder_ = rootFolder_+"PedestalMonitor_Hcal";

  // Pedestal Monitor - specific cfg variables

  // set number of ped events needed for pedestal computation to be performed

  doFCpeds_ = ps.getUntrackedParameter<bool>("PedestalMonitor_pedestalsInFC", true);

  minEntriesPerPed_ = ps.getUntrackedParameter<unsigned int>("PedestalMonitor_minEntriesPerPed",1);

  // set expected pedestal mean, width (in ADC)
  nominalPedMeanInADC_ = ps.getUntrackedParameter<double>("PedestalMonitor_nominalPedMeanInADC",3);
  nominalPedWidthInADC_ = ps.getUntrackedParameter<double>("PedestalMonitor_nominalPedWidthInADC",1);

  // Set error limits that will cause problem histograms to be filled
  maxPedMeanDiffADC_ = ps.getUntrackedParameter<double>("PedesstalMonitor_maxPedMeanDiffADC",1.);
  maxPedWidthDiffADC_ = ps.getUntrackedParameter<double>("PedestalMonitor_maxPedWidthDiffADC",1.);

  pedmon_minErrorFlag_ = ps.getUntrackedParameter<double>("PedestalMonitor_minErrorFlag", minErrorFlag_);
  pedmon_checkNevents_ = ps.getUntrackedParameter<int>("PedestalMonitor_checkNevents", checkNevents_);
  
  makeDiagnostics = ps.getUntrackedParameter<bool>("PedestalMonitor_makeDiagnosticPlots",makeDiagnostics);

  // set bins over which pedestals will be computed
  startingTimeSlice_ = ps.getUntrackedParameter<int>("PedestalMonitor_startingTimeSlice",0);
  endingTimeSlice_   = ps.getUntrackedParameter<int>("PedestalMonitor_endingTimeSlice"  ,1); 

  ievt_=0;

  if ( m_dbe ) 
    {
      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("Pedestal Task Event Number");
      meEVT_->Fill(ievt_);

      // MeanMap, RMSMap values are in ADC, because they show the cells that can create problem cell errors 
      SetupEtaPhiHists(MeanMapByDepth,"Pedestal Mean Map", "ADC");
      SetupEtaPhiHists(RMSMapByDepth, "Pedestal RMS Map", "ADC");
      
      ProblemPedestals=m_dbe->book2D(" ProblemPedestals",
				     " Problem Pedestal Rate for all HCAL",
				     etaBins_,etaMin_,etaMax_,
				     phiBins_,phiMin_,phiMax_);
      ProblemPedestals->setAxisTitle("i#eta",1);
      ProblemPedestals->setAxisTitle("i#phi",2);
      
      // Overall Problem plot appears in main directory; plots by depth appear in subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/problem_pedestals");

      SetupEtaPhiHists(ProblemPedestalsByDepth, " Problem Pedestal Rate","");

      m_dbe->setCurrentFolder(baseFolder_+"/adc/unsubtracted");
      SetupEtaPhiHists(ADCPedestalMean, "Pedestal Values Map","ADC");
      SetupEtaPhiHists( ADCPedestalRMS, "Pedestal Widths Map","ADC");
      setupDepthHists1D(ADCPedestalMean_1D, "1D Pedestal Values",
			"ADC",0,10,200);
      setupDepthHists1D(ADCPedestalRMS_1D, "1D Pedestal Widths",
			"ADC",0,10,200);

      // Subtracted pedetals (disable for now)
      /*
      m_dbe->setCurrentFolder(baseFolder_+"/adc/subtracted(BETA)");
      SetupEtaPhiHists(subADCPedestalMean, "Subtracted Pedestal Values Map",
			"ADC");
      SetupEtaPhiHists(subADCPedestalRMS, "Subtracted Pedestal Widths Map",
			"ADC");
      setupDepthHists1D(subADCPedestalMean_1D, "1D Subtracted Pedestal Values",
			"ADC",-10,10,200);
      setupDepthHists1D(subADCPedestalRMS_1D, "1D Subtracted Pedestal Widths",
			"ADC",-10,10,200);
      */

      m_dbe->setCurrentFolder(baseFolder_+"/fc/unsubtracted");
      SetupEtaPhiHists(fCPedestalMean, "Pedestal Values Map",
			"fC");
      SetupEtaPhiHists(fCPedestalRMS, "Pedestal Widths Map",
			"fC");
      setupDepthHists1D(fCPedestalMean_1D, "1D Pedestal Values",
			"fC",-5,15,200);
      setupDepthHists1D(fCPedestalRMS_1D, "1D Pedestal Widths",
			"fC",0,10,200);

      /*
      m_dbe->setCurrentFolder(baseFolder_+"/fc/subtracted(BETA)");
      SetupEtaPhiHists(subfCPedestalMean, "Subtracted Pedestal Values Map",
			"fC");
      SetupEtaPhiHists(subfCPedestalRMS, "Subtracted Pedestal Widths Map",
			"fC");
      setupDepthHists1D(subfCPedestalMean_1D, "1D Subtracted Pedestal Values",
			"fC",-10,10,200);
      setupDepthHists1D(subfCPedestalRMS_1D, "1D Subtracted Pedestal Widths",
			"fC",-10,10,200);
      */

      m_dbe->setCurrentFolder(baseFolder_+"/reference_pedestals/adc");
      SetupEtaPhiHists(ADC_PedestalFromDBByDepth, 
			"Pedestal Values from DataBase","ADC");
      SetupEtaPhiHists(ADC_WidthFromDBByDepth, 
			"Pedestal Widths from DataBase","ADC");
      setupDepthHists1D(ADC_1D_PedestalFromDBByDepth, "1D Reference Pedestal Values",
			"ADC",0,10,200);
      setupDepthHists1D(ADC_1D_WidthFromDBByDepth, "1D Reference Pedestal Widths",
			"ADC",0,10,200);


      m_dbe->setCurrentFolder(baseFolder_+"/reference_pedestals/fc");
      SetupEtaPhiHists(fC_PedestalFromDBByDepth, 
			"Pedestal Values from DataBase","fC");
		      
      SetupEtaPhiHists(fC_WidthFromDBByDepth, 
			"Pedestal Widths from DataBase","fC");
      setupDepthHists1D(fC_1D_PedestalFromDBByDepth, "1D Reference Pedestal Values",
			"fC",-5,15,200);
      setupDepthHists1D(fC_1D_WidthFromDBByDepth, "1D Reference Pedestal Widths",
			"fC",0,10,200);


      // setup zdc histograms
      if (checkZDC_)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/zdc/");
	  name.str("");
	  for (int side=0;side<2;++side)
	    {
	      for (int section=0;section<2;++section)
		{
		  for (int depth=0; depth<5;++depth)
		    {
		      if (section==1 && depth==4) continue;
		      name<<"ZDC Pedestal Side = "<<(side+1)<<" section = "<<(section+1)<<" channel = "<<(depth + 1);
		      zdc_pedestals.push_back(m_dbe->book1D(name.str().c_str(),
							    name.str().c_str(),
							    25,0,25));
		      name.str("");
		    }
		}
	    }
	}// if (checkZDC_)
      for (unsigned int zz=0;zz<zdc_pedestals.size();++zz)
	zdc_pedestals[zz]->setAxisTitle("ADC counts");
      zeroCounters();

    } // if (m_dbe)

  
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalPedestalMonitor SETUP -> "<<cpu_timer.cpuTime()<<endl;
    }
  
  return;

} // void HcalPedestalMonitor::setup(...)


// *************************************************** //


void HcalPedestalMonitor::processEvent(const HBHEDigiCollection& hbhe,
				       const HODigiCollection& ho,
				       const HFDigiCollection& hf,
				       const ZDCDigiCollection& zdc, // ZDCs not yet added
				       const HcalDbService& cond)
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  ievt_++;
  meEVT_->Fill(ievt_);

  if(!shape_) shape_ = cond.getHcalShape(); // this one is generic

  if(!m_dbe) { 
    if(fVerbosity) std::cout<<"HcalPedestalMonitor::processEvent   DQMStore not instantiated!!!"<<endl;
    return; 
  }
  
  CaloSamples tool;  // digi values in ADC will be converted to fC values stored in tool
  float ADC_myval=0;

  // dummy fills for histograms
  for (unsigned int i=0;i<MeanMapByDepth.depth.size();++i)
    {
      MeanMapByDepth.depth[i]->setBinContent(0,ievt_);
      RMSMapByDepth.depth[i]->setBinContent(0,ievt_);
      ADC_PedestalFromDBByDepth.depth[i]->setBinContent(0,ievt_);
      ADC_WidthFromDBByDepth.depth[i]->setBinContent(0,ievt_);
      fC_PedestalFromDBByDepth.depth[i]->setBinContent(0,ievt_);
      fC_WidthFromDBByDepth.depth[i]->setBinContent(0,ievt_);
      ADCPedestalMean.depth[i]->setBinContent(0,ievt_);
      ADCPedestalRMS.depth[i]->setBinContent(0,ievt_);
      fCPedestalMean.depth[i]->setBinContent(0,ievt_);
      fCPedestalRMS.depth[i]->setBinContent(0,ievt_);
      // Disable subtracted pedestals for now
      //subADCPedestalMean.depth[i]->setBinContent(0,ievt_);
      //subADCPedestalRMS.depth[i]->setBinContent(0,ievt_);
      //subfCPedestalMean.depth[i]->setBinContent(0,ievt_);
      //subfCPedestalRMS.depth[i]->setBinContent(0,ievt_);
      ProblemPedestalsByDepth.depth[i]->setBinContent(0,ievt_);
    }

  for (unsigned int i=0;i<ADC_1D_PedestalFromDBByDepth.size();++i)
    {
      ADC_1D_PedestalFromDBByDepth[i]->setBinContent(0,ievt_);
      ADC_1D_WidthFromDBByDepth[i]->setBinContent(0,ievt_);
      fC_1D_PedestalFromDBByDepth[i]->setBinContent(0,ievt_);
      fC_1D_WidthFromDBByDepth[i]->setBinContent(0,ievt_);
      ADCPedestalMean_1D[i]->setBinContent(0,ievt_);
      ADCPedestalRMS_1D[i]->setBinContent(0,ievt_);
      fCPedestalMean_1D[i]->setBinContent(0,ievt_);
      fCPedestalRMS_1D[i]->setBinContent(0,ievt_);
      // Disable subtracted pedestals for now
      //subfCPedestalMean_1D[i]->setBinContent(0,ievt_);
      //subfCPedestalRMS_1D[i]->setBinContent(0,ievt_);
      //subADCPedestalMean_1D[i]->setBinContent(0,ievt_);
      //subADCPedestalRMS_1D[i]->setBinContent(0,ievt_);
    }
  ProblemPedestals->setBinContent(0,ievt_);


  //ZDC loop
  for (ZDCDigiCollection::const_iterator j=zdc.begin();
       j!=zdc.end();++j)

    {
      const ZDCDataFrame digi = (const ZDCDataFrame)(*j);
      int zside = digi.id().zside()==1 ? 0 : 1;
      int section = digi.id().section()-1;
      int channel = digi.id().channel()-1;
      for (int k=0;k<digi.size();++k)
	{
	  int zdc_myval=digi.sample(k).adc();
	  if (zdc_myval>=0 && zdc_myval<=24)
	    ++zdc_ADC_peds[zside][section][channel][zdc_myval];
	  else // over/underflow bin
	    ++zdc_ADC_peds[zside][section][channel][25];
	  ++zdc_ADC_count[zside][section][channel];
	}
    }

  // HB/HE Loop
  for (HBHEDigiCollection::const_iterator j=hbhe.begin(); 
       j!=hbhe.end(); ++j)
    {
      
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      // Temporary fix to avoid channels not in calibration DB
      if (!digi.id().validDetId(digi.id().subdet(),digi.id().ieta(),digi.id().iphi(),digi.id().depth())) continue;
      if(!checkHB_ && (HcalSubdetector)(digi.id().subdet())==HcalBarrel) continue;
      if(!checkHE_ && (HcalSubdetector)(digi.id().subdet())==HcalEndcap) continue;
      
      calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private.
      int iEta   = digi.id().ieta();
      int iPhi   = digi.id().iphi();
      int iDepth = digi.id().depth();
      
      channelCoder_ = cond.getHcalCoder(digi.id());
      HcalCoderDb coderDB(*channelCoder_, *shape_);
      coderDB.adc2fC(digi,tool);  // convert digi ADC counts to fC in tool
      
      // Now loop over digi slices
      for (int k=0;k<digi.size();++k)
	{
	  if (k<startingTimeSlice_ || k>endingTimeSlice_)
	    continue;
	  
	  // Add ADC value to pedestal computation
	  pedcounts[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]++;
	  ADC_myval=digi.sample(k).adc();
	  ADC_pedsum[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+= ADC_myval;
	  ADC_pedsum2[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	  fC_pedsum[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+= tool[k];
	  fC_pedsum2[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+=tool[k]*tool[k];
	  
	} // for (int k=0;k<digi.size();++k)
    } // loop over digis
  
  // HO Loop

  if (checkHO_)
    {
      for (HODigiCollection::const_iterator j=ho.begin(); 
	   j!=ho.end(); ++j)
	{
	  const HODataFrame digi = (const HODataFrame)(*j);
	  //const HcalPedestalWidth* pedw = cond.getPedestalWidth(digi.id());
	  // Temporary fix to avoid channels not in calibration DB
	  if (!digi.id().validDetId(digi.id().subdet(),digi.id().ieta(),digi.id().iphi(),digi.id().depth())) continue;
	  calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private.
	  int iEta   = digi.id().ieta();
	  int iPhi   = digi.id().iphi();
	  int iDepth = digi.id().depth();
	  
	  // Convert digi ADC value to fC (stored in tool)
	  channelCoder_ = cond.getHcalCoder(digi.id());
	  HcalCoderDb coderDB(*channelCoder_, *shape_);
	  coderDB.adc2fC(digi,tool);
	  
	  // Now loop over digi slices
	  for (int k=0;k<digi.size();++k)
	    {
	      if (k<startingTimeSlice_ || k>endingTimeSlice_)
		continue;
		  
	      // Add ADC value to pedestal computation
	      pedcounts[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]++;
	      ADC_myval=digi.sample(k).adc();
	      ADC_pedsum[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+= ADC_myval;
	      ADC_pedsum2[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_pedsum[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+= tool[k];
	      fC_pedsum2[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+=tool[k]*tool[k];

	    } // for (int k=0;k<digi.size();++k)
	} // loop over digis
    } // if (checkHO_)

  // HF Loop
  
  if (checkHF_)
    {    
      for (HFDigiCollection::const_iterator j=hf.begin(); 
	   j!=hf.end(); ++j)
	{
	  const HFDataFrame digi = (const HFDataFrame)(*j);
	  // Temporary fix to avoid channels not in calibration DB
	  if (!digi.id().validDetId(digi.id().subdet(),digi.id().ieta(),digi.id().iphi(),digi.id().depth())) continue;
	  //const HcalPedestalWidth* pedw = cond.getPedestalWidth(digi.id());
	  calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private.
	  int iEta   = digi.id().ieta();
	  int iPhi   = digi.id().iphi();
	  int iDepth = digi.id().depth();
	      
	  channelCoder_ = cond.getHcalCoder(digi.id());
	  HcalCoderDb coderDB(*channelCoder_, *shape_);
	  coderDB.adc2fC(digi,tool);
	  	      
	  // Now loop over digi slices
	  for (int k=0;k<digi.size();++k)
	    {
	      if (k<startingTimeSlice_ || k>endingTimeSlice_)
		continue;

	      // Add ADC value to pedestal computation

	      // Depth values increased by 2 to avoid overlap with HE at |eta|=29

	      pedcounts[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]++;
	      ADC_myval=digi.sample(k).adc();
	      ADC_pedsum[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+= ADC_myval;
	      ADC_pedsum2[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_pedsum[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+= tool[k];
	      fC_pedsum2[CalcEtaBin(digi.id().subdet(),iEta,iDepth)][iPhi-1][iDepth-1]+=tool[k]*tool[k];

	    } // for (int k=0;k<digi.size();++k)
	} // loop over digis
    } // if (checkHF_)

  // Should we allow each subdetector to get filled separately?  (Fill hfHists every 1000 events, hoHists every 2000, etc.?)

  if (ievt_%pedmon_checkNevents_==0)
    {
      fillPedestalHistos();
    }

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalPedestalMonitor DIGI PROCESSEVENT -> "<<cpu_timer.cpuTime()<<endl;
    }

  
  return;
} // void HcalPestalMonitor::processEvent(...)


// *********************************************************** //


void HcalPedestalMonitor::fillPedestalHistos(void)
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  // Fills pedestal histograms
  if (fVerbosity>0) 
    std::cout <<"<HcalPedestalMonitor::fillPedestalHistos> Entered fillPedestalHistos routine"<<endl;
  

  // fill zdc histograms
  for (unsigned int zs=0;zs<2;++zs)
    {
      for (unsigned int sec=0;sec<2;++sec)
	{
	  for (unsigned int ch=0;ch<5;++ch)
	    {
	      unsigned int counter=9*zs+5*sec+ch;
	      if (counter>=zdc_pedestals.size())
		continue;
	      for (int a=0;a<25;++a)
		zdc_pedestals[counter]->setBinContent(a+1,zdc_ADC_peds[zs][sec][ch][a]);
	    }
	}
    }


  // Set value to be filled in problem histograms to be checkNevents (or remainder of ievt_/pedmon_checkNevents_)
  
  double fillvalue=0;

  double ADC_mean, ADC_RMS;
  
  double fC_mean, fC_RMS;
  double ADC_temp_mean, ADC_temp_RMS, fC_temp_mean, fC_temp_RMS;
  int subdet=0;
  int ieta=-9999;

  for (int i=0;i<4;++i)
    {
      ADCPedestalMean_1D[subdet]->Reset();
      ADCPedestalRMS_1D[subdet]->Reset();
      fCPedestalMean_1D[subdet]->Reset();
      fCPedestalRMS_1D[subdet]->Reset();
  
      //subtracted pedestals -- disable for now
      /*
      subADCPedestalMean_1D[subdet]->Reset();
      subADCPedestalRMS_1D[subdet]->Reset();
      subfCPedestalMean_1D[subdet]->Reset();
      subfCPedestalRMS_1D[subdet]->Reset();
      */
    }

  for (int eta=0;eta<85;++eta)
    {
      for (int phi=0;phi<72;++phi)
	{
	  for (int depth=0;depth<4;++depth) // this is one unit less "true" depth (for indexing purposes) 
	    {
	      // Skip events that don't contain required number of events
	      if (pedcounts[eta][phi][depth] < minEntriesPerPed_) continue;
	      if (pedcounts[eta][phi][depth]==0) continue;

	      subdet=depth+1;
	      ieta=CalcIeta(eta,depth+1);
	      if (ieta==-9999) continue;
	      if (depth<2)
		{
		  if (depth==1)
		    {
		      if (abs(ieta)<17) subdet=1;
		      else if (eta<13 || eta>71) subdet=4;
		      else subdet=2;
		    }
		  else if (depth==2)
		    {
		      if (abs(ieta)<13) subdet=1;
		      else if (eta<14 || eta>43) subdet=4;
		      else subdet=2;
		    }
		}
	      else if (depth==3) // depth=3; HO
		subdet=3;
	      else subdet=2; //depth==2; HE 
	      // fillvalue = fraction of events used for pedestal determination
	      // a small fillvalue causes the problem plots to get filled with a smaller value than a large fillvalue
	      if ((endingTimeSlice_-startingTimeSlice_+1)*ievt_==0) continue;
	      fillvalue = 1.*pedcounts[eta][phi][depth]/((endingTimeSlice_-startingTimeSlice_+1)*ievt_);

	      // Compute mean and RMS for raw and subtracted pedestals in units of fC and ADC (phew!)

	      ADC_mean= 1.*ADC_pedsum[eta][phi][depth]/pedcounts[eta][phi][depth]; 
	      ADC_RMS = 1.0*ADC_pedsum2[eta][phi][depth]/pedcounts[eta][phi][depth]-1.*ADC_mean*ADC_mean;
	      ADC_RMS=pow(fabs(ADC_RMS),0.5);

	      fC_mean=1.*fC_pedsum[eta][phi][depth]/pedcounts[eta][phi][depth];
	      fC_RMS=1.0*fC_pedsum2[eta][phi][depth]/pedcounts[eta][phi][depth]-1.*fC_mean*fC_mean;
	      fC_RMS=pow(fabs(fC_RMS),0.5);
	      
	      // When setting Bin Content, bins start at count of 1, not 0.

	      ADCPedestalMean.depth[depth]->setBinContent(eta+1,phi+1,ADC_mean);
	      ADCPedestalRMS.depth[depth]->setBinContent(eta+1,phi+1,ADC_RMS);
	      ADCPedestalMean_1D[subdet-1]->Fill(ADC_mean);
	      ADCPedestalRMS_1D[subdet-1]->Fill(ADC_RMS);

	      fCPedestalMean.depth[depth]->setBinContent(eta+1,phi+1,fC_mean);
	      fCPedestalMean_1D[subdet-1]->Fill(fC_mean);
	      fCPedestalRMS.depth[depth]->setBinContent(eta+1,phi+1,fC_RMS);
	      fCPedestalRMS_1D[subdet-1]->Fill(fC_RMS);

	      //subtracted pedestals
	      ADC_temp_mean = ADC_mean-ADC_PedestalFromDBByDepth.depth[depth]->getBinContent(eta+1,phi+1);
	      ADC_temp_RMS = ADC_RMS-ADC_WidthFromDBByDepth.depth[depth]->getBinContent(eta+1,phi+1);
	      // disable subtracted histograms for now
	      /* 
	      subADCPedestalMean.depth[depth]->setBinContent(eta+1,phi+1,ADC_temp_mean);
	      subADCPedestalRMS.depth[depth]->setBinContent(eta+1,phi+1,ADC_temp_RMS);
	      subADCPedestalMean_1D[subdet-1]->Fill(ADC_temp_mean);
	      subADCPedestalRMS_1D[subdet-1]->Fill(ADC_temp_RMS);
	      */
	      fC_temp_mean = fC_mean-fC_PedestalFromDBByDepth.depth[depth]->getBinContent(eta+1,phi+1);
	      fC_temp_RMS = fC_RMS-fC_WidthFromDBByDepth.depth[depth]->getBinContent(eta+1,phi+1);
	      /*
	      subfCPedestalMean.depth[depth]->setBinContent(eta+1,phi+1,fC_temp_mean);
	      subfCPedestalRMS.depth[depth]->setBinContent(eta+1,phi+1,fC_temp_RMS);
	      subfCPedestalMean_1D[subdet-1]->Fill(fC_temp_mean);
	      subfCPedestalRMS_1D[subdet-1]->Fill(fC_temp_RMS);
	      */

	      // Overall plots by depth
	      MeanMapByDepth.depth[depth]->setBinContent(eta+1,phi+1,ADC_mean);
	      RMSMapByDepth.depth[depth]->setBinContent(eta+1,phi+1,ADC_RMS);

	      // Don't flag HO SiPMs as problematic yet; just skip them
	      if (isSiPM(ieta,phi+1,depth+1)) continue;

	      // Problem Cells
	      // Problem Cells currently determined by checking ADC counts against
	      // nominal expectations of (mean=3 ADC counts, RMS = 1 count)
	      // We may instead want to compare the values to the database values in order to determine problems?
	      if  (fillvalue>pedmon_minErrorFlag_ 
		   && (fabs(ADC_mean-nominalPedMeanInADC_)>maxPedMeanDiffADC_ 
		       || fabs(ADC_RMS-nominalPedWidthInADC_)>maxPedWidthDiffADC_))
		{
		  ProblemPedestals->setBinContent(eta+1,phi+1,fillvalue);
		  ProblemPedestalsByDepth.depth[depth]->setBinContent(eta+1,phi+1,fillvalue);
		}
	    } // for (int depth)
	} // for (int phi)
    } // for (int eta)
  
  // Fill unphysical cells
  FillUnphysicalHEHFBins(ADCPedestalMean);
  FillUnphysicalHEHFBins(ADCPedestalRMS);
  FillUnphysicalHEHFBins(fCPedestalMean); 
  FillUnphysicalHEHFBins(fCPedestalRMS);  
  
  //subtracted pedestals -- disable for now
  //FillUnphysicalHEHFBins(subADCPedestalMean);
  //FillUnphysicalHEHFBins(subADCPedestalRMS); 
  
  //FillUnphysicalHEHFBins(subfCPedestalMean); 
  //FillUnphysicalHEHFBins(subfCPedestalRMS);  

  // Individual capid plots

  // Overall plots by depth
  FillUnphysicalHEHFBins(MeanMapByDepth);    
  FillUnphysicalHEHFBins(RMSMapByDepth);     
  
  FillUnphysicalHEHFBins(ProblemPedestalsByDepth); 
  FillUnphysicalHEHFBins(ProblemPedestals);



  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalPedestalHistos DIGI FILLPEDESTALHISTOS -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;

} //void HcalPedestalMonitor::fillPedestalHistos(void)


// *********************************************************** //


void HcalPedestalMonitor::done()
{
  // I'd like to put in another call to fillPedestalHistos() here, but this gets called after root file gets written?
  // update on 22 October 2008:  DQMFileSaver gets called with endRun, not endJob.  If we want the the plots to be updated, we should call them in endRun
  return;
}


// ******************************************************** //

void HcalPedestalMonitor::fillDBValues(const HcalDbService& cond)
{
  /* Fills reference pedestal mean, width plots with pedestal values from conditions database
   */
  
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }


  if(!shape_) shape_ = cond.getHcalShape(); // this one is generic

  double ADC_ped=0;
  double ADC_width=0;
  double fC_ped=0;
  double fC_width=0;
  double temp_ADC=0;
  double temp_fC=0;

  int ieta=-9999;
  for (int subdet=1; subdet<=4;++subdet)
    {
      for (int depth=0;depth<4;++depth)
	{
	  for (int eta=0;eta<ADCPedestalMean.depth[depth]->getNbinsX();++eta)
	    {
	      ieta=CalcIeta(subdet,eta,depth+1);
	      if (ieta==-9999) continue;
	      for (int phi=0;phi<ADCPedestalMean.depth[depth]->getNbinsY();++phi)
		{
		  if (!validDetId((HcalSubdetector)(subdet), ieta, phi+1, depth+1)) continue;
		  HcalDetId detid((HcalSubdetector)(subdet), ieta, phi+1, depth+1);
		  ADC_ped=0;
		  ADC_width=0;
		  fC_ped=0;
		  fC_width=0;
		  calibs_= cond.getHcalCalibrations(detid);  
		  const HcalPedestalWidth* pedw = cond.getPedestalWidth(detid);
		  channelCoder_ = cond.getHcalCoder(detid);

		  // Loop over capIDs
		  for (unsigned int capid=0;capid<4;++capid)
		    {
		      // Still need to determine how to convert widths to ADC or fC
		      // calibs_.pedestal value is always in fC, according to Radek
		      temp_fC = calibs_.pedestal(capid);
		      fC_ped+= temp_fC;
		      // convert to ADC from fC
		      temp_ADC=channelCoder_->adc(*shape_,
						  (float)calibs_.pedestal(capid),
						  capid);
		      ADC_ped+=temp_ADC;

		      if (doFCpeds_)
			{
			  temp_fC=pedw->getSigma(capid,capid);
			  fC_width+=temp_fC;
			  temp_ADC=pedw->getSigma(capid,capid)*pow(1.*channelCoder_->adc(*shape_,(float)calibs_.pedestal(capid),capid)/calibs_.pedestal(capid),2);
			  ADC_width+=temp_ADC;
			}
		      else
			{
			  temp_ADC=pedw->getSigma(capid,capid);
			  ADC_width+=temp_ADC;
			  temp_fC=pedw->getSigma(capid,capid)*pow(1.*calibs_.pedestal(capid)/channelCoder_->adc(*shape_,(float)calibs_.pedestal(capid),capid),2);
			  fC_width+=temp_fC;
			}

		    }//capid loop

		  // Pedestal values are average over four cap IDs
		  // widths are sqrt(SUM [sigma_ii^2])/4.
		  fC_ped/=4.;
		  ADC_ped/=4.;

		  ADC_1D_PedestalFromDBByDepth[subdet-1]->Fill(ADC_ped);
		  fC_1D_PedestalFromDBByDepth[subdet-1]->Fill(fC_ped);

		  // Divide width by 2, or by four?
		  // Dividing by 2 gives subtracted results closer to zero -- estimate of variance?
		  fC_width=pow(fC_width,0.5)/2.;
		  ADC_width=pow(ADC_width,0.5)/2.;

		  ADC_1D_WidthFromDBByDepth[subdet-1]->Fill(ADC_width);
		  fC_1D_WidthFromDBByDepth[subdet-1]->Fill(fC_width);

		  if (fVerbosity>1)
		    {
		      std::cout <<"<HcalPedestalMonitor::fillDBValues> HcalDet ID = "<<(HcalSubdetector)subdet<<": ("<<ieta<<", "<<phi+1<<", "<<depth<<")"<<endl;
		      std::cout <<"\tADC pedestal = "<<ADC_ped<<" +/- "<<ADC_width<<endl;
		      std::cout <<"\tfC pedestal = "<<fC_ped<<" +/- "<<fC_width<<endl;
		    }
		  // Shift HF by -/+1 when filling eta-phi histograms
		  if (subdet==4)
		    {
		      if (ieta<0) ieta--;
		      else ieta++;
		    }
		  ADC_PedestalFromDBByDepth.depth[depth]->Fill(ieta,phi+1,ADC_ped);
		  ADC_WidthFromDBByDepth.depth[depth]->Fill(ieta, phi+1, ADC_width);
		  fC_PedestalFromDBByDepth.depth[depth]->Fill(ieta,phi+1,fC_ped);
		  fC_WidthFromDBByDepth.depth[depth]->Fill(ieta, phi+1, fC_width);
		} // iphi loop
	    } // ieta loop
	} //depth loop

    } // subdet loop
  FillUnphysicalHEHFBins(ADC_PedestalFromDBByDepth);
  FillUnphysicalHEHFBins(ADC_WidthFromDBByDepth);
  FillUnphysicalHEHFBins(fC_PedestalFromDBByDepth);
  FillUnphysicalHEHFBins(fC_WidthFromDBByDepth);

  if (showTiming)
    {
      cpu_timer.stop();  
      std::cout <<"TIMER:: HcalPedestalMonitor FILLDBVALUES -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalPedestalMonitor::fillDBValues(void)


void HcalPedestalMonitor::zeroCounters(void)
{
  for (unsigned int zs=0;zs<2;++zs)
    {
      for (unsigned int sec=0;sec<2;++sec)
	{
	  for (unsigned int ch=0;ch<5;++ch)
	    {
	      zdc_ADC_count[zs][sec][ch]=0;
	      for (unsigned int tt=0;tt<25;++tt)
		{
		  zdc_ADC_peds[zs][sec][ch][tt]=0;
		} 
	    }
	}
    }

  // initialize all counters to 0
  for (unsigned int eta=0;eta<85;++eta)
    {
      for (unsigned int phi=0;phi<72;++phi)
	{
	  for (unsigned int depth=0;depth<4;++depth)
	    {
	      pedcounts[eta][phi][depth]=0;
	      ADC_pedsum[eta][phi][depth]=0;
	      ADC_pedsum2[eta][phi][depth]=0;
	      fC_pedsum[eta][phi][depth]=0;
	      fC_pedsum2[eta][phi][depth]=0;
	    } // loop over depths
	} // loop over phi
    } // loop over eta
  
}

// ******************************************************** //

#include "DQM/HcalMonitorTasks/interface/HcalPedestalMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/*
  This is the old version of the Pedestal Monitor, that
  checked presamples of all digis to estimate pedestals.
  The newer version (HcalDetDiagPedestalMonitor) runs on
  only orbit gap events (the HcalCalib stream).
  
  This code is now used only for dumping out the reference pedestals 
  at the start of a run.
*/

HcalPedestalMonitor::HcalPedestalMonitor() 
{ 
  shape_=NULL; 
} // constructor


HcalPedestalMonitor::~HcalPedestalMonitor() 
{
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
  baseFolder_ = rootFolder_+"ReferencePedestalMonitor_Hcal";

  // Pedestal Monitor - specific cfg variables
  makeDiagnostics = ps.getUntrackedParameter<bool>("ReferencePedestalMonitor_makeDiagnosticPlots",makeDiagnostics);

  // if makeDiagnostics not set, only the reference histograms will be plotted

  // doFCpeds_ not needed; pedestals assumed to be read out in fC?
  doFCpeds_ = ps.getUntrackedParameter<bool>("ReferencePedestalMonitor_pedestalsInFC", true);
  
  minEntriesPerPed_ = ps.getUntrackedParameter<unsigned int>("ReferencePedestalMonitor_minEntriesPerPed",1);

  // set expected pedestal mean, width (in ADC)
  nominalPedMeanInADC_ = ps.getUntrackedParameter<double>("ReferencePedestalMonitor_nominalPedMeanInADC",3);
  nominalPedWidthInADC_ = ps.getUntrackedParameter<double>("ReferencePedestalMonitor_nominalPedWidthInADC",1);

  // Set error limits that will cause problem histograms to be filled
  maxPedMeanDiffADC_ = ps.getUntrackedParameter<double>("ReferencePedestalMonitor_maxPedMeanDiffADC",1.);
  maxPedWidthDiffADC_ = ps.getUntrackedParameter<double>("ReferencePedestalMonitor_maxPedWidthDiffADC",1.);

  pedmon_minErrorFlag_ = ps.getUntrackedParameter<double>("ReferencePedestalMonitor_minErrorFlag", minErrorFlag_);
  pedmon_checkNevents_ = ps.getUntrackedParameter<int>("ReferencePedestalMonitor_checkNevents", checkNevents_);
  
  // set bins over which pedestals will be computed
  startingTimeSlice_ = ps.getUntrackedParameter<int>("ReferencePedestalMonitor_startingTimeSlice",0);
  endingTimeSlice_   = ps.getUntrackedParameter<int>("ReferencePedestalMonitor_endingTimeSlice"  ,1); 

  ievt_=0;

  if ( m_dbe ) 
    {
      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("Pedestal Task Event Number");
      meEVT_->Fill(ievt_);
      ProblemCells=m_dbe->book2D(" ProblemReferencePedestals",
				 " Problem Reference Pedestal Rate for all HCAL;i#eta;i#phi",
				 85,-42.5,42.5,
				 72,0.5,72.5);
      SetEtaPhiLabels(ProblemCells);

      m_dbe->setCurrentFolder(baseFolder_+"/reference_pedestals/adc");
      SetupEtaPhiHists(ADC_PedestalFromDBByDepth, 
			"Pedestal Values from DataBase","ADC");
      SetupEtaPhiHists(ADC_WidthFromDBByDepth, 
			"Pedestal Widths from DataBase","ADC");
      setupDepthHists1D(ADC_1D_PedestalFromDBByDepth, "1D Reference Pedestal Values",
			"ADC",0,15,120);
      setupDepthHists1D(ADC_1D_WidthFromDBByDepth, "1D Reference Pedestal Widths",
			"ADC",0,5,40);

      m_dbe->setCurrentFolder(baseFolder_+"/reference_pedestals/fc");
      SetupEtaPhiHists(fC_PedestalFromDBByDepth, 
			"Pedestal Values from DataBase","fC");
		      
      SetupEtaPhiHists(fC_WidthFromDBByDepth, 
			"Pedestal Widths from DataBase","fC");
      setupDepthHists1D(fC_1D_PedestalFromDBByDepth, "1D Reference Pedestal Values",
			"fC",-5,20,250);
      setupDepthHists1D(fC_1D_WidthFromDBByDepth, "1D Reference Pedestal Widths",
			"fC",0,5,100);

      // Overall Problem plot appears in main directory; plots by depth appear in subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/problem_referencepedestals");
      SetupEtaPhiHists(ProblemCellsByDepth, " Problem Reference Pedestal Rate","");
      zeroCounters();

      if (!makeDiagnostics)
	return;
      m_dbe->setCurrentFolder(baseFolder_);
      SetupEtaPhiHists(MeanMapByDepth,"Pedestal Mean Map", "ADC");
      SetupEtaPhiHists(RMSMapByDepth, "Pedestal RMS Map", "ADC");

      m_dbe->setCurrentFolder(baseFolder_+"/adc/unsubtracted");
      SetupEtaPhiHists(ADCPedestalMean, "Pedestal Values Map","ADC");
      SetupEtaPhiHists( ADCPedestalRMS, "Pedestal Widths Map","ADC");
      setupDepthHists1D(ADCPedestalMean_1D, "1D Pedestal Values",
			"ADC",0,15,200);
      setupDepthHists1D(ADCPedestalRMS_1D, "1D Pedestal Widths",
			"ADC",0,5,40);

      m_dbe->setCurrentFolder(baseFolder_+"/fc/unsubtracted");
      SetupEtaPhiHists(fCPedestalMean, "Pedestal Values Map",
			"fC");
      SetupEtaPhiHists(fCPedestalRMS, "Pedestal Widths Map",
			"fC");
      setupDepthHists1D(fCPedestalMean_1D, "1D Pedestal Values",
			"fC",-5,20,200);
      setupDepthHists1D(fCPedestalRMS_1D, "1D Pedestal Widths",
			"fC",0,5,100);
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
				       const HcalDbService& cond)
{
  if (!makeDiagnostics) // in normal running, this pedetal monitor shouldn't run
    return;

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

  for (unsigned int i=0;i<ADC_1D_PedestalFromDBByDepth.size();++i)
    {
      ADC_1D_PedestalFromDBByDepth[i]->update();
      ADC_1D_WidthFromDBByDepth[i]->update();
      fC_1D_PedestalFromDBByDepth[i]->update();
      fC_1D_WidthFromDBByDepth[i]->update();
      ADCPedestalMean_1D[i]->update();
      ADCPedestalRMS_1D[i]->update();
      fCPedestalMean_1D[i]->update();
      fCPedestalRMS_1D[i]->update();
    }
  ProblemCells->update();
  for (unsigned int i=0;i<ProblemCellsByDepth.depth.size();++i)
    ProblemCellsByDepth.depth[i]->update();
  
  if (makeDiagnostics)
    {
      // dummy fills for histograms
      for (unsigned int i=0;i<MeanMapByDepth.depth.size();++i)
	{
	  MeanMapByDepth.depth[i]->update();
	  RMSMapByDepth.depth[i]->update();
	  ADC_PedestalFromDBByDepth.depth[i]->update();
	  ADC_WidthFromDBByDepth.depth[i]->update();
	  fC_PedestalFromDBByDepth.depth[i]->update();
	  fC_WidthFromDBByDepth.depth[i]->update();
	  ADCPedestalMean.depth[i]->update();
	  ADCPedestalRMS.depth[i]->update();
	  fCPedestalMean.depth[i]->update();
	  fCPedestalRMS.depth[i]->update();
	  ProblemCellsByDepth.depth[i]->update();
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

  if (!makeDiagnostics) return; // don't bother filling histograms unless makeDiagnostics is true

  // Fills pedestal histograms
  if (fVerbosity>0) 
    std::cout <<"<HcalPedestalMonitor::fillPedestalHistos> Entered fillPedestalHistos routine"<<endl;

  // Set value to be filled in problem histograms to be checkNevents (or remainder of ievt_/pedmon_checkNevents_)
  
  double fillvalue=0;

  double ADC_mean, ADC_RMS;
  
  double fC_mean, fC_RMS;
  double ADC_temp_mean, ADC_temp_RMS, fC_temp_mean, fC_temp_RMS;
  int subdet=0;
  int ieta=-9999;


  for (int i=0;i<4;++i)
    {
      ADCPedestalMean_1D[i]->Reset();
      ADCPedestalRMS_1D[i]->Reset();
      fCPedestalMean_1D[i]->Reset();
      fCPedestalRMS_1D[i]->Reset();
  
      //subtracted pedestals -- disable for now
      /*
      subADCPedestalMean_1D[i]->Reset();
      subADCPedestalRMS_1D[i]->Reset();
      subfCPedestalMean_1D[i]->Reset();
      subfCPedestalRMS_1D[i]->Reset();
      */
    }

  ProblemCells->Reset();
  //ProblemCells->setBinContent(0,0,ievt_);
  int etabins=0;
  int phibins=0;
  int zside=0;

  for (int depth=0;depth<4;++depth)
    {
      ProblemCellsByDepth.depth[depth]->Reset();
      //ProblemCellsByDepth.depth[depth]->setBinContent(0,0,ievt_);
      subdet=depth+1;
      etabins=ProblemCellsByDepth.depth[depth]->getNbinsX();
      phibins=ProblemCellsByDepth.depth[depth]->getNbinsY();
      for (int eta=0;eta<etabins;++eta)
	{
	  ieta=CalcIeta(eta,depth+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
	    {
	      // Skip events that don't contain required number of events
	      if (pedcounts[eta][phi][depth] < minEntriesPerPed_) continue;
	      if (pedcounts[eta][phi][depth]==0) continue;

	      if (depth<2)
		{
		  if (depth==0)
		    {
		      if (abs(ieta)<17) subdet=1; // HB
		      else if (eta<13 || eta > 71) subdet=4; // HF
		      else {subdet=2;} // HE
		    }
		  else if (depth==1)
		    {
		      if (abs(ieta)<17) subdet=1; // HB
		      else if (eta<13 || eta>43) subdet=4; // HF
		      else {subdet=2;}// HE
		    }
		}
	      else if (depth==3) // "true" depth=4; HO
		subdet=3;
	      else if (depth==2)
		{
		  subdet=2; //"true" depth=3; HE 
		}
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
	      zside=0;
	      if (depth<2)
		{
		  if (isHF(eta,depth+1))
		    ieta<0 ? zside = -1 : zside = 1;
		}
	      // Problem Cells
	      // Problem Cells currently determined by checking ADC counts against
	      // nominal expectations of (mean=3 ADC counts, RMS = 1 count)
	      // We may instead want to compare the values to the database values in order to determine problems?
	      if  (fillvalue>pedmon_minErrorFlag_ 
		   && (fabs(ADC_mean-nominalPedMeanInADC_)>maxPedMeanDiffADC_ 
		       || fabs(ADC_RMS-nominalPedWidthInADC_)>maxPedWidthDiffADC_))
		{
		  ProblemCells->Fill(ieta+zside,phi+1,fillvalue);
		  ProblemCellsByDepth.depth[depth]->Fill(ieta+zside,phi+1,fillvalue);
		}
	    } // for (int phi)
	} // for (int eta)
    } // for (int depth...)
  etabins=ProblemCells->getNbinsX();
  phibins=ProblemCells->getNbinsY();

  for (int eta=0;eta<etabins;++eta)
    {
      for (int phi=0;phi<phibins;++phi)
	if (ProblemCells->getBinContent(eta+1,phi+1)>ievt_)
	  ProblemCells->setBinContent(eta+1,phi+1,ievt_);
    }

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
  
  FillUnphysicalHEHFBins(ProblemCellsByDepth); 
  FillUnphysicalHEHFBins(ProblemCells);



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
  int iphi=-9999;
  for (int subdet=1; subdet<=4;++subdet)
    {
      for (int depth=0;depth<4;++depth)
	{
	  for (int eta=0;eta<ADC_PedestalFromDBByDepth.depth[depth]->getNbinsX();++eta)
	    {
	      ieta=CalcIeta(subdet,eta,depth+1);
	      if (ieta==-9999) continue;
	      for (int phi=0;phi<ADC_PedestalFromDBByDepth.depth[depth]->getNbinsY();++phi)
		{
		  iphi=phi+1;
		  if (!validDetId((HcalSubdetector)(subdet), ieta, iphi, depth+1)) continue;
		  HcalDetId detid((HcalSubdetector)(subdet), ieta, iphi, depth+1);
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
		      std::cout <<"<HcalPedestalMonitor::fillDBValues> HcalDet ID = "<<(HcalSubdetector)subdet<<": ("<<ieta<<", "<<iphi<<", "<<depth<<")"<<endl;
		      std::cout <<"\tADC pedestal = "<<ADC_ped<<" +/- "<<ADC_width<<endl;
		      std::cout <<"\tfC pedestal = "<<fC_ped<<" +/- "<<fC_width<<endl;
		    }
		  // Shift HF by -/+1 when filling eta-phi histograms
		  int zside=0;
		  if (subdet==4)
		    {
		      if (ieta<0) zside=-1;
		      else zside=1;
		    }
		  ADC_PedestalFromDBByDepth.depth[depth]->Fill(ieta+zside,iphi,ADC_ped);
		  ADC_WidthFromDBByDepth.depth[depth]->Fill(ieta+zside, iphi, ADC_width);
		  fC_PedestalFromDBByDepth.depth[depth]->Fill(ieta+zside,iphi,fC_ped);
		  fC_WidthFromDBByDepth.depth[depth]->Fill(ieta+zside, iphi, fC_width);
		} // phi loop
	    } // eta loop
	} //depth loop

    } // subdet loop
  FillUnphysicalHEHFBins(ADC_PedestalFromDBByDepth);
  FillUnphysicalHEHFBins(ADC_WidthFromDBByDepth);
  FillUnphysicalHEHFBins(fC_PedestalFromDBByDepth);
  FillUnphysicalHEHFBins(fC_WidthFromDBByDepth);

  // Center ADC pedestal values near 3 +/- 1
  for (unsigned int i=0;i<ADC_PedestalFromDBByDepth.depth.size();++i)
  {
    ADC_PedestalFromDBByDepth.depth[i]->getTH2F()->SetMinimum(0);
    if (ADC_PedestalFromDBByDepth.depth[i]->getTH2F()->GetMaximum()<6)
      ADC_PedestalFromDBByDepth.depth[i]->getTH2F()->SetMaximum(6);
  }

  for (unsigned int i=0;i<ADC_WidthFromDBByDepth.depth.size();++i)
  {
    ADC_WidthFromDBByDepth.depth[i]->getTH2F()->SetMinimum(0);
    if (ADC_WidthFromDBByDepth.depth[i]->getTH2F()->GetMaximum()<2)
      ADC_WidthFromDBByDepth.depth[i]->getTH2F()->SetMaximum(2);
  }

  if (showTiming)
    {
      cpu_timer.stop();  
      std::cout <<"TIMER:: HcalPedestalMonitor FILLDBVALUES -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalPedestalMonitor::fillDBValues(void)


void HcalPedestalMonitor::zeroCounters(void)
{
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
  return;
}

// ******************************************************** //

#include "DQM/HcalMonitorTasks/interface/HcalPedestalMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalPedestalMonitor::HcalPedestalMonitor() 
{ 
  shape_=NULL; 
} // constructor


HcalPedestalMonitor::~HcalPedestalMonitor() 
{
  // Do we need to delete all pointers here?  If not, will he have a memory leak?  Does this even get explicitly called?  cout statements placed here didn't seem to work
  
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
    cout <<"<HcalPedestalMonitor::setup>  Setting up histograms"<<endl;

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
      setupDepthHists2D(MeanMapByDepth,"Pedestal Mean Map", "ADC");
      setupDepthHists2D(RMSMapByDepth, "Pedestal RMS Map", "ADC");
      
      ProblemPedestals=m_dbe->book2D(" ProblemPedestals",
				     " Problem Pedestal Rate for all HCAL",
				     etaBins_,etaMin_,etaMax_,
				     phiBins_,phiMin_,phiMax_);
      ProblemPedestals->setAxisTitle("i#eta",1);
      ProblemPedestals->setAxisTitle("i#phi",2);
      
      // Overall Problem plot appears in main directory; plots by depth appear in subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/problem_pedestals");

      setupDepthHists2D(ProblemPedestalsByDepth, " Problem Pedestal Rate","");

      m_dbe->setCurrentFolder(baseFolder_+"/adc/raw");
      setupDepthHists2D(rawADCPedestalMean, "Pedestal Values Map","ADC");
      setupDepthHists2D( rawADCPedestalRMS, "Pedestal Widths Map","ADC");
      setupDepthHists1D(rawADCPedestalMean_1D, "1D Pedestal Values",
			"ADC",0,10,200);
      setupDepthHists1D(rawADCPedestalRMS_1D, "1D Pedestal Widths",
			"ADC",0,10,200);
      m_dbe->setCurrentFolder(baseFolder_+"/adc/subtracted__beta_testing");
      setupDepthHists2D(subADCPedestalMean, "Subtracted Pedestal Values Map",
			"ADC");
      setupDepthHists2D(subADCPedestalRMS, "Subtracted Pedestal Widths Map",
			"ADC");
      setupDepthHists1D(subADCPedestalMean_1D, "1D Subtracted Pedestal Values",
			"ADC",-5,5,200);
      setupDepthHists1D(subADCPedestalRMS_1D, "1D Subtracted Pedestal Widths",
			"ADC",-5,5,200);

      m_dbe->setCurrentFolder(baseFolder_+"/fc/raw");
      setupDepthHists2D(rawFCPedestalMean, "Pedestal Values Map",
			"fC");
      setupDepthHists2D(rawFCPedestalRMS, "Pedestal Widths Map",
			"fC");
      setupDepthHists1D(rawFCPedestalMean_1D, "1D Pedestal Values",
			"fC",-5,15,200);
      setupDepthHists1D(rawFCPedestalRMS_1D, "1D Pedestal Widths",
			"fC",0,10,200);
      m_dbe->setCurrentFolder(baseFolder_+"/fc/subtracted__beta_testing");
      setupDepthHists2D(subFCPedestalMean, "Subtracted Pedestal Values Map",
			"fC");
      setupDepthHists2D(subFCPedestalRMS, "Subtracted Pedestal Widths Map",
			"fC");
      setupDepthHists1D(subFCPedestalMean_1D, "1D Subtracted Pedestal Values",
			"fC",-10,10,200);
      setupDepthHists1D(subFCPedestalRMS_1D, "1D Subtracted Pedestal Widths",
			"fC",-5,5,200);

      m_dbe->setCurrentFolder(baseFolder_+"/reference_pedestals/adc");
      setupDepthHists2D(ADC_PedestalFromDB, ADC_PedestalFromDBByDepth, 
			"Pedestal Values from DataBase","ADC");
      setupDepthHists2D(ADC_WidthFromDB, ADC_WidthFromDBByDepth, 
			"Pedestal Widths from DataBase","ADC");

      m_dbe->setCurrentFolder(baseFolder_+"/reference_pedestals/fc");
      setupDepthHists2D(fC_PedestalFromDB, fC_PedestalFromDBByDepth, 
			"Pedestal Values from DataBase","fC");
		      
      setupDepthHists2D(fC_WidthFromDB, fC_WidthFromDBByDepth, 
			"Pedestal Widths from DataBase","fC");

      // initialize all counters to 0
      for (int eta=0;eta<ETABINS;++eta)
	{
	  for (int phi=0;phi<PHIBINS;++phi)
	    {
	      for (int depth=0;depth<6;++depth)
		{
		  pedcounts[eta][phi][depth]=0;
		  rawpedsum[eta][phi][depth]=0;
		  rawpedsum2[eta][phi][depth]=0;
		  subpedsum[eta][phi][depth]=0;
		  subpedsum2[eta][phi][depth]=0;
		  fC_rawpedsum[eta][phi][depth]=0;
		  fC_rawpedsum2[eta][phi][depth]=0;
		  fC_subpedsum[eta][phi][depth]=0;
		  fC_subpedsum2[eta][phi][depth]=0;
		
		} // loop over depths
	    } // loop over phi
	} // loop over eta

    } // if (m_dbe)

  
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalPedestalMonitor SETUP -> "<<cpu_timer.cpuTime()<<endl;
    }
  
  return;

} // void HcalPedestalMonitor::setup(...)


// *************************************************** //


void HcalPedestalMonitor::processEvent(const HBHEDigiCollection& hbhe,
				       const HODigiCollection& ho,
				       const HFDigiCollection& hf,
				       //const ZDCDigiCollection& zdc, // ZDCs not yet added
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
    if(fVerbosity) cout<<"HcalPedestalMonitor::processEvent   DQMStore not instantiated!!!"<<endl;
    return; 
  }
  
  CaloSamples tool;  // digi values in ADC will be converted to fC values stored in tool
  float fC_myval=0;
  float ADC_myval=0;

  // HB/HE Loop
  try
    {    
      for (HBHEDigiCollection::const_iterator j=hbhe.begin(); 
	   j!=hbhe.end(); ++j)
	{
      
	  const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
	  if(!checkHB_ && (HcalSubdetector)(digi.id().subdet())==HcalBarrel) continue;
	  if(!checkHE_ && (HcalSubdetector)(digi.id().subdet())==HcalEndcap) continue;

	  calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private.
	  int iEta   = digi.id().ieta();
	  int iPhi   = digi.id().iphi();
	  int iDepth = digi.id().depth();

	  // Offset HBHE depths 1 and 2 by +4 so they don't overlap with HF
	  if (digi.id().subdet()==HcalEndcap && iDepth!=3)
	    iDepth+=4;
	
	  channelCoder_ = cond.getHcalCoder(digi.id());
	  HcalCoderDb coderDB(*channelCoder_, *shape_);
	  coderDB.adc2fC(digi,tool);  // convert digi ADC counts to fC in tool
	  	  
	  // Now loop over digi slices
	  for (int k=0;k<digi.size();++k)
	    {
	      if (k<startingTimeSlice_ || k>endingTimeSlice_)
		continue;
	      
	      unsigned int capid=digi.sample(k).capid();
	      // Add ADC value to pedestal computation
	      pedcounts[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]++;
	      ADC_myval=digi.sample(k).adc();
	      rawpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= ADC_myval;
	      rawpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_rawpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= tool[k];
	      fC_rawpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=tool[k]*tool[k];
	  	      
	      // Form subtracted pedestals
	      // calibration object ALWAYS returns pedestals in fC, according to Radek (11 Feb 2009)

	      fC_myval=tool[k]-calibs_.pedestal(capid);  // digi value in fC - pedestal (fC) for this capid
	      //  HcalQIECoder->adc takes (shape, charge, capid) and returns adc value
	      ADC_myval=digi.sample(k).adc()-(int)(channelCoder_->adc(*shape_,(float)calibs_.pedestal(capid), capid));
	      
	      //cout <<digi.sample(k).adc()<<"  "<<channelCoder_->adc(*shape_,(float)calibs_.pedestal(capid), capid)<<"  ADC = "<<ADC_myval<<endl;
	      subpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval;
	      subpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_subpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=fC_myval;
	      fC_subpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=fC_myval*fC_myval;
	    } // for (int k=0;k<digi.size();++k)
	} // loop over digis
    } // try loop
  catch (...)
    {
      if (fVerbosity>0)
	cout <<"<HcalPedestalMonitor::processEvent>  No HBHE Digis."<<endl;
    }
  
  
  // HO Loop

  try
    {    
      for (HODigiCollection::const_iterator j=ho.begin(); 
	   j!=ho.end(); ++j)
	{
	  if (!checkHO_) continue;
	  const HODataFrame digi = (const HODataFrame)(*j);
	  //const HcalPedestalWidth* pedw = cond.getPedestalWidth(digi.id());
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
		  
	      unsigned int capid=digi.sample(k).capid();
		  
	      // Add ADC value to pedestal computation
	      pedcounts[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]++;
	      ADC_myval=digi.sample(k).adc();
	      rawpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= ADC_myval;
	      rawpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_rawpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= tool[k];
	      fC_rawpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=tool[k]*tool[k];

	      // Form subtracted pedestals
	      fC_myval=tool[k]-calibs_.pedestal(capid);
	      //  HcalQIECoder->adc takes (shape, charge, capid) and returns adc value
	      ADC_myval=digi.sample(k).adc()-(int)channelCoder_->adc(*shape_,(float)calibs_.pedestal(capid), capid);

	      subpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval;
	      subpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_subpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=fC_myval;
	      fC_subpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=fC_myval*fC_myval;
		  
	    } // for (int k=0;k<digi.size();++k)
	} // loop over digis
    } // try loop
  catch (...)
    {
      if (fVerbosity > 0)
	cout <<"<HcalPedestalMonitor::processEvent>  No HO Digis."<<endl;
    }


  // HF Loop
  try
    {    
      for (HFDigiCollection::const_iterator j=hf.begin(); 
	   j!=hf.end(); ++j)
	{
	  if (!checkHO_) continue;
	  const HFDataFrame digi = (const HFDataFrame)(*j);
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

	      unsigned int capid=digi.sample(k).capid();

	      // Add ADC value to pedestal computation

	      // Depth values increased by 2 to avoid overlap with HE at |eta|=29

	      pedcounts[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]++;
	      ADC_myval=digi.sample(k).adc();
	      rawpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= ADC_myval;
	      rawpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_rawpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= tool[k];
	      fC_rawpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=tool[k]*tool[k];

	      // calibs_.pedestal values are always in fC
	      fC_myval=tool[k]-calibs_.pedestal(capid);
	      //  HcalQIECoder->adc takes (shape, charge, capid) and returns adc value
	      ADC_myval=digi.sample(k).adc()-(int)channelCoder_->adc(*shape_,(float)calibs_.pedestal(capid), capid);

	      subpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval;
	      subpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_subpedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=fC_myval;
	      fC_subpedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=fC_myval*fC_myval;

	    } // for (int k=0;k<digi.size();++k)
	} // loop over digis
    } // try loop
  catch (...)
    {
      if (fVerbosity>0)
	cout <<"<HcalPedestalMonitor::processEvent>  No HF Digis."<<endl;
    }

  // Should we allow each subdetector to get filled separately?  (Fill hfHists every 1000 events, hoHists every 2000, etc.?)

  if (ievt_%pedmon_checkNevents_==0)
    {
      fillPedestalHistos();
    }

  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalPedestalMonitor DIGI PROCESSEVENT -> "<<cpu_timer.cpuTime()<<endl;
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
    cout <<"<HcalPedestalMonitor::fillPedestalHistos> Entered fillPedestalHistos routine"<<endl;
  
  // Set value to be filled in problem histograms to be checkNevents (or remainder of ievt_/pedmon_checkNevents_)
  
  double fillvalue=0;

  int mydepth=0;
  double ADC_myval, ADC_RMS, ADC_sub_myval, ADC_sub_RMS;
  
  double fC_myval, fC_RMS, fC_sub_myval, fC_sub_RMS;
  for (int eta=0;eta<(etaBins_-2);++eta)
    {
      for (int phi=0;phi<72;++phi)
	{
	  for (int depth=0;depth<6;++depth) // this is one unit less "true" depth (for indexing purposes) 
	    {
	      mydepth=depth;
	      
	      // Skip events that don't contain required number of events
	      if (pedcounts[eta][phi][depth] < minEntriesPerPed_) continue;
	      
	      // fillvalue = fraction of events used for pedestal determination
	      // a small fillvalue causes the problem plots to get filled with a smaller value than a large fillvalue
	      fillvalue = 1.*pedcounts[eta][phi][depth]/((endingTimeSlice_-startingTimeSlice_+1)*ievt_);

	      // Compute mean and RMS for raw and subtracted pedestals in units of fC and ADC (phew!)

	      ADC_myval= 1.*rawpedsum[eta][phi][depth]/pedcounts[eta][phi][depth]; 
	      ADC_RMS = 1.0*rawpedsum2[eta][phi][depth]/pedcounts[eta][phi][depth]-1.*ADC_myval*ADC_myval;
	      ADC_RMS=pow(fabs(ADC_RMS),0.5);
	      
	      ADC_sub_myval = 1.*subpedsum[eta][phi][depth]/pedcounts[eta][phi][depth];
	      ADC_sub_RMS   = 1.0*subpedsum2[eta][phi][depth]/pedcounts[eta][phi][depth]-1.*ADC_sub_myval*ADC_sub_myval;
	      ADC_sub_RMS = pow(fabs(ADC_sub_RMS),0.5);
		  
	      fC_myval=1.*fC_rawpedsum[eta][phi][depth]/pedcounts[eta][phi][depth];
	      fC_RMS=1.0*fC_rawpedsum2[eta][phi][depth]/pedcounts[eta][phi][depth]-1.*fC_myval*fC_myval;
	      fC_RMS=pow(fabs(fC_RMS),0.5);
	      
	      /*
		if ((eta-int((etaBins_-2)/2))==29 && depth==1)
		cout <<phi<<"  "<<fC_subpedsum[eta][phi][depth]<<"  "<<fC_subpedsum2[eta][phi][depth]<<endl;
	      */
	      fC_sub_myval = 1.0*fC_subpedsum[eta][phi][depth]/pedcounts[eta][phi][depth];
	      fC_sub_RMS   = 1.0*fC_subpedsum2[eta][phi][depth]/pedcounts[eta][phi][depth]-1.*fC_sub_myval*fC_sub_myval;
	      fC_sub_RMS = pow(fabs(fC_sub_RMS),0.5);

	      // When setting Bin Content, bins start at count of 1, not 0.
	      // Also, first bins around eta,phi are empty.
	      // Thus, eta,phi must be shifted by +2 (+1 for bin count, +1 to ignore empty row)

	      // raw pedestals for HF
	      rawADCPedestalMean[mydepth]->setBinContent(eta+2,phi+2,ADC_myval);
	      rawADCPedestalRMS[mydepth]->setBinContent(eta+2,phi+2,ADC_RMS);
	      rawADCPedestalMean_1D[mydepth]->Fill(ADC_myval);
	      rawADCPedestalRMS_1D[mydepth]->Fill(ADC_RMS);
	      rawFCPedestalMean[mydepth]->setBinContent(eta+2,phi+2,fC_myval);
	      rawFCPedestalMean_1D[mydepth]->Fill(fC_myval);
	      rawFCPedestalRMS[mydepth]->setBinContent(eta+2,phi+2,fC_RMS);
	      rawFCPedestalRMS_1D[mydepth]->Fill(fC_RMS);
	      
	      //subtracted pedestals
	      subADCPedestalMean[mydepth]->setBinContent(eta+2,phi+2,ADC_sub_myval);
	      subADCPedestalRMS[mydepth]->setBinContent(eta+2,phi+2,ADC_sub_RMS);
	      subADCPedestalMean_1D[mydepth]->Fill(ADC_sub_myval);
	      subADCPedestalRMS_1D[mydepth]->Fill(ADC_sub_RMS);
	      subFCPedestalMean[mydepth]->setBinContent(eta+2,phi+2,fC_sub_myval);
	      subFCPedestalRMS[mydepth]->setBinContent(eta+2,phi+2,fC_sub_RMS);
	      subFCPedestalMean_1D[mydepth]->Fill(fC_sub_myval);
	      subFCPedestalRMS_1D[mydepth]->Fill(fC_sub_RMS);
	      
	      // Overall plots by depth
	      MeanMapByDepth[mydepth]->setBinContent(eta+2,phi+2,ADC_myval);
	      RMSMapByDepth[mydepth]->setBinContent(eta+2,phi+2,ADC_RMS);
	      
	      // Problem Cells
	      // Problem Cells currently determined by checking ADC counts against
	      // nominal expectations of (mean=3 ADC counts, RMS = 1 count)
	      // We may instead want to compare the values to the database values in order to determine problems?
	      if  (fillvalue>pedmon_minErrorFlag_ 
		   && (fabs(ADC_myval-nominalPedMeanInADC_)>maxPedMeanDiffADC_ 
		       || fabs(ADC_RMS-nominalPedWidthInADC_)>maxPedWidthDiffADC_))
		{
		  ProblemPedestals->setBinContent(eta+2,phi+2,fillvalue);
		  ProblemPedestalsByDepth[mydepth]->setBinContent(eta+2,phi+2,fillvalue);
		}
	      
		  // ZDC still to be added
		  /*
		    ADD ZDC HERE AT SOME POINT!
		  */
	      
	    } // for (int depth)
	} // for (int phi)
    } // for (int eta)
  
  // Fill unphysical cells
  FillUnphysicalHEHFBins(rawADCPedestalMean);
  FillUnphysicalHEHFBins(rawADCPedestalRMS);
  FillUnphysicalHEHFBins(rawFCPedestalMean); 
  FillUnphysicalHEHFBins(rawFCPedestalRMS);  
  
  //subtracted pedestals
  FillUnphysicalHEHFBins(subADCPedestalMean);
  FillUnphysicalHEHFBins(subADCPedestalRMS); 
  
  FillUnphysicalHEHFBins(subFCPedestalMean); 
  FillUnphysicalHEHFBins(subFCPedestalRMS);  
  
  // Overall plots by depth
  FillUnphysicalHEHFBins(MeanMapByDepth);    
  FillUnphysicalHEHFBins(RMSMapByDepth);     
  
  FillUnphysicalHEHFBins(ProblemPedestalsByDepth); 
  FillUnphysicalHEHFBins(ProblemPedestals);

  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalPedestalHistos DIGI FILLPEDESTALHISTOS -> "<<cpu_timer.cpuTime()<<endl;
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
  
  int fill_offset=0;
  double ADC_ped=0;
  double ADC_width=0;
  double fC_ped=0;
  double fC_width=0;
  for (int subdet=1; subdet<=4;++subdet)
    {
      for (int depth=1;depth<=4;++depth)
	{
	  for (int ieta=(int)etaMin_;ieta<=(int)etaMax_;++ieta)
	    {
	      for (int iphi=(int)phiMin_;iphi<=(int)phiMax_;++iphi)
		{
		  //if (!hcal.validDetId((HcalSubdetector)(subdet), ieta, iphi, depth)) continue; // implement once this is available in future version of HcalDetId.h
		  if (!validDetId((HcalSubdetector)(subdet), ieta, iphi, depth)) continue;
		  HcalDetId detid((HcalSubdetector)(subdet), ieta, iphi, depth);

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
		      fC_ped+=calibs_.pedestal(capid);
		      // convert to ADC from fC
		      ADC_ped+=channelCoder_->adc(*shape_,
						  (float)calibs_.pedestal(capid),
						  capid);

		      if (doFCpeds_)
			{
			  fC_width+=pedw->getSigma(capid,capid);
			  ADC_width+=pedw->getSigma(capid,capid)*pow(1.*channelCoder_->adc(*shape_,(float)calibs_.pedestal(capid),capid)/calibs_.pedestal(capid),2);
			}
		      else
			{
			  ADC_width+=pedw->getSigma(capid,capid);
			  fC_width+=pedw->getSigma(capid,capid)*pow(1.*calibs_.pedestal(capid)/channelCoder_->adc(*shape_,(float)calibs_.pedestal(capid),capid),2);
			}
		    }//capid loop

		  // Pedestal values are average over four cap IDs
		  // widths are sqrt(SUM [sigma_ii^2])/4.
		  fC_ped/=4.;
		  ADC_ped/=4.;
		  fC_width=pow(fC_width,0.5)/4.;
		  ADC_width=pow(ADC_width,0.5)/4.;

		  if (fVerbosity>1)
		    {
		      cout <<"<HcalPedestalMonitor::fillDBValues> HcalDet ID = "<<(HcalSubdetector)subdet<<": ("<<ieta<<", "<<iphi<<", "<<depth<<")"<<endl;
		      cout <<"\tADC pedestal = "<<ADC_ped<<" +/- "<<ADC_width<<endl;
		      cout <<"\tfC pedestal = "<<fC_ped<<" +/- "<<fC_width<<endl;
		    }
		  ADC_PedestalFromDB->Fill(ieta,iphi,ADC_ped);
		  ADC_WidthFromDB->Fill(ieta,iphi,ADC_width);
		  fC_PedestalFromDB->Fill(ieta,iphi,fC_ped);
		  fC_WidthFromDB->Fill(ieta,iphi,fC_width);

		  if (((HcalSubdetector)(subdet)==HcalEndcap) && depth<3) fill_offset=4;
		  else fill_offset=0;
		  ADC_PedestalFromDBByDepth[depth-1+fill_offset]->Fill(ieta,iphi,ADC_ped);
		  ADC_WidthFromDBByDepth[depth-1+fill_offset]->Fill(ieta, iphi, ADC_width);
		  fC_PedestalFromDBByDepth[depth-1+fill_offset]->Fill(ieta,iphi,fC_ped);
		  fC_WidthFromDBByDepth[depth-1+fill_offset]->Fill(ieta, iphi, fC_width);
		} // iphi loop
	    } // ieta loop
	} //depth loop

    } // subdet loop
   
  if (showTiming)
    {
      cpu_timer.stop();  
      cout <<"TIMER:: HcalPedestalMonitor FILLDBVALUES -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalPedestalMonitor::fillDBValues(void)


// ******************************************************** //

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
      setupDepthHists2D(ADCPedestalMean, "Pedestal Values Map","ADC");
      setupDepthHists2D( ADCPedestalRMS, "Pedestal Widths Map","ADC");
      setupDepthHists1D(ADCPedestalMean_1D, "1D Pedestal Values",
			"ADC",0,10,200);
      setupDepthHists1D(ADCPedestalRMS_1D, "1D Pedestal Widths",
			"ADC",0,10,200);
      m_dbe->setCurrentFolder(baseFolder_+"/adc/subtracted");
      setupDepthHists2D(subADCPedestalMean, "Subtracted Pedestal Values Map",
			"ADC");
      setupDepthHists2D(subADCPedestalRMS, "Subtracted Pedestal Widths Map",
			"ADC");
      setupDepthHists1D(subADCPedestalMean_1D, "1D Subtracted Pedestal Values",
			"ADC",-10,10,200);
      setupDepthHists1D(subADCPedestalRMS_1D, "1D Subtracted Pedestal Widths",
			"ADC",-10,10,200);
    
      m_dbe->setCurrentFolder(baseFolder_+"/fc/raw");
      setupDepthHists2D(fCPedestalMean, "Pedestal Values Map",
			"fC");
      setupDepthHists2D(fCPedestalRMS, "Pedestal Widths Map",
			"fC");
      setupDepthHists1D(fCPedestalMean_1D, "1D Pedestal Values",
			"fC",-5,15,200);
      setupDepthHists1D(fCPedestalRMS_1D, "1D Pedestal Widths",
			"fC",0,10,200);
      m_dbe->setCurrentFolder(baseFolder_+"/fc/subtracted");
      setupDepthHists2D(subfCPedestalMean, "Subtracted Pedestal Values Map",
			"fC");
      setupDepthHists2D(subfCPedestalRMS, "Subtracted Pedestal Widths Map",
			"fC");
      setupDepthHists1D(subfCPedestalMean_1D, "1D Subtracted Pedestal Values",
			"fC",-10,10,200);
      setupDepthHists1D(subfCPedestalRMS_1D, "1D Subtracted Pedestal Widths",
			"fC",-10,10,200);

      m_dbe->setCurrentFolder(baseFolder_+"/reference_pedestals/adc");
      setupDepthHists2D(ADC_PedestalFromDBByDepth, 
			"Pedestal Values from DataBase","ADC");
      setupDepthHists2D(ADC_WidthFromDBByDepth, 
			"Pedestal Widths from DataBase","ADC");
      setupDepthHists1D(ADC_1D_PedestalFromDBByDepth, "1D Reference Pedestal Values",
			"ADC",0,10,200);
      setupDepthHists1D(ADC_1D_WidthFromDBByDepth, "1D Reference Pedestal Widths",
			"ADC",0,10,200);


      m_dbe->setCurrentFolder(baseFolder_+"/reference_pedestals/fc");
      setupDepthHists2D(fC_PedestalFromDBByDepth, 
			"Pedestal Values from DataBase","fC");
		      
      setupDepthHists2D(fC_WidthFromDBByDepth, 
			"Pedestal Widths from DataBase","fC");
      setupDepthHists1D(fC_1D_PedestalFromDBByDepth, "1D Reference Pedestal Values",
			"fC",-5,15,200);
      setupDepthHists1D(fC_1D_WidthFromDBByDepth, "1D Reference Pedestal Widths",
			"fC",0,10,200);

      // Only make these if diagnostics flag is on

      if (makeDiagnostics)
	{
	  for (int i=0;i<4;++i)
	    {
	      // references
	      m_dbe->setCurrentFolder(baseFolder_+"/reference_pedestals/adc/capid");
	      std::vector<MonitorElement*> refADCmean;
	      name<<"ADC Reference Pedestal Mean CapID "<<i;
	      setupDepthHists2D(refADCmean,(char*)(name.str().c_str()),"ADC");
	      ADC_PedestalFromDBByDepth_bycapid.push_back(refADCmean);
	      name.str("");
	      std::vector<MonitorElement*> refADCRMS;
	      name<<"ADC Reference Pedestal Width CapID "<<i;
	      setupDepthHists2D(refADCRMS,(char*)(name.str().c_str()),"ADC");
	      ADC_WidthFromDBByDepth_bycapid.push_back(refADCRMS);
	      name.str("");
	      std::vector<MonitorElement*> refADCmean1D;
	      name<<"1D ADC Reference Pedestal Mean CapID "<<i;
	      setupDepthHists1D(refADCmean1D,(char*)(name.str().c_str()),"ADC",0,10,200);
	      ADC_PedestalFromDBByDepth_1D_bycapid.push_back(refADCmean1D);
	      name.str("");
	      std::vector<MonitorElement*> refADCRMS1D;
	      name<<"1D ADC Reference Pedestal Width CapID "<<i;
	      setupDepthHists1D(refADCRMS1D,(char*)(name.str().c_str()),"ADC",-5,15,200);
	      ADC_WidthFromDBByDepth_1D_bycapid.push_back(refADCRMS1D);
	      name.str("");


	      m_dbe->setCurrentFolder(baseFolder_+"/reference_pedestals/fc/capid");
	      std::vector<MonitorElement*> reffCmean;
	      name<<"fC Reference Pedestal Mean CapID "<<i;
	      setupDepthHists2D(reffCmean,(char*)(name.str().c_str()),"fC");
	      fC_PedestalFromDBByDepth_bycapid.push_back(reffCmean);
	      name.str("");
	      std::vector<MonitorElement*> reffCRMS;
	      name<<"fC Reference Pedestal Width CapID "<<i;
	      setupDepthHists2D(reffCRMS,(char*)(name.str().c_str()),"fC");
	      fC_WidthFromDBByDepth_bycapid.push_back(reffCRMS);
	      name.str("");
	      std::vector<MonitorElement*> reffCmean1D;
	      name<<"1D fC Reference Pedestal Mean CapID "<<i;
	      setupDepthHists1D(reffCmean1D,(char*)(name.str().c_str()),"fC",0,10,200);
	      fC_PedestalFromDBByDepth_1D_bycapid.push_back(reffCmean1D);
	      name.str("");
	      std::vector<MonitorElement*> reffCRMS1D;
	      name<<"1D fC Reference Pedestal Width CapID "<<i;
	      setupDepthHists1D(reffCRMS1D,(char*)(name.str().c_str()),"fC",0,10,200);
	      fC_WidthFromDBByDepth_1D_bycapid.push_back(reffCRMS1D);
	      name.str("");

	      // ADC
	      // unsubtracted
	      m_dbe->setCurrentFolder(baseFolder_+"/adc/raw/capid");
	      std::vector<MonitorElement*> ADCmean;
	      name<<"ADC Pedestal Mean CapID "<<i;
	      setupDepthHists2D(ADCmean,(char*)(name.str().c_str()),"ADC");
	      ADCPedestalMean_bycapid.push_back(ADCmean);
	      name.str("");
	      std::vector<MonitorElement*> ADCRMS;
	      name<<"ADC Pedestal Width CapID "<<i;
	      setupDepthHists2D(ADCRMS,(char*)(name.str().c_str()),"ADC");
	      ADCPedestalRMS_bycapid.push_back(ADCRMS);
	      name.str("");
	      std::vector<MonitorElement*> ADCmean1D;
	      name<<"1D ADC Pedestal Mean CapID "<<i;
	      setupDepthHists1D(ADCmean1D,(char*)(name.str().c_str()),"ADC",0,10,200);
	      ADCPedestalMean_1D_bycapid.push_back(ADCmean1D);
	      name.str("");
	      std::vector<MonitorElement*> ADCRMS1D;
	      name<<"1D ADC Pedestal Width CapID "<<i;
	      setupDepthHists1D(ADCRMS1D,(char*)(name.str().c_str()),"ADC",0,10,200);
	      ADCPedestalRMS_1D_bycapid.push_back(ADCRMS1D);
	      name.str("");
	      // subtracted
	      m_dbe->setCurrentFolder(baseFolder_+"/adc/subtracted/capid");
	      std::vector<MonitorElement*> subADCmean;
	      name<<"ADC Pedestal Mean Minus Reference CapID "<<i;
	      setupDepthHists2D(subADCmean,(char*)(name.str().c_str()),"ADC");
	      subADCPedestalMean_bycapid.push_back(subADCmean);
	      name.str("");
	      std::vector<MonitorElement*> subADCRMS;
	      name<<"ADC Pedestal Width Minus Reference CapID "<<i;
	      setupDepthHists2D(subADCRMS,(char*)(name.str().c_str()),"ADC");
	      subADCPedestalRMS_bycapid.push_back(subADCRMS);
	      name.str("");
	      std::vector<MonitorElement*> subADCmean1D;
	      name<<"1D ADC Pedestal Mean Minus Reference CapID "<<i;
	      setupDepthHists1D(subADCmean1D,(char*)(name.str().c_str()),"ADC",-10,10,200);
	      subADCPedestalMean_1D_bycapid.push_back(subADCmean1D);
	      name.str("");
	      std::vector<MonitorElement*> subADCRMS1D;
	      name<<"1D ADC Pedestal Width Minus Reference CapID "<<i;
	      setupDepthHists1D(subADCRMS1D,(char*)(name.str().c_str()),"ADC",-10,10,200);
	      subADCPedestalRMS_1D_bycapid.push_back(subADCRMS1D);
	      name.str("");

	      m_dbe->setCurrentFolder(baseFolder_+"/fc/raw/capid");
	      std::vector<MonitorElement*> fCmean;
	      name<<"fC Pedestal Mean CapID "<<i;
	      setupDepthHists2D(fCmean,(char*)(name.str().c_str()),"fC");
	      fCPedestalMean_bycapid.push_back(fCmean);
	      name.str("");
	      std::vector<MonitorElement*> fCRMS;
	      name<<"fC Pedestal Width CapID "<<i;
	      setupDepthHists2D(fCRMS,(char*)(name.str().c_str()),"fC");
	      fCPedestalRMS_bycapid.push_back(fCRMS);
	      name.str("");
	      std::vector<MonitorElement*> fCmean1D;
	      name<<"1D fC Pedestal Mean CapID "<<i;
	      setupDepthHists1D(fCmean1D,(char*)(name.str().c_str()),"fC",-5,15,200);
	      fCPedestalMean_1D_bycapid.push_back(fCmean1D);
	      name.str("");
	      std::vector<MonitorElement*> fCRMS1D;
	      name<<"1D fC Pedestal Width CapID "<<i;
	      setupDepthHists1D(fCRMS1D,(char*)(name.str().c_str()),"fC",0,10,200);
	      fCPedestalRMS_1D_bycapid.push_back(fCRMS1D);
	      name.str("");

	      // subtracted
	      m_dbe->setCurrentFolder(baseFolder_+"/fc/subtracted/capid");
	      std::vector<MonitorElement*> subfCmean;
	      name<<"fC Pedestal Mean Minus Reference CapID "<<i;
	      setupDepthHists2D(subfCmean,(char*)(name.str().c_str()),"fC");
	      subfCPedestalMean_bycapid.push_back(subfCmean);
	      name.str("");
	      std::vector<MonitorElement*> subfCRMS;
	      name<<"fC Pedestal Width Minus Reference CapID "<<i;
	      setupDepthHists2D(subfCRMS,(char*)(name.str().c_str()),"fC");
	      subfCPedestalRMS_bycapid.push_back(subfCRMS);
	      name.str("");
	      std::vector<MonitorElement*> subfCmean1D;
	      name<<"1D fC Pedestal Mean Minus Reference CapID "<<i;
	      setupDepthHists1D(subfCmean1D,(char*)(name.str().c_str()),"fC",-10,10,200);
	      subfCPedestalMean_1D_bycapid.push_back(subfCmean1D);
	      name.str("");
	      std::vector<MonitorElement*> subfCRMS1D;
	      name<<"1D fC Pedestal Width Minus Reference CapID "<<i;
	      setupDepthHists1D(subfCRMS1D,(char*)(name.str().c_str()),"fC",-10,10,200);
	      subfCPedestalRMS_1D_bycapid.push_back(subfCRMS1D);
	      name.str("");
	    } // loop over capids
	} // if (makeDiagnostics)



      // initialize all counters to 0
      for (unsigned int eta=0;eta<ETABINS;++eta)
	{
	  for (unsigned int phi=0;phi<PHIBINS;++phi)
	    {
	      for (unsigned int depth=0;depth<6;++depth)
		{
		  pedcounts[eta][phi][depth]=0;
		  ADC_pedsum[eta][phi][depth]=0;
		  ADC_pedsum2[eta][phi][depth]=0;
		  fC_pedsum[eta][phi][depth]=0;
		  fC_pedsum2[eta][phi][depth]=0;
		  for (unsigned int capid=0;capid<4;++capid)
		    {
		      pedcounts_bycapid[eta][phi][depth][capid]=0;
		      ADC_pedsum_bycapid[eta][phi][depth][capid]=0;
		      ADC_pedsum2_bycapid[eta][phi][depth][capid]=0;
		      fC_pedsum_bycapid[eta][phi][depth][capid]=0;
		      fC_pedsum2_bycapid[eta][phi][depth][capid]=0;
		    } // loop over capids;
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
	      ADC_pedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= ADC_myval;
	      ADC_pedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_pedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= tool[k];
	      fC_pedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=tool[k]*tool[k];
	      
	      pedcounts_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]++;
	      ADC_pedsum_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+= ADC_myval;
	      ADC_pedsum2_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+=ADC_myval*ADC_myval;
	      fC_pedsum_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+= tool[k];
	      fC_pedsum2_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+=tool[k]*tool[k];
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
	      ADC_pedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= ADC_myval;
	      ADC_pedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_pedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= tool[k];
	      fC_pedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=tool[k]*tool[k];
	      
	      pedcounts_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]++;
	      ADC_pedsum_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+= ADC_myval;
	      ADC_pedsum2_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+=ADC_myval*ADC_myval;
	      fC_pedsum_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+= tool[k];
	      fC_pedsum2_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+=tool[k]*tool[k];
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
	      ADC_pedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= ADC_myval;
	      ADC_pedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=ADC_myval*ADC_myval;
	      fC_pedsum[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+= tool[k];
	      fC_pedsum2[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1]+=tool[k]*tool[k];

	      pedcounts_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]++;
	      ADC_pedsum_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+= ADC_myval;
	      ADC_pedsum2_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+=ADC_myval*ADC_myval;
	      fC_pedsum_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+= tool[k];
	      fC_pedsum2_bycapid[iEta+(int)((etaBins_-2)/2)][iPhi-1][iDepth-1][capid]+=tool[k]*tool[k];

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

  double ADC_mean, ADC_RMS;
  
  double fC_mean, fC_RMS;
  double ADC_temp_mean, ADC_temp_RMS, fC_temp_mean, fC_temp_RMS;
  for (int eta=0;eta<(etaBins_-2);++eta)
    {
      for (int phi=0;phi<72;++phi)
	{
	  for (int depth=0;depth<6;++depth) // this is one unit less "true" depth (for indexing purposes) 
	    {
	      // Skip events that don't contain required number of events
	      if (pedcounts[eta][phi][depth] < minEntriesPerPed_) continue;
	      
	      // fillvalue = fraction of events used for pedestal determination
	      // a small fillvalue causes the problem plots to get filled with a smaller value than a large fillvalue
	      fillvalue = 1.*pedcounts[eta][phi][depth]/((endingTimeSlice_-startingTimeSlice_+1)*ievt_);

	      // Compute mean and RMS for raw and subtracted pedestals in units of fC and ADC (phew!)

	      ADC_mean= 1.*ADC_pedsum[eta][phi][depth]/pedcounts[eta][phi][depth]; 
	      ADC_RMS = 1.0*ADC_pedsum2[eta][phi][depth]/pedcounts[eta][phi][depth]-1.*ADC_mean*ADC_mean;
	      ADC_RMS=pow(fabs(ADC_RMS),0.5);
	      		  
	      fC_mean=1.*fC_pedsum[eta][phi][depth]/pedcounts[eta][phi][depth];
	      fC_RMS=1.0*fC_pedsum2[eta][phi][depth]/pedcounts[eta][phi][depth]-1.*fC_mean*fC_mean;
	      fC_RMS=pow(fabs(fC_RMS),0.5);
	      
	      /*
		if ((eta-int((etaBins_-2)/2))==29 && depth==1)
		cout <<phi<<"  "<<fC_subpedsum[eta][phi][depth]<<"  "<<fC_subpedsum2[eta][phi][depth]<<endl;
	      */
	      

	      // When setting Bin Content, bins start at count of 1, not 0.
	      // Also, first bins around eta,phi are empty.
	      // Thus, eta,phi must be shifted by +2 (+1 for bin count, +1 to ignore empty row)

	      ADCPedestalMean[depth]->setBinContent(eta+2,phi+2,ADC_mean);
	      ADCPedestalRMS[depth]->setBinContent(eta+2,phi+2,ADC_RMS);
	      ADCPedestalMean_1D[depth]->Fill(ADC_mean);
	      ADCPedestalRMS_1D[depth]->Fill(ADC_RMS);
	      cout <<"SET MEAN"<<eta+2<<"  "<<phi+2<<"  "<<fC_mean<<endl;
	      fCPedestalMean[depth]->setBinContent(eta+2,phi+2,fC_mean);
	      fCPedestalMean_1D[depth]->Fill(fC_mean);
	      fCPedestalRMS[depth]->setBinContent(eta+2,phi+2,fC_RMS);
	      fCPedestalRMS_1D[depth]->Fill(fC_RMS);
	      
	      //subtracted pedestals
	      ADC_temp_mean = ADC_mean-ADC_PedestalFromDBByDepth[depth]->getBinContent(eta+2,phi+2);
	      ADC_temp_RMS = ADC_RMS-ADC_WidthFromDBByDepth[depth]->getBinContent(eta+2,phi+2);
	      subADCPedestalMean[depth]->setBinContent(eta+2,phi+2,ADC_temp_mean);
	      subADCPedestalRMS[depth]->setBinContent(eta+2,phi+2,ADC_temp_RMS);
	      subADCPedestalMean_1D[depth]->Fill(ADC_temp_mean);
	      subADCPedestalRMS_1D[depth]->Fill(ADC_temp_RMS);
	      fC_temp_mean = fC_mean-fC_PedestalFromDBByDepth[depth]->getBinContent(eta+2,phi+2);
	      fC_temp_RMS = fC_RMS-fC_WidthFromDBByDepth[depth]->getBinContent(eta+2,phi+2);
	      subfCPedestalMean[depth]->setBinContent(eta+2,phi+2,fC_temp_mean);
	      subfCPedestalRMS[depth]->setBinContent(eta+2,phi+2,fC_temp_RMS);
	      subfCPedestalMean_1D[depth]->Fill(fC_temp_mean);
	      subfCPedestalRMS_1D[depth]->Fill(fC_temp_RMS);
	      
	      // Overall plots by depth
	      MeanMapByDepth[depth]->setBinContent(eta+2,phi+2,ADC_mean);
	      RMSMapByDepth[depth]->setBinContent(eta+2,phi+2,ADC_RMS);
	      
	      // Problem Cells
	      // Problem Cells currently determined by checking ADC counts against
	      // nominal expectations of (mean=3 ADC counts, RMS = 1 count)
	      // We may instead want to compare the values to the database values in order to determine problems?
	      if  (fillvalue>pedmon_minErrorFlag_ 
		   && (fabs(ADC_mean-nominalPedMeanInADC_)>maxPedMeanDiffADC_ 
		       || fabs(ADC_RMS-nominalPedWidthInADC_)>maxPedWidthDiffADC_))
		{
		  ProblemPedestals->setBinContent(eta+2,phi+2,fillvalue);
		  ProblemPedestalsByDepth[depth]->setBinContent(eta+2,phi+2,fillvalue);
		}
	      
		  // ZDC still to be added
		  /*
		    ADD ZDC HERE AT SOME POINT!
		  */



	      // individual plots by capid -- plot only if makeDiagnostics flag is on
	      if (makeDiagnostics)
		{
		  for (int capid=0;capid<4;++capid)
		    {
		      // Require individual pedestals be at least minEntriesPerPed?  Or only 
		      // half of that (since only 2 peds tested per event in default settings?)
		      if (pedcounts_bycapid[eta][phi][depth][capid] < (minEntriesPerPed_)) continue;
		      

		      //fillvalue = 1.*pedcounts_bycapid[eta][phi][depth][capid]/((endingTimeSlice_-startingTimeSlice_+1)*ievt_);

		      ADC_mean= 1.*ADC_pedsum_bycapid[eta][phi][depth][capid]/pedcounts_bycapid[eta][phi][depth][capid]; 
		      ADC_RMS = 1.0*ADC_pedsum2_bycapid[eta][phi][depth][capid]/pedcounts_bycapid[eta][phi][depth][capid]-1.*ADC_mean*ADC_mean;
		      ADC_RMS=pow(fabs(ADC_RMS),0.5);
		      
		      fC_mean=1.*fC_pedsum_bycapid[eta][phi][depth][capid]/pedcounts_bycapid[eta][phi][depth][capid];
		      fC_RMS=1.0*fC_pedsum2_bycapid[eta][phi][depth][capid]/pedcounts_bycapid[eta][phi][depth][capid]-1.*fC_mean*fC_mean;
		      fC_RMS=pow(fabs(fC_RMS),0.5);

		      ADCPedestalMean_bycapid[capid][depth]->setBinContent(eta+2,phi+2,ADC_mean);
		      ADCPedestalRMS_bycapid[capid][depth]->setBinContent(eta+2,phi+2,ADC_RMS);
		      ADCPedestalMean_1D_bycapid[capid][depth]->Fill(ADC_mean);
		      ADCPedestalRMS_1D_bycapid[capid][depth]->Fill(ADC_RMS);

		      fCPedestalMean_bycapid[capid][depth]->setBinContent(eta+2,phi+2,fC_mean);
		      fCPedestalMean_1D_bycapid[capid][depth]->Fill(fC_mean);
		      fCPedestalRMS_bycapid[capid][depth]->setBinContent(eta+2,phi+2,fC_RMS);
		      fCPedestalRMS_1D_bycapid[capid][depth]->Fill(fC_RMS);
		      
		      //subtracted pedestals
		      ADC_temp_mean = ADC_mean-ADC_PedestalFromDBByDepth_bycapid[capid][depth]->getBinContent(eta+2,phi+2);
		      ADC_temp_RMS = ADC_RMS-ADC_WidthFromDBByDepth_bycapid[capid][depth]->getBinContent(eta+2,phi+2);
		      subADCPedestalMean_bycapid[capid][depth]->setBinContent(eta+2,phi+2,ADC_temp_mean);
		      subADCPedestalRMS_bycapid[capid][depth]->setBinContent(eta+2,phi+2,ADC_temp_RMS);
		      subADCPedestalMean_1D_bycapid[capid][depth]->Fill(ADC_temp_mean);
		      subADCPedestalRMS_1D_bycapid[capid][depth]->Fill(ADC_temp_RMS);
		      fC_temp_mean = fC_mean-fC_PedestalFromDBByDepth_bycapid[capid][depth]->getBinContent(eta+2,phi+2);
		      fC_temp_RMS = fC_RMS-fC_WidthFromDBByDepth_bycapid[capid][depth]->getBinContent(eta+2,phi+2);

		      subfCPedestalMean_bycapid[capid][depth]->setBinContent(eta+2,phi+2,fC_temp_mean);
		      subfCPedestalRMS_bycapid[capid][depth]->setBinContent(eta+2,phi+2,fC_temp_RMS);
		      subfCPedestalMean_1D_bycapid[capid][depth]->Fill(fC_temp_mean);
		      subfCPedestalRMS_1D_bycapid[capid][depth]->Fill(fC_temp_RMS);
		    } // loop on capids

		}  // if (makeDiagnostics)

	    } // for (int depth)
	} // for (int phi)
    } // for (int eta)
  
  // Fill unphysical cells
  FillUnphysicalHEHFBins(ADCPedestalMean);
  FillUnphysicalHEHFBins(ADCPedestalRMS);
  FillUnphysicalHEHFBins(fCPedestalMean); 
  FillUnphysicalHEHFBins(fCPedestalRMS);  
  
  //subtracted pedestals
  FillUnphysicalHEHFBins(subADCPedestalMean);
  FillUnphysicalHEHFBins(subADCPedestalRMS); 
  
  FillUnphysicalHEHFBins(subfCPedestalMean); 
  FillUnphysicalHEHFBins(subfCPedestalRMS);  

  // Individual capid plots

  for (unsigned int i=0;i<ADC_PedestalFromDBByDepth_bycapid.size();++i)
    {
      for (unsigned int capid=0;capid<ADC_PedestalFromDBByDepth_bycapid[i].size();++capid)
	{
      // Why doesn't this work here?

	  FillUnphysicalHEHFBins(ADCPedestalMean_bycapid[capid]);
	  FillUnphysicalHEHFBins(ADCPedestalRMS_bycapid[capid]);
	  FillUnphysicalHEHFBins(fCPedestalMean_bycapid[capid]);
	  FillUnphysicalHEHFBins(fCPedestalRMS_bycapid[capid]);
	}
    }

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
  double temp_ADC=0;
  double temp_fC=0;
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
		  
		  if (((HcalSubdetector)(subdet)==HcalEndcap) && depth<3) fill_offset=4;
		  else fill_offset=0;

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
		      if (makeDiagnostics)
			{
			  ADC_PedestalFromDBByDepth_bycapid[capid][depth-1+fill_offset]->Fill(ieta,iphi,temp_ADC);
			  fC_PedestalFromDBByDepth_bycapid[capid][depth-1+fill_offset]->Fill(ieta,iphi,temp_fC);
			  ADC_PedestalFromDBByDepth_1D_bycapid[capid][depth-1+fill_offset]->Fill(temp_ADC);
			  fC_PedestalFromDBByDepth_1D_bycapid[capid][depth-1+fill_offset]->Fill(temp_fC);
			}

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

		      if (makeDiagnostics)
			{
			  ADC_WidthFromDBByDepth_bycapid[capid][depth-1+fill_offset]->Fill(ieta,iphi,temp_ADC);
			  fC_WidthFromDBByDepth_bycapid[capid][depth-1+fill_offset]->Fill(ieta,iphi,temp_fC);
			  ADC_WidthFromDBByDepth_1D_bycapid[capid][depth-1+fill_offset]->Fill(temp_ADC);
			  fC_WidthFromDBByDepth_1D_bycapid[capid][depth-1+fill_offset]->Fill(temp_fC);	
			}
		    }//capid loop

		  // Pedestal values are average over four cap IDs
		  // widths are sqrt(SUM [sigma_ii^2])/4.
		  fC_ped/=4.;
		  ADC_ped/=4.;

		  ADC_1D_PedestalFromDBByDepth[depth-1+fill_offset]->Fill(ADC_ped);
		  fC_1D_PedestalFromDBByDepth[depth-1+fill_offset]->Fill(fC_ped);

		  // Divide width by 2, or by four?
		  // Dividing by 2 gives subtracted results closer to zero -- estimate of variance?
		  fC_width=pow(fC_width,0.5)/2.;
		  ADC_width=pow(ADC_width,0.5)/2.;

		  ADC_1D_WidthFromDBByDepth[depth-1+fill_offset]->Fill(ADC_width);
		  fC_1D_WidthFromDBByDepth[depth-1+fill_offset]->Fill(fC_width);

		  if (fVerbosity>1)
		    {
		      cout <<"<HcalPedestalMonitor::fillDBValues> HcalDet ID = "<<(HcalSubdetector)subdet<<": ("<<ieta<<", "<<iphi<<", "<<depth<<")"<<endl;
		      cout <<"\tADC pedestal = "<<ADC_ped<<" +/- "<<ADC_width<<endl;
		      cout <<"\tfC pedestal = "<<fC_ped<<" +/- "<<fC_width<<endl;
		    }
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
  FillUnphysicalHEHFBins(ADC_PedestalFromDBByDepth);
  FillUnphysicalHEHFBins(ADC_WidthFromDBByDepth);
  FillUnphysicalHEHFBins(fC_PedestalFromDBByDepth);
  FillUnphysicalHEHFBins(fC_WidthFromDBByDepth);

  for (unsigned int i=0;i<ADC_PedestalFromDBByDepth_bycapid.size();++i)
    {
      for (unsigned int capid=0;capid<ADC_PedestalFromDBByDepth_bycapid[i].size();++capid)
	{
	  // Why does this crash here, but not in lxplus?
	  FillUnphysicalHEHFBins(ADC_PedestalFromDBByDepth_bycapid[capid]);
	  FillUnphysicalHEHFBins(ADC_WidthFromDBByDepth_bycapid[capid]);
	  FillUnphysicalHEHFBins(fC_PedestalFromDBByDepth_bycapid[capid]);
	  FillUnphysicalHEHFBins(fC_WidthFromDBByDepth_bycapid[capid]);
	  
	}
    }
  if (showTiming)
    {
      cpu_timer.stop();  
      cout <<"TIMER:: HcalPedestalMonitor FILLDBVALUES -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalPedestalMonitor::fillDBValues(void)


// ******************************************************** //

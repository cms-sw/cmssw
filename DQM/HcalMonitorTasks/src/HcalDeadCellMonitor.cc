#include "DQM/HcalMonitorTasks/interface/HcalDeadCellMonitor.h"

#define OUT if(fverbosity_)cout

using namespace std;

HcalDeadCellMonitor::HcalDeadCellMonitor()
{
  ievt_=0;
} //constructor

HcalDeadCellMonitor::~HcalDeadCellMonitor()
{
} //destructor


/* ------------------------------------ */ 

void HcalDeadCellMonitor::setup(const edm::ParameterSet& ps,
				DQMStore* dbe)
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  if (fVerbosity>0)
    cout <<"<HcalPedestalMonitor::setup>  Setting up histograms"<<endl;

  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"DeadCellMonitor";
  
  // Dead Cell Monitor - specific cfg variables

  if (fVerbosity>1)
    cout <<"<HcalPedestalMonitor::setup>  Getting variable values from cfg files"<<endl;
  // determine whether database pedestals are in FC or ADC
  doFCpeds_ = ps.getUntrackedParameter<bool>("DeadCellMonitor_pedestalsInFC", true);

  // deadmon_makeDiagnostics_ will take on base taks value unless otherwise specified
  deadmon_makeDiagnostics_ = ps.getUntrackedParameter<bool>("DeadCellMonitor_makeDiagnosticPlots",makeDiagnostics);
  
  // Set checkNevents values
  deadmon_checkNevents_ = ps.getUntrackedParameter<int>("DeadCellMonitor_checkNevents",checkNevents_);
  deadmon_checkNevents_occupancy_ = ps.getUntrackedParameter<int>("DeadCellMonitor_checkNevents_occupancy",deadmon_checkNevents_);
  deadmon_checkNevents_pedestal_  = ps.getUntrackedParameter<int>("DeadCellMonitor_checkNevents_pedestal" ,deadmon_checkNevents_);
  deadmon_checkNevents_neighbor_  = ps.getUntrackedParameter<int>("DeadCellMonitor_checkNevents_neighbor" ,deadmon_checkNevents_);
  deadmon_checkNevents_energy_    = ps.getUntrackedParameter<int>("DeadCellMonitor_checkNevents_energy"   ,deadmon_checkNevents_);
 
  // Set which dead cell checks will be performed
  deadmon_test_occupancy_ = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_occupancy",true);
  deadmon_test_pedestal_ = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_pedestal",true);
  deadmon_test_neighbor_ = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_neighbor",true);

  deadmon_minErrorFlag_ = ps.getUntrackedParameter<double>("DeadCellMonitor_minErrorFlag",0.0);

  nsigma_ = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_Nsigma",-10);
  HBnsigma_ = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_HB_Nsigma",nsigma_);
  HEnsigma_ = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_HE_Nsigma",nsigma_);
  HOnsigma_ = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_HO_Nsigma",nsigma_);
  HFnsigma_ = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_HF_Nsigma",nsigma_);

  // Set initial event # to 0
  ievt_=0;


  // Set up histograms
  if (m_dbe)
    {
      if (fVerbosity>1)
	cout <<"<HcalPedestalMonitor::setup>  Setting up histograms"<<endl;

      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("Dead Cell Task Event Number");
      meEVT_->Fill(ievt_);

      // Create problem cell plots
      // Overall plot gets an initial " " in its name
      ProblemDeadCells=m_dbe->book2D(" ProblemDeadCells",
                                     "Problem Dead Cell Rate for all HCAL",
                                     etaBins_,etaMin_,etaMax_,
                                     phiBins_,phiMin_,phiMax_);
      ProblemDeadCells->setAxisTitle("i#eta",1);
      ProblemDeadCells->setAxisTitle("i#phi",2);
      
      
      // Overall Problem plot appears in main directory; plots by depth appear \in subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/problem_deadcells");
      setupDepthHists2D(ProblemDeadCellsByDepth, "Problem Dead Cell Rate","");

      // Set up plots for each failure mode of dead cells
      stringstream units; // We'll need to set the titles individually, rather than passing units to setupDepthHists2D (since this also would affect the name of the histograms)
      m_dbe->setCurrentFolder(baseFolder_+"/dead_unoccupied");
      //units<<"("<<deadmon_checkNevents_occupancy_<<" consec. events)";
      units.str("");
      setupDepthHists2D(UnoccupiedDeadCellsByDepth,
			"Dead Cells with No Digis",
			(char*)units.str().c_str());
      units.str("");
      
      m_dbe->setCurrentFolder(baseFolder_+"/dead_pedestaltest");
      /*
      if (nsigma_>0)
	units<<"(< pedestal + "<<nsigma_<<")";
      else
	units<<"(< pedestal - "<<nsigma_<<")";
      */
      setupDepthHists2D(BelowPedestalDeadCellsByDepth,"Dead Cells Failing Pedestal Test",
			(char*)units.str().c_str());
      units.str("");
      
      m_dbe->setCurrentFolder(baseFolder_+"/dead_neighbortest");
      //units<<"("<<deadmon_checkNevents_neighbor_<<" consec. events)"; // set title later?
      setupDepthHists2D(BelowNeighborsDeadCellsByDepth,"Dead Cells Failing Neighbor Test",
			(char*)units.str().c_str());
      units.str("");
			      
    } // if (m_dbe)

  return;
} //void HcalDeadCellMonitor::setup(...)

/* --------------------------- */

void HcalDeadCellMonitor::reset(){}  // reset function is empty for now

/* --------------------------- */

void HcalDeadCellMonitor::createMaps(const HcalDbService& cond)
{
  
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  if (fVerbosity>0)
    cout <<"<HcalDeadCellMonitor::createMaps>:  Making pedestal maps"<<endl;
  float ped=0;
  float width=0;
  HcalCalibrations calibs;
  const HcalQIEShape* shape = cond.getHcalShape();

  double myNsigma=0;

  for (int ieta=(int)etaMin_;ieta<=(int)etaMax_;++ieta)
    {
      for (int iphi=(int)phiMin_;iphi<=(int)phiMax_;++iphi)
	{
	  for (int depth=1;depth<=4;++depth)
	    {
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth))
		    continue;
		  HcalDetId hcal((HcalSubdetector)(subdet), ieta, iphi, depth);
		  
		  if (hcal.subdet()==HcalBarrel)
		    myNsigma=HBnsigma_;
		  else if (hcal.subdet()==HcalEndcap)
		    myNsigma=HEnsigma_;
		  else if (hcal.subdet()==HcalOuter)
		    myNsigma=HOnsigma_;
		  else if (hcal.subdet()==HcalForward)
		    myNsigma=HFnsigma_;
		  
		  calibs=cond.getHcalCalibrations(hcal);
		  const HcalPedestalWidth* pedw = cond.getPedestalWidth(hcal);
		   
		  ped=0.;
		  width=0.;

		  // loop over capids
		  for (int capid=0;capid<4;++capid)
		    {
		      if (doFCpeds_)
			{
			  // pedestals in fC
			  const HcalQIECoder* channelCoder=cond.getHcalCoder(hcal);

			  // Convert pedestals to ADC
			  ped+=channelCoder->adc(*shape,
						 (float)calibs.pedestal(capid),
						 capid);

			  // Okay, this definitely isn't right.  Need to figure out how to convert from fC to ADC properly
			  // Right now, take width as half the difference between (ped+width)- (ped-width), converting each to ADC

			  width+=0.5*(channelCoder->adc(*shape,
							(float)calibs.pedestal(capid)+(float)pow((double)pedw->getWidth(capid),(double)0.5),
							capid)
				      - channelCoder->adc(*shape,
							  (float)calibs.pedestal(capid)-(float)pow((double)pedw->getWidth(capid),(double)0.5),
							  capid));
			} // if (doFCpeds_) // (pedestals in fC)
		      else
			{
			  // pedestals in ADC
			  ped+=calibs.pedestal(capid);
			  width+=pedw->getWidth(capid); // add in quadrature?  Make use of correlations?
			} // else //pedestals in ADC
		    } // for (int capid=0;capid<4;++capid)

		  ped/=4.;  // pedestal value is average over capids
		  if (doFCpeds_)
		    width/=4.;
		  else
		    width=pow((double)width/4.,(double)0.5); // getWidth returns width^2

		  pedestals_[hcal]=ped;
		  widths_[hcal]=width;
		  if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::createMaps>  Pedestal Value -- ID = "<<(HcalSubdetector)subdet<<"  ("<<ieta<<", "<<iphi<<", "<<depth<<"): "<<ped<<"; width = "<<width<<endl;
		  pedestal_thresholds_[hcal]=ped+myNsigma*width;
		} // for (int subdet=1,...)
	    } // for (int depth=1;...)
	} // for (int phi ...)
    } // for (int ieta...)
  
  return;
} // void HcalDeadCellMonitor::createMaps




/* ------------------------- */

void HcalDeadCellMonitor::done()
{
  if (fVerbosity==0)
    return;

  int eta,phi;
  float binval;
  int mydepth;

  char* subdet;
  cout <<"<HcalDeadCellMonitor>  Summary of Dead Cells in Run: "<<endl;
  cout <<"(Error rate must be >= "<<deadmon_minErrorFlag_*100.<<"% )"<<endl;  

  for (int ieta=1;ieta<=etaBins_;++ieta)
    {
      for (int iphi=1;iphi<=phiBins_;++iphi)
        {
          eta=ieta+int(etaMin_)-1;
          phi=iphi+int(phiMin_)-1;

          for (int d=0;d<6;++d)
            {
	      binval=ProblemDeadCellsByDepth[d]->getBinContent(ieta,iphi);
	      if (binval<=deadmon_minErrorFlag_) continue;

	      // Set subdetector labels for output
	      if (d<2) // HB/HF
		{
		  if (abs(eta)<29)
		    subdet="HB";
		  else
		    subdet="HF";
		}
	      else if (d==3)
		{
		  if (abs(eta)==43)
		    subdet="ZDC";
		  else
		    subdet="HO";
		}
	      else
		subdet="HE";
	      // Set correct depth label
	      if (d>3)
		mydepth=d-3;
	      else
		mydepth=d+1;
	      cout <<"Dead Cell "<<subdet<<"("<<eta<<", "<<phi<<", "<<mydepth<<"):  "<<binval*100.<<"%"<<endl;
	    } // for (int d=0;d<6;++d) // loop over depth histograms
	} // for (int iphi=1;iphi<=phiBins_;++iphi)
    } // for (int ieta=1;ieta<=etaBins_;++ieta)

  return;

} // void HcalDeadCellMonitor::done()



/* --------------------------------- */

void HcalDeadCellMonitor::clearME()
{
  // I don't think this function gets cleared any more.  
  // And need to add code to clear out subfolders as well?
  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
    }
  return;
} // void HcalDeadCellMonitor::clearME()

/* -------------------------------- */


void HcalDeadCellMonitor::processEvent(const HBHERecHitCollection& hbHits,
				       const HORecHitCollection& hoHits,
				       const HFRecHitCollection& hfHits,
				       //const ZDCRecHitCollection& zdcHits,
				       const HBHEDigiCollection& hbhedigi,
				       const HODigiCollection& hodigi,
				       const HFDigiCollection& hfdigi,
				       //const ZDCDigiCollection& zdcdigi,
				       const HcalDbService& cond
				       )
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent> Processing event..."<<endl;

  if (deadmon_test_occupancy_ || deadmon_test_pedestal_)
    processEvent_digi(hbhedigi,hodigi,hfdigi,cond);
  

   if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor PROCESSEVENT -> "<<cpu_timer.cpuTime()<<endl;
    }
  return;
} // void HcalDeadCellMonitor::processEvent(...)

/* --------------------------------------- */


void HcalDeadCellMonitor::processEvent_digi( const HBHEDigiCollection& hbhedigi,
					     const HODigiCollection& hodigi,
					     const HFDigiCollection& hfdigi,
					     //const ZDCDigiCollection& zdcdigi, 
					     const HcalDbService& cond
					     )
{
  
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent_digi> Processing digi..."<<endl;

  ievt_++;

  // Variables used in pedestal check
  float digival=0;
  float maxval=0;
  int maxbin=0;
  float ADCsum=0;

  // Variables used in occupancy check
  int ieta=0;
  int iphi=0;
  int depth=0;

  HcalCalibrationWidths widths;
  HcalCalibrations calibs;
  const HcalQIEShape* shape=cond.getHcalShape();

  // Loop over HBHE digis

  if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent_digi> Processing HBHE..."<<endl;

  for (HBHEDigiCollection::const_iterator j=hbhedigi.begin();
       j!=hbhedigi.end(); ++j)
    {
      digival=0;
      maxval=0;
      maxbin=0;
      ADCsum=0;
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);

      ieta=digi.id().ieta();
      iphi=digi.id().iphi();
      depth=digi.id().depth();

      //if (deadmon_test_occupancy_) // do this for every digi?  Or just ignore occupancy array when filling histos?
      occupancy[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]++;

      if (!deadmon_test_pedestal_)
	continue;
      
      HcalDetId myid = digi.id();
      cond.makeHcalCalibrationWidth(digi.id(),&widths);
      calibs = cond.getHcalCalibrations(digi.id());

      // Find digi time slice with maximum (pedestal-subtracted) ADC count
      for (int i=0;i<digi.size();++i)
	{
	  int thisCapid = digi.sample(i).capid();
	  if (doFCpeds_)
	    {
	      const HcalQIECoder* coder  = cond.getHcalCoder(digi.id());
	      digival = coder->charge(*shape,digi.sample(i).adc(),digi.sample(i).capid())-calibs.pedestal(thisCapid);
	    }
	  else
	    digival=digi.sample(i).adc()-calibs.pedestal(thisCapid);
	  
	  // Find maximum pedestal-subtracted digi value
	  if (digival>maxval)
	    {
	      maxval=digival;
	      maxbin=i;
	    }
	} // for (int i=0;i<digi.size();++i)
      
      // We'll assume steeply-peaked distribution, so that charge deposit occurs
      // in slices (i-1) -> (i+2) around maximum deposit time i
      
      for (int i=max(0,maxbin-1);i<=min(digi.size()-1,maxbin+2);++i)
	{
	  ADCsum+=digi.sample(i).adc();

	} // for (int i=max(0,maxbin-1);...)      

      // Compare ADCsum to minimum expected value (pedestal+nsigma)

      // Search for digi in map of pedestal+threshold values
      if (pedestal_thresholds_.find(myid)!=pedestal_thresholds_.end())
	{
	  if (ADCsum < pedestal_thresholds_[myid])
	    belowpedestal[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]++;
	}
      else if (ADCsum==0)
	belowpedestal[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]++;
    } // for (HBHEDigiCollection...)

  // Loop over HO
  if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent_digi> Processing HO..."<<endl;

  for (HODigiCollection::const_iterator j=hodigi.begin();
       j!=hodigi.end(); ++j)
    {
      digival=0;
      maxval=0;
      maxbin=0;
      ADCsum=0;
      const HODataFrame digi = (const HODataFrame)(*j);

      ieta=digi.id().ieta();
      iphi=digi.id().iphi();
      depth=digi.id().depth();

      //if (deadmon_test_occupancy_) // do this for every digi?  Or just ignore occupancy array when filling histos?
      occupancy[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]++;

      if (!deadmon_test_pedestal_)
	continue;
      
      HcalDetId myid = digi.id();
      cond.makeHcalCalibrationWidth(digi.id(),&widths);
      calibs = cond.getHcalCalibrations(digi.id());

      for (int i=0;i<digi.size();++i)
	{
	  int thisCapid = digi.sample(i).capid();
	  if (doFCpeds_)
	    {
	      const HcalQIECoder* coder  = cond.getHcalCoder(digi.id());
	      digival = coder->charge(*shape,digi.sample(i).adc(),digi.sample(i).capid())-calibs.pedestal(thisCapid);
	    }
	  else
	    digival=digi.sample(i).adc()-calibs.pedestal(thisCapid);
	  
	  // Find maximum pedestal-subtracted digi value
	  if (digival>maxval)
	    {
	      maxval=digival;
	      maxbin=i;
	    }
	} // for (int i=0;i<digi.size();++i)
      
      // We'll assume steeply-peaked distribution, so that charge deposit occurs
      // in slices (i-1) -> (i+2) around maximum deposit time i
      
      for (int i=max(0,maxbin-1);i<=min(digi.size()-1,maxbin+2);++i)
	{
	  ADCsum+=digi.sample(i).adc();

	} // for (int i=max(0,maxbin-1);...)      

      // Search for digi in map of pedestal+threshold values
      if (pedestal_thresholds_.find(myid)!=pedestal_thresholds_.end())
	{
	  if (ADCsum < pedestal_thresholds_[myid])
	    belowpedestal[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]++;
	}
      else if (ADCsum==0)
	belowpedestal[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]++;

    } // for (HODigiCollection...)

  
  // Loop over HF
  if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent_digi> Processing HF..."<<endl;

  for (HFDigiCollection::const_iterator j=hfdigi.begin();
       j!=hfdigi.end(); ++j)
    {
      digival=0;
      maxval=0;
      maxbin=0;
      ADCsum=0;
      const HFDataFrame digi = (const HFDataFrame)(*j);

      ieta=digi.id().ieta();
      iphi=digi.id().iphi();
      depth=digi.id().depth()+2; // offset depth by 2 for HF

      //if (deadmon_test_occupancy_) // do this for every digi?  Or just ignore occupancy array when filling histos?
      occupancy[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]++;

      if (!deadmon_test_pedestal_)
	continue;
      
      HcalDetId myid = digi.id();
      cond.makeHcalCalibrationWidth(digi.id(),&widths);
      calibs = cond.getHcalCalibrations(digi.id());

      for (int i=0;i<digi.size();++i)
	{
	  int thisCapid = digi.sample(i).capid();
	  if (doFCpeds_)
	    {
	      const HcalQIECoder* coder  = cond.getHcalCoder(digi.id());
	      digival = coder->charge(*shape,digi.sample(i).adc(),digi.sample(i).capid())-calibs.pedestal(thisCapid);
	    }
	  else
	    digival=digi.sample(i).adc()-calibs.pedestal(thisCapid);
	  
	  // Find maximum pedestal-subtracted digi value
	  if (digival>maxval)
	    {
	      maxval=digival;
	      maxbin=i;
	    }
	} // for (int i=0;i<digi.size();++i)
      
      // We'll assume steeply-peaked distribution, so that charge deposit occurs
      // in slices (i-1) -> (i+2) around maximum deposit time i
      
      for (int i=max(0,maxbin-1);i<=min(digi.size()-1,maxbin+2);++i)
	{
	  ADCsum+=digi.sample(i).adc();

	} // for (int i=max(0,maxbin-1);...)      

      // Search for digi in map of pedestal+threshold values
      if (pedestal_thresholds_.find(myid)!=pedestal_thresholds_.end())
	{
	  if (ADCsum < pedestal_thresholds_[myid])
	    belowpedestal[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]++;
	}
      else if (ADCsum==0)
	belowpedestal[ieta+(int)((etaBins_-2)/2)][iphi-1][depth-1]++;

    } // for (HFDigiCollection...)

  // Fill histograms 
  if (ievt_%deadmon_checkNevents_occupancy_==0)
    {
    if (deadmon_test_occupancy_)
      {
	if (fVerbosity) cout <<"<HcalDeadCellMonitor::processEvent_digi> Filling DeadCell Occupancy plots"<<endl;
	fillNevents_occupancy();
      }
    }

  if (ievt_%deadmon_checkNevents_pedestal_==0)
    {
      if( deadmon_test_pedestal_)
	{
	  if (fVerbosity) cout <<"<HcalDeadCellMonitor::processEvent_digi> Filling DeadCell Pedestal plots"<<endl;
	  fillNevents_pedestal();
	}
    }

  // Fill problem cells
  if (((ievt_%deadmon_checkNevents_occupancy_==0)&&deadmon_test_occupancy_)||
      ((ievt_%deadmon_checkNevents_pedestal_ ==0)&&deadmon_test_pedestal_ )||
      ((ievt_%deadmon_checkNevents_neighbor_ ==0)&&deadmon_test_neighbor_ )||
      ((ievt_%deadmon_checkNevents_energy_   ==0)&&deadmon_test_energy_))
    {
      fillNevents_problemCells();
    }

   if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor PROCESSEVENT_DIGI -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} // void HcalDeadCellMonitor::processEvent_digi


/* ----------------------------------- */

void HcalDeadCellMonitor::fillNevents_occupancy(void)
{
  // Fill Histograms showing digi cells with no occupancy

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    cout <<"<HcalDeadCellMonitor::fillNevents_occupancy> FILLING OCCUPANCY PLOTS"<<endl;

  int mydepth=0;
  int ieta=0;
  int iphi=0;
  for (int eta=0;eta<(etaBins_-2);++eta)
    {
      ieta=eta-int((etaBins_-2)/2);
      for (int phi=0;phi<72;++phi)
        {
	  iphi=phi+1;
	  for (int depth=0;depth<4;++depth) // this is one unit less "true" depth (for indexing purposes)
            {
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1))
		    continue;
		  mydepth=depth;
		  if (subdet==4) // remember that HF's elements stored in depths (2,3), not (0,1)
		    mydepth=depth+2;
		  if (occupancy[eta][phi][mydepth]==0)
		    {
		      if (fVerbosity>0) cout <<"DEAD CELL; NO OCCUPANCY = "<<subdet<<" eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<endl;
		      if (subdet==2 && depth<2) // HE depth positions(0,1) found -- shift up to positions (4,5)
			mydepth=depth+4;
		      else
			mydepth=depth; // switches back HF to its correct depth
		      // no digi was found for the N events; set histogram error rate
		      int oldevts=(ievt_/deadmon_checkNevents_occupancy_);
		      if (ievt_%deadmon_checkNevents_occupancy_==0)
			oldevts-=1;
		      oldevts*=deadmon_checkNevents_occupancy_;
		      int newevts=ievt_-oldevts;
		      if (newevts<0) newevts=0;
		      if (fVerbosity>2)
			{
			  cout <<"\t MYDEPTH = "<<mydepth<<endl;
			  cout <<"\t oldevents = "<<oldevts<<"  new = "<<newevts<<endl;
			  cout <<"\t\t"<<(oldevts*UnoccupiedDeadCellsByDepth[mydepth]->getBinContent(eta+2,phi+2)+newevts)*1./ievt_<<endl;
			}
		      // BinContent starts at 1, not 0 (offset by 0)
		      // Offset by another 1 due to empty bins at edges
		      UnoccupiedDeadCellsByDepth[mydepth]->setBinContent( eta+2,phi+2,
									  (oldevts*UnoccupiedDeadCellsByDepth[mydepth]->getBinContent(eta+2,phi+2)+newevts)*1./ievt_);
		    }
		  else //reset counter
		    occupancy[eta][phi][depth]=0;
		} // for (int subdet=1;subdet<=4;++subdet)

	    } // for (int depth=0;depth<4;++depth)
	} // for (int phi=0;...)
    } // for (int eta=0;...)

  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_OCCUPANCY -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;


} // void HcalDeadCellMonitor::fillNevents_occupancy(void)




/* ----------------------------------- */

void HcalDeadCellMonitor::fillNevents_pedestal(void)
{
  // Fill Histograms showing digi cells below pedestal values

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    cout <<"<HcalDeadCellMonitor::fillNevents_pedestal> FILLING OCCUPANCY PLOTS"<<endl;

  int mydepth=0;
  int ieta=0;
  int iphi=0;
  for (int eta=0;eta<(etaBins_-2);++eta)
    {
      ieta=eta-int((etaBins_-2)/2);
      for (int phi=0;phi<72;++phi)
        {
	  iphi=phi+1;
	  for (int depth=0;depth<4;++depth) // this is one unit less "true" depth (for indexing purposes)
            {
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth+1))
		    continue;
		  
		  int oldevts=(ievt_/deadmon_checkNevents_pedestal_);
		  if (ievt_%deadmon_checkNevents_pedestal_==0)
		    oldevts-=1;
		  oldevts*=deadmon_checkNevents_pedestal_;
		  int newevts=ievt_-oldevts;
		  if (newevts<0) newevts=0; // shouldn't happen
		  //cout <<"GOOD  "<<subdet<<" ("<<eta<<", "<<phi<<", "<<mydepth<<") = "<<belowpedestal[eta][phi][mydepth]<<endl;
		  mydepth=depth;
		  if (subdet==4) // remember that HF's elements stored in depths (2,3), not (0,1)
		    mydepth=depth+2;

		  if (belowpedestal[eta][phi][mydepth]<(unsigned int)newevts)
		    {
		      belowpedestal[eta][phi][mydepth]=0;
		      continue; // cells must be below pedestal threshold for all the full range 'newevts' to be considered dead
		    }


		  if (fVerbosity>0) cout <<"DEAD CELL; BELOW PEDESTAL THRESHOLD = "<<subdet<<" eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<endl;
		  if (subdet==2 && depth<2) // HE depth positions(0,1) found -- shift up to positions (4,5)
		    mydepth=depth+4;
		  else
		    mydepth=depth; // switches back HF to its correct depth
		  // no digi was found for the N events; set histogram error rate
		  
		  if (fVerbosity>0)
		    {
		      cout <<"\t MYDEPTH = "<<mydepth<<endl;
		      cout <<"\t oldevents = "<<oldevts<<"  new = "<<newevts<<endl;
		      cout <<"\t\t"<<(oldevts*BelowPedestalDeadCellsByDepth[mydepth]->getBinContent(eta+2,phi+2)+newevts)*1./ievt_<<endl;
		    }
		  // BinContent starts at 1, not 0 (offset by 0)
		  // Offset by another 1 due to empty bins at edges
		  BelowPedestalDeadCellsByDepth[mydepth]->setBinContent( eta+2,phi+2,
									 (oldevts*BelowPedestalDeadCellsByDepth[mydepth]->getBinContent(eta+2,phi+2)+newevts)*1./ievt_);
		  //reset counter
		  belowpedestal[eta][phi][depth]=0;
		} // for (int subdet=1;subdet<=4;++subdet)
	      
	    } // for (int depth=0;depth<4;++depth)
	} // for (int phi=0;...)
    } // for (int eta=0;...)
  
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_BELOWPEDESTAL -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;


} // void HcalDeadCellMonitor::fillNevents_pedestal(void)


void HcalDeadCellMonitor::fillNevents_problemCells(void)
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    cout <<"<HcalDeadCellMonitor::fillNevents_problemCells> FILLING PROBLEM CELL PLOTS"<<endl;

  int ieta=0;
  int iphi=0;

  double problemvalue=0;
  double sumproblemvalue=0; // summed over all depths
  for (int eta=0;eta<(etaBins_-2);++eta)
    {
      ieta=eta-int((etaBins_-2)/2);
      for (int phi=0;phi<72;++phi)
        {
	  iphi=phi+1;
	  sumproblemvalue=0;
	  for (int mydepth=0;mydepth<6;++mydepth)
	    {
	      // total bad fraction is sum of fractions from individual tests
	      // (eventually, do we want to be more careful about how we handle this, in case checkNevents is
	      //  drastically different for the different tests?)
	      problemvalue=0;
	      if (deadmon_test_occupancy_)
		{
		  problemvalue+=UnoccupiedDeadCellsByDepth[mydepth]->getBinContent(eta+2,phi+2);
		  sumproblemvalue+=UnoccupiedDeadCellsByDepth[mydepth]->getBinContent(eta+2,phi+2);
		}
	      if (deadmon_test_pedestal_)
		{
		  problemvalue+=BelowPedestalDeadCellsByDepth[mydepth]->getBinContent(eta+2,phi+2);
		  sumproblemvalue+=BelowPedestalDeadCellsByDepth[mydepth]->getBinContent(eta+2,phi+2);
		}
	      if (deadmon_test_neighbor_)
		{
		  problemvalue+=BelowNeighborsDeadCellsByDepth[mydepth]->getBinContent(eta+2,phi+2);
		  sumproblemvalue+=BelowNeighborsDeadCellsByDepth[mydepth]->getBinContent(eta+2,phi+2);
		}
	      if (deadmon_test_energy_)
		{
		  problemvalue+=BelowEnergyThresholdCellsByDepth[mydepth]->getBinContent(eta+2,phi+2);
		  sumproblemvalue+=BelowEnergyThresholdCellsByDepth[mydepth]->getBinContent(eta+2,phi+2);
		}
	      problemvalue=min(1.,problemvalue);
	      ProblemDeadCellsByDepth[mydepth]->setBinContent(eta+2,phi+2,problemvalue);
	    } // for (int mydepth=0;mydepth<6;...)
	  sumproblemvalue=min(1.,sumproblemvalue);
	  ProblemDeadCells->setBinContent(eta+2,phi+2,sumproblemvalue);
	} // loop on phi=0;phi<72
    } // loop on eta=0; eta<(etaBins_-2)
  
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_PROBLEMCELLS -> "<<cpu_timer.cpuTime()<<endl;
    }

} // void HcalDeadCellMonitor::fillNevents_problemCells(void)

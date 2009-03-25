#include "DQM/HcalMonitorTasks/interface/HcalDeadCellMonitor.h"

#define OUT if(fverbosity_)cout
#define BITSHIFT 5

using namespace std;

HcalDeadCellMonitor::HcalDeadCellMonitor()
{
  ievt_=0;
  // Default initialization
  showTiming   = false;
  fVerbosity   = 0;
  deadmon_makeDiagnostics_ = false;
} //constructor

HcalDeadCellMonitor::~HcalDeadCellMonitor()
{
} //destructor


/* ------------------------------------ */ 

void HcalDeadCellMonitor::setup(const edm::ParameterSet& ps,
				DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  baseFolder_ = rootFolder_+"DeadCellMonitor_Hcal";
  if (fVerbosity>0)
    cout <<"<HcalDeadCellMonitor::setup>  Setting up histograms"<<endl;

  // Assume subdetectors not present until shown otherwise
  HBpresent_=false;
  HEpresent_=false;
  HOpresent_=false;
  HFpresent_=false;

  // Dead Cell Monitor - specific cfg variables

  if (fVerbosity>1)
    cout <<"<HcalDeadCellMonitor::setup>  Getting variable values from cfg files"<<endl;
  // determine whether database pedestals are in FC or ADC
  doFCpeds_ = ps.getUntrackedParameter<bool>("DeadCellMonitor_pedestalsInFC", true);

  // deadmon_makeDiagnostics_ will take on base task value unless otherwise specified
  deadmon_makeDiagnostics_ = ps.getUntrackedParameter<bool>("DeadCellMonitor_makeDiagnosticPlots",makeDiagnostics);
  
  // Set checkNevents values
  deadmon_checkNevents_ = ps.getUntrackedParameter<int>("DeadCellMonitor_checkNevents",checkNevents_);
   
  // Set which dead cell checks will be performed
  deadmon_test_occupancy_         = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_occupancy", true);
  deadmon_test_rechit_occupancy_  = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_rechit_occupancy", true);

  deadmon_test_pedestal_          = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_pedestal",  true);
  deadmon_test_neighbor_          = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_neighbor",  true);
  deadmon_test_energy_            = ps.getUntrackedParameter<bool>("DeadCellMonitor_test_energy",    true);

  deadmon_minErrorFlag_ = ps.getUntrackedParameter<double>("DeadCellMonitor_minErrorFlag",0.0);

  // pedestal test -- cell must be below pedestal+nsigma for a number of consecutive events to be considered dead
  nsigma_       = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_Nsigma",         -10);
  HBnsigma_     = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_HB_Nsigma",  nsigma_);
  HEnsigma_     = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_HE_Nsigma",  nsigma_);
  HOnsigma_     = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_HO_Nsigma",  nsigma_);
  HFnsigma_     = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_HF_Nsigma",  nsigma_);
  ZDCnsigma_    = ps.getUntrackedParameter<double>("DeadCellMonitor_pedestal_ZDC_Nsigma", nsigma_);

  // rechit energy test -- cell must be below threshold value for a number of consecutive events to be considered dead
  energyThreshold_       = ps.getUntrackedParameter<double>("DeadCellMonitor_energyThreshold",                  1);
  HBenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HB_energyThreshold",energyThreshold_);
  HEenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HE_energyThreshold",energyThreshold_);
  HOenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HO_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HF_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("DeadCellMonitor_HF_energyThreshold",            -999);
  ZDCenergyThreshold_    = ps.getUntrackedParameter<double>("DeadCellMonitor_ZDC_energyThreshold",           -999);

  // neighboring-cell tests
  defaultNeighborParams_.DeltaIphi = ps.getUntrackedParameter<int>("DeadCellMonitor_neighbor_deltaIphi", 1);
  defaultNeighborParams_.DeltaIeta = ps.getUntrackedParameter<int>("DeadCellMonitor_neighbor_deltaIeta", 1);
  defaultNeighborParams_.DeltaDepth = ps.getUntrackedParameter<int>("DeadCellMonitor_neighbor_deltaDepth", 0);
  defaultNeighborParams_.maxCellEnergy = ps.getUntrackedParameter<double>("DeadCellMonitor_neighbor_maxCellEnergy",3.);
  defaultNeighborParams_.minNeighborEnergy = ps.getUntrackedParameter<double>("DeadCellMonitor_neighbor_minNeighborEnergy",1.);
  defaultNeighborParams_.minGoodNeighborFrac = ps.getUntrackedParameter<double>("DeadCellMonitor_neighbor_minGoodNeighborFrac",0.7);
  defaultNeighborParams_.maxEnergyFrac = ps.getUntrackedParameter<double>("DeadCellMonitor_neighbor_maxEnergyFrac",0.2);
  setupNeighborParams(ps,HBNeighborParams_ ,"HB");
  setupNeighborParams(ps,HENeighborParams_ ,"HE");
  setupNeighborParams(ps,HONeighborParams_ ,"HO");
  setupNeighborParams(ps,HFNeighborParams_ ,"HF");
  setupNeighborParams(ps,ZDCNeighborParams_,"ZDC");
  HFNeighborParams_.DeltaIphi*=2; // HF cell segmentation is 10 degrees, not 5 (mostly).  Need to multiply by 2 to convert from cell range to degree format

  // Set initial event # to 0
  ievt_=0;

  // zero all counters
  for (int i=0;i<ETABINS;++i)
    {
      for (int j=0;j<PHIBINS;++j)
	{
	  for (int k=0;k<4;++k)
	    {
	      occupancy[i][j][k]=0;
	      rechit_occupancy[i][j][k]=0;
	      abovepedestal[i][j][k]=0;
	      belowneighbors[i][j][k]=0;
	      aboveenergy[i][j][k]=0;
	    }
	}
    }

  // Set up histograms
  if (m_dbe)
    {
      if (fVerbosity>1)
	cout <<"<HcalDeadCellMonitor::setup>  Setting up histograms"<<endl;

      m_dbe->setCurrentFolder(baseFolder_);
      meEVT_ = m_dbe->bookInt("Dead Cell Task Event Number");
      meEVT_->Fill(ievt_);

      // Create problem cell plots
      // Overall plot gets an initial " " in its name
      ProblemDeadCells=m_dbe->book2D(" ProblemDeadCells",
                                     " Problem Dead Cell Rate for all HCAL",
                                     etaBins_,etaMin_,etaMax_,
                                     phiBins_,phiMin_,phiMax_);
      ProblemDeadCells->setAxisTitle("i#eta",1);
      ProblemDeadCells->setAxisTitle("i#phi",2);
      // Only show problem cells that are above problem threshold
      (ProblemDeadCells->getTH2F())->SetMinimum(deadmon_minErrorFlag_);
      (ProblemDeadCells->getTH2F())->SetMaximum(1.);
      
      // Overall Problem plot appears in main directory; plots by depth appear \in subdirectory
      m_dbe->setCurrentFolder(baseFolder_+"/problem_deadcells");
      setupDepthHists2D(ProblemDeadCellsByDepth, " Problem Dead Cell Rate","");
      setMinMaxHists2D(ProblemDeadCellsByDepth,deadmon_minErrorFlag_,1.);

      // Set up plots for each failure mode of dead cells
      stringstream units; // We'll need to set the titles individually, rather than passing units to setupDepthHists2D (since this also would affect the name of the histograms)
      m_dbe->setCurrentFolder(baseFolder_+"/dead_unoccupied_digi");
      //units<<"("<<deadmon_checkNevents_<<" consec. events)";
      setupDepthHists2D(UnoccupiedDeadCellsByDepth,
			"Dead Cells with No Digis","");
      setMinMaxHists2D(UnoccupiedDeadCellsByDepth,0.,1.);

      m_dbe->setCurrentFolder(baseFolder_+"/dead_unoccupied_rechit");
      setupDepthHists2D(UnoccupiedRecHitsByDepth,
			"Dead Cells with No Rec Hits","");
      setMinMaxHists2D(UnoccupiedRecHitsByDepth,0.,1.);

      m_dbe->setCurrentFolder(baseFolder_+"/dead_pedestaltest");
      setupDepthHists2D(BelowPedestalDeadCellsByDepth,"Dead Cells Failing Pedestal Test","");
      setMinMaxHists2D(BelowPedestalDeadCellsByDepth,0.,1.);
      // set more descriptive titles for pedestal plots
      units.str("");
      units<<"Dead Cells Failing Pedestal Test Depth 1 -- HB < ped + "<<HBnsigma_<<" #sigma, HF < ped + "<<HFnsigma_<<" #sigma";
      BelowPedestalDeadCellsByDepth[0]->setTitle(units.str().c_str());
      units.str("");
      units<<"Dead Cells Failing Pedestal Test Depth 2 -- HB < ped + "<<HBnsigma_<<" #sigma, HF < ped + "<<HFnsigma_<<" #sigma";
      BelowPedestalDeadCellsByDepth[1]->setTitle(units.str().c_str());
      units.str("");
      units<<"Dead Cells Failing Pedestal Test Depth 3 -- HE < ped + "<<HEnsigma_<<" #sigma";
      BelowPedestalDeadCellsByDepth[2]->setTitle(units.str().c_str());
      units.str("");
      units<<"Dead Cells Failing Pedestal Test Depth 4 -- HO < ped + "<<HOnsigma_<<" #sigma, ZDC TBD";
      BelowPedestalDeadCellsByDepth[3]->setTitle(units.str().c_str());
      units.str("");
      units<<"Dead Cells Failing Pedestal Test Depth 1 -- HE < ped + "<<HEnsigma_<<" #sigma";
      BelowPedestalDeadCellsByDepth[4]->setTitle(units.str().c_str());
      units.str("");
      units<<"Dead Cells Failing Pedestal Test Depth 2 -- HE < ped + "<<HEnsigma_<<" #sigma";
      BelowPedestalDeadCellsByDepth[5]->setTitle(units.str().c_str());
      units.str("");

      m_dbe->setCurrentFolder(baseFolder_+"/dead_neighbortest");
      setupDepthHists2D(BelowNeighborsDeadCellsByDepth,"Dead Cells Failing Neighbor Test","");
      setMinMaxHists2D(BelowNeighborsDeadCellsByDepth,0.,1.);

      m_dbe->setCurrentFolder(baseFolder_+"/dead_energytest");
      setupDepthHists2D(BelowEnergyThresholdCellsByDepth,"Dead Cells Failing Energy Threshold Test","");
      setMinMaxHists2D(BelowEnergyThresholdCellsByDepth,0.,1.);
      // set more descriptive titles for threshold plots
      units.str("");
      units<<"Dead Cells with Consistent Low Energy Depth 1 -- HB <"<<HBenergyThreshold_<<" GeV, HF <"<<HFenergyThreshold_<<" GeV";
      BelowEnergyThresholdCellsByDepth[0]->setTitle(units.str().c_str());
      units.str("");
      units<<"Dead Cells with Consistent Low Energy Depth 2 -- HB <"<<HBenergyThreshold_<<" GeV, HF <"<<HFenergyThreshold_<<" GeV";
      BelowEnergyThresholdCellsByDepth[1]->setTitle(units.str().c_str());
      units.str("");
      units<<"Dead Cells with Consistent Low Energy Depth 3 -- HE <"<<HEenergyThreshold_<<" GeV";
      BelowEnergyThresholdCellsByDepth[2]->setTitle(units.str().c_str());
      units.str("");
      units<<"Dead Cells with Consistent Low Energy Depth 4 -- HO <"<<HOenergyThreshold_<<" GeV, ZDC TBD";
      BelowEnergyThresholdCellsByDepth[3]->setTitle(units.str().c_str());
      units.str("");
      units<<"Dead Cells with Consistent Low Energy Depth 1 -- HE <"<<HEenergyThreshold_<<" GeV";
      BelowEnergyThresholdCellsByDepth[4]->setTitle(units.str().c_str());
      units.str("");
      units<<"Dead Cells with Consistent Low Energy Depth 2 -- HE <"<<HEenergyThreshold_<<" GeV";
      BelowEnergyThresholdCellsByDepth[5]->setTitle(units.str().c_str());
      units.str("");

      if (deadmon_makeDiagnostics_)
	{
	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/pedestal");
	  d_HBnormped=m_dbe->book1D("HB_normped","HB Dead Cell pedestal diagnostic ",300,-10,20);
	  d_HEnormped=m_dbe->book1D("HE_normped","HE Dead Cell pedestal diagnostic",300,-10,20);
	  d_HOnormped=m_dbe->book1D("HO_normped","HO Dead Cell pedestal diagnostic",300,-10,20);
	  d_HFnormped=m_dbe->book1D("HF_normped","HF Dead Cell pedestal diagnostic",300,-10,20);

	  d_HBnormped->setAxisTitle("(avg ADC-pedestal)/#sigma",1);
	  d_HEnormped->setAxisTitle("(avg ADC-pedestal)/#sigma",1);
	  d_HOnormped->setAxisTitle("(avg ADC-pedestal)/#sigma",1);
	  d_HFnormped->setAxisTitle("(avg ADC-pedestal)/#sigma",1);

	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/energythreshold");
	  d_HBrechitenergy=m_dbe->book1D("HB_rechitenergy","HB rechit energy",500,-10,40);
	  d_HErechitenergy=m_dbe->book1D("HE_rechitenergy","HE rechit energy",500,-10,40);
	  d_HOrechitenergy=m_dbe->book1D("HO_rechitenergy","HO rechit energy",500,-10,40);
	  d_HFrechitenergy=m_dbe->book1D("HF_rechitenergy","HF rechit energy",500,-10,40);

	  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/neighborcells");
	  d_HBenergyVsNeighbor=m_dbe->book2D("HB_energyVsNeighbor","HB rec hit energy vs.  #Sigma Neighbors",100,0,25,100,-5,15);
	  d_HEenergyVsNeighbor=m_dbe->book2D("HE_energyVsNeighbor","HE rec hit energy vs.  #Sigma Neighbors",100,0,25,100,-5,15);
	  d_HOenergyVsNeighbor=m_dbe->book2D("HO_energyVsNeighbor","HO rec hit energy vs.  #Sigma Neighbors",100,0,25,100,-5,15);
	  d_HFenergyVsNeighbor=m_dbe->book2D("HF_energyVsNeighbor","HF rec hit energy vs.  #Sigma Neighbors",100,0,25,100,-5,15);

	} // if (deadmon_makeDiagnostics_)
    } // if (m_dbe)

  return;
} //void HcalDeadCellMonitor::setup(...)

/* --------------------------- */
void HcalDeadCellMonitor::setupNeighborParams(const edm::ParameterSet& ps,
					      neighborParams& N,
					      char* type)
{
  // sets up parameters for neighboring-cell algorithm for each subdetector
  ostringstream myname;
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_deltaIphi";
  N.DeltaIphi = ps.getUntrackedParameter<int>(myname.str().c_str(),
					      defaultNeighborParams_.DeltaIphi);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_deltaIeta";
  N.DeltaIeta = ps.getUntrackedParameter<int>(myname.str().c_str(),
					      defaultNeighborParams_.DeltaIeta);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_deltaDepth";
  N.DeltaDepth = ps.getUntrackedParameter<int>(myname.str().c_str(),
					       defaultNeighborParams_.DeltaDepth);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_maxCellEnergy";
  N.maxCellEnergy = ps.getUntrackedParameter<double>(myname.str().c_str(),
						     defaultNeighborParams_.maxCellEnergy);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_minNeighborEnergy";
  N.minNeighborEnergy = ps.getUntrackedParameter<double>(myname.str().c_str(),
							 defaultNeighborParams_.minNeighborEnergy);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_minGoodNeighborFrac";
  N.minGoodNeighborFrac = ps.getUntrackedParameter<double>(myname.str().c_str(),
							   defaultNeighborParams_.minGoodNeighborFrac);
  myname.str("");
  myname<<"DeadCellMonitor_"<<type<<"_neighbor_maxEnergyFrac";
  N.maxEnergyFrac = ps.getUntrackedParameter<double>(myname.str().c_str(),
						     defaultNeighborParams_.maxEnergyFrac);
  return;
} // void HcalDeadCellMonitor::setupNeighborParams

/* --------------------------- */

void HcalDeadCellMonitor::reset(){}  // reset function is empty for now

/* --------------------------- */

void HcalDeadCellMonitor::createMaps(const HcalDbService& cond)
{

  // Creates maps for pedestals, widths, and pedestals+Nsigma*widths, using HcalDetIds as keys
  
  if (!deadmon_test_pedestal_) return; // no need to create maps if we're not running the pedestal-based dead cell finder

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  if (fVerbosity>0)
    cout <<"<HcalDeadCellMonitor::createMaps>:  Making pedestal maps"<<endl;
  double ped=0;
  double width=0;
  HcalCalibrations calibs;
  const HcalQIEShape* shape = cond.getHcalShape();

  double myNsigma=0;
  double myADC=0;

  for (int ieta=static_cast<int>(etaMin_);ieta<=static_cast<int>(etaMax_);++ieta)
    {
      for (int iphi=static_cast<int>(phiMin_);iphi<=static_cast<int>(phiMax_);++iphi)
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
		  myADC=0.;
		  // loop over capids
		  for (int capid=0;capid<4;++capid)
		    {
		      // pedestals from calibs.pedestal are always in fC
		      const HcalQIECoder* channelCoder=cond.getHcalCoder(hcal);
		      
		      // Convert pedestals to ADC
		      myADC=channelCoder->adc(*shape,
					      (float)calibs.pedestal(capid),
					      capid);
		      ped+=myADC;

		      // Now the tricky part -- need to convert widths to ADC if provided in fC
		      if (doFCpeds_)
			{
			  // Form width by summing the diagonal terms of the covariance matrix (sigma_ii),
			  // and scale by ADC/fC ratio to convert from fC^2 to ADC^2
			  width+=(pedw->getSigma(capid,capid)*pow(myADC/calibs.pedestal(capid),2));
			}
		      else
			width+=pedw->getSigma(capid,capid);
		    } // for (int capid=0;capid<4;++capid)

		  ped/=4.;  // pedestal value is average over capids
		  width=pow(width,0.5)/4.;

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

void HcalDeadCellMonitor::done(std::map<HcalDetId, unsigned int>& myqual)
{
  if (dump2database==0) 
    return;

  // Dump to ascii file for database -- now taken care of through ChannelStatus objects
  /*
  char buffer [1024];
  
  ofstream fOutput("hcalDeadCells.txt", ios::out);
  sprintf (buffer, "# %15s %15s %15s %15s %8s %10s\n", "eta", "phi", "dep", "det", "value", "DetId");
  fOutput << buffer;
  */

  int eta,phi;
  float binval;
  int mydepth;

  int subdet;
  char* subdetname;
  if (fVerbosity>1)
    {
      cout <<"<HcalDeadCellMonitor>  Summary of Dead Cells in Run: "<<endl;
      cout <<"(Error rate must be >= "<<deadmon_minErrorFlag_*100.<<"% )"<<endl;  
    }
  for (int ieta=1;ieta<=etaBins_;++ieta)
    {
      for (int iphi=1;iphi<=phiBins_;++iphi)
        {
          eta=ieta+int(etaMin_)-1;
          phi=iphi+int(phiMin_)-1;

          for (int d=0;d<6;++d)
            {
	      binval=ProblemDeadCellsByDepth[d]->getBinContent(ieta,iphi);
	     
	      // Set subdetector labels for output
	      if (d<2) // HB/HF
		{
		  if (abs(eta)<29)
		    {
		      subdetname="HB";
		      subdet=1;
		    }
		  else
		    {
		      subdetname="HF";
		      subdet=4;
		    }
		}
	      else if (d==3)
		{
		  if (abs(eta)==43)
		    {
		      subdetname="ZDC";
		      subdet=7; // correct value??
		    }
		  else
		    {
		      subdetname="HO";
		      subdet=3;
		    }
		}
	      else
		{
		  subdetname="HE";
		  subdet=2;
		}
	      // Set correct depth label
	      if (d>3)
		mydepth=d-3;
	      else
		mydepth=d+1;
	      HcalDetId myid((HcalSubdetector)(subdet), eta, phi, mydepth);
	      if (!validDetId((HcalSubdetector)(subdet), eta, phi, mydepth))
		continue;

	      if (fVerbosity>0 && binval>deadmon_minErrorFlag_)
		cout <<"Dead Cell "<<subdet<<"("<<eta<<", "<<phi<<", "<<mydepth<<"):  "<<binval*100.<<"%"<<endl;
	      int value = 0;
	      if (binval>deadmon_minErrorFlag_)
		value=1;
	      
	      if (value==1)
	      if (myqual.find(myid)==myqual.end())
		{
		  myqual[myid]=(value<<BITSHIFT);  // deadcell shifted to bit 5
		}
	      else
		{
		  int mask=(1<<BITSHIFT);
		  if (value==1)
		    myqual[myid] |=mask;

		  else
		    myqual[myid] &=~mask;
		  if (value==1 && fVerbosity>1) cout <<"myqual = "<<std::hex<<myqual[myid]<<std::dec<<"  MASK = "<<std::hex<<mask<<std::dec<<endl;
		}
	      /*
	      sprintf(buffer, "  %15i %15i %15i %15s %8X %10X \n",eta,phi,mydepth,subdetname,(value<<BITSHIFT),int(myid.rawId()));
	      fOutput<<buffer;
	      */
	    } // for (int d=0;d<6;++d) // loop over depth histograms
	} // for (int iphi=1;iphi<=phiBins_;++iphi)
    } // for (int ieta=1;ieta<=etaBins_;++ieta)
  //fOutput.close();

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

  ++ievt_;
  if (m_dbe) meEVT_->Fill(ievt_);
  
  HOpresent_ = (hodigi.size()>0||hoHits.size()>0);
  HFpresent_ = (hfdigi.size()>0||hfHits.size()>0);
  if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent> Processing event..."<<endl;

  // Do Digi-Based dead cell searches 

  if (deadmon_test_occupancy_ || deadmon_test_pedestal_)
    processEvent_digi(hbhedigi,hodigi,hfdigi,cond);

  // Search for "dead" cells below a certain energy
  if (deadmon_test_energy_ || deadmon_test_rechit_occupancy_)
    {
      processEvent_rechitenergy(hbHits, hoHits,hfHits);
    }

  // Search for cells that are "dead" compared to their neighbors
  if (deadmon_test_neighbor_)
    {
      processEvent_rechitneighbors(hbHits, hoHits, hfHits);
    }
  
  // Fill problem cells
  if (((ievt_%deadmon_checkNevents_ ==0) && deadmon_test_occupancy_ )||
      ((ievt_%deadmon_checkNevents_  ==0) && deadmon_test_pedestal_  )||
      ((ievt_%deadmon_checkNevents_  ==0) && deadmon_test_neighbor_  )||
      ((ievt_%deadmon_checkNevents_    ==0) && deadmon_test_energy_    ))
    {
      fillNevents_problemCells();
    }

  return;
} // void HcalDeadCellMonitor::processEvent(...)

/* --------------------------------------- */

void HcalDeadCellMonitor::fillDeadHistosAtEndRun()
{
  // Fill histograms one last time at endRun call
  
  /*
    I'm not sure I like this feature.  Suppose checkNevents=500, and the end run occurs at 501?
    Then the occupancy plot would create errors for whichever digis were not found in a single event.
    That's not desired behavior.
    We could just exclude the occupancy test from running here, but I'm not sure that's the best solution either.
    For now (28 Oct. 2008), just disable this functionality.  We'll come back to it if necessary.
  */
  return;

  if (deadmon_test_occupancy_ && ievt_%deadmon_checkNevents_>0) fillNevents_occupancy();
  if (deadmon_test_pedestal_  && ievt_%deadmon_checkNevents_ >0) fillNevents_pedestal();
  if (deadmon_test_neighbor_  && ievt_%deadmon_checkNevents_ >0) fillNevents_neighbor();
  if (deadmon_test_energy_    && ievt_%deadmon_checkNevents_   >0) fillNevents_energy();
  if (deadmon_test_occupancy_ || deadmon_test_pedestal_ || 
      deadmon_test_neighbor_  || deadmon_test_energy_)  
   {
     fillNevents_problemCells();
     FillUnphysicalHEHFBins(ProblemDeadCellsByDepth);
   }
}

/* --------------------------------------- */


void HcalDeadCellMonitor::processEvent_rechitenergy( const HBHERecHitCollection& hbheHits,
						     const HORecHitCollection& hoHits,
						     const HFRecHitCollection& hfHits)
						
{
  // Looks at rechits of cells and compares to threshold energies.
  // Cells below thresholds get marked as dead candidates

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

 if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent_rechitenergy> Processing rechits..."<<endl;
 if (deadmon_test_neighbor_)   rechitEnergies_.clear();

 // loop over HBHE
 for (HBHERecHitCollection::const_iterator HBHEiter=hbheHits.begin(); HBHEiter!=hbheHits.end(); ++HBHEiter) 
   { // loop over all hits
     float en = HBHEiter->energy();
     //float ti = HBHEiter->time();

     HcalDetId id(HBHEiter->detid().rawId());
     int ieta = id.ieta();
     int iphi = id.iphi();
     int depth = id.depth();
     if (id.subdet()==HcalBarrel)
       {
	 HBpresent_=true;
	 if (!checkHB_) continue;
	 if (deadmon_makeDiagnostics_) d_HBrechitenergy->Fill(en);
	 ++rechit_occupancy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
	 if (en>=HBenergyThreshold_)
	   ++aboveenergy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
       }
     else //if (id.subdet()==HcalEndcap)
       {
	 HEpresent_=true;
	 if (!checkHE_) continue;
	 if (depth<3) depth=depth+4; // shift HE depths 1 & 2 up by 4
	 if (deadmon_makeDiagnostics_) d_HErechitenergy->Fill(en);
	 ++rechit_occupancy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
	 if (en>=HEenergyThreshold_)
	   ++aboveenergy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
       }
     if (deadmon_test_neighbor_) rechitEnergies_[id]=en;
   } //for (HBHERecHitCollection::const_iterator HBHEiter=...)

 // loop over HO
 if (checkHO_)
   {
     for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) 
       { // loop over all hits
	 float en = HOiter->energy();
	 
	 HcalDetId id(HOiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();
	 ++rechit_occupancy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
	 if (deadmon_makeDiagnostics_) d_HOrechitenergy->Fill(en);
	 if (en>=HOenergyThreshold_)
	   ++aboveenergy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
	 if (deadmon_test_neighbor_) rechitEnergies_[id]=en;
       }
   } // if (checkHO_)
 
 // loop over HF
 if (checkHF_)
   {
     for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter) 
       { // loop over all hits
	 float en = HFiter->energy();
	 
	 HcalDetId id(HFiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();
	 if (deadmon_makeDiagnostics_) d_HFrechitenergy->Fill(en);
	 ++rechit_occupancy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
	 if (en>=HFenergyThreshold_)
	   ++aboveenergy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1]; 
	 if (deadmon_test_neighbor_) rechitEnergies_[id]=en;
       }
   } // if (checkHF_)
 
 
 // Fill histograms 
  if (ievt_%deadmon_checkNevents_==0)
    {
	if (fVerbosity>0) cout <<"<HcalDeadCellMonitor::processEvent_rechitenergy> Filling DeadCell Energy plots"<<endl;
	fillNevents_energy();
    }

 if (showTiming)
   {
     cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor PROCESSEVENT_RECHITENERGY -> "<<cpu_timer.cpuTime()<<endl;
   }
 return;
} // void HcalDeadCellMonitor::processEvent_rechitenergy

/* --------------------------------------- */


void HcalDeadCellMonitor::processEvent_rechitneighbors( const HBHERecHitCollection& hbheHits,
							const HORecHitCollection& hoHits,
							const HFRecHitCollection& hfHits
							)
{
  // Compares energy to energy of neighboring cells.
  // Perhaps promising, but energies tend to be centered around 0 (positive AND negative)
  // negative-energy rechits make this method pretty useless.  Keep it disabled for now.

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

 if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent_rechitneighbors> Processing rechits..."<<endl;

 // if Energy test wasn't run, need to create map of Detid:rechitenergy here
 if (!deadmon_test_energy_)
   {
     rechitEnergies_.clear(); // clear old map
     for (HBHERecHitCollection::const_iterator HBHEiter=hbheHits.begin(); HBHEiter!=hbheHits.end(); ++HBHEiter) 
       { // loop over all hits
	 float en = HBHEiter->energy();
	 HcalDetId id(HBHEiter->detid().rawId());
	 if (id.subdet()==HcalBarrel)
	   {
	     HBpresent_=true;
	     if (!checkHB_)
	       continue;
	   }
	 else
	   {
	     HEpresent_=true;
	     if (!checkHE_)
	       continue;
	   }
	 rechitEnergies_[id]=en;
       }
     // HO
     if (checkHO_)
       {
	 for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) 
	   { // loop over all hits
	     float en = HOiter->energy();
	     HcalDetId id(HOiter->detid().rawId());
	     rechitEnergies_[id]=en;
	   }
       } // if (checkHO_)
     //HF
     if (checkHF_)
       {
	 for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter) 
	   { // loop over all hits
	     float en = HFiter->energy();
	     HcalDetId id(HFiter->detid().rawId());
	     rechitEnergies_[id]=en;
	   }
       } // if (checkHF_)
     
   } // if (!deadmon_test_energy_)   
 
 //Now do "real" loop, checking against each cell against its neighbors
 
 /* Note:  This works a little differently than the other tests.  The other tests check that a cell consistently
    fails its test condition for N consecutive events.  The neighbor test will flag a cell for every event in which
    it's significantly less than its neighbors, regardless of whether that condition persists for a number of events.
 */
 
 int ieta, iphi, depth;
 float en;

 int cellsfound=0;
 int allneighbors=0;
 float enNeighbor=0;

 // loop over HBHE
 for (HBHERecHitCollection::const_iterator HBHEiter=hbheHits.begin(); 
      HBHEiter!=hbheHits.end(); 
      ++HBHEiter) 
   { // loop over all hits
     
     en = HBHEiter->energy();
     HcalDetId id(HBHEiter->detid().rawId());
     ieta = id.ieta();
     iphi = id.iphi();
     depth = id.depth();

     if (id.subdet()==HcalBarrel)
       {
	 HBpresent_=true;
	 if (!checkHB_) continue;
	 // Search keys for neighboring cells
	 if (en>HBNeighborParams_.maxCellEnergy) // cells above maxCellEnergy not considered dead
	   continue;
	 allneighbors=0;
	 cellsfound=0;
	 enNeighbor=0;
	 for (int nD=-1*HBNeighborParams_.DeltaDepth;nD<=HBNeighborParams_.DeltaDepth;++nD)
	   {
	     for (int nP =-1*HBNeighborParams_.DeltaIphi;nP<=HBNeighborParams_.DeltaIphi;++nP)
	       {
		 for (int nE =-1*HBNeighborParams_.DeltaIeta;nE<=HBNeighborParams_.DeltaIeta;++nE)
		   {
		     if (nD==0 && nE==0 && nP==0) 
		       continue; // don't count the cell itself
		     int myphi=nP+iphi;
		     if (myphi>72) myphi-=72; // allow for wrapping of cells
		     if (myphi<=0) myphi+=72;
		     if (!validDetId((HcalSubdetector)(1),nE+ieta, myphi, nD+depth)) continue;
		     HcalDetId myid((HcalSubdetector)(1), nE+ieta, myphi, nD+depth); // HB
		     ++allneighbors;
		     if (rechitEnergies_.find(myid)==rechitEnergies_.end())
		       continue;
		     if (rechitEnergies_[myid]<HBNeighborParams_.minNeighborEnergy)
		       continue;
		     ++cellsfound;
		     enNeighbor+=rechitEnergies_[myid];
		   } // loop over nE (neighbor eta)
	       } // loop over nP (neighbor phi)
	   } // loop over nD depths

	 if (deadmon_makeDiagnostics_)
	   d_HBenergyVsNeighbor->Fill(enNeighbor,en);
	 
	 // Case 1:  Not enough good neighbors found
	 if (1.*cellsfound/allneighbors<HBNeighborParams_.minGoodNeighborFrac)
	   continue;
	 // Case 2:  energy/(avg. neighbor energy) too large for cell to be considered dead
	 if (1.*en/(enNeighbor/allneighbors)>HENeighborParams_.maxEnergyFrac)
	   continue;
	 // Case 3:  Tests passed; cell marked as dead
	 ++belowneighbors[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
       }
     else //if (id.subdet()==HcalEndcap)
       {
	 HEpresent_=true;
	 if (!checkHE_) continue;
	 // Search keys for neighboring cells
	 if (en>HENeighborParams_.maxCellEnergy) // cells above maxCellEnergy not considered dead
	   continue;
	 allneighbors=0;
	 cellsfound=0;
	 enNeighbor=0;
	 int HEDeltaIphi = HENeighborParams_.DeltaIphi;
	 // now correct for boundaries
	 if (abs(ieta)>20) HEDeltaIphi*=2; // double iphi boundary range when segmentation switches to 10 degrees
	 // This still needs to be worked on to properly deal with boundaries
	 for (int nD=-1*HENeighborParams_.DeltaDepth;nD<=HENeighborParams_.DeltaDepth;++nD)
	   {
	     for (int nP =-1*HEDeltaIphi;nP<=HEDeltaIphi;++nP)
	       {
		 for (int nE =-1*HENeighborParams_.DeltaIeta;nE<=HENeighborParams_.DeltaIeta;++nE)
		   {
		     if (nD==0 && nE==0 && nP==0) 
		       continue; // don't count the cell itself
		     
		     int myphi=nP+iphi;
                     if (myphi>72) myphi-=72; // allow for wrapping of cells
		     if (myphi<=0) myphi+=72;
		     if (!validDetId((HcalSubdetector)(2),nE+ieta, myphi, nD+depth)) continue;
                     HcalDetId myid((HcalSubdetector)(2), nE+ieta, myphi, nD+depth); // HE
		     ++allneighbors;
		     if (rechitEnergies_.find(myid)==rechitEnergies_.end())
		       continue;
		     if (rechitEnergies_[myid]<HENeighborParams_.minNeighborEnergy)
		       continue;
		     ++cellsfound;
		     enNeighbor+=rechitEnergies_[myid];
		   } // loop over nE (neighbor eta)
	       } // loop over nP (neighbor phi)
	   } // loop over nD depths

	 if (deadmon_makeDiagnostics_)
	   d_HEenergyVsNeighbor->Fill(enNeighbor,en);
	 
	 // Case 1:  Not enough good neighbors found
	 if (1.*cellsfound/allneighbors<HENeighborParams_.minGoodNeighborFrac)
	   continue;
	 // Case 2:  energy/(avg. neighbor energy) too large for cell to be considered dead
	 if (1.*en/(enNeighbor/allneighbors)>HENeighborParams_.maxEnergyFrac)
	   continue;
	 // Case 3:  Tests passed; cell marked as dead
	 if (depth<3) depth+=4; // shift HE depths 1 & 2 up by 4
	 ++belowneighbors[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
       }
   } //for (HBHERecHitCollection::const_iterator HBHEiter=...)

 // loop over HO
 if (checkHO_)
   {
     for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) 
       { // loop over all hits
	 float en = HOiter->energy();
	 HcalDetId id(HOiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();

	 // Search keys for neighboring cells
	 if (en>HONeighborParams_.maxCellEnergy) // cells above maxCellEnergy not considered dead
	   continue;
	 allneighbors=0;
	 cellsfound=0;
	 enNeighbor=0;
	 for (int nD=-1*HONeighborParams_.DeltaDepth;nD<=HONeighborParams_.DeltaDepth;++nD)
	   {
	     for (int nP =-1*HONeighborParams_.DeltaIphi;nP<=HONeighborParams_.DeltaIphi;++nP)
	       {
		 for (int nE =-1*HONeighborParams_.DeltaIeta;nE<=HONeighborParams_.DeltaIeta;++nE)
		   {
		     if (nD==0 && nE==0 && nP==0) 
		       continue; // don't count the cell itself
		     int myphi=nP+iphi;
		     if (myphi>72) myphi-=72; // allow for wrapping of cells
		     if (myphi<=0) myphi+=72;
                     if (!validDetId((HcalSubdetector)(3),nE+ieta, myphi, nD+depth)) continue;
                     HcalDetId myid((HcalSubdetector)(3), nE+ieta, myphi, nD+depth); // HO
		     ++allneighbors;
		     if (rechitEnergies_.find(myid)==rechitEnergies_.end())
		       continue;
		     if (rechitEnergies_[myid]<HONeighborParams_.minNeighborEnergy)
		       continue;
		     ++cellsfound;
		     enNeighbor+=rechitEnergies_[myid];
		   } // loop over nE (neighbor eta)
	       } // loop over nP (neighbor phi)
	   } // loop over nD depths

	 if (deadmon_makeDiagnostics_)
	   d_HOenergyVsNeighbor->Fill(enNeighbor,en);
	 
	 // Case 1:  Not enough good neighbors found
	 if (1.*cellsfound/allneighbors<HONeighborParams_.minGoodNeighborFrac)
	   continue;
	 // Case 2:  energy/(avg. neighbor energy) too large for cell to be considered dead
	 if (1.*en/(enNeighbor/allneighbors)>HONeighborParams_.maxEnergyFrac)
	   continue;
	 // Case 3:  Tests passed; cell marked as dead
	 ++belowneighbors[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
       }
   } // if (checkHO_)
 
 // loop over HF
 if (checkHF_)
   {
     for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter) 
       { // loop over all hits
	 float en = HFiter->energy();
	 HcalDetId id(HFiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();

	  // Search keys for neighboring cells
	 if (en>HFNeighborParams_.maxCellEnergy) // cells above maxCellEnergy not considered dead
	   continue;
	 allneighbors=0;
	 cellsfound=0;
	 enNeighbor=0;
	 int HFDeltaIphi = HFNeighborParams_.DeltaIphi;
	 if (abs(ieta)>39) HFDeltaIphi*=2;  // double phi range when segmentation switches to 20 degrees
	 // Still need to create a more robust handling of boundary cases
	 for (int nD=-1*HFNeighborParams_.DeltaDepth;nD<=HFNeighborParams_.DeltaDepth;++nD)
	   {
	     for (int nP =-1*HFDeltaIphi;nP<=HFDeltaIphi;++nP)
	       {
		 for (int nE =-1*HFNeighborParams_.DeltaIeta;nE<=HFNeighborParams_.DeltaIeta;++nE)
		   {
		     if (nD==0 && nE==0 && nP==0) 
		       continue; // don't count the cell itself
		     int myphi=nP+iphi;
                     if (myphi>72) myphi-=72; // allow for wrapping of cells
                     if (myphi<=0) myphi+=72;
		     if (!validDetId((HcalSubdetector)(4),nE+ieta, myphi, nD+depth)) continue;
		     HcalDetId myid((HcalSubdetector)(4), nE+ieta, myphi, nD+depth); // HF
		     
		     ++allneighbors;
		     if (rechitEnergies_.find(myid)==rechitEnergies_.end())
		       continue;
		     if (rechitEnergies_[myid]<HFNeighborParams_.minNeighborEnergy)
		       continue;
		     ++cellsfound;
		     enNeighbor+=rechitEnergies_[myid];
		   } // loop over nE (neighbor eta)
	       } // loop over nP (neighbor phi)
	   } // loop over nD depths

	 if (deadmon_makeDiagnostics_)
	   d_HFenergyVsNeighbor->Fill(enNeighbor,en);
	 
	 // Case 1:  Not enough good neighbors found
	 if (1.*cellsfound/allneighbors<HFNeighborParams_.minGoodNeighborFrac)
	   continue;
	 // Case 2:  energy/(avg. neighbor energy) too large for cell to be considered dead
	 if (1.*en/(enNeighbor/allneighbors)>HFNeighborParams_.maxEnergyFrac)
	   continue;
	 // Case 3:  Tests passed; cell marked as dead
	 ++belowneighbors[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
       }
   } // if (checkHF_)
 
 
 // Fill histograms 
  if (ievt_%deadmon_checkNevents_==0)
    {
	if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent_rechitneighbor> Filling DeadCell Neighbor plots"<<endl;
	fillNevents_neighbor();
    }

 if (showTiming)
   {
     cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor PROCESSEVENT_RECHITNEIGHBOR -> "<<cpu_timer.cpuTime()<<endl;
   }
 return;
} // void HcalDeadCellMonitor::processEvent_rechitneighbor


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

  if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent_digi> Processing digis..."<<endl;

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
      if ((HcalSubdetector)(digi.id().subdet())==HcalBarrel)
	{
	  HBpresent_=true;
	  if (!checkHB_)
	    continue;
	}
      else 
	{
	  HEpresent_=true;
	  if (!checkHE_)
	    continue;
	  if (depth<3) depth+=4; // shift HE depths 1&2 up by 4
	}

      
      ++occupancy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];

      if (!deadmon_test_pedestal_)
	continue;
      
      HcalDetId myid = digi.id();
      cond.makeHcalCalibrationWidth(digi.id(),&widths);
      //const HcalCalibrationWidths widths = cond.getHcalCalibrationWidths(digi.id() );

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
      
      int bins=0;
      for (int i=max(0,maxbin-1);i<=min(digi.size()-1,maxbin+2);++i)
	{
	  ADCsum+=digi.sample(i).adc();
	  ++bins;
	} // for (int i=max(0,maxbin-1);...)      

      // Compare ADCsum to minimum expected value (pedestal+nsigma)
      // we want to compare the average over the sum to the average (ped+nsigma)
      ADCsum*=1./bins;

      // Search for digi in map of pedestal+threshold values
      if (pedestal_thresholds_.find(myid)!=pedestal_thresholds_.end())
	{
	  if (ADCsum >= pedestal_thresholds_[myid])
	    ++abovepedestal[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
	  if (deadmon_makeDiagnostics_)
	    {
	      if (widths_[myid]==0) continue;

	      if (myid.subdet()==HcalBarrel)
		d_HBnormped->Fill(1.*(ADCsum-pedestals_[myid])/widths_[myid]);
	      else
		d_HEnormped->Fill(1.*(ADCsum-pedestals_[myid])/widths_[myid]);
	    } // if (deadmon_makeDiagnostics)
	}
      else if (ADCsum>0) // if pedestal can't be found, just make sure ADC counts are non-zero
	++abovepedestal[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
    } // for (HBHEDigiCollection...)

  // Loop over HO
  if (checkHO_)
    {
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
	  ++occupancy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
	  
	  if (!deadmon_test_pedestal_)
	    continue;
	  
	  HcalDetId myid = digi.id();
	  //const HcalCalibrationWidths widths = cond.getHcalCalibrationWidths(digi.id() );
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
      
	  int bins=0;
	  for (int i=max(0,maxbin-1);i<=min(digi.size()-1,maxbin+2);++i)
	    {
	      ADCsum+=digi.sample(i).adc();
	      ++bins;
	    } // for (int i=max(0,maxbin-1);...)      
	  
	  ADCsum*=1./bins;
	  // Search for digi in map of pedestal+threshold values
	  if (pedestal_thresholds_.find(myid)!=pedestal_thresholds_.end())
	    {
	      if (ADCsum >= pedestal_thresholds_[myid])
		++abovepedestal[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
	      if (deadmon_makeDiagnostics_)
		{
		  if (widths_[myid]==0) continue;
		  d_HOnormped->Fill(1.*(ADCsum-pedestals_[myid])/widths_[myid]);
		} // if (deadmon_makeDiagnostics)
	    }
	  else if (ADCsum>0)
	    ++abovepedestal[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];

	} // for (HODigiCollection...)
    } // if (checkHO_)

  if (checkHF_)
    {
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
	  depth=digi.id().depth();

	  //if (deadmon_test_occupancy_) // do this for every digi?  Or just ignore occupancy array when filling histos?
	  ++occupancy[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];

	  if (!deadmon_test_pedestal_)
	    continue;
      
	  HcalDetId myid = digi.id();
	  //const HcalCalibrationWidths widths = cond.getHcalCalibrationWidths(digi.id() );
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
      
	  int bins=0;
	  for (int i=max(0,maxbin-1);i<=min(digi.size()-1,maxbin+2);++i)
	    {
	      ADCsum+=digi.sample(i).adc();
	      ++bins;
	    } // for (int i=max(0,maxbin-1);...)      

	  ADCsum*=1./bins;
	  // Search for digi in map of pedestal+threshold values
	  if (pedestal_thresholds_.find(myid)!=pedestal_thresholds_.end())
	    {
	      if (ADCsum >= pedestal_thresholds_[myid])
		++abovepedestal[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];
	      if (deadmon_makeDiagnostics_)
		{
		  if (widths_[myid]==0) continue;
		  d_HFnormped->Fill(1.*(ADCsum-pedestals_[myid])/widths_[myid]);
		} // if (deadmon_makeDiagnostics)
	    }
	  else if (ADCsum>0)
	    ++abovepedestal[static_cast<int>(ieta+(etaBins_-2)/2)][iphi-1][depth-1];

	} // for (HFDigiCollection...)
    } // if (checkHF_)

  // Fill histograms 
  if (ievt_%deadmon_checkNevents_==0)
    {
    if (deadmon_test_occupancy_)
      {
	if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent_digi> Filling DeadCell Occupancy plots"<<endl;
	fillNevents_occupancy();
      }
    }

  if (ievt_%deadmon_checkNevents_==0)
    {
      if( deadmon_test_pedestal_)
	{
	  if (fVerbosity>1) cout <<"<HcalDeadCellMonitor::processEvent_digi> Filling DeadCell Pedestal plots"<<endl;
	  fillNevents_pedestal();
	}
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
	  for (int depth=1;depth<=4;++depth) 
            {
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth))
		    continue;
		  // Ignore subdetectors that weren't in run
		  if ((subdet==1 && !HBpresent_) || (subdet==2 &&!HEpresent_)||(subdet==3 &&!HOpresent_) || (subdet==4 &&!HFpresent_)) continue;
		  // ignore subdetectors we explicitly mask off 
		  if ((!checkHB_ && subdet==1) ||
		      (!checkHE_ && subdet==2) ||
		      (!checkHO_ && subdet==3) ||
		      (!checkHF_ && subdet==4)) continue;
		  mydepth=depth-1 ; // my depth starts at 0, not 1
		  if (subdet==2 && depth<3) // remember that HE depths 1 and 2 are shifter up by 4 in occupancy array
		    mydepth=mydepth+4;
		  if (occupancy[eta][phi][mydepth]==0)
		    {
		      if (fVerbosity>0) cout <<"DEAD CELL; NO OCCUPANCY: subdet = "<<subdet<<", eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<endl;
		      // no digi was found for the N events; Fill cell as bad for all N events (N = deadmon_checkNevents_);
		      UnoccupiedDeadCellsByDepth[mydepth]->Fill(ieta,iphi,deadmon_checkNevents_);
		    }
		  else //reset counter
		    occupancy[eta][phi][depth]=0;
		} // for (int subdet=1;subdet<=4;++subdet)

	    } // for (int depth=1;depth<=4;++depth)
	} // for (int phi=0;...)
    } // for (int eta=0;...)
  FillUnphysicalHEHFBins(UnoccupiedDeadCellsByDepth);
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
	  for (int depth=1;depth<=4;++depth) 
            {
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth))
		    continue;
		  // Ignore subdetectors that weren't in run
                  if ((subdet==1 && !HBpresent_) || (subdet==2 &&!HEpresent_)||(subdet==3 &&!HOpresent_) || (subdet==4 &&!HFpresent_)) continue;
		  
		  if ((!checkHB_ && subdet==1) ||
		      (!checkHE_ && subdet==2) ||
		      (!checkHO_ && subdet==3) ||
		      (!checkHF_ && subdet==4)) continue;
		  
		  if (occupancy[eta][phi][mydepth]==0)
		    {
		      abovepedestal[eta][phi][mydepth]=0; // reset counter (shouldn't be necessary)
		      continue; // no cell found; it gets flagged as bad in occupancy tests, so don't check it here as well
		    }

		  mydepth=depth-1; // my depth starts at 0, not 1
		  if (subdet==2) // remember that HE depths 1 & 2 are shifter up by 4
		    mydepth=mydepth+4;

		  // Now that we have a valid cell, check whether it was ever above the pedestal threshold
		  if (abovepedestal[eta][phi][mydepth]>0)
		    {
		      abovepedestal[eta][phi][mydepth]=0;
		      continue; // cell was above pedestal threshold at least once; ignore it
		    }


		  if (fVerbosity>0) cout <<"DEAD CELL; BELOW PEDESTAL THRESHOLD = "<<subdet<<" eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<endl;

		  // no digi was found for the N events; fill as bad for all N events
		  BelowPedestalDeadCellsByDepth[mydepth]->Fill(ieta,iphi,deadmon_checkNevents_);
		  //reset counter -- shouldn't be necessary (counter should already be 0).
		  abovepedestal[eta][phi][depth]=0;
		} // for (int subdet=1;subdet<=4;++subdet)
	      
	    } // for (int depth=1;depth<=4;++depth)
	} // for (int phi=0;...)
    } // for (int eta=0;...)
  FillUnphysicalHEHFBins(BelowPedestalDeadCellsByDepth);
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_BELOWPEDESTAL -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;


} // void HcalDeadCellMonitor::fillNevents_pedestal(void)


/* ----------------------------------- */

void HcalDeadCellMonitor::fillNevents_energy(void)
{
  // Fill Histograms showing rec hits with low energy

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    cout <<"<HcalDeadCellMonitor::fillNevents_energy> BELOW-ENERGY-THRESHOLD PLOTS"<<endl;

  int mydepth=0;
  int ieta=0;
  int iphi=0;
  for (int eta=0;eta<(etaBins_-2);++eta)
    {
      ieta=eta-int((etaBins_-2)/2);
      for (int phi=0;phi<72;++phi)
        {
	  iphi=phi+1;
	  for (int depth=1;depth<=4;++depth) 
            {
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth))
		    continue;
		  // Ignore subdetectors that weren't in run
                  if ((subdet==1 && !HBpresent_) || (subdet==2 &&!HEpresent_)||(subdet==3 &&!HOpresent_) || (subdet==4 &&!HFpresent_)) continue;

		  if ((!checkHB_ && subdet==1) ||
		      (!checkHE_ && subdet==2) ||
		      (!checkHO_ && subdet==3) ||
		      (!checkHF_ && subdet==4)) continue;
		  mydepth=depth-1; // my depth index starts at 0, not 1
		  if (subdet==2 && depth<3) // remember that HE depths 1 & 2 are shifted by 4
		    mydepth=mydepth+4;
		  
		  if (rechit_occupancy[eta][phi][mydepth]==0 && deadmon_test_rechit_occupancy_) // no rechits found; 
		    {
		      UnoccupiedRecHitsByDepth[mydepth]->Fill(ieta,iphi,deadmon_checkNevents_);
		      aboveenergy[eta][phi][mydepth]=0; // shouldn't be necessary
		      continue;
		    }

		  if (!deadmon_test_energy_) continue;
		  if (aboveenergy[eta][phi][mydepth]>0)
		    {
		      // Cell exceeded energy threshold at least once in this pass;  ignore it and reset counter
		      aboveenergy[eta][phi][mydepth]=0;
		      continue;
		    }

		  if (fVerbosity>2) 
		    cout <<"DEAD CELL; BELOW ENERGY THRESHOLD = "<<subdet<<" eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<endl;
		  
		  // Cell is below energy for all 'checkNevents_' consecutive events; update histogram
		  // BinContent starts at 1, not 0 (offset by 0)
		  // Offset by another 1 due to empty bins at edges

		  BelowEnergyThresholdCellsByDepth[mydepth]->Fill(ieta,iphi,deadmon_checkNevents_);

		} // for (int subdet=1;subdet<=4;++subdet)
	      // reset counters
	      aboveenergy[eta][phi][depth]=0;
	      rechit_occupancy[eta][phi][depth]=0;
	    } // for (int depth=1;depth<=4;++depth)
	} // for (int phi=0;...)
    } // for (int eta=0;...)
    FillUnphysicalHEHFBins(UnoccupiedRecHitsByDepth);
    FillUnphysicalHEHFBins(BelowEnergyThresholdCellsByDepth);
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_ENERGY -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;


} // void HcalDeadCellMonitor::fillNevents_energy(void)



/* ----------------------------------- */

void HcalDeadCellMonitor::fillNevents_neighbor(void)
{
  // Fill Histograms showing rec hits with energy much less than neighbors' average

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>0)
    cout <<"<HcalDeadCellMonitor::fillNevents_neighbor> FILLING BELOW-NEIGHBOR-ENERGY PLOTS"<<endl;

  int mydepth=0;
  int ieta=0;
  int iphi=0;
  for (int eta=0;eta<(etaBins_-2);++eta)
    {
      ieta=eta-int((etaBins_-2)/2);
      for (int phi=0;phi<72;++phi)
        {
	  iphi=phi+1;
	  for (int depth=1;depth<=4;++depth) 
            {
	      for (int subdet=1;subdet<=4;++subdet)
		{
		  if (!validDetId((HcalSubdetector)subdet, ieta, iphi, depth))
		    continue;
		  // Ignore subdetectors that weren't in run
                  if ((subdet==1 && !HBpresent_) || (subdet==2 &&!HEpresent_)||(subdet==3 &&!HOpresent_) || (subdet==4 &&!HFpresent_)) continue;
		  if ((!checkHB_ && subdet==1) ||
		      (!checkHE_ && subdet==2) ||
		      (!checkHO_ && subdet==3) ||
		      (!checkHF_ && subdet==4)) continue;
		  mydepth=depth-1; // my depth starts at 0, not 1
		  if (subdet==2 && depth<3) // remember that HE depths 1 & 2 are shifted by +4
		    mydepth=mydepth+4;
		  if (rechit_occupancy[eta][phi][mydepth]==0) // no rechits found; ignore test
		    {
		      belowneighbors[eta][phi][mydepth]=0; // shouldn't be necessary
		      continue;
		    }
		  if (belowneighbors[eta][phi][mydepth]>0)
		    {
		      if (fVerbosity>2) cout <<"DEAD CELL; BELOW NEIGHBORS = "<<subdet<<" eta = "<<ieta<<", phi = "<<iphi<<" depth = "<<depth+1<<endl;
		      BelowNeighborsDeadCellsByDepth[mydepth]->Fill(ieta,iphi,belowneighbors[eta][phi][mydepth]);
		      //reset counter
		      belowneighbors[eta][phi][depth]=0;
		    } // if (belowneighbors[eta][phi][mydepth]>0)
		} // for (int subdet=1;subdet<=4;++subdet)

	    } // for (int depth=0;depth<4;++depth)
	} // for (int phi=0;...)
    } // for (int eta=0;...)
  FillUnphysicalHEHFBins(BelowNeighborsDeadCellsByDepth);
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_NEIGHBOR -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;


} // void HcalDeadCellMonitor::fillNevents_neighbor(void)






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
	      if (deadmon_test_rechit_occupancy_)
		{
		  problemvalue+=UnoccupiedRecHitsByDepth[mydepth]->getBinContent(eta+2,phi+2);
		  sumproblemvalue+=UnoccupiedRecHitsByDepth[mydepth]->getBinContent(eta+2,phi+2);
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

	      //problemvalue=min(1.,problemvalue);
	      ProblemDeadCellsByDepth[mydepth]->setBinContent(eta+2,phi+2,problemvalue);
	    } // for (int mydepth=0;mydepth<6;...)
	  //sumproblemvalue=min(1.,sumproblemvalue);
	  ProblemDeadCells->setBinContent(eta+2,phi+2,sumproblemvalue);
	} // loop on phi=0;phi<72
    } // loop on eta=0; eta<(etaBins_-2)
  
  if (showTiming)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalDeadCellMonitor FILLNEVENTS_PROBLEMCELLS -> "<<cpu_timer.cpuTime()<<endl;
    }

} // void HcalDeadCellMonitor::fillNevents_problemCells(void)

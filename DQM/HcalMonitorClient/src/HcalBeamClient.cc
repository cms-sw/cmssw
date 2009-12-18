#include <DQM/HcalMonitorClient/interface/HcalBeamClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/MonitorElement.h"
#include <math.h>
#include <iostream>
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"

HcalBeamClient::HcalBeamClient(){} // constructor

void HcalBeamClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName)
{
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  beamclient_checkNevents_ = ps.getUntrackedParameter<int>("BeamClient_checkNevents",100);

  minErrorFlag_ = ps.getUntrackedParameter<double>("BeamClient_minErrorFlag",0.0);

  beamclient_makeDiagnostics_ = ps.getUntrackedParameter<bool>("BeamClient_makeDiagnosticPlots",false);

  // Set histograms to NULL

  CenterOfEnergyRadius=0;
  CenterOfEnergy=0;
  COEradiusVSeta=0;
  HBCenterOfEnergyRadius=0;
  HBCenterOfEnergy=0;
  HECenterOfEnergyRadius=0;
  HECenterOfEnergy=0;
  HOCenterOfEnergyRadius=0;
  HOCenterOfEnergy=0;
  HFCenterOfEnergyRadius=0;
  HFCenterOfEnergy=0;
  
  Etsum_eta_L=0;
  Etsum_eta_S=0;
  Etsum_phi_L=0;
  Etsum_phi_S=0;
  Etsum_ratio_p=0;
  Etsum_ratio_m=0;
  Etsum_map_L=0;
  Etsum_map_S=0;
  Etsum_ratio_map=0;
  Etsum_rphi_L=0;
  Etsum_rphi_S=0;
  Energy_Occ=0;

  Occ_rphi_L=0;
  Occ_rphi_S=0;
  Occ_eta_L=0;
  Occ_eta_S=0;
  Occ_phi_L=0;
  Occ_phi_S=0;
  Occ_map_L=0;
  Occ_map_S=0;
  
  HFlumi_ETsum_perwedge=0;
  HFlumi_Occupancy_above_thr_r1=0;
  HFlumi_Occupancy_between_thrs_r1=0;
  HFlumi_Occupancy_below_thr_r1=0;
  HFlumi_Occupancy_above_thr_r2=0;
  HFlumi_Occupancy_between_thrs_r2=0;
  HFlumi_Occupancy_below_thr_r2=0;

  if (beamclient_makeDiagnostics_)
    {
      for (int i=0;i<83;++i)
	{
	  HB_CenterOfEnergyRadius[i]=0;
	  HE_CenterOfEnergyRadius[i]=0;
	  HO_CenterOfEnergyRadius[i]=0;
	  HF_CenterOfEnergyRadius[i]=0;
	}
    }
  subdets_.push_back("HB HF Depth 1 ");
  subdets_.push_back("HB HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO ZDC ");
  subdets_.push_back("HE Depth 1 ");
  subdets_.push_back("HE Depth 2 ");
  return;
} // HcalBeamClient::init;

HcalBeamClient::~HcalBeamClient()
{
  this->cleanup();
} // destructor


void HcalBeamClient::beginJob(){

  if ( debug_>1 ) cout << "HcalBeamClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;
  this->setup();

  if (!dbe_) return;

  // Set up problem cell histograms

  stringstream mydir;
  mydir<<rootFolder_<<"/BeamMonitor_Hcal";
  dbe_->setCurrentFolder(mydir.str().c_str());
  ProblemCells=dbe_->book2D(" ProblemBeamMonitor",
				" Problem Beam Cell Rate for all HCAL;i#eta;i#phi",
				85,-42.5,42.5,
				72,0.5,72.5);
  SetEtaPhiLabels(ProblemCells);
  (ProblemCells->getTH2F())->SetMinimum(0);
  (ProblemCells->getTH2F())->SetMaximum(1);
  mydir<<"/problem_beammonitor";
  dbe_->setCurrentFolder(mydir.str().c_str());

  ProblemCellsByDepth.setup(dbe_," Problem BeamMonitor Rate");
  for  (unsigned int d=0;d<ProblemCellsByDepth.depth.size();++d)
    {
      if (ProblemCellsByDepth.depth[d]!=0) 
	{
	  (ProblemCellsByDepth.depth[d]->getTH2F())->SetMinimum(0);
	  (ProblemCellsByDepth.depth[d]->getTH2F())->SetMaximum(1);
	}
    }    
  return;
} // void HcalBeamClient::beginJob(const EventSetup& eventSetup);


void HcalBeamClient::beginRun(void)
{
  if ( debug_>1 ) cout << "HcalBeamClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
} // void HcalBeamClient::beginRun(void)


void HcalBeamClient::endJob(void) 
{
  if ( debug_>1 ) cout << "HcalBeamClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();
  return;
} // void HcalBeamClient::endJob(void)


void HcalBeamClient::endRun(void) 
{
  if ( debug_>1 ) cout << "HcalBeamClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();
  return;
} // void HcalBeamClient::endRun(void)


void HcalBeamClient::setup(void) 
{
  return;
} // void HcalBeamClient::setup(void)

void HcalBeamClient::cleanup(void) 
{
  // leave deletions to framework
  if(1<0 && cloneME_)
    {
      if (beamclient_makeDiagnostics_)
	{
	  for (int i=0;i<83;++i)
	    {
	      if (HB_CenterOfEnergyRadius[i])   delete HB_CenterOfEnergyRadius[i];
	      if (HE_CenterOfEnergyRadius[i])   delete HE_CenterOfEnergyRadius[i];
	      if (HO_CenterOfEnergyRadius[i])   delete HO_CenterOfEnergyRadius[i];
	      if (HF_CenterOfEnergyRadius[i])   delete HF_CenterOfEnergyRadius[i];
	      
	    }
	}
      if (CenterOfEnergyRadius)       delete CenterOfEnergyRadius;
      if (CenterOfEnergy)             delete CenterOfEnergy;
      if (COEradiusVSeta)             delete COEradiusVSeta;
      if (HBCenterOfEnergyRadius)     delete HBCenterOfEnergyRadius;
      if (HBCenterOfEnergy)           delete HBCenterOfEnergy;
      if (HECenterOfEnergyRadius)     delete HECenterOfEnergyRadius;
      if (HECenterOfEnergy)           delete HECenterOfEnergy;
      if (HOCenterOfEnergyRadius)     delete HOCenterOfEnergyRadius;
      if (HOCenterOfEnergy)           delete HOCenterOfEnergy;
      if (HFCenterOfEnergyRadius)     delete HFCenterOfEnergyRadius;
      if (HFCenterOfEnergy)           delete HFCenterOfEnergy;
      
      if (Etsum_eta_L)                delete Etsum_eta_L;
      if (Etsum_eta_S)                delete Etsum_eta_S;
      if (Etsum_phi_L)                delete Etsum_phi_L;
      if (Etsum_phi_S)                delete Etsum_phi_S;
      if (Etsum_ratio_p)              delete Etsum_ratio_p;
      if (Etsum_ratio_m)              delete Etsum_ratio_m;
      if (Etsum_map_L)                delete Etsum_map_L;
      if (Etsum_map_S)                delete Etsum_map_S;
      if (Etsum_ratio_map)            delete Etsum_ratio_map;
      if (Etsum_rphi_L)               delete Etsum_rphi_L;
      if (Etsum_rphi_S)               delete Etsum_rphi_S;
      if (Energy_Occ)                 delete Energy_Occ;
      
      if (Occ_rphi_L) delete Occ_rphi_L;
      if (Occ_rphi_S) delete Occ_rphi_S;
      if (Occ_eta_L)  delete Occ_eta_L;
      if (Occ_eta_S)  delete Occ_eta_S;
      if (Occ_phi_L)  delete Occ_phi_L;
      if (Occ_phi_S)  delete Occ_phi_S;
      if (Occ_map_L)  delete Occ_map_L;
      if (Occ_map_S)  delete Occ_map_S;
      
      if (HFlumi_ETsum_perwedge) delete HFlumi_ETsum_perwedge;
      if (HFlumi_Occupancy_above_thr_r1)     delete HFlumi_Occupancy_above_thr_r1;
      if (HFlumi_Occupancy_between_thrs_r1)  delete HFlumi_Occupancy_between_thrs_r1;
      if (HFlumi_Occupancy_below_thr_r1)     delete HFlumi_Occupancy_below_thr_r1;
      if (HFlumi_Occupancy_above_thr_r2)     delete HFlumi_Occupancy_above_thr_r2;
      if (HFlumi_Occupancy_between_thrs_r2)  delete HFlumi_Occupancy_between_thrs_r2;
      if (HFlumi_Occupancy_below_thr_r2)     delete HFlumi_Occupancy_below_thr_r2;
      
    } // if (cloneME_)

  // Set individual pointer to NULL
  ProblemCells=0;
  CenterOfEnergyRadius=0;
  CenterOfEnergy=0;
  COEradiusVSeta=0;
  HBCenterOfEnergyRadius=0;
  HBCenterOfEnergy=0;
  HECenterOfEnergyRadius=0;
  HECenterOfEnergy=0;
  HOCenterOfEnergyRadius=0;
  HOCenterOfEnergy=0;
  HFCenterOfEnergyRadius=0;
  HFCenterOfEnergy=0;

  for (unsigned int i=0;i<ProblemCellsByDepth.depth.size();++i)
    ProblemCellsByDepth.depth[i]=0;
  
  Etsum_eta_L=0;
  Etsum_eta_S=0;
  Etsum_phi_L=0;
  Etsum_phi_S=0;
  Etsum_ratio_p=0;
  Etsum_ratio_m=0;
  Etsum_map_L=0;
  Etsum_map_S=0;
  Etsum_ratio_map=0;
  Etsum_rphi_L=0;
  Etsum_rphi_S=0;
  Energy_Occ=0;

  Occ_rphi_L=0;
  Occ_rphi_S=0;
  Occ_eta_L=0;
  Occ_eta_S=0;
  Occ_phi_L=0;
  Occ_phi_S=0;
  Occ_map_L=0;
  Occ_map_S=0;
  
  HFlumi_ETsum_perwedge=0;
  HFlumi_Occupancy_above_thr_r1=0;
  HFlumi_Occupancy_between_thrs_r1=0;
  HFlumi_Occupancy_below_thr_r1=0;
  HFlumi_Occupancy_above_thr_r2=0;
  HFlumi_Occupancy_between_thrs_r2=0;
  HFlumi_Occupancy_below_thr_r2=0;

  if (beamclient_makeDiagnostics_)
    {
      for (int i=0;i<83;++i)
	{
	  HB_CenterOfEnergyRadius[i]=0;
	  HE_CenterOfEnergyRadius[i]=0;
	  HO_CenterOfEnergyRadius[i]=0;
	  HF_CenterOfEnergyRadius[i]=0;
	}
    }
  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  return;
} // void HcalBeamClient::cleanup(void)


void HcalBeamClient::report()
{
  if(!dbe_) return;
  if ( debug_>1 ) cout << "HcalBeamClient: report" << endl;
  this->setup();

  stringstream name;
  name<<process_.c_str()<<rootFolder_.c_str()<<"/BeamMonitor_Hcal/BeamMonitor Event Number";
  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) 
    {
      string s = me->valueString();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
      if ( debug_>1 ) cout << "Found '" << name.str().c_str() << "'" << endl;
    }
  getHistograms();

  return;
} // HcalBeamClient::report()


void HcalBeamClient::getHistograms()
{
  if(!dbe_) return;

  ostringstream name;  

  name<<process_.c_str()<<"BeamMonitor_Hcal/CenterOfEnergyRadius";
  CenterOfEnergyRadius = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/CenterOfEnergy";
  CenterOfEnergy = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/COEradiusVSeta";
  COEradiusVSeta = getTProfile( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");

  name<<process_.c_str()<<"BeamMonitor_Hcal/HB/HBCenterOfEnergyRadius";
  HBCenterOfEnergyRadius = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/HB/HBCenterOfEnergy";
  HBCenterOfEnergy = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  
  if (beamclient_makeDiagnostics_)
    {
      for (int i=-16;i<=16;++i)
	{
	  if (i==0) continue;
	  name<<process_.c_str()<<"BeamMonitor_Hcal/HB/HB_CenterOfEnergyRadius_ieta"<<i;
	  HB_CenterOfEnergyRadius[i+16]= getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
	  name.str("");
	}
    }
  name<<process_.c_str()<<"BeamMonitor_Hcal/HE/HECenterOfEnergyRadius";
  HECenterOfEnergyRadius = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/HE/HECenterOfEnergy";
  HECenterOfEnergy = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");

  if (beamclient_makeDiagnostics_)
    {
      for (int i=-29;i<=29;++i)
	{
	  if (abs(i)<17) continue;
	  name<<process_.c_str()<<"BeamMonitor_Hcal/HE/HE_CenterOfEnergyRadius_ieta"<<i;
	  HE_CenterOfEnergyRadius[i+29]= getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
	  name.str("");
	}
    }
  name<<process_.c_str()<<"BeamMonitor_Hcal/HO/HOCenterOfEnergyRadius";
  HOCenterOfEnergyRadius = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/HO/HOCenterOfEnergy";
  HOCenterOfEnergy = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  
  if (beamclient_makeDiagnostics_)
    {
      for (int i=-15;i<=15;++i)
	{
	  if (i==0) continue;
	  name<<process_.c_str()<<"BeamMonitor_Hcal/HO/HO_CenterOfEnergyRadius_ieta"<<i;
	  HO_CenterOfEnergyRadius[i+15]= getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
	  name.str("");
	}
    }
  name<<process_.c_str()<<"BeamMonitor_Hcal/HF/HFCenterOfEnergyRadius";
  HFCenterOfEnergyRadius = getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/HF/HFCenterOfEnergy";
  HFCenterOfEnergy = getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  
  if (beamclient_makeDiagnostics_)
    {
      for (int i=-41;i<=41;++i)
	{
	  if (abs(i)<29) continue;
	  name<<process_.c_str()<<"BeamMonitor_Hcal/HF/HF_CenterOfEnergyRadius_ieta"<<i;
	  HF_CenterOfEnergyRadius[i+41]= getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
	  name.str("");
	}
    }
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Et Sum vs Eta Long Fiber";
  Etsum_eta_L=getTProfile( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Et Sum vs Eta Short Fiber";
  Etsum_eta_S=getTProfile( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Et Sum vs Phi Long Fiber";
  Etsum_phi_L=getTProfile( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Et Sum vs Phi Short Fiber";
  Etsum_phi_S=getTProfile( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs fm HF+";
  Etsum_ratio_p=getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs Energy";
  Energy_Occ=getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs fm HF-";
  Etsum_ratio_m=getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/EtSum 2D phi and eta Long Fiber";
  Etsum_map_L=getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/EtSum 2D phi and eta Short Fiber";
  Etsum_map_S=getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/EtSum 2D phi and radius Long Fiber";
  Etsum_rphi_L=getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/EtSum 2D phi and radius Short Fiber";
  Etsum_rphi_S=getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Abnormal fm";
  Etsum_ratio_map=getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ 2D phi and radius Short Fiber";
  Occ_rphi_S=getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ 2D phi and radius Long Fiber";
  Occ_rphi_L=getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs Eta Short Fiber";
  Occ_eta_S=getTProfile( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs Eta Long Fiber";
  Occ_eta_L=getTProfile( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs Phi Short Fiber";
  Occ_phi_S=getTProfile( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs Phi Long Fiber";
  Occ_phi_L=getTProfile( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ_map Long Fiber";
  Occ_map_L=getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ_map Short Fiber";
  Occ_map_S=getTH2F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");

  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi ET-sum per wedge";
  HFlumi_ETsum_perwedge=getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy above threshold ring1";
  HFlumi_Occupancy_above_thr_r1=getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy between thresholds ring1";
  HFlumi_Occupancy_between_thrs_r1=getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy below threshold ring1";
  HFlumi_Occupancy_below_thr_r1=getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy above threshold ring2";
  HFlumi_Occupancy_above_thr_r2=getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy between thresholds ring2";
  HFlumi_Occupancy_between_thrs_r2=getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy below threshold ring2";
  HFlumi_Occupancy_below_thr_r2=getTH1F( name.str(), process_, rootFolder_, dbe_, debug_, cloneME_);
  name.str("");

  return;
}  // void HcalBeamClient::getHistograms()


void HcalBeamClient::analyze(void)
{
  ++jevt_;
  if ( jevt_ % 10 == 0 ) 
    {
      if ( debug_>1 ) cout << "<HcalBeamClient::analyze>  Running analyze "<<endl;
    }
  //getHistograms();
  return;
} // void HcalBeamClient::analyze(void)

void HcalBeamClient::createTests()
{
  // Removed a bunch of code that was in older versions of HcalBeamClient
  // tests should now be handled from outside
  if(!dbe_) return;
  return;
} // void HcalBeamClient::createTests()

void HcalBeamClient::resetAllME()
{
  if(!dbe_) return;
  
  ostringstream name;

  // Reset individual histograms

  if (ProblemCells!=0)
    ProblemCells->Reset();
  
  for  (unsigned int d=0;d<ProblemCellsByDepth.depth.size();++d)
    {
      if (ProblemCellsByDepth.depth[d]!=0) 
	ProblemCellsByDepth.depth[d]->Reset();
    }
  
  name<<process_.c_str()<<"BeamMonitor_Hcal/CenterOfEnergyRadius";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/CenterOfEnergy";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/COEradiusVSeta";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/HB/CenterOfEnergyRadius";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/HB/CenterOfEnergy";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  if (beamclient_makeDiagnostics_)
    {
      for (int i=-16;i<=16;++i)
	{
	  if (i==0) continue;
	  name<<process_.c_str()<<"BeamMonitor_Hcal/HB/HB_CenterOfEnergyRadius_ieta"<<i;
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
    }
  name<<process_.c_str()<<"BeamMonitor_Hcal/HE/CenterOfEnergyRadius";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/HE/CenterOfEnergy";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  if (beamclient_makeDiagnostics_)
    {
      for (int i=-29;i<=29;++i)
	{
	  if (abs(i)<17) continue;
	  name<<process_.c_str()<<"BeamMonitor_Hcal/HE/HE_CenterOfEnergyRadius_ieta"<<i;
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
    }
  name<<process_.c_str()<<"BeamMonitor_Hcal/HO/CenterOfEnergyRadius";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/HO/CenterOfEnergy";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  
  if (beamclient_makeDiagnostics_)
    {
      for (int i=-15;i<=15;++i)
	{
	  if (i==0) continue;
	  name<<process_.c_str()<<"BeamMonitor_Hcal/HO/HO_CenterOfEnergyRadius_ieta"<<i;
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
    }
  name<<process_.c_str()<<"BeamMonitor_Hcal/HF/CenterOfEnergyRadius";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/HF/CenterOfEnergy";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  if (beamclient_makeDiagnostics_)
    {
      for (int i=-41;i<=41;++i)
	{
	  if (i==0) continue;
	  name<<process_.c_str()<<"BeamMonitor_Hcal/HF/HF_CenterOfEnergyRadius_ieta"<<i;
	  resetME(name.str().c_str(),dbe_);
	  name.str("");
	}
    }
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Et Sum vs Eta Long Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Et Sum vs Eta Short Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Et Sum vs Phi Long Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Et Sum vs Phi Short Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs fm HF+";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs Energy";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs fm HF-";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/EtSum 2D phi and eta Long Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/EtSum 2D phi and eta Short Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/EtSum 2D phi and radius Long Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/EtSum 2D phi and radius Short Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Abnormal fm";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ 2D phi and radius Short Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ 2D phi and radius Long Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs Eta Short Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs Eta Long Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs Phi Short Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ vs Phi Long Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ_map Long Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/Occ_map Short Fiber";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi ET-sum per wedge";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy above threshold ring1";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy between thresholds ring1";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy below threshold ring1";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy above threshold ring2";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy between thresholds ring2";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  name<<process_.c_str()<<"BeamMonitor_Hcal/Lumi/HF lumi Occupancy below threshold ring2";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  return;
} // void HcalBeamClient::resetAllME()



void HcalBeamClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>1) cout << "Preparing HcalBeamClient html output ..." << endl;
  getHistograms();
  string client = "BeamMonitor";

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Beam Monitor Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Beam Monitor</span></h2> " << endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal Beam Monitor Status</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  htmlFile << "</h3>" << endl;

  htmlFile << "<table align=\"center\" border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  htmlAnyHisto(runNo,ProblemCells->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
  gStyle->SetPalette(1);
  htmlFile<<"</tr><tr align=\"center\" border=\"0\" cellspacing=\"0\" cellpadding=\"10\" >"<<endl;
  htmlAnyHisto(runNo,CenterOfEnergy,"normalized x coordinate","normalized y coordinate", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<endl;
  htmlFile<<"<tr align=\"center\"><td> There are as yet no criteria for marking beam monitor plots.  The plots in the link below are for diagnostic purposes only."<<endl;
 
  htmlFile<<"</td>"<<endl;
  htmlFile<<"</tr></table>"<<endl;
  htmlFile<<"<hr><table align=\"center\" border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile<<"<tr><td align=center><a href=\"Expert_"<< htmlName<<"\"><h2>Detailed Beam Monitor Plots</h2> </a></br></td>"<<endl;
  htmlFile<<"</tr></table><br><hr>"<<endl;
  
  // Now print out problem cells
  htmlFile <<"<br>"<<endl;
  htmlFile << "<h2><strong>Hcal Problem Cells</strong></h2>" << endl;
  htmlFile << "(A problem cell is listed below if its failure rate exceeds "<<(100.*minErrorFlag_)<<"%).<br><br>"<<endl;
  htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile <<"<td> Problem Beam Monitor Cells<br>(ieta, iphi, depth)</td><td align=\"center\"> Fraction of Events <br>in which cells are bad (%)</td></tr>"<<endl;

  if (ProblemCells==0)
    {
      if (debug_) cout <<"<HcalBeamClient::htmlOutput>  ERROR: can't find Problem Beam Monitor plot!"<<endl;
      return;
    }

  int ieta=0;
  int iphi=0;

  ostringstream name;
  int etabins=0, phibins=0;
  for (unsigned int depth=0;depth<ProblemCellsByDepth.depth.size(); ++depth)
    {
      for (int eta=0;eta<etabins;++eta)
        {
	  ieta=CalcIeta(eta,depth+1);
	  if (ieta==-9999) continue;
	  for (int phi=0;phi<phibins;++phi)
            {
	      iphi=phi+1;
	      if (abs(eta)>20 && phi%2!=1) continue;
	      if (abs(eta)>39 && phi%4!=3) continue;
	      if (ProblemCellsByDepth.depth[depth]==0)
		  continue;
	      if (ProblemCellsByDepth.depth[depth]->getBinContent(ieta,iphi)>minErrorFlag_)
		{
		  if (depth<2)
                    {
                      if (isHB(eta,depth+1)) name <<"HB";
                      else if (isHE(eta,depth+1)) name<<"HE";
                      else if (isHF(eta,depth+1)) name<<"HF";
                    }
                  else if (depth==2) name <<"HE";
                  else if (depth==3) name<<"HO";

		  htmlFile<<"<td>"<<name.str().c_str()<<" ("<<eta<<", "<<phi<<", "<<depth+1<<")</td><td align=\"center\">"<<ProblemCellsByDepth.depth[depth]->getBinContent(ieta,iphi)*100.<<"</td></tr>"<<std::endl;
		  
		  name.str("");	  name.str("");

		}

	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // for (int depth=0;...)
  
  
  // html page footer
  htmlFile <<"</table> " << endl;
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();
  htmlExpertOutput(runNo, htmlDir, htmlName);

  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalBeamClient HTMLOUTPUT  -> "<<cpu_timer.cpuTime()<<endl;
    }

  return;
} //void HcalBeamClient::htmlOutput(int runNo, ...) 


void HcalBeamClient::htmlExpertOutput(int runNo, string htmlDir, string htmlName)
{

  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>1) 
    cout <<" <HcalBeamClient::htmlExpertOutput>  Preparing Expert html output ..." <<endl;
  
  string client = "BeamMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_); // does this do anything?

  ofstream htmlFile;
  htmlFile.open((htmlDir +"Expert_"+ htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Beam Monitor Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile <<"<a name=\"EXPERT_BEAM_TOP\" href = \".\"> Back to Main HCAL DQM Page </a><br>"<<endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Beam Monitor Status Page </a><br>"<<endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Beam Monitor</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table width=100%  border = 1>"<<endl;
  htmlFile << "<tr><td align=\"center\" colspan=\"2\"><a href=\"#OVERALL_PROBLEMS\">PROBLEM CELLS BY DEPTH </a></td></tr>"<<endl;
  htmlFile << "<tr><td align=\"center\" colspan=\"2\">"<<endl;
  htmlFile<<"<br><a href=\"#CENTEROFENERGY\">Center-of-Energy Plots </a>"<<endl;
  htmlFile<<"<br><a href=\"#LONGSHORT\">Long/Short Fiber Plots </a>"<<endl;
  htmlFile<<"<br><a href=\"#LUMI\">Other HF Lumi Plots </a>"<<endl;
  htmlFile<<"</td></tr>"<<endl;
  if (beamclient_makeDiagnostics_)
    {
      htmlFile<<"<tr><td align=\"center\">"<<endl;
      htmlFile<<"<br><a href=\"#HB\">HB Individual Eta Plots </a>"<<endl;
      htmlFile<<"</td><td align=\"center\">"<<endl;
      htmlFile<<"<br><a href=\"#HE\">HE Individual Eta Plots </a>"<<endl;
      htmlFile<<"</td></tr><tr><td align=\"center\">"<<endl;
      htmlFile<<"<br><a href=\"#HO\">HO Individual Eta Plots </a>"<<endl;
      htmlFile<<"</td><td align=\"center\">"<<endl;
      htmlFile<<"<br><a href=\"#HF\">HF Individual Eta Plots </a>"<<endl;
    }
  htmlFile << "</td></tr>"<<endl;
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><br>"<<endl;


  // Plot overall errors
  htmlFile << "<h2><strong><a name=\"OVERALL_PROBLEMS\">Eta-Phi Maps of Problem Cells By Depth</strong></h2>"<<endl;
  htmlFile <<" These plots of problem cells should be empty, until specific beam-monitor tests are devised<br>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_BEAM_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  
  for (unsigned int i=0;i<ProblemCellsByDepth.depth.size()/2;++i)
    {
      htmlFile << "<tr align=\"left\">" << endl;
      htmlAnyHisto(runNo,ProblemCellsByDepth.depth[2*i]->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ProblemCellsByDepth.depth[2*i+1]->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<endl;
    }
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><hr><br>"<<endl;
  

  // Plot Beam Center of Energy
  htmlFile << "<h2><strong><a name=\"CENTEROFENERGY\">Center-of-Energy plots for all subdetectors</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_BEAM_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  
  htmlFile << "<tr>" << endl;
  htmlAnyHisto(runNo,COEradiusVSeta,"i#eta","normalized radius", 92, htmlFile, htmlDir);
  htmlFile <<"</tr></table>"<<endl;
  htmlFile <<"<table border=\"0\" cellspacing=\"0\" cellpadding=\"10\"> "<<endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlAnyHisto(runNo,CenterOfEnergy,"normalized x coordinate","normalized y coordinate", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,CenterOfEnergyRadius,"normalized radius","", 92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,HBCenterOfEnergy,"normalized x coordinate","normalized y coordinate", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,HBCenterOfEnergyRadius,"normalized radius","", 92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,HECenterOfEnergy,"normalized x coordinate","normalized y coordinate", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,HECenterOfEnergyRadius,"normalized radius","", 92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,HOCenterOfEnergy,"normalized x coordinate","normalized y coordinate", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,HOCenterOfEnergyRadius,"normalized radius","", 92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,HFCenterOfEnergy,"normalized x coordinate","normalized y coordinate", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,HFCenterOfEnergyRadius,"normalized radius","", 92, htmlFile, htmlDir);
  htmlFile <<"</tr>"<<endl;
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><hr><br>"<<endl;
  
  // Plot Long/Short Fiber Plots
  htmlFile << "<h2><strong><a name=\"LONGSHORT\">HF long/short Fiber plots</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_BEAM_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,Occ_eta_L,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,Occ_eta_S,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,Occ_phi_L,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,Occ_phi_S,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,Occ_map_L,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,Occ_map_S,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,Occ_rphi_L,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,Occ_rphi_S,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,Etsum_eta_L,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,Etsum_eta_S,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,Etsum_phi_L,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,Etsum_phi_S,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,Etsum_map_L,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,Etsum_map_S,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,Etsum_rphi_L,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,Etsum_rphi_S,"","",92, htmlFile, htmlDir);

  htmlFile <<"</tr>"<<endl;
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><hr><br>"<<endl;
  
  // Plot other Lumi Plots
  htmlFile << "<h2><strong><a name=\"LUMI\">Other HF Lumi plots</strong></h2>"<<endl;
  htmlFile <<"<a href= \"#EXPERT_BEAM_TOP\" > Back to Top</a><br>"<<endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  gStyle->SetPalette(1);
  
  htmlFile << "<tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,Energy_Occ,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,Etsum_ratio_map,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,Etsum_ratio_m,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,Etsum_ratio_p,"","",92, htmlFile, htmlDir);
  htmlFile<<"</tr></table>"<<endl;
  htmlFile<<"<table border=\"0\" cellspacing=\"0\" cellpadding=\"10\"><tr>"<<endl;
  htmlAnyHisto(runNo,HFlumi_ETsum_perwedge,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,HFlumi_Occupancy_above_thr_r1,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,HFlumi_Occupancy_above_thr_r2,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,HFlumi_Occupancy_between_thrs_r1,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,HFlumi_Occupancy_between_thrs_r2,"","",92, htmlFile, htmlDir);
  htmlFile << "</tr><tr align=\"left\">" << endl;
  htmlAnyHisto(runNo,HFlumi_Occupancy_below_thr_r1,"","",92, htmlFile, htmlDir);
  htmlAnyHisto(runNo,HFlumi_Occupancy_below_thr_r2,"","",92, htmlFile, htmlDir);
  htmlFile <<"</tr>"<<endl;
  htmlFile <<"</table>"<<endl;
  htmlFile <<"<br><hr><br>"<<endl;

  if (beamclient_makeDiagnostics_)
    {
      htmlFile << "<h2><strong><a name=\"HB\">HB individual ieta plots</strong></h2>"<<endl;
      htmlFile <<"<a href= \"#EXPERT_BEAM_TOP\" > Back to Top</a><br>"<<endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\"> " << endl;
      gStyle->SetPalette(1);
  
      for (int i=0;i<16;++i)
	{
	  htmlFile << "<tr align=\"left\">" << endl;
	  htmlAnyHisto(runNo,HB_CenterOfEnergyRadius[i],"","",92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,HB_CenterOfEnergyRadius[32-i],"","",92, htmlFile, htmlDir);
	}
      htmlFile <<"</tr>"<<endl;
      htmlFile <<"</table>"<<endl;
      htmlFile <<"<br><hr><br>"<<endl;

      htmlFile << "<h2><strong><a name=\"HE\">HE individual ieta plots</strong></h2>"<<endl;
      htmlFile <<"<a href= \"#EXPERT_BEAM_TOP\" > Back to Top</a><br>"<<endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\"> " << endl;
      gStyle->SetPalette(1);
  
      for (int i=0;i<13;++i)
	{
	  htmlFile << "<tr align=\"left\">" << endl;
	  htmlAnyHisto(runNo,HE_CenterOfEnergyRadius[i],"","",92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,HE_CenterOfEnergyRadius[58-i],"","",92, htmlFile, htmlDir);
	}
      htmlFile <<"</tr>"<<endl;
      htmlFile <<"</table>"<<endl;
      htmlFile <<"<br><hr><br>"<<endl;
 
      htmlFile << "<h2><strong><a name=\"HO\">HO individual ieta plots</strong></h2>"<<endl;
      htmlFile <<"<a href= \"#EXPERT_BEAM_TOP\" > Back to Top</a><br>"<<endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\"> " << endl;
      gStyle->SetPalette(1);
  
      for (int i=0;i<15;++i)
	{
	  htmlFile << "<tr align=\"left\">" << endl;
	  htmlAnyHisto(runNo,HO_CenterOfEnergyRadius[i],"","",92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,HO_CenterOfEnergyRadius[30-i],"","",92, htmlFile, htmlDir);
	}
      htmlFile <<"</tr>"<<endl;
      htmlFile <<"</table>"<<endl;
      htmlFile <<"<br><hr><br>"<<endl;

      htmlFile << "<h2><strong><a name=\"HF\">HF individual ieta plots</strong></h2>"<<endl;
      htmlFile <<"<a href= \"#EXPERT_BEAM_TOP\" > Back to Top</a><br>"<<endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\"> " << endl;
      gStyle->SetPalette(1);
  
      for (int i=0;i<13;++i)
	{
	  htmlFile << "<tr align=\"left\">" << endl;
	  htmlAnyHisto(runNo,HF_CenterOfEnergyRadius[i],"","",92, htmlFile, htmlDir);
	  htmlAnyHisto(runNo,HF_CenterOfEnergyRadius[82-i],"","",92, htmlFile, htmlDir);
	}
      htmlFile <<"</tr>"<<endl;
      htmlFile <<"</table>"<<endl;
      htmlFile <<"<br><hr><br>"<<endl;

    } // if (beamclient_makeDiagnostics_)

  // Footer 
  htmlFile <<"<br><hr><br><a href= \"#EXPERT_BEAM_TOP\" > Back to Top of Page </a><br>"<<endl;
  htmlFile <<"<a href = \".\"> Back to Main HCAL DQM Page </a><br>"<<endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to Beam Monitor Status Page </a><br>"<<endl;

  // 

  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();

  if (showTiming_)
    {
      cpu_timer.stop();  cout <<"TIMER:: HcalBeamClient  HTMLEXPERTOUTPUT ->"<<cpu_timer.cpuTime()<<endl;
    }
  return;
} // void HcalBeamClient::htmlExpertOutput(...)


void HcalBeamClient::loadHistograms(TFile* infile)
{
  // deprecated function; no longer used
  return;

} // loadHistograms(...)

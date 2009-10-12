#include <DQM/HcalMonitorClient/interface/HcalRecHitClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <math.h>
#include <iostream>

HcalRecHitClient::HcalRecHitClient(){} // constructor 

void HcalRecHitClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  // Get variable values from cfg file

  rechitclient_checkNevents_ = ps.getUntrackedParameter<int>("RecHitClient_checkNevents",100);

  minErrorFlag_ = ps.getUntrackedParameter<double>("RecHitClient_minErrorFlag",0.0);

  rechitclient_makeDiagnostics_ = ps.getUntrackedParameter<bool>("RecHitClient_makeDiagnosticPlots",false);

  // Set histograms to NULL
  ProblemCells=0;
  meHBEnergy_1D=0;
  meHEEnergy_1D=0;
  meHOEnergy_1D=0;
  meHFEnergy_1D=0;
  meHBEnergyRMS_1D=0;
  meHEEnergyRMS_1D=0;
  meHOEnergyRMS_1D=0;
  meHFEnergyRMS_1D=0;

  for (int i=0;i<4;++i)
    {
      // Set each array's pointers to NULL
      OccupancyByDepth[i]         =0;
      OccupancyThreshByDepth[i]   =0;
      SumEnergyByDepth[i]            =0;
      SumEnergy2ByDepth[i]           =0;
      SumEnergyThreshByDepth[i]      =0;
      SumTimeByDepth[i]              =0;
      SumTimeThreshByDepth[i]        =0;
    }  

  if (rechitclient_makeDiagnostics_)
    {
      d_HBEnergy                  =0;
      d_HBTotalEnergy             =0;
      d_HBTime                    =0;
      d_HBOccupancy               =0;
      d_HBThreshEnergy            =0;
      d_HBThreshTotalEnergy       =0;
      d_HBThreshTime              =0;
      d_HBThreshOccupancy         =0;
      
      d_HEEnergy                  =0;
      d_HETotalEnergy             =0;
      d_HETime                    =0;
      d_HEOccupancy               =0;
      d_HEThreshEnergy            =0;
      d_HEThreshTotalEnergy       =0;
      d_HEThreshTime              =0;
      d_HEThreshOccupancy         =0;
      
      d_HOEnergy                  =0;
      d_HOTotalEnergy             =0;
      d_HOTime                    =0;
      d_HOOccupancy               =0;
      d_HOThreshEnergy            =0;
      d_HOThreshTotalEnergy       =0;
      d_HOThreshTime              =0;
      d_HOThreshOccupancy         =0;
      
      d_HFEnergy                  =0;
      d_HFTotalEnergy             =0;
      d_HFTime                    =0;
      d_HFOccupancy               =0;
      d_HFThreshEnergy            =0;
      d_HFThreshTotalEnergy       =0;
      d_HFThreshTime              =0;
      d_HFThreshOccupancy         =0;
    } // if (rechitclient_makeDiagnostics_)

  subdets_.push_back("HB HE HF Depth 1 ");
  subdets_.push_back("HB HE HF Depth 2 ");
  subdets_.push_back("HE Depth 3 ");
  subdets_.push_back("HO Depth 4 ");

  return;
} // void HcalRecHitClient::init(...)


HcalRecHitClient::~HcalRecHitClient()
{
  this->cleanup();
} // destructor


void HcalRecHitClient::beginJob(){

  if ( debug_>1 ) std::cout << "HcalRecHitClient: beginJob" << std::endl;

  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  if (!dbe_) return;
  stringstream mydir;
  mydir<<rootFolder_<<"/RecHitMonitor_Hcal";
  dbe_->setCurrentFolder(mydir.str().c_str());

  // Create problem cell plots
  ProblemCells=dbe_->book2D(" ProblemRecHits",
                               "Problem RecHit Rate for all HCAL",
                               85,-42.5,42.5,
                               72,0.5,72.5);
  SetEtaPhiLabels(ProblemCells);
  
  // Overall Problem plot appears in main directory; plots by depth appear in subdirectory
  mydir<<"/problem_rechits";
  dbe_->setCurrentFolder(mydir.str().c_str());
  ProblemCellsByDepth.setup(dbe_,"Problem RecHit Rate");
  mydir.str("");
  
  mydir<<rootFolder_<<"/RecHitMonitor_Hcal/rechit_info";
  dbe_->setCurrentFolder(mydir.str().c_str());
  meEnergyByDepth.setup(dbe_,"RecHit Average Energy","GeV");
  meTimeByDepth.setup(dbe_,"RecHit Average Time","nS");
  mydir.str("");
  
  mydir<<rootFolder_<<"/RecHitMonitor_Hcal/rechit_info_threshold";
  dbe_->setCurrentFolder(mydir.str().c_str());
  meEnergyThreshByDepth.setup(dbe_,"Above Threshold RecHit Average Energy","GeV");
  meTimeThreshByDepth.setup(dbe_,"Above Threshold RecHit Average Time","nS");
  mydir.str("");
  
  mydir<<rootFolder_<<"/RecHitMonitor_Hcal/rechit_1D_plots";
  dbe_->setCurrentFolder(mydir.str().c_str());
  meHBEnergy_1D=dbe_->book1D("HB_energy_1D","HB Average Energy Per RecHit;Energy (GeV)",400,-5,15);
  meHEEnergy_1D=dbe_->book1D("HE_energy_1D","HE Average Energy Per RecHit;Energy (GeV)",400,-5,15);
  meHOEnergy_1D=dbe_->book1D("HO_energy_1D","HO Average Energy Per RecHit;Energy (GeV)",600,-10,20);
  meHFEnergy_1D=dbe_->book1D("HF_energy_1D","HF Average Energy Per RecHit;Energy (GeV)",400,-5,15);

  meHBEnergyRMS_1D=dbe_->book1D("HB_energy_RMS_1D","HB Energy RMS Per RecHit;Energy (GeV)",500,0,5);
  meHEEnergyRMS_1D=dbe_->book1D("HE_energy_RMS_1D","HE Energy RMS Per RecHit;Energy (GeV)",500,0,5);
  meHOEnergyRMS_1D=dbe_->book1D("HO_energy_RMS_1D","HO Energy RMS Per RecHit;Energy (GeV)",500,0,5);
  meHFEnergyRMS_1D=dbe_->book1D("HF_energy_RMS_1D","HF Energy RMS Per RecHit;Energy (GeV)",500,0,5);
  
  return;
} // void HcalRecHitClient::beginJob(const EventSetup& eventSetup);


void HcalRecHitClient::beginRun(void)
{
  if ( debug_>1 ) std::cout << "HcalRecHitClient: beginRun" << std::endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
} // void HcalRecHitClient::beginRun(void)


void HcalRecHitClient::endJob(void) 
{
  if ( debug_>1 ) std::cout << "HcalRecHitClient: endJob, ievt = " << ievt_ << std::endl;

  this->cleanup();
  return;
} // void HcalRecHitClient::endJob(void)


void HcalRecHitClient::endRun(void) 
{
  if ( debug_>1 ) std::cout << "HcalRecHitClient: endRun, jevt = " << jevt_ << std::endl;

  this->cleanup();
  return;
} // void HcalRecHitClient::endRun(void)


void HcalRecHitClient::setup(void) 
{
  return;
} // void HcalRecHitClient::setup(void)


void HcalRecHitClient::cleanup(void) 
{
  // leave deletions to framework
  if(1<0 && cloneME_)
    {
      // delete individual histogram pointers

      for (int i=0;i<4;++i)
	{
	  // delete pointers within arrays of histograms
	  if (OccupancyByDepth[i])                delete OccupancyByDepth[i];
	  if (OccupancyThreshByDepth[i])          delete OccupancyThreshByDepth[i];
	  if (SumEnergyByDepth[i])                delete SumEnergyByDepth[i];
          if (SumEnergy2ByDepth[i])               delete SumEnergy2ByDepth[i];
	  if (SumEnergyThreshByDepth[i])          delete SumEnergyThreshByDepth[i];
	  if (SumTimeByDepth[i])                  delete SumTimeByDepth[i];
	  if (SumTimeThreshByDepth[i])            delete SumTimeThreshByDepth[i];

	}
      
      if (rechitclient_makeDiagnostics_)
	{
	  if (d_HBEnergy)                        delete d_HBEnergy;
	  if (d_HBTotalEnergy)                   delete d_HBTotalEnergy;
	  if (d_HBTime)                          delete d_HBTime;
	  if (d_HBOccupancy)                     delete d_HBOccupancy;
	  if (d_HBThreshEnergy)                  delete d_HBThreshEnergy;
	  if (d_HBThreshTotalEnergy)             delete d_HBThreshTotalEnergy;
	  if (d_HBThreshTime)                    delete d_HBThreshTime;
	  if (d_HBThreshOccupancy)               delete d_HBThreshOccupancy;

	  if (d_HEEnergy)                        delete d_HEEnergy;
	  if (d_HETotalEnergy)                   delete d_HETotalEnergy;
	  if (d_HETime)                          delete d_HETime;
	  if (d_HEOccupancy)                     delete d_HEOccupancy;
	  if (d_HEThreshEnergy)                  delete d_HEThreshEnergy;
	  if (d_HEThreshTotalEnergy)             delete d_HEThreshTotalEnergy;
	  if (d_HEThreshTime)                    delete d_HEThreshTime;
	  if (d_HEThreshOccupancy)               delete d_HEThreshOccupancy;

	  if (d_HOEnergy)                        delete d_HOEnergy;
	  if (d_HOTotalEnergy)                   delete d_HOTotalEnergy;
	  if (d_HOTime)                          delete d_HOTime;
	  if (d_HOOccupancy)                     delete d_HOOccupancy;
	  if (d_HOThreshEnergy)                  delete d_HOThreshEnergy;
	  if (d_HOThreshTotalEnergy)             delete d_HOThreshTotalEnergy;
	  if (d_HOThreshTime)                    delete d_HOThreshTime;
	  if (d_HOThreshOccupancy)               delete d_HOThreshOccupancy;

	  if (d_HFEnergy)                        delete d_HFEnergy;
	  if (d_HFTotalEnergy)                   delete d_HFTotalEnergy;
	  if (d_HFTime)                          delete d_HFTime;
	  if (d_HFOccupancy)                     delete d_HFOccupancy;
	  if (d_HFThreshEnergy)                  delete d_HFThreshEnergy;
	  if (d_HFThreshTotalEnergy)             delete d_HFThreshTotalEnergy;
	  if (d_HFThreshTime)                    delete d_HFThreshTime;
	  if (d_HFThreshOccupancy)               delete d_HFThreshOccupancy;

	} // if (rechitclient_makeDiagnostics_)
      
    } // if (cloneME_)

  /*
  // Set individual pointers to NULL
  ProblemRecHits = 0;
  h_HBEnergy_1D=0;
  h_HEEnergy_1D=0;
  h_HOEnergy_1D=0;
  h_HFEnergy_1D=0;

  for (int i=0;i<4;++i)
    {
      ProblemRecHitsByDepth[i]    =0;
      OccupancyByDepth[i]          =0;
      OccupancyThreshByDepth[i]   =0;
      EnergyByDepth[i]            =0;
      EnergyThreshByDepth[i]      =0;
      TimeByDepth[i]              =0;
      TimeThreshByDepth[i]        =0;
      SumEnergyByDepth[i]            =0;
      SumEnergyThreshByDepth[i]      =0;
      SumTimeByDepth[i]              =0;
      SumTimeThreshByDepth[i]        =0;
    }
  
  if (rechitclient_makeDiagnostics_)
    {
      d_HBEnergy                  =0;
      d_HBTotalEnergy             =0;
      d_HBTime                    =0;
      d_HBOccupancy               =0;
      d_HBThreshEnergy            =0;
      d_HBThreshTotalEnergy       =0;
      d_HBThreshTime              =0;
      d_HBThreshOccupancy         =0;
      
      d_HEEnergy                  =0;
      d_HETotalEnergy             =0;
      d_HETime                    =0;
      d_HEOccupancy               =0;
      d_HEThreshEnergy            =0;
      d_HEThreshTotalEnergy       =0;
      d_HEThreshTime              =0;
      d_HEThreshOccupancy         =0;
      
      d_HOEnergy                  =0;
      d_HOTotalEnergy             =0;
      d_HOTime                    =0;
      d_HOOccupancy               =0;
      d_HOThreshEnergy            =0;
      d_HOThreshTotalEnergy       =0;
      d_HOThreshTime              =0;
      d_HOThreshOccupancy         =0;
      
      d_HFEnergy                  =0;
      d_HFTotalEnergy             =0;
      d_HFTime                    =0;
      d_HFOccupancy               =0;
      d_HFThreshEnergy            =0;
      d_HFThreshTotalEnergy       =0;
      d_HFThreshTime              =0;
      d_HFThreshOccupancy         =0;
    } // if (rechitclient_makeDiagnostics_)
  */

  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  return;
} // void HcalRecHitClient::cleanup(void)


void HcalRecHitClient::report()
{
  if(!dbe_) return;
  if ( debug_>1 ) std::cout << "HcalRecHitClient: report" << std::endl;
  this->setup();

//   getHistograms();

  return;
} // HcalRecHitClient::report()


void HcalRecHitClient::getHistograms()
{
  if(!dbe_) return;

  stringstream name;
  name<<process_.c_str()<<rootFolder_<<"/RecHitMonitor_Hcal/RecHit Task Event Number";

  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) 
    {
      string s = me->valueString();
      ievt_ = -1;
      sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
      if ( debug_>1 ) std::cout << "Found '" << name.str().c_str() << "'" << std::endl;
    }
  name.str("");

  // Grab individual histograms
  getEtaPhiHists(rootFolder_,"RecHitMonitor_Hcal/rechit_info/","RecHit Occupancy", OccupancyByDepth);
  getEtaPhiHists(rootFolder_,"RecHitMonitor_Hcal/rechit_info_threshold/","Above Threshold RecHit Occupancy", OccupancyThreshByDepth);
  getEtaPhiHists(rootFolder_,"RecHitMonitor_Hcal/rechit_info/sumplots/","RecHit Summed Energy", SumEnergyByDepth, "GeV");
  getEtaPhiHists(rootFolder_,"RecHitMonitor_Hcal/rechit_info/sumplots/","RecHit Summed Energy2", SumEnergy2ByDepth, "GeV");
  getEtaPhiHists(rootFolder_,"RecHitMonitor_Hcal/rechit_info_threshold/sumplots/","Above Threshold RecHit Summed Energy", SumEnergyThreshByDepth, "GeV");
  getEtaPhiHists(rootFolder_,"RecHitMonitor_Hcal/rechit_info/sumplots/","RecHit Summed Time", SumTimeByDepth, "nS");
  getEtaPhiHists(rootFolder_,"RecHitMonitor_Hcal/rechit_info_threshold/sumplots/","Above Threshold RecHit Summed Time", SumTimeThreshByDepth, "nS");
  
  if (rechitclient_makeDiagnostics_)
    {
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hb/HB_energy";
      d_HBEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hb/HB_total_energy";
      d_HBTotalEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hb/HB_time";
      d_HBTime=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hb/HB_occupancy";
      d_HBOccupancy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hb/HB_energy_thresh";
      d_HBThreshEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hb/HB_total_energy_thresh";
      d_HBThreshTotalEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hb/HB_time_thresh";
      d_HBThreshTime=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hb/HB_occupancy_thresh";
      d_HBThreshOccupancy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");

      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/he/HE_energy";
      d_HEEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/he/HE_total_energy";
      d_HETotalEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/he/HE_time";
      d_HETime=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/he/HE_occupancy";
      d_HEOccupancy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/he/HE_energy_thresh";
      d_HEThreshEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/he/HE_total_energy_thresh";
      d_HEThreshTotalEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/he/HE_time_thresh";
      d_HEThreshTime=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/he/HE_occupancy_thresh";
      d_HEThreshOccupancy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");

      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/ho/HO_energy";
      d_HOEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/ho/HO_total_energy";
      d_HOTotalEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/ho/HO_time";
      d_HOTime=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/ho/HO_occupancy";
      d_HOOccupancy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/ho/HO_energy_thresh";
      d_HOThreshEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/ho/HO_total_energy_thresh";
      d_HOThreshTotalEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/ho/HO_time_thresh";
      d_HOThreshTime=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/ho/HO_occupancy_thresh";
      d_HOThreshOccupancy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");

      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hf/HF_energy";
      d_HFEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hf/HF_total_energy";
      d_HFTotalEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hf/HF_time";
      d_HFTime=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hf/HF_occupancy";
      d_HFOccupancy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hf/HF_energy_thresh";
      d_HFThreshEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hf/HF_total_energy_thresh";
      d_HFThreshTotalEnergy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hf/HF_time_thresh";
      d_HFThreshTime=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
      name<<process_.c_str()<<"RecHitMonitor_Hcal/diagnostics/hf/HF_occupancy_thresh";
      d_HFThreshOccupancy=getTH1F(name.str(),process_,rootFolder_,dbe_,debug_,cloneME_);
      name.str("");
    } // if (rechitclient_makeDiagnostics_)

  return;
} //void HcalRecHitClient::getHistograms()


void HcalRecHitClient::analyze(void)
{
  jevt_++;
  if ( jevt_ % 10 == 0 ) 
    {
      if ( debug_>1 ) std::cout << "<HcalRecHitClient::analyze>  Running analyze "<<std::endl;
    }
    
  getHistograms();
  
  meHBEnergy_1D->Reset();
  meHEEnergy_1D->Reset();
  meHOEnergy_1D->Reset();
  meHFEnergy_1D->Reset();
  
  meHBEnergyRMS_1D->Reset();
  meHEEnergyRMS_1D->Reset();
  meHOEnergyRMS_1D->Reset();
  meHFEnergyRMS_1D->Reset();

  // Fill Average Energy, Time plots
  if (ievt_>0)
    {
      for (int mydepth=0;mydepth<4;++mydepth)
        {
          for (int eta=0;eta<OccupancyByDepth[mydepth]->GetNbinsX();++eta)
            {
              for (int phi=0;phi<72;++phi)
                {

                  if (OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1)>0)
                    {

                      if (isHB(eta,mydepth+1)) {
                        if (validDetId(HcalBarrel, CalcIeta(HcalBarrel, eta, mydepth+1), phi+1, mydepth+1)) {
                            meHBEnergy_1D->Fill(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
                            meHBEnergyRMS_1D->Fill(sqrt(pow(SumEnergy2ByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1)-pow(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)));
                        }
                      } else if (isHE(eta,mydepth+1)) {
                        if (validDetId(HcalEndcap, CalcIeta(HcalEndcap, eta, mydepth+1), phi+1, mydepth+1)) {
                            
                            meHEEnergy_1D->Fill(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
                            meHEEnergyRMS_1D->Fill(sqrt(pow(SumEnergy2ByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1)-pow(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)));
                        }
                      } else if (isHO(eta,mydepth+1)) {
                         if (validDetId(HcalOuter, CalcIeta(HcalOuter, eta, mydepth+1), phi+1, mydepth+1)) {
                             meHOEnergy_1D->Fill(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
                             meHOEnergyRMS_1D->Fill(sqrt(pow(SumEnergy2ByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1)-pow(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)));
                         }
                      } else if (isHF(eta,mydepth+1)) {
                         if (validDetId(HcalForward, CalcIeta(HcalForward, eta, mydepth+1), phi+1, mydepth+1)) {
                             meHFEnergy_1D->Fill(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
                             meHFEnergyRMS_1D->Fill(sqrt(pow(SumEnergy2ByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1)-pow(SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1),2)));
                         }
                      }
                      
                      meEnergyByDepth.depth[mydepth]->setBinContent(eta+1, phi+1, SumEnergyByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
                      meTimeByDepth.depth[mydepth]->setBinContent(eta+1, phi+1, SumTimeByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyByDepth[mydepth]->GetBinContent(eta+1, phi+1));
                    }
                    
                  if (OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)>0)
                    {

                      meEnergyThreshByDepth.depth[mydepth]->setBinContent(eta+1, phi+1, SumEnergyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1));
                      meTimeThreshByDepth.depth[mydepth]->setBinContent(eta+1, phi+1, SumTimeThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1)/OccupancyThreshByDepth[mydepth]->GetBinContent(eta+1, phi+1));
                    }
                } // for (int phi=0;phi<72;++phi)
            } // for (int eta=0;eta<OccupancyByDepth...;++eta)
        } // for (int mydepth=0;...)
    } // if (ievt_>0)
  
  return;
} // void HcalRecHitClient::analyze(void)


void HcalRecHitClient::createTests()
{
  // Removed a bunch of code that was in older versions of HcalRecHitClient
  // tests should now be handled from outside
  if(!dbe_) return;
  return;
} // void HcalRecHitClient::createTests()


void HcalRecHitClient::resetAllME()
{
  if(!dbe_) return;
  
  stringstream name;

  // Reset counter?  Is this what we want to do, or do we want to implement a separate counter from the 'overall' one?  This also won't work, since the next call to ievt within HcalMonitor will simply fill with the ievt stored there.  Or will it clear that as well, since evt # is a pointer within the Monitor?
  // We also need the parameters that call resetAllME to also reset the counters used to fill the histograms.  Can we just use a fill command for the histograms and clear the counters when the fill is complete?  
  // Don't seem to be able to reset a counter.  Hmm, investigate further at some point.

  /*
  name<<process_.c_str()<<rootFolder_<<"/RecHitMonitor_Hcal/RecHit Task Event Number";
  resetME(name.str().c_str(),dbe_);
  name.str("");
  */

  // Reset individual histograms
  name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/ ProblemRecHits";
  resetME(name.str().c_str(),dbe_);
  name.str("");

  for (int i=0;i<4;++i)
    {
      // Reset arrays of histograms
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/problem_rechits/"<<subdets_[i]<<"Problem RecHit Rate";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/rechit_info/"<<subdets_[i]<<"RecHit Occupancy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/rechit_info_threshold/"<<subdets_[i]<<"Above Threshold RecHit Occupancy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/rechit_info/"<<subdets_[i]<<"RecHit Average Energy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/rechit_info_threshold/"<<subdets_[i]<<"Above Threshold RecHit Average Energy GeV";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/rechit_info/"<<subdets_[i]<<"RecHit Average Time";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/rechit_info_threshold/"<<subdets_[i]<<"Above Threshold RecHit Average Time nS";
      resetME(name.str().c_str(),dbe_);
      name.str("");
    } // for (int i=0;i<4;++i)

 if (rechitclient_makeDiagnostics_)
    {
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hb/HB_energy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hb/HB_total_energy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hb/HB_time";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hb/HB_occupancy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hb/HB_energy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hb/HB_total_energy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hb/HB_time_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hb/HB_occupancy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/he/HE_energy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/he/HE_total_energy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/he/HE_time";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/he/HE_occupancy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/he/HE_energy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/he/HE_total_energy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/he/HE_time_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/he/HE_occupancy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/ho/HO_energy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/ho/HO_total_energy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/ho/HO_time";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/ho/HO_occupancy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/ho/HO_energy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/ho/HO_total_energy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/ho/HO_time_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/ho/HO_occupancy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");

      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hf/HF_energy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hf/HF_total_energy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hf/HF_time";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hf/HF_occupancy";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hf/HF_energy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hf/HF_total_energy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hf/HF_time_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
      name<<process_.c_str()<<rootFolder_<<"RecHitMonitor_Hcal/diagnostics/hf/HF_occupancy_thresh";
      resetME(name.str().c_str(),dbe_);
      name.str("");
    } // if (rechitclient_makeDiagnostics_)

 return;
} // void HcalRecHitClient::resetAllME()


void HcalRecHitClient::htmlOutput(int runNo, string htmlDir, string htmlName)
{
  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>1) std::cout << "Preparing HcalRecHitClient html output ..." << std::endl;
  //getHistograms(); // not needed here; grabbed in report()
  string client = "RecHitMonitor";

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal RecHit Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal RecHits</span></h2> " << std::endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<h2><strong>Hcal RecHit Status</strong></h2>" << std::endl;


  htmlFile << "<table align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  htmlAnyHisto(runNo,ProblemCells->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
  htmlFile<<"</tr>"<<std::endl;
  htmlFile<<"<tr align=\"center\"><td> There are as yet no criteria for marking a rec hit as 'bad'.  The plots in the link below are for diagnostic purposes only."<<std::endl;
 
  htmlFile<<"</td>"<<std::endl;
  htmlFile<<"</tr></table>"<<std::endl;
  htmlFile<<"<hr><table align=\"center\" border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile<<"<tr><td align=center><a href=\"Expert_"<< htmlName<<"\"><h2>Detailed RecHit Plots</h2> </a></br></td>"<<std::endl;
  htmlFile<<"</tr></table><br><hr>"<<std::endl;
  
  // Now print out problem cells
  htmlFile <<"<br>"<<std::endl;
  htmlFile << "<h2><strong>Hcal Problem RecHits</strong></h2>" << std::endl;
  htmlFile << "(A problem cell is listed below if its failure rate exceeds "<<(100.*minErrorFlag_)<<"%).<br><br>"<<std::endl;
  htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  htmlFile << "<tr align=\"center\">" << std::endl;
  htmlFile <<"<td> Problem RecHits<br>(ieta, iphi, depth)</td><td align=\"center\"> Fraction of Events <br>in which cells are bad (%)</td></tr>"<<std::endl;

  if (ProblemCells==0)
    {
      if (debug_) std::cout <<"<HcalRecHitClient::htmlOutput>  ERROR: can't find Problem RecHit plot!"<<std::endl;
      return;
    }

  int ieta,iphi;

  stringstream name;
  for (int depth=0;depth<4; ++depth)
    {
      for (int eta=0;eta<ProblemCellsByDepth.depth[depth]->getTH2F()->GetNbinsX();++eta)
        {
	  ieta=CalcIeta(eta,depth+1);
          for (int phi=0; phi<ProblemCellsByDepth.depth[depth]->getTH2F()->GetNbinsY();++phi)
            {
	      iphi=phi+1;
	      if (abs(eta)>20 && phi%2!=1) continue;
	      if (abs(eta)>39 && phi%4!=3) continue;
	      
	      if (ProblemCellsByDepth.depth[depth]==0)
		{
		  continue;
		}
	      if (ProblemCellsByDepth.depth[depth]->getTH2F()->GetBinContent(eta+1,phi+1)>minErrorFlag_)
		{
		  if (isHB(eta,depth+1)) name<<"HB";
		  else if (isHE(eta,depth+1)) name<<"HE";
		  else if (isHF(eta,depth+1)) name<<"HF";
		  else if (isHO(eta,depth+1)) name<<"HO";
		  else continue;

		  htmlFile<<"<td>"<<name.str().c_str()<<" ("<<eta<<", "<<phi<<", "<<depth+1<<")</td><td align=\"center\">"<<ProblemCellsByDepth.depth[depth]->getTH2F()->GetBinContent(ieta,iphi)*100.<<"</td></tr>"<<std::endl;

		  name.str("");
		}
	    } // for (int iphi=1;...)
	} // for (int ieta=1;...)
    } // for (int depth=0;...)
  
  
  // html page footer
  htmlFile <<"</table> " << std::endl;
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;

  htmlFile.close();
  htmlExpertOutput(runNo, htmlDir, htmlName);

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalRecHitClient HTMLOUTPUT  -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} //void HcalRecHitClient::htmlOutput(int runNo, ...) 


void HcalRecHitClient::htmlExpertOutput(int runNo, string htmlDir, string htmlName)
{

  if (showTiming_)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (debug_>1) 
    std::cout <<" <HcalRecHitClient::htmlExpertOutput>  Preparing Expert html output ..." <<std::endl;
  
  string client = "RecHitMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_); // does this do anything?

  ofstream htmlFile;
  htmlFile.open((htmlDir +"Expert_"+ htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal RecHit Task output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile <<"<a name=\"EXPERT_RECHIT_TOP\" href = \".\"> Back to Main HCAL DQM Page </a><br>"<<std::endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to RecHit Status Page </a><br>"<<std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << std::endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal RecHits</span></h2> " << std::endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << std::endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << std::endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << std::endl;
  htmlFile << "<hr>" << std::endl;

  htmlFile << "<table width=100%  border = 1>"<<std::endl;
  htmlFile << "<tr><td align=\"center\" colspan=1><a href=\"#OVERALL_PROBLEMS\">PROBLEM CELLS BY DEPTH </a></td></tr>"<<std::endl;
  htmlFile << "<tr><td align=\"center\">"<<std::endl;
  htmlFile<<"<br><a href=\"#OCC_PLOTS\">RecHit Occupancy Plots </a>"<<std::endl;
  htmlFile<<"<br><a href=\"#ENERGY_PLOTS\">RecHit Energy Plots</a>"<<std::endl;
  htmlFile<<"<br><a href=\"#TIME_PLOTS\">RecHit Time Plots </a>"<<std::endl;

  htmlFile << "</td></tr>"<<std::endl;
  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><br>"<<std::endl;


  // Plot overall errors
  htmlFile << "<h2><strong><a name=\"OVERALL_PROBLEMS\">Eta-Phi Maps of Problem Cells By Depth</strong></h2>"<<std::endl;
  htmlFile <<" These plots of problem cells combine results from all rec hit tests<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_RECHIT_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(20,pcol_error_); // set palette to standard error color scheme
  
  // Depths are stored as:  0:  HB/HE/HF depth 1, 1:  HB/HE/HF depth 2, 2:  HE depth 3, 3:  HO
  for (int i=0;i<2;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,ProblemCellsByDepth.depth[2*i]->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,ProblemCellsByDepth.depth[2*i+1]->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }

  htmlFile <<"</table>"<<std::endl;
  htmlFile <<"<br><hr><br>"<<std::endl;
  
  // Occupancy Plots
  htmlFile << "<h2><strong><a name=\"OCC_PLOTS\">Occupancy Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows rechit occupancy of each cell (future version will show average occupancy/per event?)<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_RECHIT_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,OccupancyByDepth[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,OccupancyThreshByDepth[i],"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;
  if (rechitclient_makeDiagnostics_)
    {
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HBOccupancy,"HB Occupancy","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HBThreshOccupancy,"HB Occupancy","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HEOccupancy,"HE Occupancy","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HEThreshOccupancy,"HE Occupancy","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HOOccupancy,"HO Occupancy","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HOThreshOccupancy,"HO Occupancy","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HFOccupancy,"HF Occupancy","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HFThreshOccupancy,"HF Occupancy","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile <<"</table>"<<std::endl;
    }

  htmlFile <<"<br><hr><br>"<<std::endl;

  // energy Plots
  htmlFile << "<h2><strong><a name=\"ENERGY_PLOTS\">Energy Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows average rechit energy of each cell per event<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_RECHIT_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,meEnergyByDepth.depth[i]->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,meEnergyThreshByDepth.depth[i]->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;

  htmlFile <<"<br><hr><br>"<<std::endl;

  // 1D Plots
  htmlFile << "<h2><strong><a name=\"1DENERGY_PLOTS\">1D Energy Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows a 1D distribution of average cell energy<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_RECHIT_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo, meHBEnergy_1D->getTH1F(),"Energy (GeV)","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo, meHBEnergyRMS_1D->getTH1F(),"Energy (GeV)","", 92, htmlFile, htmlDir);
  htmlFile <<"</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo, meHEEnergy_1D->getTH1F(),"Energy (GeV)","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo, meHEEnergyRMS_1D->getTH1F(),"Energy (GeV)","", 92, htmlFile, htmlDir);
  htmlFile <<"</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo, meHOEnergy_1D->getTH1F(),"Energy (GeV)","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo, meHOEnergyRMS_1D->getTH1F(),"Energy (GeV)","", 92, htmlFile, htmlDir);
  htmlFile <<"</tr>"<<std::endl;
  htmlFile << "<tr align=\"left\">" << std::endl;
  htmlAnyHisto(runNo, meHFEnergy_1D->getTH1F(),"Energy (GeV)","", 92, htmlFile, htmlDir);
  htmlAnyHisto(runNo, meHFEnergyRMS_1D->getTH1F(),"Energy (GeV)","", 92, htmlFile, htmlDir);
  htmlFile <<"</tr>"<<std::endl;

  htmlFile <<"</table>"<<std::endl;

  if (rechitclient_makeDiagnostics_)
    {
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HBEnergy,"HB Energy","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HBThreshEnergy,"HB Energy","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HEEnergy,"HE Energy","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HEThreshEnergy,"HE Energy","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HOEnergy,"HO Energy","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HOThreshEnergy,"HO Energy","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HFEnergy,"HF Energy","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HFThreshEnergy,"HF Energy","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile <<"</table>"<<std::endl;
    }

  htmlFile <<"<br><hr><br>"<<std::endl;

  htmlFile << "<h2><strong><a name=\"TIME_PLOTS\">Time Plots</strong></h2>"<<std::endl;
  htmlFile <<"This shows average rechit time of each cell per event<br>"<<std::endl;
  htmlFile <<"<a href= \"#EXPERT_RECHIT_TOP\" > Back to Top</a><br>"<<std::endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  gStyle->SetPalette(1);
  for (int i=0;i<4;++i)
    {
      htmlFile << "<tr align=\"left\">" << std::endl;
      htmlAnyHisto(runNo,meTimeByDepth.depth[i]->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,meTimeThreshByDepth.depth[i]->getTH2F(),"i#eta","i#phi", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
    }
  htmlFile <<"</table>"<<std::endl;
  if (rechitclient_makeDiagnostics_)
    {
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HBTime,"HB Time","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HBThreshTime,"HB Time","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HETime,"HE Time","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HEThreshTime,"HE Time","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HOTime,"HO Time","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HOThreshTime,"HO Time","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile<<"<tr align=\"left\">"<<std::endl;
      htmlAnyHisto(runNo,d_HFTime,"HF Time","", 92, htmlFile, htmlDir);
      htmlAnyHisto(runNo,d_HFThreshTime,"HF Time","", 92, htmlFile, htmlDir);
      htmlFile <<"</tr>"<<std::endl;
      htmlFile <<"</table>"<<std::endl;
    }

  htmlFile <<"<br><hr><br>"<<std::endl;



  htmlFile <<"<br><hr><br><a href= \"#EXPERT_RECHIT_TOP\" > Back to Top of Page </a><br>"<<std::endl;
  htmlFile <<"<a href = \".\"> Back to Main HCAL DQM Page </a><br>"<<std::endl;
  htmlFile <<"<a href= \""<<htmlName.c_str()<<"\" > Back to RecHit Status Page </a><br>"<<std::endl;

  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  
  htmlFile.close();

  if (showTiming_)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalRecHitClient  HTMLEXPERTOUTPUT ->"<<cpu_timer.cpuTime()<<std::endl;
    }
  return;
} // void HcalRecHitClient::htmlExpertOutput(...)



void HcalRecHitClient::loadHistograms(TFile* infile)
{

  return;

} // void HcalRecHitClient::loadHistograms(...)



bool HcalRecHitClient::hasErrors_Temp()
{
  int problemcount=0;
  int etabins=0;
  int phibins=0;

  for (int depth=0;depth<4; ++depth)
    {
      if (ProblemCellsByDepth.depth[depth]==0) continue;
      etabins  = ProblemCellsByDepth.depth[depth]->getTH2F()->GetNbinsX();
      phibins  = ProblemCellsByDepth.depth[depth]->getTH2F()->GetNbinsY();
      for (int ieta=0;ieta<etabins;++ieta)
        {
          for (int iphi=0; iphi<phibins;++iphi)
            {
	      if (ProblemCellsByDepth.depth[depth]->getTH2F()->GetBinContent(ieta+1,iphi+1)>minErrorFlag_)
		problemcount++;
	    } // for (int iphi=0;...)
	} // for (int ieta=0;...)
    } // for (int depth=0;...)

  if (problemcount>=100) return true;
  return false;

} // bool HcalRecHitClient::hasErrors_Temp()


bool HcalRecHitClient::hasWarnings_Temp()
{
  int problemcount=0;
  int etabins=0;
  int phibins=0;

  for (int depth=0;depth<4; ++depth)
    {
      if (ProblemCellsByDepth.depth[depth]==0) continue;
      etabins  = ProblemCellsByDepth.depth[depth]->getTH2F()->GetNbinsX();
      phibins  = ProblemCellsByDepth.depth[depth]->getTH2F()->GetNbinsY();
      for (int ieta=0;ieta<etabins;++ieta)
        {
          for (int iphi=0; iphi<phibins;++iphi)
            {
	      if (ProblemCellsByDepth.depth[depth]->getTH2F()->GetBinContent(ieta+1,iphi+1)>minErrorFlag_)
		problemcount++;
	    } // for (int iphi=0;...)
	} // for (int ieta=0;...)
    } // for (int depth=0;...)

  if (problemcount>0) return true;
  return false;

} // bool HcalRecHitClient::hasWarnings_Temp()

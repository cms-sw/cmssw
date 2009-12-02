#include <DQM/HcalMonitorClient/interface/HcalDetDiagNoiseMonitorClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <math.h>
#include <iostream>

HcalDetDiagNoiseMonitorClient::HcalDetDiagNoiseMonitorClient(){}

void HcalDetDiagNoiseMonitorClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName) {
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);
  return;
}

HcalDetDiagNoiseMonitorClient::~HcalDetDiagNoiseMonitorClient() {
  this->cleanup();
}

void HcalDetDiagNoiseMonitorClient::beginJob() {

  if ( debug_ ) std::cout << "HcalDetDiagNoiseMonitorClient: beginJob" << std::endl;
  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  if (!dbe_) return;

  if(!Online_) {

    stringstream mydir;
    mydir<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring";
    dbe_->setCurrentFolder(mydir.str().c_str());
    std::string title = "MET Threshold vs Rate All Events";
    Met_AllEvents_Rate = dbe_->book1D("MET_Rate_All_Events","MET_Rate_All_Events",200,0,2000);
    Met_AllEvents_Rate->setAxisTitle(title);
    mydir.str("");

    mydir<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/SelectedForNoiseMonitoring/HcalNoiseCategory";
    dbe_->setCurrentFolder(mydir.str().c_str());
    title="MET Threshold vs Rate passing selections & Categorized as 'Hcal Noise'";
    Met_passingTrigger_HcalNoiseCategory_Rate = dbe_->book1D("Hcal_Noise_MET_Rate_pass_selections","Hcal_Noise_MET_Rate_pass_selections",200,0,2000);
    Met_passingTrigger_HcalNoiseCategory_Rate->setAxisTitle(title);
    mydir.str("");

    mydir<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/SelectedForNoiseMonitoring/PhysicsCategory";
    dbe_->setCurrentFolder(mydir.str().c_str());
    title="MET Threshold vs Rate passing selections & Categorized as 'Physics'";
    Met_passingTrigger_PhysicsCategory_Rate = dbe_->book1D("Physics_MET_Rate_pass_selections","Physics_MET_Rate_pass_selections",200,0,2000);
    Met_passingTrigger_PhysicsCategory_Rate->setAxisTitle(title);
    mydir.str("");

    mydir<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/SelectedForNoiseMonitoring";
    dbe_->setCurrentFolder(mydir.str().c_str());
    title = "Jet E_{T} Threshold vs Rate - events passing selections";
    Jets_Et_passing_selections_Rate = dbe_->book1D("Jets_Et_passing_selections_Rate","Jets_Et_passing_selections_Rate",200,0,2000);
    Jets_Et_passing_selections_Rate->setAxisTitle(title);
    mydir.str("");

    mydir<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/SelectedForNoiseMonitoring/HcalNoiseCategory";
    dbe_->setCurrentFolder(mydir.str().c_str());
    title = "'Noise' Jet E_{T} Threshold vs Rate - events passing selections";
    Noise_Jets_Et_passing_selections_Rate = dbe_->book1D("Noise_Jets_Et_passing_selections_Rate","Noise_Jets_Et_passing_selections_Rate",200,0,2000);
    Noise_Jets_Et_passing_selections_Rate->setAxisTitle(title);
    mydir.str("");

  }

  return;
}

void HcalDetDiagNoiseMonitorClient::beginRun(void) {
  if ( debug_ ) std::cout << "HcalDetDiagNoiseMonitorClient: beginRun" << std::endl;
  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
}

void HcalDetDiagNoiseMonitorClient::endJob(void) {
  if ( debug_ ) std::cout << "HcalDetDiagNoiseMonitorClient: endJob, ievt = " << ievt_ << std::endl;
  this->cleanup();
  return;
}


void HcalDetDiagNoiseMonitorClient::endRun(void)  {
  if ( debug_ ) std::cout << "HcalDetDiagNoiseMonitorClient: endRun, jevt = " << jevt_ << std::endl;

  if(!Online_) {

    stringstream name;
    name<<process_.c_str()<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/MET_All_Events";
    MonitorElement* metall = dbe_->get(name.str().c_str());
    if(metall) {
      ievt_ = -1;
      if ( debug_ ) {std::cout << "Found '" << name.str().c_str() << "'" << std::endl;}
    }
    name.str("");

    name<<process_.c_str()<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/SelectedForNoiseMonitoring/HcalNoiseCategory/Hcal_Noise_MET_pass_selections";
    MonitorElement* metnoise = dbe_->get(name.str().c_str());
    if(metnoise) {
      ievt_ = -1;
      if ( debug_ ) {std::cout << "Found '" << name.str().c_str() << "'" << std::endl;}
    }
    name.str("");

    name<<process_.c_str()<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/SelectedForNoiseMonitoring/PhysicsCategory/Physics_MET_pass_selections";
    MonitorElement* metphysics = dbe_->get(name.str().c_str());
    if(metphysics) {
      ievt_ = -1;
      if ( debug_ ) {std::cout << "Found '" << name.str().c_str() << "'" << std::endl;}
    }
    name.str("");

    name<<process_.c_str()<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/NLumiSections";
    MonitorElement* nLS = dbe_->get(name.str().c_str());
    if(nLS) {
      ievt_ = -1;
      if ( debug_ ) {std::cout << "Found '" << name.str().c_str() << "'" << std::endl;}
    }
    name.str("");

    name<<process_.c_str()<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/SelectedForNoiseMonitoring/Jets_Et_passing_selections";
    MonitorElement* jetetall = dbe_->get(name.str().c_str());
    if(jetetall) {
      ievt_ = -1;
      if ( debug_ ) {std::cout << "Found '" << name.str().c_str() << "'" << std::endl;}
    }
    name.str("");

    name<<process_.c_str()<<rootFolder_<<"/HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/SelectedForNoiseMonitoring/HcalNoiseCategory/Noise_Jets_Et_passing_selections";
    MonitorElement* jetetnoise = dbe_->get(name.str().c_str());
    if(jetetnoise) {
      ievt_ = -1;
      if ( debug_ ) {std::cout << "Found '" << name.str().c_str() << "'" << std::endl;}
    }
    name.str("");

    if(nLS->getBinContent(1) > 0) {
      for(int bin1=0; bin1<=201; bin1++) {
        double integral=0.0;
        for(int bin2=bin1; bin2<=201; bin2++) {integral += metall->getBinContent(bin2);}
        Met_AllEvents_Rate->setBinContent(bin1, integral / (nLS->getBinContent(1) * 93.0));
      }
      for(int bin1=0; bin1<=201; bin1++) {
        double integral=0.0;
        for(int bin2=bin1; bin2<=201; bin2++) {integral += metnoise->getBinContent(bin2);}
        Met_passingTrigger_HcalNoiseCategory_Rate->setBinContent(bin1, integral / (nLS->getBinContent(1) * 93.0));
      }
      for(int bin1=0; bin1<=201; bin1++) {
        double integral=0.0;
        for(int bin2=bin1; bin2<=201; bin2++) {integral += metphysics->getBinContent(bin2);}
        Met_passingTrigger_PhysicsCategory_Rate->setBinContent(bin1, integral / (nLS->getBinContent(1) * 93.0));
      }
      for(int bin1=0; bin1<=201; bin1++) {
        double integral=0.0;
        for(int bin2=bin1; bin2<=201; bin2++) {integral += jetetall->getBinContent(bin2);}
        Jets_Et_passing_selections_Rate->setBinContent(bin1, integral / (nLS->getBinContent(1) * 93.0));
      }
      for(int bin1=0; bin1<=201; bin1++) {
        double integral=0.0;
        for(int bin2=bin1; bin2<=201; bin2++) {integral += jetetnoise->getBinContent(bin2);}
        Noise_Jets_Et_passing_selections_Rate->setBinContent(bin1, integral / (nLS->getBinContent(1) * 93.0));
      }
    }

  }

  this->cleanup();
  return;
}

void HcalDetDiagNoiseMonitorClient::setup(void) {
  return;
}

void HcalDetDiagNoiseMonitorClient::cleanup(void) {
  // leave deletions to framework
  if(1<0 && cloneME_) {
    // delete individual histogram pointers      
  }
  dqmReportMapErr_.clear(); 
  dqmReportMapWarn_.clear(); 
  dqmReportMapOther_.clear();
  dqmQtests_.clear();
  return;
}

void HcalDetDiagNoiseMonitorClient::report() {
  if(!dbe_) return;
  if ( debug_ ) std::cout << "HcalDetDiagNoiseMonitorClient: report" << std::endl;
  this->setup();
  return;
}

void HcalDetDiagNoiseMonitorClient::getHistograms() {
  if(!dbe_) return;

  return;
}

void HcalDetDiagNoiseMonitorClient::analyze(void) {
  jevt_++;
  if ( jevt_ % 10 == 0 ) {
    if ( debug_ ) std::cout << "<HcalDetDiagNoiseMonitorClient::analyze>  Running analyze "<<std::endl;
  }
  return;
}

void HcalDetDiagNoiseMonitorClient::createTests() {
  if(!dbe_) return;
  return;
}

void HcalDetDiagNoiseMonitorClient::resetAllME()
{
  if(!dbe_) return;

  // Reset individual histograms  
  if(!Online_) {
    stringstream name;
    name<<process_.c_str()<<rootFolder_<<"HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/MET_Rate_All_Events";
    resetME(name.str().c_str(),dbe_);
    name.str("");
    name<<process_.c_str()<<rootFolder_<<"HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/SelectedForNoiseMonitoring/HcalNoiseCategory/Hcal_Noise_MET_Rate_pass_selections";
    resetME(name.str().c_str(),dbe_);
    name.str("");
    name<<process_.c_str()<<rootFolder_<<"HcalNoiseMonitor/MetExpressStreamNoiseMonitoring/SelectedForNoiseMonitoring/PhysicsCategory/Physics_MET_Rate_pass_selections";
    resetME(name.str().c_str(),dbe_);
    name.str("");
  }

  return;
}

void HcalDetDiagNoiseMonitorClient::htmlOutput(int runNo, string htmlDir, string htmlName) {
  if (showTiming_) {
    cpu_timer.reset(); cpu_timer.start();
  }
  if (showTiming_) {
    cpu_timer.stop();  std::cout <<"TIMER:: HcalDetDiagNoiseMonitorClient HTMLOUTPUT  -> "<<cpu_timer.cpuTime()<<std::endl;
  }
  return;
}

void HcalDetDiagNoiseMonitorClient::htmlExpertOutput(int runNo, string htmlDir, string htmlName) {
  if (showTiming_) {
    cpu_timer.reset(); cpu_timer.start();
  }
  if (showTiming_) {
    cpu_timer.stop();  std::cout <<"TIMER:: HcalDetDiagNoiseMonitorClient  HTMLEXPERTOUTPUT ->"<<cpu_timer.cpuTime()<<std::endl;
  }
  return;
}

void HcalDetDiagNoiseMonitorClient::loadHistograms(TFile* infile) {
  return;
}

#include "DQM/L1TMonitorClient/interface/L1TGMTClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <sstream>

using namespace edm;
using namespace std;

L1TGMTClient::L1TGMTClient(const edm::ParameterSet& ps)
{
  parameters_=ps;
  initialize();
}

L1TGMTClient::~L1TGMTClient(){
  LogInfo("TriggerDQM")<<"[TriggerDQM]: ending... ";
}

//--------------------------------------------------------
void L1TGMTClient::initialize(){ 

  counterLS_=0; 
  counterEvt_=0; 
  
  // get back-end interface
  dbe_ = Service<DQMStore>().operator->();
  
  // base folder for the contents of this job
  monitorName_ = parameters_.getUntrackedParameter<string>("monitorName","");
  cout << "Monitor name = " << monitorName_ << endl;
  prescaleLS_ = parameters_.getUntrackedParameter<int>("prescaleLS", -1);
  cout << "DQM lumi section prescale = " << prescaleLS_ << " lumi section(s)"<< endl;
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  cout << "DQM event prescale = " << prescaleEvt_ << " events(s)"<< endl;
  output_dir_ = parameters_.getUntrackedParameter<string>("output_dir","");
  cout << "DQM output dir = " << output_dir_ << endl;
  input_dir_ = parameters_.getUntrackedParameter<string>("input_dir","");
  cout << "DQM input dir = " << input_dir_ << endl;
  
  LogInfo( "TriggerDQM");

      
}

//--------------------------------------------------------
void L1TGMTClient::beginJob(void){

  LogInfo("TriggerDQM")<<"[TriggerDQM]: Begin Job";

}

//--------------------------------------------------------
void L1TGMTClient::beginRun(const Run& r, const EventSetup& context) {

  // get backendinterface
  dbe_ = Service<DQMStore>().operator->();  

  dbe_->setCurrentFolder(output_dir_);

  // booking
  eff_eta_dtcsc = bookClone1DVB("eff_eta_dtcsc","efficiency DTCSC vs eta","eta_DTCSC_and_RPC");
  eff_eta_dtcsc->setAxisTitle("eta",1);
  eff_eta_dtcsc->getTH1F()->Sumw2();
  
  eff_eta_rpc   = bookClone1DVB("eff_eta_rpc","efficiency RPC vs eta","eta_DTCSC_and_RPC");
  eff_eta_rpc->setAxisTitle("eta",1);
  eff_eta_rpc->getTH1F()->Sumw2();

  
  eff_phi_dtcsc = bookClone1D("eff_phi_dtcsc","efficiency DTCSC vs phi","phi_DTCSC_and_RPC");
  eff_phi_dtcsc->setAxisTitle("phi (deg)",1);
  eff_phi_dtcsc->getTH1F()->Sumw2();
  
  eff_phi_rpc   = bookClone1D("eff_phi_rpc","efficiency RPC vs phi","phi_DTCSC_and_RPC");
  eff_phi_rpc->setAxisTitle("phi (deg)",1);
  eff_phi_rpc->getTH1F()->Sumw2();
  
  
  eff_etaphi_dtcsc = bookClone2D("eff_etaphi_dtcsc","efficiency DTCSC vs eta and phi","etaphi_DTCSC_and_RPC");
  eff_etaphi_dtcsc->setAxisTitle("eta",1);
  eff_etaphi_dtcsc->setAxisTitle("phi (deg)",2);
  eff_etaphi_dtcsc->getTH2F()->Sumw2();
  
  eff_etaphi_rpc   = bookClone2D("eff_etaphi_rpc","efficiency RPC vs eta and phi","etaphi_DTCSC_and_RPC");
  eff_etaphi_rpc->setAxisTitle("eta",1);
  eff_etaphi_rpc->setAxisTitle("phi (deg)",2);
  eff_etaphi_rpc->getTH2F()->Sumw2();
  
}

//--------------------------------------------------------
void L1TGMTClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
   // clientHisto->Reset();
}
//--------------------------------------------------------

void L1TGMTClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c){
  counterLS_++;
  if (prescaleLS_<1) return;
  if (prescaleLS_>0 && counterLS_%prescaleLS_ != 0) return;
//  cout << "L1TGMTClient::endLumi" << endl;

  process();
}             
//--------------------------------------------------------
void L1TGMTClient::analyze(const Event& e, const EventSetup& context){
   counterEvt_++;
   if (prescaleEvt_<1) return;
   if (prescaleEvt_>0 && counterEvt_%prescaleEvt_ != 0) return;
   process();
}

//--------------------------------------------------------
void L1TGMTClient::process() {
  
//  cout << "L1TGMTClient: processing..." << endl;
  
  makeEfficiency1D(eff_eta_dtcsc,"eta_DTCSC_and_RPC","eta_RPC_only");
  makeEfficiency1D(eff_eta_rpc  ,"eta_DTCSC_and_RPC","eta_DTCSC_only");
  
  makeEfficiency1D(eff_phi_dtcsc,"phi_DTCSC_and_RPC","phi_RPC_only");
  makeEfficiency1D(eff_phi_rpc  ,"phi_DTCSC_and_RPC","phi_DTCSC_only");
  
  makeEfficiency2D(eff_etaphi_dtcsc,"etaphi_DTCSC_and_RPC","etaphi_RPC_only");
  makeEfficiency2D(eff_etaphi_rpc  ,"etaphi_DTCSC_and_RPC","etaphi_DTCSC_only");
  
}
//--------------------------------------------------------
void L1TGMTClient::endRun(const Run& r, const EventSetup& context){
  process();
}

//--------------------------------------------------------
void L1TGMTClient::endJob(){
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void L1TGMTClient::makeRatio1D(MonitorElement* mer, string h1Name, string h2Name) {
   TH1F* h1 = get1DHisto(input_dir_+"/"+h1Name,dbe_);
   TH1F* h2 = get1DHisto(input_dir_+"/"+h2Name,dbe_);
   TH1F* hr = mer->getTH1F();
   
   if(hr && h1 && h2) {
     hr->Divide(h1,h2,1.,1.," ");
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void L1TGMTClient::makeEfficiency1D(MonitorElement* meeff, string heName, string hiName) {
   TH1F* he = get1DHisto(input_dir_+"/"+heName,dbe_);
   TH1F* hi = get1DHisto(input_dir_+"/"+hiName,dbe_);
   TH1F* heff = meeff->getTH1F();
   
   if(heff && he && hi) {
     TH1F* hall = (TH1F*) he->Clone("hall");
     hall->Add(hi);
     heff->Divide(he,hall,1.,1.,"B");
     delete hall;
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void L1TGMTClient::makeEfficiency2D(MonitorElement* meeff, string heName, string hiName) {
   TH2F* he = get2DHisto(input_dir_+"/"+heName,dbe_);
   TH2F* hi = get2DHisto(input_dir_+"/"+hiName,dbe_);
   TH2F* heff = meeff->getTH2F();
  
   if(heff && he && hi) {
     TH2F* hall = (TH2F*) he->Clone("hall");
     hall->Add(hi);
     heff->Divide(he,hall,1.,1.,"B");
     delete hall;
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
TH1F* L1TGMTClient::get1DHisto(string meName, DQMStore* dbi) {
  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogInfo("TriggerDQM") << "L1TGMT: " << meName << " NOT FOUND.";
    return NULL;
  }
  return me_->getTH1F();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
TH2F* L1TGMTClient::get2DHisto(string meName, DQMStore* dbi) {
  MonitorElement * me_ = dbi->get(meName);

  if (!me_) { 
    LogInfo("TriggerDQM") << "L1TGMT: " << meName << " NOT FOUND.";
    return NULL;
  }
  return me_->getTH2F();
}
//////////////////////////////////////////////////////////////////////////////////////////////////
MonitorElement* L1TGMTClient::bookClone1D(string name, string title, string hrefName) {
  MonitorElement* me;
  
  TH1F* href = get1DHisto(input_dir_+"/"+hrefName,dbe_);
  if(href) {
    const unsigned nbx = href->GetNbinsX();
    const double xmin = href->GetXaxis()->GetXmin();
    const double xmax = href->GetXaxis()->GetXmax();
    me = dbe_->book1D(name,title,nbx,xmin,xmax);
  } else {
    me = NULL;
  }
  
  return me;
}
//////////////////////////////////////////////////////////////////////////////////////////////////
MonitorElement* L1TGMTClient::bookClone1DVB(string name, string title, string hrefName) {
  MonitorElement* me;
  
  TH1F* href = get1DHisto(input_dir_+"/"+hrefName,dbe_);
  if(href) {
    int nbx = href->GetNbinsX();
    if(nbx>99) nbx=99;
    float xbins[100];
    for(int i=0; i<nbx; i++) {
      xbins[i]=href->GetBinLowEdge(i+1);
    }
    xbins[nbx]=href->GetXaxis()->GetXmax();
    me = dbe_->book1D(name,title,nbx,xbins);
  } else {
    me = NULL;
  }
  
  return me;
}
//////////////////////////////////////////////////////////////////////////////////////////////////
MonitorElement* L1TGMTClient::bookClone2D(string name, string title, string hrefName) {
  MonitorElement* me;
  
  TH2F* href = get2DHisto(input_dir_+"/"+hrefName,dbe_);
  if(href) {
    const unsigned nbx = href->GetNbinsX();
    const double xmin = href->GetXaxis()->GetXmin();
    const double xmax = href->GetXaxis()->GetXmax();
    const unsigned nby = href->GetNbinsY();
    const double ymin = href->GetYaxis()->GetXmin();
    const double ymax = href->GetYaxis()->GetXmax();
    me = dbe_->book2D(name,title,nbx,xmin,xmax,nby,ymin,ymax);
  } else {
    me = NULL;
  }
  
  return me;
}
//////////////////////////////////////////////////////////////////////////////////////////////////




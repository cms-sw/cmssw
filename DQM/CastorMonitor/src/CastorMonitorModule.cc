#include "DQM/CastorMonitor/interface/CastorMonitorModule.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

//**************************************************************//
//***************** CastorMonitorModule       ******************//
//***************** Author: Dmytro Volyanskyy ******************//
//***************** Date  : 22.11.2008 (first version) *********// 
////---- simple event filter which directs events to monitoring tasks: 
////---- access unpacked data from each event and pass them to monitoring tasks 
////---- revision: 06.10.2010 (Dima Volyanskyy)
////---- last revision: 31.05.2011 (Panos Katsas)
////---- LS1 upgrade: 04.06.2013 (Pedro Cipriano)
//**************************************************************//

//---- critical revision 26.06.2014 (Vladimir Popov)

//==================================================================//
//======================= Constructor ==============================//
CastorMonitorModule::CastorMonitorModule(const edm::ParameterSet& ps):
  fVerbosity{ps.getUntrackedParameter<int>("debug", 0)}
{
  if(fVerbosity>0) std::cout<<"CastorMonitorModule Constructor(start)"<<std::endl;

  inputLabelRaw_ 	= ps.getParameter<edm::InputTag>("rawLabel");
  inputLabelReport_     = ps.getParameter<edm::InputTag>("unpackerReportLabel");
  inputLabelDigi_ 	= ps.getParameter<edm::InputTag>("digiLabel");
  inputLabelRecHitCASTOR_  = ps.getParameter<edm::InputTag>("CastorRecHitLabel");
  NBunchesOrbit		= ps.getUntrackedParameter<int>("nBunchesOrbit",3563);
  showTiming_ 		= ps.getUntrackedParameter<bool>("showTiming",false);
//inputLabelCastorTowers_  = ps.getParameter<edm::InputTag>("CastorTowerLabel"); 
//dump2database_   	= ps.getUntrackedParameter<bool>("dump2database",false);

  irun_=0; 
  ilumisec_=0; 
  ievent_=0; 
  itime_=0;
  ibunch_=0;

  DigiMon_ = NULL; 
  RecHitMon_ = NULL;
  LedMon_ = NULL;
 
 //---------------------- DigiMonitor ----------------------// 
  if ( ps.getUntrackedParameter<bool>("DigiMonitor", false) ) {
    if(fVerbosity>0) std::cout << "CastorMonitorModule: Digi monitor flag is on...." << std::endl;
    DigiMon_ = new CastorDigiMonitor(ps);
//    DigiMon_ = new CastorDigiMonitor();
    DigiMon_->setup(ps);
  }

 ////----------- RecHitMonitor ----// 
  if ( ps.getUntrackedParameter<bool>("RecHitMonitor", false) ) {
    if(fVerbosity>0) std::cout << "CastorMonitorModule: RecHit monitor flag is on...." << std::endl;
    RecHitMon_ = new CastorRecHitMonitor(ps);
    RecHitMon_->setup(ps);
  }
////--------- LEDMonitor -----// 
  if ( ps.getUntrackedParameter<bool>("LEDMonitor", false) ) {
    if(fVerbosity>0) std::cout << "CastorMonitorModule: LED monitor flag is on...." << std::endl;
    LedMon_ = new CastorLEDMonitor(ps);
    LedMon_->setup(ps);
  }
 //-------------------------------------------------------------//
  
  std::string subsystemname = ps.getUntrackedParameter<std::string>("subSystemFolder","Castor");
  if(fVerbosity>1) std::cout << "===>CastorMonitor name = " << subsystemname << std::endl;

  ievt_ = 0;
  
  if(fVerbosity>0) std::cout<<"CastorMonitorModule Constructor(end)"<< std::endl;
}

//======================= Destructor ===============================//
CastorMonitorModule::~CastorMonitorModule() { 
  if (DigiMon_ != NULL) { delete DigiMon_; }
  if (RecHitMon_ != NULL) { delete RecHitMon_; }
  if (LedMon_ != NULL) { delete LedMon_; }
}

void CastorMonitorModule::dqmBeginRun(const edm::Run& iRun,
                                      const edm::EventSetup& iSetup) {
  iSetup.get<CastorDbRecord>().get(conditions_);

  ////---- get Castor Pedestal Values from the DB
  iSetup.get<CastorPedestalsRcd>().get(dbPedestals);
  if(!dbPedestals.isValid() && fVerbosity>0) {
    std::cout<<"CASTOR has no CastorPedestals in the CondDB"<<std::endl;
  }
}

//=============== bookHistograms =================//
void CastorMonitorModule::bookHistograms(DQMStore::IBooker& ibooker,
	const edm::Run& iRun, const edm::EventSetup& iSetup)
{
 if(fVerbosity>0) std::cout<<"CastorMonitorModule::beginRun (start)" << std::endl;

 if (DigiMon_ != NULL) { DigiMon_->bookHistograms(ibooker,iRun,iSetup);}
 if (RecHitMon_ != NULL) { RecHitMon_->bookHistograms(ibooker,iRun,iSetup); }
 if (LedMon_ != NULL) { LedMon_->bookHistograms(ibooker,iRun,iSetup); }

//std::cout<<"CastorMonitorModule::bookHist(): CastorCurrentFolder:"<<rootFolder_<<std::endl;
  ibooker.setCurrentFolder(rootFolder_ + "CastorEventProducts"); 
 char s[60];
  sprintf(s,"CastorEventProducts");
    CastorEventProduct = ibooker.book1D(s,s,4,-0.5,3.5);
    CastorEventProduct->getTH1F()->GetXaxis()->SetBinLabel(1,"FEDs/3");
    CastorEventProduct->getTH1F()->GetXaxis()->SetBinLabel(2,"RawData");
    CastorEventProduct->getTH1F()->GetXaxis()->SetBinLabel(3,"CastorDigi");
    CastorEventProduct->getTH1F()->GetXaxis()->SetBinLabel(4,"CastorRecHits");

// reset();
 if(fVerbosity>0) 
 std::cout<<"CastorMonitorModule::bookHistogram(end)"<< std::endl;
 return;
}

void CastorMonitorModule::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					       const edm::EventSetup& context) { }

void CastorMonitorModule::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					     const edm::EventSetup& context) {}

void CastorMonitorModule::endRun(const edm::Run& r, const edm::EventSetup& context)
{}

//========================== analyze  ===============================//
void CastorMonitorModule::analyze(const edm::Event& iEvent, const edm::EventSetup& eventSetup)
{
  if (fVerbosity>0) std::cout <<"  "<<std::endl;
  if (fVerbosity>0)  std::cout <<"CastorMonitorModule::analyze (start)"<<std::endl;

  using namespace edm;

  irun_     = iEvent.id().run();
  ilumisec_ = iEvent.luminosityBlock();
  ievent_   = iEvent.id().event();
  itime_    = iEvent.time().value();
  ibunch_   = iEvent.bunchCrossing() % NBunchesOrbit;

  if (fVerbosity>1) { 
//std::cout << " CastorMonitorModule: evts: "<<nevt_ <<", run: "<<irun_<<", LS: "<<ilumisec_<<std::endl;
  std::cout <<" evt:"<<ievent_<<", time: "<<itime_ <<"\t total count = "<<ievt_<<std::endl; 
  }

  ievt_++;

  bool rawOK_    = true;
  bool digiOK_   = true;
  bool rechitOK_ = true;

  edm::Handle<FEDRawDataCollection> RawData;  
  iEvent.getByLabel(inputLabelRaw_,RawData);
  if (!RawData.isValid()) {
    rawOK_=false;
    if (fVerbosity>0)  std::cout << "RAW DATA NOT FOUND!" << std::endl;
  }
  
  edm::Handle<HcalUnpackerReport> report; 
  iEvent.getByLabel(inputLabelReport_,report);  
  if (!report.isValid()) {
    rawOK_=false;
    if (fVerbosity>0)  std::cout << "UNPACK REPORT HAS FAILED!" << std::endl;
  }
  else 
  {
    const std::vector<int> feds =  (*report).getFedsUnpacked();    
    fedsUnpacked = float(feds.size())/3.;
  }
  
  edm::Handle<CastorDigiCollection> CastorDigi;
  iEvent.getByLabel(inputLabelDigi_,CastorDigi);
  if (!CastorDigi.isValid()) {
    digiOK_=false;
    if (fVerbosity>0)  std::cout << "DIGI DATA NOT FOUND!" << std::endl;
  }
  
  edm::Handle<CastorRecHitCollection> CastorHits;
  iEvent.getByLabel(inputLabelRecHitCASTOR_,CastorHits);
  if (!CastorHits.isValid()) {
    rechitOK_ = false;
    if (fVerbosity>0)  std::cout << "RECO DATA NOT FOUND!" << std::endl;
  }

 CastorEventProduct->Fill(0,int(fedsUnpacked));
 CastorEventProduct->Fill(1,int(rawOK_));
 CastorEventProduct->Fill(2,int(digiOK_));
 CastorEventProduct->Fill(3,int(rechitOK_));

  // if((DigiMon_!=NULL) && (evtMask&DO_CASTOR_PED_CALIBMON) && digiOK_) 
  if(digiOK_) DigiMon_->processEvent(*CastorDigi,*conditions_,ibunch_);
  if (showTiming_){
      cpu_timer.stop();
      if (DigiMon_!=NULL) std::cout <<"TIMER:: DIGI MONITOR ->"<<cpu_timer.cpuTime()<<std::endl;
      cpu_timer.reset(); cpu_timer.start();
    }

 if(rechitOK_) RecHitMon_->processEvent(*CastorHits);
 if (showTiming_){
      cpu_timer.stop();
      if (RecHitMon_!=NULL) std::cout <<"TIMER:: RECHIT MONITOR ->"<<cpu_timer.cpuTime()<<std::endl;
      cpu_timer.reset(); cpu_timer.start();
    }

  if(digiOK_) LedMon_->processEvent(*CastorDigi,*conditions_);
   if (showTiming_){
       cpu_timer.stop();
       if (LedMon_!=NULL) std::cout <<"TIMER:: LED MONITOR ->"<<cpu_timer.cpuTime()<<std::endl;
       cpu_timer.reset(); cpu_timer.start();
     }

 if(fVerbosity>1 && ievt_%100 == 0)
    std::cout << "CastorMonitorModule: processed "<<ievt_<<" events"<<std::endl;
 if (fVerbosity>0)  std::cout <<"CastorMonitorModule::analyze (end)"<<std::endl;
 return;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include <DQM/CastorMonitor/interface/CastorMonitorModule.h>
#include "DQMServices/Core/interface/DQMStore.h"

DEFINE_FWK_MODULE(CastorMonitorModule);

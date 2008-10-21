#include "DQM/SiStripHistoricInfoClient/interface/HistoricOfflineClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <string>
#include "TNamed.h"

//---- default constructor / destructor
HistoricOfflineClient::HistoricOfflineClient(const edm::ParameterSet& iConfig) { dqmStore_ = edm::Service<DQMStore>().operator->(); dqmStore_->setVerbose(0); }
HistoricOfflineClient::~HistoricOfflineClient() {}

//---- called each event
void HistoricOfflineClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
 
  if(firstEventInRun){
    firstEventInRun=false;
    pSummary_->setTimeValue(iEvent.time().value());
  }
  ++nevents;
}

//---- called each BOR
void HistoricOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){
  pSummary_ = new SiStripPerformanceSummary();
  std::cout<<"HistoricOfflineClient::beginRun() nevents = "<<nevents<<std::endl;
  pSummary_->clear(); // just in case
  pSummary_->setRunNr(run.run());
  firstEventInRun=true;
}

//---- called each EOR
void HistoricOfflineClient::endRun(const edm::Run& run , const edm::EventSetup& iSetup){
  firstEventInRun=false;
  retrievePointersToModuleMEs(iSetup);
  fillSummaryObjects(run);
  std::cout<<"HistoricOfflineClient::endRun() nevents = "<<nevents<<std::endl;
  pSummary_->print();
  writeToDB(run);
}

//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::beginJob(const edm::EventSetup&) {
  nevents = 0;
}

//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::endJob() {
  if ( parameters.getUntrackedParameter<bool>("writeHisto", true) ){
    std::string outputfile = parameters.getUntrackedParameter<std::string>("outputFile", "historicOffline.root");
    std::cout<<"HistoricOfflineClient::endJob() outputFile = "<<outputfile<<std::endl;
    dqmStore_->save(outputfile);
  }
}

//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::retrievePointersToModuleMEs(const edm::EventSetup& iSetup) {
  // take from eventSetup the SiStripDetCabling object
  edm::ESHandle<SiStripDetCabling> tkmechstruct;
  iSetup.get<SiStripDetCablingRcd>().get(tkmechstruct);
  // get list of active detectors from SiStripDetCabling - this will change and be taken from a SiStripDetControl object
  std::vector<uint32_t> activeDets;
  activeDets.clear(); // just in case
  tkmechstruct->addActiveDetectorsRawIds(activeDets);
  /// get all MonitorElements tagged as <tag>
  ClientPointersToModuleMEs.clear();
  for(std::vector<uint32_t>::const_iterator idet = activeDets.begin(); idet != activeDets.end(); ++idet){
    std::vector<MonitorElement *> local_mes =  dqmStore_->get(*idet); // get tagged MEs
    ClientPointersToModuleMEs.insert(std::make_pair(*idet, local_mes));
  }
  std::cout<<"HistoricOfflineClient::retrievePointersToModuleMEs() ClientPointersToModuleMEs.size()="<<ClientPointersToModuleMEs.size()<<std::endl;
}

//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::fillSummaryObjects(const edm::Run& run) const {
  std::cout<<"HistoricOfflineClient::fillSummaryObjects() called. ClientPointersToModuleMEs.size()="<<ClientPointersToModuleMEs.size()<<" runnr="<<run.run()<<std::endl;
  for(std::map<uint32_t , std::vector<MonitorElement *> >::const_iterator imapmes = ClientPointersToModuleMEs.begin(); imapmes != ClientPointersToModuleMEs.end(); imapmes++){
     uint32_t local_detid = imapmes->first;
     std::vector<MonitorElement*> locvec = imapmes->second;
//     std::cout<<"HistoricOfflineClient::fillSummaryObjects() detailed. detid="<<local_detid<<" histos.size()="<<locvec.size()<<std::endl;
     for(std::vector<MonitorElement*>::const_iterator imep = locvec.begin(); imep != locvec.end() ; imep++){
        std::string MEName = (*imep)->getName();
      //std::cout<<"HistoricOfflineClient::fillSummaryObjects() //detailed. detid="<<local_detid<<" MEName="<<MEName<<std::endl;
       if( MEName.find("ClusterWidth__") != std::string::npos){  //std::cout<<"ClusterWidth "<<(*imep)->getMean()<<std::endl;
         pSummary_->setClusterSize(local_detid, (*imep)->getMean(), (*imep)->getRMS());
       }
       if( MEName.find("ClusterCharge__") != std::string::npos){ //std::cout<<"ClusterCharge "<<(*imep)->getMean()<<std::endl;
         pSummary_->setClusterCharge(local_detid, (*imep)->getMean(), (*imep)->getRMS());
       }
       if( MEName.find("ModuleLocalOccupancy__") != std::string::npos){ 
         pSummary_->setOccupancy(local_detid, (*imep)->getMean(), (*imep)->getRMS());
         float percover = CalculatePercentOver(*imep);
         if (percover>-198.) pSummary_->setPercentNoisyStrips(local_detid, CalculatePercentOver(*imep)); // set percentage only if sensible value
       }
     }
  }
}

//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::writeToDB(const edm::Run& run) const {
  unsigned int l_run  = run.run();
  std::cout<<"HistoricOfflineClient::writeToDB()  run="<<l_run<<std::endl;
  //now write SiStripPerformanceSummary data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( mydbservice.isAvailable() ){
    if( mydbservice->isNewTagRequest("SiStripPerformanceSummaryRcd") ){
      mydbservice->createNewIOV<SiStripPerformanceSummary>(pSummary_,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripPerformanceSummaryRcd");
    } else {
      mydbservice->appendSinceTime<SiStripPerformanceSummary>(pSummary_,mydbservice->currentTime(),"SiStripPerformanceSummaryRcd");
    }
  }else{
    edm::LogError("writeToDB")<<"Service is unavailable"<<std::endl;
  }
}

//-----------------------------------------------------------------------------------------------
void HistoricOfflineClient::writeToDB(edm::EventID evid, edm::Timestamp evtime) const {
  unsigned int l_run        = evid.run();
  unsigned int l_event      = evid.event();
  unsigned long long l_tval = evtime.value();
  std::cout<<"HistoricOfflineClient::writeToDB()  run="<<l_run<<" event="<<l_event<<" time="<<l_tval<<std::endl;
  //now write SiStripPerformanceSummary data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( mydbservice.isAvailable() ){
    if( mydbservice->isNewTagRequest("SiStripPerformanceSummaryRcd") ){
      mydbservice->createNewIOV<SiStripPerformanceSummary>(pSummary_,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripPerformanceSummaryRcd");      
    } else {
      mydbservice->appendSinceTime<SiStripPerformanceSummary>(pSummary_,mydbservice->currentTime(),"SiStripPerformanceSummaryRcd");
    }
  }else{
    edm::LogError("writeToDB")<<"Service is unavailable"<<std::endl;
  }
}

//-----------------------------------------------------------------------------------------------
float HistoricOfflineClient::CalculatePercentOver(MonitorElement * me) const{
  if (me->kind() == MonitorElement::DQM_KIND_TH1F) {
    TH1F * root_ob = me->getTH1F();
    if(root_ob){
      float percsum=0.;
      TAxis * ta = root_ob->GetXaxis();
      unsigned int maxbins  = ta->GetNbins();
      unsigned int upperbin = root_ob->FindBin(root_ob->GetMean()+3.*root_ob->GetRMS()); // bin where +3 RMS from mean ends
      if(upperbin<=maxbins){
         percsum = root_ob->Integral(upperbin,maxbins) / root_ob->Integral();
         return percsum;
      }
    }
  }
  return -199.; // nonsense value
}

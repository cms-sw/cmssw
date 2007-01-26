#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
#include "IOVPayloadEndOfJob.h"
IOVPayloadEndOfJob::IOVPayloadEndOfJob(const edm::ParameterSet& iConfig ):
  m_record(iConfig.getParameter< std::string >("record")){
  std::cout<<"IOVPayloadEndOfJob::IOVPayloadEndOfJob"<<std::endl;
}
IOVPayloadEndOfJob::~IOVPayloadEndOfJob(){
  std::cout<<"IOVPayloadEndOfJob::~IOVPayloadEndOfJob"<<std::endl;
}
void IOVPayloadEndOfJob::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup){
  //
}
void IOVPayloadEndOfJob::endJob(){ 
  std::cout<<"IOVPayloadEndOfJob::endJob "<<std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"Service is unavailable"<<std::endl;
    return;
  }
  try{
    std::string tag=mydbservice->tag(m_record);
    Pedestals* myped=new Pedestals;
    for(int ichannel=1; ichannel<=5; ++ichannel){
      Pedestals::Item item;
      item.m_mean=1.11*ichannel;
      item.m_variance=1.12*ichannel;
      myped->m_pedestals.push_back(item);
    }
    if( mydbservice->isNewTagRequest(m_record) ){
      //create 
      cond::Time_t firstTillTime=mydbservice->endOfTime();
      mydbservice->createNewIOV<Pedestals>(myped,firstTillTime,m_record);
    }else{
      //append 
      cond::Time_t thisPayload_valid_since=5;
      mydbservice->appendSinceTime<Pedestals>(myped,thisPayload_valid_since,m_record);
    }
    std::cout<<myped->m_pedestals[1].m_mean<<std::endl;
  }catch(const cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"caught std::exception "<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Unknown error"<<std::endl;
  }
}
DEFINE_FWK_MODULE(IOVPayloadEndOfJob);

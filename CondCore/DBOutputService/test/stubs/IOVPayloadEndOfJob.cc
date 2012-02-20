#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondFormats/Calibration/interface/Pedestals.h"

#include "CondFormats/Common/interface/GenericSummary.h"

#include "IOVPayloadEndOfJob.h"
#include <cstdlib>

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
    if( mydbservice->isNewTagRequest(m_record) ){
      for(int ichannel=1; ichannel<=5; ++ichannel){
	Pedestals::Item item;
	item.m_mean=1.11*ichannel;
	item.m_variance=1.12*ichannel;
	myped->m_pedestals.push_back(item);
      }
      //create 
      cond::Time_t firstSinceTime=mydbservice->beginOfTime();
       std::cout<<"firstSinceTime is begin of time "<<firstSinceTime<<std::endl;
       mydbservice->writeOne(myped,new cond::GenericSummary("first"),firstSinceTime,m_record);
    }else{
      //append 
      cond::Time_t current=mydbservice->currentTime();
      std::cout<<"current time"<<current<<std::endl;
      if(current>=5){
	std::cout<<"appending payload"<<std::endl;
	for(int ichannel=1; ichannel<=5; ++ichannel){
	  Pedestals::Item item;
	  item.m_mean=0.15*ichannel;
	  item.m_variance=0.32*ichannel;
	  myped->m_pedestals.push_back(item);
	}
	cond::Time_t thisPayload_valid_since=current;
	std::cout<<"appeding since time "<<thisPayload_valid_since<<std::endl;
	mydbservice->writeOne(myped,new cond::GenericSummary("second"),thisPayload_valid_since,m_record);
	std::cout<<"done"<<std::endl;
	//std::cout<<myped->m_pedestals[1].m_mean<<std::endl;
      }
    }
  }catch(const cond::Exception& er){
    throw cms::Exception("DataBaseUnitTestFailure","failed IOVPayloadEndOfJob",er);
    //std::cout<<er.what()<<std::endl;
  }catch(const cms::Exception& er){
    throw cms::Exception("DataBaseUnitTestFailure","failed IOVPayloadEndOfJob",er);
  }/*catch(const std::exception& er){
    std::cout<<"caught std::exception "<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Unknown error"<<std::endl;
    }*/
}
DEFINE_FWK_MODULE(IOVPayloadEndOfJob);

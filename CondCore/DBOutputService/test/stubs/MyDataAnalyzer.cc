#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
#include "MyDataAnalyzer.h"
#include <cstdlib>
MyDataAnalyzer::MyDataAnalyzer(const edm::ParameterSet& iConfig ):
  m_record(iConfig.getParameter< std::string >("record")),
  m_LoggingOn(false){
  m_LoggingOn=iConfig.getUntrackedParameter< bool >("loggingOn");
  std::cout<<"MyDataAnalyzer::MyDataAnalyzer"<<std::endl;
}
MyDataAnalyzer::~MyDataAnalyzer(){
  std::cout<<"MyDataAnalyzer::~MyDataAnalyzer"<<std::endl;
}
void MyDataAnalyzer::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup){
  //
}
void MyDataAnalyzer::endJob(){ 
  std::cout<<"MyDataAnalyzer::endJob "<<std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"Service is unavailable"<<std::endl;
    return;
  }
  try{
    mydbservice->setLogHeaderForRecord(m_record,"mynullsource","this is zhen's dummy test");
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
      cond::Time_t firstTillTime=mydbservice->endOfTime();
      std::cout<<"firstTillTime is end of time "<<firstTillTime<<std::endl;
      mydbservice->createNewIOV<Pedestals>(myped,firstTillTime,m_record,m_LoggingOn);
    }else{
      //append 
      std::cout<<"appending payload"<<std::endl;
      for(int ichannel=1; ichannel<=5; ++ichannel){
	Pedestals::Item item;
	item.m_mean=0.15*ichannel;
	item.m_variance=0.32*ichannel;
	myped->m_pedestals.push_back(item);
      }
      cond::Time_t thisPayload_valid_since=5;
      std::cout<<"appeding since time "<<thisPayload_valid_since<<std::endl;
      mydbservice->appendSinceTime<Pedestals>(myped,thisPayload_valid_since,m_record,m_LoggingOn);
      std::cout<<"done"<<std::endl;
    }
  }catch(const cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"caught std::exception "<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Unknown error"<<std::endl;
  }
}
DEFINE_FWK_MODULE(MyDataAnalyzer);

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/Common/interface/GenericSummary.h"

#include "Timestamp.h"

Timestamp::Timestamp(const edm::ParameterSet& iConfig ):
  m_record(iConfig.getParameter< std::string >("record")){
  std::cout<<"Timestamp::Timestamp"<<std::endl;
}
Timestamp::~Timestamp(){
  std::cout<<"Timestamp::~Timestamp"<<std::endl;
}
void Timestamp::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup){
  std::cout<<"Timestamp::analyze "<<std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"Service is unavailable"<<std::endl;
    return;
  }
  cond::Time_t itime=(cond::Time_t)evt.time().value();
  std::string tag=mydbservice->tag(m_record);
  std::cout<<"tag "<<tag<<std::endl;
  std::cout<<"time "<<itime<<std::endl;
  Pedestals* myped=new Pedestals;
  for(int ichannel=1; ichannel<=5; ++ichannel){
    Pedestals::Item item;
    item.m_mean=1.11*ichannel+itime;
    item.m_variance=1.12*ichannel+itime;
    myped->m_pedestals.push_back(item);
    }
  std::cout<<myped->m_pedestals[1].m_mean<<std::endl;
  std::cout<<"currentTime "<<mydbservice->currentTime()<<std::endl;
  if(mydbservice->currentTime()%5==0){
    mydbservice->writeOne(myped,new cond::GenericSummary("5"),
			  mydbservice->currentTime(),m_record,false);
  }
}
void Timestamp::endJob(){ 
  
}
DEFINE_FWK_MODULE(Timestamp);

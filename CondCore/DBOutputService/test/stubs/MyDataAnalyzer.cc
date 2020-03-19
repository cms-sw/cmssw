#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
//#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/CondDB/interface/Exception.h"
//#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondFormats/Calibration/interface/Pedestals.h"

#include "MyDataAnalyzer.h"
#include <cstdlib>
#include <iostream>
MyDataAnalyzer::MyDataAnalyzer(const edm::ParameterSet& iConfig)
    : m_record(iConfig.getParameter<std::string>("record")), m_LoggingOn(false) {
  m_LoggingOn = iConfig.getUntrackedParameter<bool>("loggingOn");
  std::cout << "MyDataAnalyzer::MyDataAnalyzer" << std::endl;
}
MyDataAnalyzer::~MyDataAnalyzer() { std::cout << "MyDataAnalyzer::~MyDataAnalyzer" << std::endl; }
void MyDataAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  //
}
void MyDataAnalyzer::endJob() {
  std::cout << "MyDataAnalyzer::endJob " << std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    std::cout << "Service is unavailable" << std::endl;
    return;
  }
  try {
    std::string tag = mydbservice->tag(m_record);
    Pedestals* myped = new Pedestals;
    if (mydbservice->isNewTagRequest(m_record)) {
      for (int ichannel = 1; ichannel <= 5; ++ichannel) {
        Pedestals::Item item;
        item.m_mean = 1.11 * ichannel;
        item.m_variance = 1.12 * ichannel;
        myped->m_pedestals.push_back(item);
      }
      //create
      cond::Time_t firstTillTime = mydbservice->endOfTime();
      cond::Time_t firstSinceTime = mydbservice->beginOfTime();
      std::cout << "firstSinceTime is begin of time " << firstSinceTime << std::endl;
      std::cout << "firstTillTime is end of time " << firstTillTime << std::endl;
      mydbservice->writeOne(myped, firstSinceTime, m_record, m_LoggingOn);
    } else {
      //append
      std::cout << "appending payload" << std::endl;
      for (int ichannel = 1; ichannel <= 5; ++ichannel) {
        Pedestals::Item item;
        item.m_mean = 0.15 * ichannel;
        item.m_variance = 0.32 * ichannel;
        myped->m_pedestals.push_back(item);
      }
      cond::Time_t thisPayload_valid_since = 5;
      std::cout << "appeding since time " << thisPayload_valid_since << std::endl;
      mydbservice->writeOne(myped, thisPayload_valid_since, m_record, m_LoggingOn);
      std::cout << "done" << std::endl;
    }
    //example for log reading
    /**
    const cond::Logger& logreader=mydbservice->queryLog();
    cond::LogDBEntry result;
    logreader.LookupLastEntryByProvenance("mynullsource",result);
    std::cout<<"Here's my last log entry \n";
    std::cout<<"logId "<<result.logId<<"\n";
    std::cout<<"destinationDB "<<result.destinationDB<<"\n";
    std::cout<<"provenance "<<result.provenance<<"\n";
    std::cout<<"usertext "<<result.usertext<<"\n";
    std::cout<<"iovtag "<<result.iovtag<<"\n";
    std::cout<<"iovtimetype "<<result.iovtimetype<<"\n";
    std::cout<<"payloadIdx "<<result.payloadIdx<<"\n";
    std::cout<<"payloadClass "<<result.payloadClass<<"\n";
    std::cout<<"payloadToken "<<result.payloadToken<<"\n";
    std::cout<<"exectime "<<result.exectime<<"\n";
    std::cout<<"execmessage "<<result.execmessage<<std::endl;
    //example for retrieving tag info
    cond::TagInfo taginfo;
    mydbservice->tagInfo(m_record,taginfo);
    std::cout<<"Here's the tag info for "<<m_record<<"\n";
    std::cout<<"tag name "<<taginfo.name<<"\n";
    std::cout<<"iov token "<<taginfo.token<<"\n";
    std::cout<<"last since "<<taginfo.lastInterval.first<<"\n";
    std::cout<<"last till "<<taginfo.lastInterval.second<<std::endl;
    std::cout<<"last payload "<<taginfo.lastPayloadToken<<std::endl;
    //example for instantiate object by token
    cond::DbSession pooldb=mydbservice->session();
    pooldb.transaction().start(true);
    std::shared_ptr<Pedestals> myinstance = pooldb.getTypedObject<Pedestals>( taginfo.lastPayloadToken);
    std::cout<<"size of items "<<myinstance->m_pedestals.size()<<std::endl;
    pooldb.transaction().commit();    
    **/
  } catch (const cond::Exception& er) {
    throw cms::Exception("DBOutputServiceUnitTestFailure", "failed MyDataAnalyzer", er);
    //std::cout<<er.what()<<std::endl;
  } catch (const cms::Exception& er) {
    throw cms::Exception("DBOutputServiceUnitTestFailure", "failed MyDataAnalyzer", er);
  } /*catch(const std::exception& er){
    std::cout<<"caught std::exception "<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Unknown error"<<std::endl;
    }*/
}
DEFINE_FWK_MODULE(MyDataAnalyzer);

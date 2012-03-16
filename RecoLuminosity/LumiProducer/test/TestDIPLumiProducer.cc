#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/NoDataException.h"
#include "FWCore/Framework/interface/NoRecordException.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "RecoLuminosity/LumiProducer/interface/DIPLumiSummary.h"
#include "RecoLuminosity/LumiProducer/interface/DIPLumiSummaryRcd.h"

#include <iostream>


namespace edm {
  class EventSetup;
}

using namespace std;
using namespace edm;

class TestDIPLumiProducer : public edm::EDAnalyzer{
public:
  
  explicit TestDIPLumiProducer(edm::ParameterSet const&);
  virtual ~TestDIPLumiProducer();
  
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c);
};

// -----------------------------------------------------------------

TestDIPLumiProducer::TestDIPLumiProducer(edm::ParameterSet const& ps)
{
}

// -----------------------------------------------------------------

TestDIPLumiProducer::~TestDIPLumiProducer()
{
}

// -----------------------------------------------------------------

void TestDIPLumiProducer::analyze(edm::Event const& e,edm::EventSetup const&)
{
}

// -----------------------------------------------------------------

void TestDIPLumiProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, EventSetup const& es){
  std::cout <<" I AM IN RUN NUMBER "<<lumiBlock.run() <<std::endl;
  std::cout <<" ---LS NUMBER "<< lumiBlock.luminosityBlock()<<std::endl;
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DIPLumiSummaryRcd"));
  if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    std::cout <<"Record \"DIPLumiSummaryRcd"<<"\" does not exist "<<std::endl;
  }
  try{
    edm::ESHandle<DIPLumiSummary> datahandle;
    unsigned long long cache_id=es.get<DIPLumiSummaryRcd>().cacheIdentifier();
    std::cout<<cache_id<<std::endl;
    es.get<DIPLumiSummaryRcd>().get(datahandle);
    if(datahandle.isValid()){
      const DIPLumiSummary* mydata=datahandle.product();
      std::cout<<*mydata<<std::endl;
    }else{
      std::cout<<"no valid data found"<<std::endl;
    }
  }catch(const edm::eventsetup::NoRecordException<DIPLumiSummaryRcd>& er){
    std::cout<<"no data found"<<std::endl;
  }
}

DEFINE_FWK_MODULE(TestDIPLumiProducer);

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
#include "RecoLuminosity/LumiProducer/interface/DIPLumiDetail.h"
#include "RecoLuminosity/LumiProducer/interface/DIPLuminosityRcd.h"

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
  std::cout <<" I AM IN RUN NUMBER "<<lumiBlock.run() <<" LS NUMBER "<< lumiBlock.luminosityBlock()<<std::endl;
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DIPLuminosityRcd"));
  if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    std::cout <<"Record \"DIPLuminosityRcd"<<"\" does not exist "<<std::endl;
  }
  try{
    edm::ESHandle<DIPLumiSummary> datahandle;
    es.getData(datahandle);
    if(datahandle.isValid()){
      const DIPLumiSummary* mydata=datahandle.product();
      if(!mydata->isNull()){
	std::cout<<mydata<<std::endl;
      }else{
	std::cout<<"data empty"<<std::endl;
      }
    }else{
      std::cout<<"no valid record found"<<std::endl;
    }
  }catch(const edm::eventsetup::NoRecordException<DIPLuminosityRcd>& er){
    std::cout<<"no data found"<<std::endl;
  }catch(const cms::Exception& ee){
    std::cout<<ee.what()<<std::endl;
  }
  //try{
  //  std::cout<<"looking at detail"<<std::endl;
  //  edm::ESHandle<DIPLumiDetail> pDetail;
  //  es.getData(pDetail);
  // if(pDetail.isValid()){
  //    const DIPLumiDetail* ddata=pDetail.product();
  //    if(!mydata->isNull()){
  //      std::cout<<*ddata<<std::endl;
  //    }else{
  //	  std::cout<<"data empty"<<std::endl;
  //    }
  //  }else{
  //    std::cout<<"no valid data found"<<std::endl;
  //  }
  //}catch(const edm::eventsetup::NoRecordException<DIPLuminosityRcd>& er){
  //  std::cout<<"no data found"<<std::endl;
  //}
}

DEFINE_FWK_MODULE(TestDIPLumiProducer);

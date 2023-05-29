#include "FWCore/Framework/interface/one/EDAnalyzer.h"
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

class TestDIPLumiProducer : public edm::one::EDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  explicit TestDIPLumiProducer(edm::ParameterSet const&);

  void beginLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) override {}
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) override;

private:
  edm::ESGetToken<DIPLumiSummary, DIPLuminosityRcd> token_;
};

// -----------------------------------------------------------------

TestDIPLumiProducer::TestDIPLumiProducer(edm::ParameterSet const& ps) : token_(esConsumes()) {}

// -----------------------------------------------------------------

void TestDIPLumiProducer::analyze(edm::Event const& e, edm::EventSetup const&) {}

// -----------------------------------------------------------------

void TestDIPLumiProducer::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, EventSetup const& es) {
  std::cout << " I AM IN RUN NUMBER " << lumiBlock.run() << " LS NUMBER " << lumiBlock.luminosityBlock() << std::endl;
  edm::eventsetup::EventSetupRecordKey recordKey(
      edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DIPLuminosityRcd"));
  if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    std::cout << "Record \"DIPLuminosityRcd"
              << "\" does not exist " << std::endl;
  }
  try {
    edm::ESHandle<DIPLumiSummary> datahandle = es.getHandle(token_);
    if (datahandle.isValid()) {
      const DIPLumiSummary* mydata = datahandle.product();
      if (!mydata->isNull()) {
        std::cout << "from Run " << mydata->fromRun() << " from LS " << mydata->fromLS() << std::endl;
        std::cout << mydata->intgDelLumiByLS() << std::endl;
      } else {
        std::cout << "data empty" << std::endl;
      }
    } else {
      std::cout << "no valid record found" << std::endl;
    }
  } catch (const edm::eventsetup::NoRecordException<DIPLuminosityRcd>& er) {
    std::cout << "no data found" << std::endl;
  } catch (const cms::Exception& ee) {
    std::cout << ee.what() << std::endl;
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

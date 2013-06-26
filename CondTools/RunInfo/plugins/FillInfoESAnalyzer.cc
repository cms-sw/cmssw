#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/RunInfo/interface/FillInfo.h"
#include "CondFormats/DataRecord/interface/FillInfoRcd.h"

namespace edmtest
{
  class FillInfoESAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  FillInfoESAnalyzer(edm::ParameterSet const& p) { 
      std::cout<<"FillInfoESAnalyzer"<<std::endl;
    }
    explicit  FillInfoESAnalyzer(int i) {
      std::cout<<"FillInfoESAnalyzer "<<i<<std::endl; 
    }
    virtual ~FillInfoESAnalyzer() {  
      std::cout<<"~FillInfoESAnalyzer "<<std::endl;
    }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  };
   
  void
   FillInfoESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context){
    std::cout<<"###FillInfoESAnalyzer::analyze"<<std::endl;
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("FillInfoRcd"));
    if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout <<"Record \"FillInfoRcd"<<"\" does not exist "<<std::endl;
    }
    edm::ESHandle<FillInfo> sum;
    std::cout<<"got eshandle"<<std::endl;
    context.get<FillInfoRcd>().get(sum);
    std::cout<<"got context"<<std::endl;
    const FillInfo* summary=sum.product();
    std::cout<<"got FillInfo* "<< std::endl;
    std::cout<< "print  result" << std::endl;
    std::cout << *summary;
  }
  DEFINE_FWK_MODULE(FillInfoESAnalyzer);
}


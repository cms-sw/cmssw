#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/RunInfo/interface/RunSummary.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

using namespace std;





namespace edmtest
{
  class RunSummaryESAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  RunSummaryESAnalyzer(edm::ParameterSet const& p) 
    { 
      std::cout<<"RunSummaryESAnalyzer"<<std::endl;
    }
    explicit  RunSummaryESAnalyzer(int i) 
    { std::cout<<"RunSummaryESAnalyzer "<<i<<std::endl; }
    ~RunSummaryESAnalyzer() override {  
      std::cout<<"~RunSummaryESAnalyzer "<<std::endl;
    }
    // virtual void beginJob();
    // virtual void beginRun(const edm::Run&, const edm::EventSetup& context);
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  private:
  };
   
  
  /* void
  RunSummaryESAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& context){
    std::cout<<"###RunSummaryESAnalyzer::beginRun"<<std::endl;
    edm::ESHandle<RunSummary> RunSummary_lumiarray;
    std::cout<<"got eshandle"<<std::endl;
    context.get<RunSummaryRcd>().get(RunSummary_lumiarray);
    std::cout<<"got data"<<std::endl;
  }
  
  void
  RunSummaryESAnalyzer::beginJob(){
    std::cout<<"###RunSummaryESAnalyzer::beginJob"<<std::endl;
   
  }
 
  */
  void
   RunSummaryESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context){
    using namespace edm::eventsetup;
    std::cout<<"###RunSummaryESAnalyzer::analyze"<<std::endl;
     
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunSummaryRcd"));
    if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout <<"Record \"RunSummaryRcd"<<"\" does not exist "<<std::endl;
    }
    edm::ESHandle<RunSummary> sum;
    std::cout<<"got eshandle"<<std::endl;
    context.get<RunSummaryRcd>().get(sum);
    std::cout<<"got context"<<std::endl;
    const RunSummary* summary=sum.product();
    std::cout<<"got RunSummary* "<< std::endl;

  
    std::cout<< "print  result" << std::endl;
    summary->printAllValues();
    std::vector<std::string> subdet = summary->getSubdtIn();
    std::cout<<"subdetector in the run "<< std::endl;
    for (size_t i=0; i<subdet.size(); i++){
        std::cout<<"--> " << subdet[i] << std::endl;
    }
  }
  DEFINE_FWK_MODULE(RunSummaryESAnalyzer);
}



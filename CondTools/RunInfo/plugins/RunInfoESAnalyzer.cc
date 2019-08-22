#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

using namespace std;

namespace edmtest {
  class RunInfoESAnalyzer : public edm::EDAnalyzer {
  public:
    explicit RunInfoESAnalyzer(edm::ParameterSet const& p) { std::cout << "RunInfoESAnalyzer" << std::endl; }
    explicit RunInfoESAnalyzer(int i) { std::cout << "RunInfoESAnalyzer " << i << std::endl; }
    ~RunInfoESAnalyzer() override { std::cout << "~RunInfoESAnalyzer " << std::endl; }
    //     virtual void beginJob();
    //  virtual void beginRun(const edm::Run&, const edm::EventSetup& context);
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  /* void
  RunInfoESAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& context){
    std::cout<<"###RunInfoESAnalyzer::beginRun"<<std::endl;
    edm::ESHandle<RunInfo> RunInfo_lumiarray;
    std::cout<<"got eshandle"<<std::endl;
    context.get<RunInfoRcd>().get(RunInfo_lumiarray);
    std::cout<<"got data"<<std::endl;
  }
  
  void
  RunInfoESAnalyzer::beginJob(){
    std::cout<<"###RunInfoESAnalyzer::beginJob"<<std::endl;
   
  }
 
  */
  void RunInfoESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    std::cout << "###RunInfoESAnalyzer::analyze" << std::endl;

    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout << "Record \"RunInfoRcd"
                << "\" does not exist " << std::endl;
    }
    edm::ESHandle<RunInfo> sum;
    std::cout << "got eshandle" << std::endl;
    context.get<RunInfoRcd>().get(sum);
    std::cout << "got context" << std::endl;
    const RunInfo* summary = sum.product();
    std::cout << "got RunInfo* " << std::endl;

    std::cout << "print  result" << std::endl;
    summary->printAllValues();
    /*  std::vector<std::string> subdet = summary->getSubdtIn();
    std::cout<<"subdetector in the run "<< std::endl;
    for (size_t i=0; i<subdet.size(); i++){
        std::cout<<"--> " << subdet[i] << std::endl;
    }
    */
  }
  DEFINE_FWK_MODULE(RunInfoESAnalyzer);
}  // namespace edmtest

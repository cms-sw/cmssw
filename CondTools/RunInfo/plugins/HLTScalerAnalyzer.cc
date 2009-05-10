#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/RunInfo/interface/HLTScaler.h"
#include "CondFormats/DataRecord/interface/HLTScalerRcd.h"

using namespace std;

namespace edmtest
{
  class HLTScalerAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  HLTScalerAnalyzer(edm::ParameterSet const& p) 
    { 
      std::cout<<"HLTScalerAnalyzer"<<std::endl;
    }
    explicit  HLTScalerAnalyzer(int i) 
    { std::cout<<"HLTScalerAnalyzer "<<i<<std::endl; }
    virtual ~HLTScalerAnalyzer() {  
      std::cout<<"~HLTScalerAnalyzer "<<std::endl;
    }
    virtual void beginJob(const edm::EventSetup& context);
    virtual void beginRun(const edm::Run&, const edm::EventSetup& context);
    virtual void beginLuminosityBlock(edm::LuminosityBlock const& iLBlock,
				      edm::EventSetup const& context);
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
  };
  void
  HLTScalerAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& context){
    /*std::cout<<"###LuminosityInfoAnalyzer::beginRun"<<std::endl;
    edm::ESHandle<lumi::LuminosityInfo> pPeds;
    std::cout<<"got eshandle"<<std::endl;
    context.get<LuminosityInfoRcd>().get(pPeds);
    */
  }
  void
  HLTScalerAnalyzer::beginJob(const edm::EventSetup& context){
    std::cout<<"###HLTScalerAnalyzer::beginJob"<<std::endl;
  }
  void
  HLTScalerAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& iLBlock,
					       edm::EventSetup const& context){
    std::cout<<"###HLTScalerAnalyzer::beginLuminosityBlock"<<std::endl;
    using namespace edm::eventsetup;
    // Context is not used.
    //std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    //    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    std::cout<<"lumiblock id value "<<iLBlock.id().value()<<std::endl;
    std::cout<<"lumiblock in run "<<iLBlock.run()<<std::endl;
    std::cout<<"lumiblock number "<<iLBlock.id().luminosityBlock()<<std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("HLTScalerRcd"));
    if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout <<"Record \"HLTScalerRcd"<<"\" does not exist "<<std::endl;
    }
    edm::ESHandle<lumi::HLTScaler> pPeds;
    context.get<HLTScalerRcd>().get(pPeds);
    const lumi::HLTScaler* myped=pPeds.product();
    
  }

  void
  HLTScalerAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context){
  }
  DEFINE_FWK_MODULE(HLTScalerAnalyzer);
}

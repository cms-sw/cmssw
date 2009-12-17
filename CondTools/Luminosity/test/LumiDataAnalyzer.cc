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
#include "CondFormats/Luminosity/interface/LumiSectionData.h"
#include "CondFormats/DataRecord/interface/LumiSectionDataRcd.h"

using namespace std;

namespace edmtest
{
  class LumiDataAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  LumiDataAnalyzer(edm::ParameterSet const& p) 
    { 
      //std::cout<<"LumiDataAnalyzer"<<std::endl;
    }
    explicit  LumiDataAnalyzer(int i) 
    { std::cout<<"LumiDataAnalyzer "<<i<<std::endl; }
    virtual ~LumiDataAnalyzer() {  
      //std::cout<<"~LumiDataAnalyzer "<<std::endl;
    }
    virtual void beginJob();
    virtual void beginRun(const edm::Run&, const edm::EventSetup& context);
    virtual void beginLuminosityBlock(edm::LuminosityBlock const& iLBlock,
				      edm::EventSetup const& context);
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
  };
  void
  LumiDataAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& context){
    /*std::cout<<"###LuminosityInfoAnalyzer::beginRun"<<std::endl;
      edm::ESHandle<lumi::LuminosityInfo> pPeds;
      std::cout<<"got eshandle"<<std::endl;
      context.get<LuminosityInfoRcd>().get(pPeds);
    */
  }
  void
  LumiDataAnalyzer::beginJob(){
    // std::cout<<"###LumiDataAnalyzer::beginJob"<<std::endl;
  }
  void
  LumiDataAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& iLBlock,
					 edm::EventSetup const& context){
    std::cout<<"###LumiDataAnalyzer::beginLuminosityBlock"<<std::endl;
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout<<"lumiblock id value "<<iLBlock.id().value()<<std::endl;
    std::cout<<"lumiblock in run "<<iLBlock.run()<<std::endl;
    std::cout<<"lumiblock number "<<iLBlock.id().luminosityBlock()<<std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("LumiSectionDataRcd"));
    if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout <<"Record \"LumiSectionDataRcd"<<"\" does not exist "<<std::endl;
    }
    edm::ESHandle<lumi::LumiSectionData> pPeds;
    context.get<LumiSectionDataRcd>().get(pPeds);
    const lumi::LumiSectionData* myped=pPeds.product();
    std::cout<<"data pointer "<<myped<<std::endl;
    std::cout<<"lumi average "<<myped->lumiAverage()<<std::endl;
  }

  void
  LumiDataAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context){
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
  }
  DEFINE_FWK_MODULE(LumiDataAnalyzer);
}

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
#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParam.h"
#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParamRcd.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include <iostream>


namespace edm {
  class EventSetup;
}

using namespace std;
using namespace edm;

class TestLumiCorrectionSource : public edm::EDAnalyzer{
public:
  
  explicit TestLumiCorrectionSource(edm::ParameterSet const&);
  virtual ~TestLumiCorrectionSource();
  
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c);
};

// -----------------------------------------------------------------

TestLumiCorrectionSource::TestLumiCorrectionSource(edm::ParameterSet const& ps)
{
}

// -----------------------------------------------------------------

TestLumiCorrectionSource::~TestLumiCorrectionSource()
{
}

// -----------------------------------------------------------------

void TestLumiCorrectionSource::analyze(edm::Event const& e,edm::EventSetup const&)
{
}

// -----------------------------------------------------------------

void TestLumiCorrectionSource::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, EventSetup const& es){
  std::cout <<" I AM IN RUN NUMBER "<<lumiBlock.run() <<" LS NUMBER "<< lumiBlock.luminosityBlock()<<std::endl;
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("LumiCorrectionParamRcd"));
  if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    std::cout <<"Record \"LumiCorrectionParamRcd"<<"\" does not exist "<<std::endl;
  }
  try{
    edm::Handle<LumiSummary> lumisummary;
    lumiBlock.getByLabel("lumiProducer", lumisummary);
    float instlumi=lumisummary->avgInsDelLumi();
    std::cout<<"raw data tag "<<lumisummary->lumiVersion()<<std::endl;;
    float correctedinstlumi=instlumi;
    float recinstlumi=lumisummary->avgInsRecLumi();
    float corrfac=1.;
    edm::ESHandle<LumiCorrectionParam> datahandle;
    es.getData(datahandle);
    if(datahandle.isValid()){
      const LumiCorrectionParam* mydata=datahandle.product();
      std::cout<<"correctionparams "<<*mydata<<std::endl;
      corrfac=mydata->getCorrection(instlumi);
    }else{
      std::cout<<"no valid record found"<<std::endl;
    }
    correctedinstlumi=instlumi*corrfac;
    std::cout<<"correctedinstlumi "<<correctedinstlumi<<std::endl;
    float correctedinstRecLumi=recinstlumi*corrfac; 
    std::cout<<"corrected rec instlumi "<<correctedinstRecLumi<<std::endl;
  }catch(const edm::eventsetup::NoRecordException<LumiCorrectionParamRcd>& er){
    std::cout<<"no data found"<<std::endl;
  }catch(const cms::Exception& ee){
    std::cout<<ee.what()<<std::endl;
  }
}

DEFINE_FWK_MODULE(TestLumiCorrectionSource);

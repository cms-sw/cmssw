// system include files
#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/CondDB/interface/Time.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelAnalysisMask.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"

using namespace std;

class CTPPSPixelDAQMappingAnalyzer : public edm::EDAnalyzer {
   public:

    string label_;
    cond::Time_t daqmappingiov_;
    cond::Time_t analysismaskiov_;

    explicit  CTPPSPixelDAQMappingAnalyzer(edm::ParameterSet const& iConfig):
    label_(iConfig.getUntrackedParameter<string>("label", "RPix")),
    daqmappingiov_(iConfig.getParameter<unsigned long long>("daqmappingiov")),
    analysismaskiov_(iConfig.getParameter<unsigned long long>("analysismaskiov"))
    {}
    explicit  CTPPSPixelDAQMappingAnalyzer(int i) {}
    virtual ~CTPPSPixelDAQMappingAnalyzer() {}
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  };


  void
  CTPPSPixelDAQMappingAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context){

    using namespace edm;

    /*edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("CTPPSPixelDAQMappingRcd"));
    if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout <<"Record \"CTPPSPixelDAQMappingRcd"<<"\" does not exist "<<std::endl;
    }*/


    //this part gets the handle of the event source and the record (i.e. the Database)
    if (e.id().run() == daqmappingiov_){
      ESHandle<CTPPSPixelDAQMapping> mapping;
      context.get<CTPPSPixelDAQMappingRcd>().get("RPix", mapping);

      // print mapping
      /*printf("* DAQ mapping\n");
      for (const auto &p : mapping->ROCMapping)
      std::cout << "    " << p.first << " -> " << p.second << std::endl;*/
    }

    //edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("CTPPSPixelAnalysisMaskRcd"));
    //if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      //std::cout <<"Record \"CTPPSPixelAnalysisMaskRcd"<<"\" does not exist "<<std::endl;
    //}

	  if (e.id().run() == analysismaskiov_){
      // get analysis mask to mask channels
      ESHandle<CTPPSPixelAnalysisMask> analysisMask;
      context.get<CTPPSPixelAnalysisMaskRcd>().get(label_, analysisMask);

      // print mask
      /*printf("* mask\n");
      for (const auto &p : analysisMask->analysisMask)
        cout << "    " << p.first
        << ": fullMask=" << p.second.fullMask
        << ", number of masked channels " << p.second.maskedPixels.size() << endl;
      */
	  }

  }
  DEFINE_FWK_MODULE(CTPPSPixelDAQMappingAnalyzer);

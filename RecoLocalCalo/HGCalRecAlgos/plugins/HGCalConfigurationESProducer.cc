#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/HGCalObjects/interface/HGCalConfiguration.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h" // depends on HGCalElectronicsMappingRcd
//#include <nlohmann/json.hpp>
//using json = nlohmann::json;


/**
   @short plugin parses HGCAL electronics configuration from JSON file
 */
class HGCalConfigurationESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  explicit HGCalConfigurationESProducer(const edm::ParameterSet& iConfig)
    : //edm::ESProducer(iConfig),
      filename_(iConfig.getParameter<std::string>("filename")),
      passthroughMode_(iConfig.getParameter<int32_t>("passthroughMode")),
      cbHeaderMarker_(iConfig.getParameter<int32_t>("cbHeaderMarker")),
      slinkHeaderMarker_(iConfig.getParameter<int32_t>("slinkHeaderMarker")),
      econdHeaderMarker_(iConfig.getParameter<int32_t>("econdHeaderMarker")),
      charMode_(iConfig.getParameter<int32_t>("charMode")) {
    auto cc = setWhatProduced(this);
    //findingRecord<HGCalModuleConfigurationRcd>();
    indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::ESInputTag>("indexSource",edm::ESInputTag(""))->setComment("Label for module indexer to set SoA size");
    //desc.add<edm::FileInPath>("filename")->setComment("JSON file with configuration parameters");
    desc.add<std::string>("filename","")->setComment("JSON file with configuration parameters");
    desc.add<int32_t>("passthroughMode",-1)->setComment("Manual override for mismatch passthrough mode");
    desc.add<int32_t>("cbHeaderMarker",-1)->setComment("Manual override for capture block header marker (BEO)");
    desc.add<int32_t>("slinkHeaderMarker",-1)->setComment("Manual override for S-link header marker (BEO)");
    desc.add<int32_t>("econdHeaderMarker",-1)->setComment("Manual override for ECON-D header marker (BEO)");
    desc.add<int32_t>("charMode",-1)->setComment("Manual override for ECON-D characterization mode");
    descriptions.addWithDefaultLabel(desc);
  }

  template<typename T>
  static T getval(T value, int32_t value_override) {
    // get value, and override if value_override>=0
    return (value_override<0 ? value : (T) value_override);
  }

  //std::optional<HGCalConfiguration> produce(const HGCalModuleConfigurationRcd& iRecord) {
  std::shared_ptr<HGCalConfiguration> produce(const HGCalModuleConfigurationRcd& iRecord) {
    std::cout << "HGCalConfigurationESProducer::produce: fname=" << filename_ << std::endl;
    auto const& moduleMap = iRecord.getRecord<HGCalElectronicsMappingRcd>().get(indexToken_);
    // TODO: retrieve values from custom JSON format (see HGCalCalibrationESProducer)
    ////std::ifstream infile(filename_);
    ////json calib_data = json::parse(infile);
    config_ = HGCalConfiguration();
    auto feds = moduleMap.fedReadoutSequences_;
    for (std::size_t fedid = 0; fedid < feds.size(); ++fedid) {
      // https://github.com/CMS-HGCAL/cmssw/blob/dev/hackathon_base_CMSSW_14_1_X/CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h
      // https://github.com/CMS-HGCAL/cmssw/blob/dev/hackathon_base_CMSSW_14_1_X/EventFilter/HGCalRawToDigi/plugins/HGCalRawToDigi.cc#L107-L111
      auto fedrs = moduleMap.fedReadoutSequences_[fedid];
      if (fedrs.readoutTypes_.size() == 0)
        continue; // non-existent FED
      HGCalFedConfig_t fed;
      fed.mismatchPassthroughMode = getval(false,passthroughMode_);   // ignore ECON-D packet mismatches
      fed.cbHeaderMarker          = getval(0,cbHeaderMarker_);    // begin of event marker/identifier for capture block
      fed.slinkHeaderMarker       = getval(0,slinkHeaderMarker_); // begin of event marker/identifier for S-link
      for (std::size_t modid = 0; modid < fedrs.readoutTypes_.size(); ++modid) { 
        HGCalECONDConfig_t mod;
        mod.headerMarker = getval(0,econdHeaderMarker_); // begin of event marker/identifier for capture block
        fed.econds.push_back(mod); // add to vector of HGCalECONDConfig_t ECON-D modules
      }
      config_.feds.push_back(fed); // add to vector of HGCalFedConfig_t FEDs
    }
    return std::shared_ptr<HGCalConfiguration>(&config_, edm::do_nothing_deleter());
  }  // end of produce()

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval& oValidity) override {
    oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
  }

  edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexToken_;
  const std::string filename_; // JSON file
  HGCalConfiguration config_;
  int32_t passthroughMode_; // for manual override
  int32_t cbHeaderMarker_; // for manual override
  int32_t slinkHeaderMarker_; // for manual override
  int32_t econdHeaderMarker_; // for manual override
  int32_t charMode_; // for manual override
};


DEFINE_FWK_EVENTSETUP_SOURCE(HGCalConfigurationESProducer);

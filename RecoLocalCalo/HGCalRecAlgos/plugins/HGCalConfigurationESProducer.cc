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
#include <string> // for std::to_string
#include <fstream> // needed to read json file with std::ifstream
#include <nlohmann/json.hpp>
using json = nlohmann::json;


/**
   @short plugin parses HGCAL electronics configuration from JSON file
 */
class HGCalConfigurationESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  explicit HGCalConfigurationESProducer(const edm::ParameterSet& iConfig)
    : //edm::ESProducer(iConfig),
      fedjson_(iConfig.getParameter<std::string>("fedjson")),
      modjson_(iConfig.getParameter<std::string>("modjson")),
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
    //desc.add<edm::FileInPath>("filename")->setComment("JSON file with FED configuration parameters");
    desc.add<std::string>("fedjson","")->setComment("JSON file with FED configuration parameters");
    desc.add<std::string>("modjson","")->setComment("JSON file with ECOND configuration parameters");
    desc.addOptional<int32_t>("passthroughMode",-1)->setComment("Manual override for mismatch passthrough mode");
    desc.addOptional<int32_t>("cbHeaderMarker",-1)->setComment("Manual override for capture block header marker (BEO, e.g 0x7f)");
    desc.addOptional<int32_t>("slinkHeaderMarker",-1)->setComment("Manual override for S-link header marker (BEO, e.g 0x55)");
    desc.addOptional<int32_t>("econdHeaderMarker",-1)->setComment("Manual override for ECON-D header marker (BEO, e.g 0x154)");
    desc.addOptional<int32_t>("charMode",-1)->setComment("Manual override for ECON-D characterization mode");
    descriptions.addWithDefaultLabel(desc);
  }

  static int32_t gethex(std::string value, int32_t value_override) {
    // get value, and override if value_override>=0
    //std::cout << "HGCalConfigurationESProducer::gethex: value=" << value << ", override=" << value_override << std::endl;
    return (value_override>=0 ? value_override : std::stoi(value,NULL,16));
  }

  static int32_t getint(int32_t value, int32_t value_override) {
    // get value, and override if value_override>=0
    //std::cout << "HGCalConfigurationESProducer::getint: value=" << value << ", override=" << value_override << std::endl;
    return (value_override>=0 ? value_override : value);
  }

  //template<typename T>
  //static T getval(T value, int32_t value_override) {
  //  // get value, and override if value_override>=0
  //  std::cout << "HGCalConfigurationESProducer::getval: value=" << value << ", override=" << value_override << std::endl;
  //  return (value_override>=0 ? (T) value_override : value);
  //}

  //std::optional<HGCalConfiguration> produce(const HGCalModuleConfigurationRcd& iRecord) {
  std::shared_ptr<HGCalConfiguration> produce(const HGCalModuleConfigurationRcd& iRecord) {
    std::cout << "HGCalConfigurationESProducer::produce: fedjson=" << fedjson_ << std::endl;
    std::cout << "HGCalConfigurationESProducer::produce: modjson=" << modjson_ << std::endl;
    auto const& moduleMap = iRecord.getRecord<HGCalElectronicsMappingRcd>().get(indexToken_);

    // Retrieve values from custom JSON format (see HGCalCalibrationESProducer)
    std::ifstream fedfile(fedjson_); // e.g. HGCalCommissioning/LocalCalibration/data/config_feds.json
    std::ifstream modfile(modjson_); // e.g. HGCalCommissioning/LocalCalibration/data/config_econds.json
    json fed_config_data = json::parse(fedfile);
    json mod_config_data = json::parse(modfile);

    // Loop over FEDs in indexer & fill configuration structs: FED > ECON-D > eRx
    config_ = HGCalConfiguration(); // container class holding FED structs of ECON-D structs of eRx structs
    auto feds = moduleMap.fedReadoutSequences_;
    //std::cout << "HGCalConfigurationESProducer::produce:   nfeds=" << feds.size() << std::endl;
    for (std::size_t fedid = 0; fedid < feds.size(); ++fedid) {
      // follow indexing by HGCalMappingModuleIndexer
      // https://github.com/CMS-HGCAL/cmssw/blob/dev/hackathon_base_CMSSW_14_1_X/CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h
      // https://github.com/CMS-HGCAL/cmssw/blob/dev/hackathon_base_CMSSW_14_1_X/EventFilter/HGCalRawToDigi/plugins/HGCalRawToDigi.cc#L107-L111

      // Fill FED configurations
      auto fedrs = moduleMap.fedReadoutSequences_[fedid];
      if (fedrs.readoutTypes_.size() == 0) // check if FED exists (non-empty)
        continue; // skip non-existent FED
      //std::cout << "HGCalConfigurationESProducer::produce:   fed=" << fedid << std::endl;
      HGCalFedConfig_t fed;
      std::string sfedid = std::to_string(fedid);
      if (!fed_config_data.contains(sfedid))
        edm::LogWarning("HGCalConfigurationESProducer") << " Did not find FED index " << sfedid
                                                        << " in JSON file " << fedjson_ << "...";
      fed.mismatchPassthroughMode = getint(fed_config_data[sfedid]["mismatchPassthroughMode"],passthroughMode_);   // ignore ECON-D packet mismatches
      fed.cbHeaderMarker          = gethex(fed_config_data[sfedid]["cbHeaderMarker"],         cbHeaderMarker_);    // begin of event marker/identifier for capture block
      fed.slinkHeaderMarker       = gethex(fed_config_data[sfedid]["slinkHeaderMarker"],      slinkHeaderMarker_); // begin of event marker/identifier for S-link

      // Loop over ECON-D modules in JSON
      //for (std::size_t modid = 0; modid < fedrs.readoutTypes_.size(); ++modid) { 
      for (const std::string typecode: fed_config_data[sfedid]["econds"]) { // loop over module typecodes in JSON file (e.g. "ML-F3PT-TX-0003")
        //std::cout << "HGCalConfigurationESProducer::produce:     typecode=" << typecode << std::endl;
        const auto& [fedid2,modid] = moduleMap.getFedAndModuleIndex(typecode);
        if (fedid != fedid2)
          edm::LogWarning("HGCalConfigurationESProducer") << " FED index from HGCalMappingModuleIndexer (" << fedid
                                                          << ") does not match that of the JSON file (" << fedid2 << ", " << fedjson_ 
                                                          << ") for ECON-D module with typecode " << typecode << " and id=" << modid << "!";
        if (modid >= fed.econds.size())
          fed.econds.resize(modid+1);
        HGCalECONDConfig_t mod;
        mod.headerMarker = gethex(mod_config_data[typecode]["headerMarker"],econdHeaderMarker_); // begin of event marker/identifier for capture block
        fed.econds[modid] = mod; // add to FED's vector of HGCalECONDConfig_t ECON-D modules
      }

      config_.feds.push_back(fed); // add to config's vector of HGCalFedConfig_t FEDs
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
  HGCalConfiguration config_; // container class holding FED structs of ECON-D structs of eRx structs
  const std::string fedjson_; // JSON file
  const std::string modjson_; // JSON file
  int32_t passthroughMode_; // for manual override
  int32_t cbHeaderMarker_; // for manual override
  int32_t slinkHeaderMarker_; // for manual override
  int32_t econdHeaderMarker_; // for manual override
  int32_t charMode_; // for manual override
};


DEFINE_FWK_EVENTSETUP_SOURCE(HGCalConfigurationESProducer);

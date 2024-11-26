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
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"  // depends on HGCalElectronicsMappingRcd
#include <string>                                                          // for std::to_string
#include <fstream>  // needed to read json file with std::ifstream
#include <nlohmann/json.hpp>
using json = nlohmann::json;

/**
   @short plugin parses HGCAL electronics configuration from JSON file
 */
class HGCalConfigurationESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  explicit HGCalConfigurationESProducer(const edm::ParameterSet& iConfig)
      :  //edm::ESProducer(iConfig),
        fedjson_(iConfig.getParameter<std::string>("fedjson")),
        modjson_(iConfig.getParameter<std::string>("modjson")) {
    if (iConfig.exists("bePassthroughMode"))
      bePassthroughMode_ = iConfig.getParameter<int32_t>("bePassthroughMode");
    if (iConfig.exists("econPassthroughMode"))
      econPassthroughMode_ = iConfig.getParameter<int32_t>("econPassthroughMode");
    if (iConfig.exists("cbHeaderMarker"))
      cbHeaderMarker_ = iConfig.getParameter<int32_t>("cbHeaderMarker");
    if (iConfig.exists("slinkHeaderMarker"))
      slinkHeaderMarker_ = iConfig.getParameter<int32_t>("slinkHeaderMarker");
    if (iConfig.exists("econdHeaderMarker"))
      econdHeaderMarker_ = iConfig.getParameter<int32_t>("econdHeaderMarker");
    if (iConfig.exists("charMode"))
      charMode_ = iConfig.getParameter<int32_t>("charMode");
    if (iConfig.exists("gain"))
      gain_ = iConfig.getParameter<int32_t>("gain");
    auto cc = setWhatProduced(this);
    //findingRecord<HGCalModuleConfigurationRcd>();
    indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::ESInputTag>("indexSource", edm::ESInputTag(""))
        ->setComment("Label for module indexer to set SoA size");
    //desc.add<edm::FileInPath>("filename")->setComment("JSON file with FED configuration parameters");
    desc.add<std::string>("fedjson", "")->setComment("JSON file with FED configuration parameters");
    desc.add<std::string>("modjson", "")->setComment("JSON file with ECOND configuration parameters");
    desc.addOptional<int32_t>("bePassthroughMode", -1)
        ->setComment("Manual override for mismatch passthrough mode in the BE");
    desc.addOptional<int32_t>("econPassthroughMode", -1)->setComment("Manual override passthrough mode in the ECON-D");
    desc.addOptional<int32_t>("cbHeaderMarker", -1)
        ->setComment("Manual override for capture block header marker (BEO, e.g 0x7f)");
    desc.addOptional<int32_t>("slinkHeaderMarker", -1)
        ->setComment("Manual override for S-link header marker (BEO, e.g 0x55)");
    desc.addOptional<int32_t>("econdHeaderMarker", -1)
        ->setComment("Manual override for ECON-D header marker (BEO, e.g 0x154)");
    desc.addOptional<int32_t>("charMode", -1)->setComment("Manual override for ROC characterization mode");
    desc.addOptional<int32_t>("gain", -1)->setComment("Manual override for ROC gain");
    descriptions.addWithDefaultLabel(desc);
  }

  static bool checkkeys(const json& data,
                        const std::string& firstkey,
                        const std::vector<std::string>& keys,
                        const std::string& fname) {
    // check if json contains key
    bool iscomplete = true;
    for (auto const& key : keys) {
      if (not data[firstkey].contains(key)) {
        edm::LogWarning("HGCalConfigurationESProducer::checkkeys")
            << " JSON is missing key '" << key << "' for " << firstkey << "!"
            << " Please check file " << fname;
        iscomplete = false;
      }
    }
    return iscomplete;
  }

  static int32_t gethex(const std::string& value, const int32_t value_override) {
    // get value, and override if value_override>=0
    return (value_override >= 0 ? value_override : std::stoi(value, nullptr, 16));
  }

  static int32_t getint(const int32_t value, const int32_t value_override) {
    // get value, and override if value_override>=0
    return (value_override >= 0 ? value_override : value);
  }

  std::unique_ptr<HGCalConfiguration> produce(const HGCalModuleConfigurationRcd& iRecord) {
    auto const& moduleMap = iRecord.get(indexToken_);

    // retrieve values from custom JSON format (see HGCalCalibrationESProducer)
    edm::FileInPath fedfip(fedjson_);  // e.g. HGCalCommissioning/LocalCalibration/data/config_feds.json
    edm::FileInPath modfip(modjson_);  // e.g. HGCalCommissioning/LocalCalibration/data/config_econds.json
    std::ifstream fedfile(fedfip.fullPath());
    std::ifstream modfile(modfip.fullPath());
    const json fed_config_data = json::parse(fedfile);
    const json mod_config_data = json::parse(modfile);

    // consistency check
    uint32_t nfeds = moduleMap.getNFED();
    uint32_t ntot_mods = 0, ntot_rocs = 0;
    const std::vector<std::string> fedkeys = {
        "mismatchPassthroughMode", "cbHeaderMarker", "slinkHeaderMarker", "econds"};
    const std::vector<std::string> modkeys = {"headerMarker", "passthrough", "Gain", "CalibrationSC"};
    if (nfeds != fed_config_data.size())
      edm::LogWarning("HGCalConfigurationESProducer")
          << "Total number of FEDs found in JSON file " << fedjson_ << " (" << fed_config_data.size()
          << ") does not match indexer (" << nfeds << ")";

    // loop over FEDs in indexer & fill configuration structs: FED > ECON-D > eRx
    // follow indexing by HGCalMappingModuleIndexer
    // HGCalConfiguration = container class holding FED structs of ECON-D structs of eRx structs
    std::unique_ptr<HGCalConfiguration> config_ = std::make_unique<HGCalConfiguration>();
    for (std::size_t fedid = 0; fedid < moduleMap.getMaxFEDSize(); ++fedid) {
      // sanity checks
      if (moduleMap.getFEDReadoutSequences()[fedid].readoutTypes_.size() == 0)  // check if FED exists (non-empty)
        continue;                                                               // skip non-existent FED
      std::string sfedid = std::to_string(fedid);                               // key in JSON dictionary must be string
      if (!fed_config_data.contains(sfedid))
        edm::LogWarning("HGCalConfigurationESProducer")
            << " Did not find FED index " << sfedid << " in JSON file " << fedjson_ << "...";
      checkkeys(fed_config_data, sfedid, fedkeys, fedjson_);  // check required keys are in the JSON, warn otherwise

      // fill FED configurations
      HGCalFedConfig fed;
      fed.mismatchPassthroughMode = getint(fed_config_data[sfedid]["mismatchPassthroughMode"],
                                           bePassthroughMode_);  // ignore ECON-D packet mismatches
      fed.cbHeaderMarker = gethex(fed_config_data[sfedid]["cbHeaderMarker"],
                                  cbHeaderMarker_);  // begin of event marker/identifier for capture block
      fed.slinkHeaderMarker = gethex(fed_config_data[sfedid]["slinkHeaderMarker"],
                                     slinkHeaderMarker_);  // begin of event marker/identifier for S-link

      // loop over ECON-D modules in JSON
      for (const std::string typecode :
           fed_config_data[sfedid]["econds"]) {  // loop over module typecodes in JSON file (e.g. "ML-F3PT-TX-0003")

        // sanity checks for ECON-Ds
        ntot_mods++;
        const auto& [fedid2, imod] = moduleMap.getIndexForFedAndModule(typecode);
        if (fedid != fedid2)
          edm::LogWarning("HGCalConfigurationESProducer")
              << " FED index from HGCalMappingModuleIndexer (" << fedid << ") does not match that of the JSON file ("
              << fedid2 << ", " << fedjson_ << ") for ECON-D module with typecode " << typecode << " and id=" << imod
              << "!";
        checkkeys(mod_config_data, typecode, modkeys, modjson_);  // check required keys are in the JSON, warn otherwise
        if (imod >= fed.econds.size())
          fed.econds.resize(imod + 1);

        // fill ECON-D configuration
        HGCalECONDConfig mod;
        mod.headerMarker = gethex(mod_config_data[typecode]["headerMarker"],
                                  econdHeaderMarker_);  // begin of event marker/identifier for capture block
        mod.passThrough = getint(mod_config_data[typecode]["passthrough"], econPassthroughMode_);

        // sanity checks for eRx half-ROCs
        uint32_t nrocs = moduleMap.getMaxERxSize(fedid, imod);
        uint32_t nrocs2 = mod_config_data[typecode]["Gain"].size();
        if (nrocs != nrocs2)
          edm::LogWarning("HGCalConfigurationESProducer")
              << " Number of eRx ROCs for ECON-D " << typecode << " in " << fedjson_ << " (" << nrocs2
              << ") does not match that of the indexer for fedid" << fedid << " & imod=" << imod << " (" << nrocs
              << ")!";
        mod.rocs.resize(nrocs);

        // fill eRX (half-ROC) configuration
        for (uint32_t iroc = 0; iroc < nrocs; iroc++) {
          ntot_rocs++;
          HGCalROCConfig roc;
          roc.gain = (uint8_t)mod_config_data[typecode]["Gain"][iroc];
          //roc.charMode = getint(mod_config_data[typecode]["characMode"],charMode_);
          roc.charMode = getint(mod_config_data[typecode]["CalibrationSC"][iroc], charMode_);
          mod.rocs[iroc] = roc;  // add to ECON-D's vector<HGCalROCConfig> of eRx half-ROCs
        }
        fed.econds[imod] = mod;  // add to FED's vector<HGCalECONDConfig> of ECON-D modules
      }

      config_->feds.push_back(fed);  // add to config's vector of HGCalFedConfig FEDs
    }

    // consistency check
    if (ntot_mods != moduleMap.getMaxModuleSize())
      edm::LogWarning("HGCalConfigurationESProducer")
          << "Total number of ECON-D modules found in JSON file " << modjson_ << " (" << ntot_mods
          << ") does not match indexer (" << moduleMap.getMaxModuleSize() << ")";
    if (ntot_rocs != moduleMap.getMaxERxSize())
      edm::LogWarning("HGCalConfigurationESProducer")
          << "Total number of eRx half-ROCs found in JSON file " << modjson_ << " (" << ntot_rocs
          << ") does not match indexer (" << moduleMap.getMaxERxSize() << ")";

    return config_;
  }  // end of produce()

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval& oValidity) override {
    oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
  }

  edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexToken_;
  const std::string fedjson_;         // JSON file
  const std::string modjson_;         // JSON file
  int32_t bePassthroughMode_ = -1;    // for manual override
  int32_t cbHeaderMarker_ = -1;       // for manual override
  int32_t slinkHeaderMarker_ = -1;    // for manual override
  int32_t econdHeaderMarker_ = -1;    // for manual override
  int32_t econPassthroughMode_ = -1;  // for manual override
  int32_t charMode_ = -1;             // for manual override
  int32_t gain_ = -1;                 // for manual override
};

DEFINE_FWK_EVENTSETUP_SOURCE(HGCalConfigurationESProducer);

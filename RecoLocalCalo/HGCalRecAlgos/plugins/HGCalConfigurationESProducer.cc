#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/HGCalObjects/interface/HGCalConfiguration.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"  // depends on HGCalElectronicsMappingRcd
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalESProducerTools.h"    // for json, search_modkey, search_fedkey

#include <string>   // for std::to_string
#include <fstream>  // needed to read json file with std::ifstream

/**
 * @short ESProducer to parse HGCAL electronics configuration from JSON file
 */
class HGCalConfigurationESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  explicit HGCalConfigurationESProducer(const edm::ParameterSet& iConfig)
      :  //edm::ESProducer(iConfig),
        fedjson_(iConfig.getParameter<edm::FileInPath>("fedjson")),
        modjson_(iConfig.getParameter<edm::FileInPath>("modjson")) {
    if (iConfig.exists("bePassthroughMode"))
      bePassthroughMode_ = iConfig.getParameter<int32_t>("bePassthroughMode");
    if (iConfig.exists("cbHeaderMarker"))
      cbHeaderMarker_ = iConfig.getParameter<int32_t>("cbHeaderMarker");
    if (iConfig.exists("slinkHeaderMarker"))
      slinkHeaderMarker_ = iConfig.getParameter<int32_t>("slinkHeaderMarker");
    if (iConfig.exists("econdHeaderMarker"))
      econdHeaderMarker_ = iConfig.getParameter<int32_t>("econdHeaderMarker");
    if (iConfig.exists("charMode"))
      charMode_ = iConfig.getParameter<int32_t>("charMode");
    auto cc = setWhatProduced(this);
    indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::ESInputTag>("indexSource", edm::ESInputTag(""))
        ->setComment("Label for module indexer to set SoA size");
    desc.add<edm::FileInPath>("fedjson")->setComment("JSON file with FED configuration parameters");
    desc.add<edm::FileInPath>("modjson")->setComment("JSON file with ECOND configuration parameters");
    desc.addOptional<int32_t>("bePassthroughMode", -1)
        ->setComment("Manual override for mismatch passthrough mode in the BE");
    desc.addOptional<int32_t>("cbHeaderMarker", -1)
        ->setComment("Manual override for capture block header marker (BEO, e.g 0x7f)");
    desc.addOptional<int32_t>("slinkHeaderMarker", -1)
        ->setComment("Manual override for S-link header marker (BEO, e.g 0x55)");
    desc.addOptional<int32_t>("econdHeaderMarker", -1)
        ->setComment("Manual override for ECON-D header marker (BEO, e.g 0x154)");
    desc.addOptional<int32_t>("charMode", -1)->setComment("Manual override for ROC characterization mode");
    descriptions.addWithDefaultLabel(desc);
  }

  // @short get hexadecimal value, and override if value_override>=0
  static int32_t gethex(const std::string& value, const int32_t value_override) {
    return (value_override >= 0 ? value_override : std::stoi(value, nullptr, 16));
  }

  // @short get integer value, and override if value_override>=0
  static int32_t getint(const int32_t value, const int32_t value_override) {
    return (value_override >= 0 ? value_override : value);
  }

  std::unique_ptr<HGCalConfiguration> produce(const HGCalModuleConfigurationRcd& iRecord) {
    auto const& moduleMap = iRecord.get(indexToken_);
    edm::LogInfo("HGCalConfigurationESProducer")
        << "produce: fedjson_=" << fedjson_ << ",\n         modjson_=" << modjson_;

    // retrieve values from custom JSON format (see HGCalCalibrationESProducer)
    std::string fedjsonurl(fedjson_.fullPath());
    std::string modjsonurl(modjson_.fullPath());
    std::ifstream fedfile(fedjsonurl);
    std::ifstream modfile(modjsonurl);
    const json fed_config_data = json::parse(fedfile, nullptr, true, /*ignore_comments*/ true);
    const json mod_config_data = json::parse(modfile, nullptr, true, /*ignore_comments*/ true);

    // consistency check
    uint32_t nfeds = moduleMap.numFEDs();
    uint32_t ntot_mods = 0, ntot_rocs = 0;
    const std::vector<std::string> fedkeys = {"mismatchPassthroughMode", "cbHeaderMarker", "slinkHeaderMarker"};
    const std::vector<std::string> modkeys = {"headerMarker", "CalibrationSC"};
    if (nfeds != fed_config_data.size())
      edm::LogWarning("HGCalConfigurationESProducer")
          << "Total number of FEDs found in JSON file " << fedjsonurl << " (" << fed_config_data.size()
          << ") does not match indexer (" << nfeds << ")";

    // loop over FEDs in indexer & fill configuration structs: FED > ECON-D > eRx
    // follow indexing by HGCalMappingModuleIndexer
    // HGCalConfiguration = container class holding FED structs of ECON-D structs of eRx structs
    std::unique_ptr<HGCalConfiguration> config_ = std::make_unique<HGCalConfiguration>();
    config_->feds.resize(moduleMap.maxFEDSize());
    for (std::size_t fedid = 0; fedid < moduleMap.maxFEDSize(); ++fedid) {
      // sanity checks
      if (moduleMap.fedReadoutSequences()[fedid].readoutTypes_.empty())              // check if FED exists (non-empty)
        continue;                                                                    // skip non-existent FED
      const auto fedkey = hgcal::search_fedkey(fedid, fed_config_data, fedjsonurl);  // search matching key
      hgcal::check_keys(
          fed_config_data, fedkey, fedkeys, fedjsonurl);  // check required keys are in the JSON, warn otherwise

      // fill FED configurations
      HGCalFedConfig fed;
      fed.mismatchPassthroughMode = getint(fed_config_data[fedkey]["mismatchPassthroughMode"],
                                           bePassthroughMode_);  // ignore ECON-D packet mismatches
      fed.cbHeaderMarker = gethex(fed_config_data[fedkey]["cbHeaderMarker"],
                                  cbHeaderMarker_);  // begin of event marker/identifier for capture block
      fed.slinkHeaderMarker = gethex(fed_config_data[fedkey]["slinkHeaderMarker"],
                                     slinkHeaderMarker_);  // begin of event marker/identifier for S-link

      // loop over module typecodes (e.g. "ML-F3PT-TX-0003")
      for (const auto& [typecode, ids] : moduleMap.typecodeMap()) {
        auto [fedid_, imod] = ids;
        if (fedid_ != fedid)
          continue;

        // sanity checks for ECON-Ds
        ntot_mods++;
        const auto modkey = hgcal::search_modkey(typecode, mod_config_data, modjsonurl);  // search matching key
        hgcal::check_keys(
            mod_config_data, modkey, modkeys, modjsonurl);  // check required keys are in the JSON, warn otherwise
        if (imod >= fed.econds.size())
          fed.econds.resize(imod + 1);

        // fill ECON-D configuration
        // headerMarker: begin of event marker/identifier for capture block
        HGCalECONDConfig mod;
        mod.headerMarker = gethex(mod_config_data[modkey]["headerMarker"], econdHeaderMarker_);

        // sanity checks for eRx half-ROCs
        uint32_t nrocs = moduleMap.getNumERxs(fedid, imod);
        uint32_t nrocs2 = mod_config_data[modkey]["CalibrationSC"].size();
        if (nrocs != nrocs2)
          edm::LogWarning("HGCalConfigurationESProducer")
              << " Number of eRx ROCs for ECON-D " << typecode << " in " << fedjsonurl << " (" << nrocs2
              << ") does not match that of the indexer for fedid" << fedid << " & imod=" << imod << " (" << nrocs
              << ")!";
        mod.rocs.resize(nrocs);

        // fill eRX (half-ROC) configuration
        for (uint32_t iroc = 0; iroc < nrocs; iroc++) {
          ntot_rocs++;
          HGCalROCConfig roc;
          roc.charMode = getint(mod_config_data[modkey]["CalibrationSC"][iroc], charMode_);
          mod.rocs[iroc] = roc;  // add to ECON-D's vector<HGCalROCConfig> of eRx half-ROCs
        }
        fed.econds[imod] = mod;  // add to FED's vector<HGCalECONDConfig> of ECON-D modules
      }

      config_->feds[fedid] = fed;  // add to config's vector of HGCalFedConfig FEDs
    }

    // consistency check
    if (ntot_mods != moduleMap.maxModulesCount())
      edm::LogWarning("HGCalConfigurationESProducer")
          << "Total number of ECON-D modules found in JSON file " << modjson_ << " (" << ntot_mods
          << ") does not match indexer (" << moduleMap.maxModulesCount() << ")";
    if (ntot_rocs != moduleMap.maxERxSize())
      edm::LogWarning("HGCalConfigurationESProducer")
          << "Total number of eRx half-ROCs found in JSON file " << modjson_ << " (" << ntot_rocs
          << ") does not match indexer (" << moduleMap.maxERxSize() << ")";

    return config_;
  }  // end of produce()

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval& oValidity) override {
    oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
  }

  edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexToken_;
  const edm::FileInPath fedjson_;   // JSON file
  const edm::FileInPath modjson_;   // JSON file
  int32_t bePassthroughMode_ = -1;  // for manual override
  int32_t cbHeaderMarker_ = -1;     // for manual override
  int32_t slinkHeaderMarker_ = -1;  // for manual override
  int32_t econdHeaderMarker_ = -1;  // for manual override
  int32_t charMode_ = -1;           // for manual override
};

DEFINE_FWK_EVENTSETUP_SOURCE(HGCalConfigurationESProducer);

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

#include "CondFormats/HGCalObjects/interface/HGCalTriggerConfiguration.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexerTrigger.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"  // depends on HGCalElectronicsMappingRcd
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalESProducerTools.h"    // for json, search_modkey, search_fedkey

#include <string>   // for std::to_string
#include <fstream>  // needed to read json file with std::ifstream
#include <regex>

/**
 * @short ESProducer to parse HGCAL electronics configuration from JSON file
 */
class HGCalTriggerConfigurationESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  explicit HGCalTriggerConfigurationESProducer(const edm::ParameterSet& iConfig)
      :  //edm::ESProducer(iConfig),
        fedjson_(iConfig.getParameter<edm::FileInPath>("fedjson")),
        modjson_(iConfig.getParameter<edm::FileInPath>("modjson")),
        emulatorOn_(iConfig.getParameter<bool>("emulatorOn")) {

    auto cc = setWhatProduced(this);
    indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::ESInputTag>("indexSource", edm::ESInputTag(""))
        ->setComment("Label for module indexer to set SoA size");
    desc.add<edm::FileInPath>("fedjson")->setComment("JSON file with FED configuration parameters");
    desc.add<edm::FileInPath>("modjson")->setComment("JSON file with ECONT configuration parameters");
    desc.add<bool>("emulatorOn", false)->setComment("if true parse the emulator configuration parameters");

    descriptions.addWithDefaultLabel(desc);
  }

  // @short get hexadecimal value, and override if value_override>=0
  static int32_t gethex(const std::string& value, const int32_t value_override) {
    int32_t ret = (value_override >= 0 ? value_override : std::stoi(value, nullptr, 16));
    return ret;
  }

  // @short get integer value, and override if value_override>=0
  static int32_t getint(const int32_t value, const int32_t value_override) {
    return (value_override >= 0 ? value_override : value);
  }

  std::unique_ptr<HGCalTriggerConfiguration> produce(const HGCalModuleConfigurationRcd& iRecord) {
    auto const& moduleMap = iRecord.get(indexToken_);
    edm::LogInfo("HGCalTriggerConfigurationESProducer")
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
    const std::vector<std::string> fedkeys = {"tdaqHeaderMarker", "neconts", "econtSwapOffset"};
    const std::vector<std::string> modkeys = {
        "dropLSB", "select", "stc_type", "eporttx_numen", "use_sum", "mux"};
    const std::vector<std::string> emulkeys = {"density", "adc_th", "calv"};
    
    if (nfeds != fed_config_data.size())
      edm::LogWarning("HGCalTriggerConfigurationESProducer")
          << "Total number of FEDs found in JSON file " << fedjsonurl << " (" << fed_config_data.size()
          << ") does not match indexer (" << nfeds << ")";

    // loop over FEDs in indexer & fill configuration structs: FED > ECON-D > eRx
    // follow indexing by HGCalMappingModuleIndexer
    // HGCalTriggerConfiguration = container class holding FED structs of ECON-D structs of eRx structs
    std::unique_ptr<HGCalTriggerConfiguration> config_ = std::make_unique<HGCalTriggerConfiguration>();
    config_->feds.resize(moduleMap.maxFEDSize());
    for (const auto& tfed : moduleMap.fedReadoutSequences()) {
      if (tfed.readoutTypes_.size() == 0)
        continue;
      auto fedid = tfed.id;

      // sanity checks
      const auto fedkey = hgcal::search_fedkey(fedid, fed_config_data, fedjsonurl);  // search matching key
      hgcal::check_keys(
          fed_config_data, fedkey, fedkeys, fedjsonurl);  // check required keys are in the JSON, warn otherwise

      // count trigger blocks and compare to size of ECONTs list
      uint32_t nTDAQ = uint32_t(fed_config_data[fedkey]["neconts"].size());

      //count econs and address swaps and compare to baseline expectations from mapping
      uint32_t totalECONTs = 0;
      for (std::size_t itdaq = 0; itdaq < nTDAQ; itdaq++) {
        totalECONTs += uint32_t(fed_config_data[fedkey]["neconts"][itdaq]);
      }
      /*if (moduleMap.getNumModules(fedid) != fed_config_data[fedkey]["econtSwapOffset"].size() ||
          moduleMap.getNumModules(fedid) !=
              totalECONTs){ // check if length of sawp offsets, number of ECONTs in FED read from module locator, and number of econts summed mathces
        std::cout << " total ECONTs " << totalECONTs << " do not match " << moduleMap.getNumModules(fedid) << "in module locator"
                  << " or SWAP offset size "  << fed_config_data[fedkey]["econtSwapOffset"].size() 
                  << " do not match " << moduleMap.getNumModules(fedid) << "in module locator" << std::endl;
        continue;
      } *///TODO readd taking into acount TL type with 2 econts 

      std::cout << fedid << " has " << nTDAQ << " nTDAQ and " << totalECONTs << " ECONTs" << std::endl;
      // fill FED configurations
      HGCalTriggerFedConfig fedConfig;

      // fill econtSwapOffset
      fedConfig.econtSwapOffset.resize(moduleMap.getNumModules(fedid));
      for (std::size_t iecont = 0; iecont < moduleMap.getNumModules(fedid); iecont++) {
        fedConfig.econtSwapOffset[iecont] = int(fed_config_data[fedkey]["econtSwapOffset"][iecont]);
      }
      // fill elinksMap (only if present in cfg)
      if(hgcal::check_keys(fed_config_data, fedkey, {"elinksMap"}, fedjsonurl)) { 
        for (auto itdaq = fed_config_data[fedkey]["elinksMap"].begin(); itdaq!=fed_config_data[fedkey]["elinksMap"].end(); ++itdaq) {
          fedConfig.elinksMap[(uint8_t)stoi(itdaq.key())].resize(itdaq.value().size());
          for (std::size_t ielink = 0; ielink < itdaq.value().size(); ielink++) {   
            fedConfig.elinksMap[(uint8_t)stoi(itdaq.key())][ielink] = itdaq.value()[ielink];
          }
        }
      }
      // fill TDAQ configurations
      fedConfig.tdaqs.resize(nTDAQ);
      totalECONTs = 0;
      for (std::size_t itdaq = 0; itdaq < nTDAQ; itdaq++) {
        HGCalTDAQConfig tdaqConfig;
        tdaqConfig.tdaqBlockHeaderMarker =
            std::stoul(std::string(fed_config_data[fedkey]["tdaqHeaderMarker"]), nullptr, 16);
        uint32_t nECONT = uint32_t(fed_config_data[fedkey]["neconts"][itdaq]);
        tdaqConfig.econts.resize(nECONT);
        for (const auto& [typecode, ids] : moduleMap.typecodeMap()) {
          auto [fedid_, imod] = ids;
          if ((fedid_ != fedid) || !(totalECONTs <= imod && imod < totalECONTs + nECONT)) {
            continue;
          }
          const auto modkey = hgcal::search_modkey(typecode, mod_config_data, modjsonurl);  // search matching key

	  bool isSiPM = std::regex_match(typecode, std::regex(R"(T[LH]-.*)"));
          if (isSiPM) {
             if (nECONT != 2){
                  throw cms::Exception("Configuration") << "SiPM module " << modkey
                          << " requires exactly 2 ECON-Ts, but nECONT = " << nECONT;

            } 
          }
          uint32_t iecont = imod - totalECONTs;
          
          const auto& modcfg = !isSiPM    ? mod_config_data[modkey]    : mod_config_data[modkey][std::to_string(iecont)];
          tdaqConfig.econts.resize(nECONT);
          if (!isSiPM) {

          hgcal::check_keys(
              mod_config_data, modkey, modkeys, modjsonurl);  // check required unpacker keys are in the JSON, warn otherwise

          if (emulatorOn_) {
            edm::LogInfo("HGCalTriggerConfigurationESProducer")
            << "Emulator enabled for this run " << std::endl;
            hgcal::check_keys(
                mod_config_data, modkey, emulkeys, modjsonurl);  // If emulator is enabled, check required emulator keys are in the JSON, warn otherwise
          }



          //sanity check
          size_t nTC_mux = mod_config_data[modkey]["mux"].size();
          //size_t nTC = moduleMap.getNumChannels(typecode);
          size_t nTC_off = mod_config_data[modkey]["offset"].size();
          if (nTC_off != nTC_mux) {
            edm::LogInfo("HGCalTriggerConfigurationESProducer") << "Size of mux " << nTC_mux 
            << " does not match size of offset " << nTC_off 
            << ", skipped, check cfg file" << std::endl;
            continue;
          }
          HGCalECONTConfig econtConfig;

          econtConfig.dropLSB = uint8_t(mod_config_data[modkey]["dropLSB"]);
          econtConfig.select = uint8_t(mod_config_data[modkey]["select"]);
          econtConfig.stcType = uint8_t(mod_config_data[modkey]["stc_type"]);
          econtConfig.eportTxNumen = uint8_t(mod_config_data[modkey]["eporttx_numen"]);
          econtConfig.sumType = uint8_t(mod_config_data[modkey]["use_sum"]);
          if (emulatorOn_) {
            econtConfig.density = uint8_t(mod_config_data[modkey]["density"]);
            size_t nTC_calv = mod_config_data[modkey]["calv"].size();
            econtConfig.calv.resize(nTC_calv);
            //sanity check
            if (nTC_calv != nTC_mux) {
              edm::LogInfo("HGCalTriggerConfigurationESProducer") << "Size of mux " << nTC_mux 
              << " does not match size of calv " << nTC_calv
              << ", skipped, check cfg file" << std::endl;
              continue;
            }
          }
          econtConfig.tcMux.resize(nTC_mux);
          econtConfig.offset.resize(nTC_off);
          for (std::size_t iTC = 0; iTC < nTC_mux; iTC++) {
            if (emulatorOn_)  econtConfig.calv[iTC] = mod_config_data[modkey]["calv"][iTC];
            econtConfig.tcMux[iTC] = mod_config_data[modkey]["mux"][iTC];
            econtConfig.offset[iTC] = mod_config_data[modkey]["offset"][iTC];  
            //econtConfig.offset[iTC] = calculateCellOffset();  //TODO: change this when we know how to calcualte
          }
          if (emulatorOn_) {
            size_t nhgcroc = mod_config_data[modkey]["adc_th"].size(); // number of hgrocs per econt
            econtConfig.hgcrocs.resize(nhgcroc);
            for (std::size_t ihgcroc = 0; ihgcroc < nhgcroc; ihgcroc++){
              HGCalROCTrigConfig hgcrocConfig;
              hgcrocConfig.adc_th = mod_config_data[modkey]["adc_th"][ihgcroc];
              econtConfig.hgcrocs[ihgcroc] = hgcrocConfig;

            }
          }
          // Caculate module number in the TDAQ
          uint32_t iecont = imod - totalECONTs;
          tdaqConfig.econts.resize(nECONT);  //resize so length is the number of econTs
          tdaqConfig.econts[iecont] = econtConfig;


        }
        else { //SiPM
            for (int econtIdx = 0; econtIdx < 2; ++econtIdx) {
                const auto& modcfg = mod_config_data[modkey][std::to_string(econtIdx)];
                size_t nTC_calv = modcfg["calv"].size();
                size_t nTC_mux = modcfg["mux"].size();
                //size_t nTC = moduleMap.getNumChannels(typecode);
                size_t nTC = modcfg["mux"].size();
               if (nTC != nTC_mux || nTC != nTC_calv) {
                continue;
               }
          HGCalECONTConfig econtConfig;

          econtConfig.density = uint8_t(modcfg["density"]);
          econtConfig.dropLSB = uint8_t(modcfg["dropLSB"]);
          econtConfig.select = uint8_t(modcfg["select"]);
          econtConfig.stcType = uint8_t(modcfg["stc_type"]);
          econtConfig.eportTxNumen = uint8_t(modcfg["eporttx_numen"]);
          econtConfig.sumType = uint8_t(modcfg["use_sum"]);

          econtConfig.calv.resize(nTC);
          econtConfig.tcMux.resize(nTC);
          econtConfig.offset.resize(nTC);
          for (std::size_t iTC = 0; iTC < nTC; iTC++) {
            econtConfig.calv[iTC] = modcfg["calv"][iTC];
            econtConfig.tcMux[iTC] = modcfg["mux"][iTC];
            econtConfig.offset[iTC] = calculateCellOffset();  //TODO: change this when we know how to calcualte
          }
          tdaqConfig.econts[econtIdx] = econtConfig;

            }
          }
        }

        fedConfig.tdaqs[itdaq] = tdaqConfig;

        totalECONTs += uint32_t(fed_config_data[fedkey]["neconts"][itdaq]);
      }

      config_->feds[fedid] = fedConfig;
    }

    LogDebug("HGCalTriggerConfigurationESProducer") << *config_;
    return config_;
  }  // end of produce()

private:
  uint32_t calculateCellOffset() { return 0; }
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval& oValidity) override {
    oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
  }

  edm::ESGetToken<HGCalMappingModuleIndexerTrigger, HGCalElectronicsMappingRcd> indexToken_;
  const edm::FileInPath fedjson_;  // JSON file
  const edm::FileInPath modjson_;  // JSON file
  bool emulatorOn_ = false;

};

DEFINE_FWK_EVENTSETUP_SOURCE(HGCalTriggerConfigurationESProducer);

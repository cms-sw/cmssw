// Author: Izaak Neutelings (March 2024)
// Based on: https://github.com/CMS-HGCAL/cmssw/blob/hgcal-condformat-HGCalNANO-13_2_0_pre3_linearity/RecoLocalCalo/HGCalRecAlgos/plugins/alpaka/HGCalRecHitProducer.cc

// includes for CMSSW
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

// includes for Alpaka
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// includes for HGCal, calibration, and configuration parameters
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/HGCalObjects/interface/HGCalConfiguration.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalESProducerTools.h"  // for json, search_fedkey

// standard includes
#include <string>
#include <vector>
#include <iomanip>  // for std::setw

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class HGCalRecHitESProducersTest : public stream::EDProducer<> {
  public:
    explicit HGCalRecHitESProducersTest(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const int maxchans_, maxmods_, maxfeds_;
    const std::string fedjson_;  // JSON file of FED configuration
    void produce(device::Event&, device::EventSetup const&) override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    edm::ESWatcher<HGCalModuleConfigurationRcd> configWatcher_;
    edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexerToken_;
    edm::ESGetToken<HGCalConfiguration, HGCalModuleConfigurationRcd> configToken_;
    device::ESGetToken<hgcalrechit::HGCalCalibParamDevice, HGCalModuleConfigurationRcd> calibParamToken_;
  };

  HGCalRecHitESProducersTest::HGCalRecHitESProducersTest(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        maxchans_(iConfig.getParameter<int>("maxchans")),
        maxmods_(iConfig.getParameter<int>("maxmods")),
        maxfeds_(iConfig.getParameter<int>("maxfeds")),
        fedjson_(iConfig.getParameter<std::string>("fedjson")) {
    std::cout << "HGCalRecHitESProducersTest::HGCalRecHitESProducersTest" << std::endl;
    indexerToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
    configToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("configSource"));
    calibParamToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("calibParamSource"));
  }

  void HGCalRecHitESProducersTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add("indexSource", edm::ESInputTag{})->setComment("Label for module indexer to set SoA size");
    desc.add("configSource", edm::ESInputTag{})->setComment("Label for HGCal configuration for unpacking raw data");
    desc.add("calibParamSource", edm::ESInputTag{})->setComment("Label for calibration parameters");
    desc.add<int>("maxchans", 500)->setComment("Maximum number of channels to print");
    desc.add<int>("maxmods", 8)->setComment("Maximum number of modules to print");
    desc.add<int>("maxfeds", 25)->setComment("Maximum number of FED IDs to test");
    desc.add<std::string>("fedjson", "")->setComment("JSON file with FED configuration parameters");
    descriptions.addWithDefaultLabel(desc);
  }

  void HGCalRecHitESProducersTest::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
    std::cout << "HGCalRecHitESProducersTest::beginRun" << std::endl;
  }

  static std::string int2hex(int value) {
    std::stringstream stream;
    stream << "0x" << std::hex << value;
    return stream.str();
  }

  using CalibParamView = hgcalrechit::HGCalCalibParamSoALayout<>::ConstViewTemplateFreeParams<128, false, true, true>;
  static void printCalPars(const CalibParamView& calibView, const int idx) {
    const auto& calib = calibView[idx];
    std::cout << std::setw(6) << idx << std::setw(7) << int2hex(idx) << std::dec << std::setw(12) << calib.ADC_ped()
              << std::setw(11) << calib.CM_slope() << std::setw(9) << calib.CM_ped() << std::setw(11)
              << calib.EM_scale() << std::endl;
  }

  void HGCalRecHitESProducersTest::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    std::cout << "HGCalRecHitESProducersTest::produce" << std::endl;
    auto queue = iEvent.queue();
    auto const& moduleIndexer = iSetup.getData(indexerToken_);
    auto const& config = iSetup.getData(configToken_);  // HGCalConfiguration
    auto const& calibParamDevice = iSetup.getData(calibParamToken_);
    //printf("HGCalRecHitESProducersTest::produce: time to load calibParamDevice from calib ESProducers: %f seconds\n", duration(start,now()));

    // Check if there are new conditions and read them
    std::string line = "HGCalRecHitESProducersTest::produce " + std::string(90, '-');
    if (configWatcher_.check(iSetup)) {
      std::cout << line << std::endl;
      std::cout << "HGCalRecHitESProducersTest::produce: moduleIndexer.maxDataSize()=" << moduleIndexer.maxDataSize()
                << ", moduleIndexer.maxERxSize()=" << moduleIndexer.maxERxSize() << std::endl;

      // ESProducer for global HGCal configuration (structs) with header markers, etc.
      auto nfeds = config.feds.size();  // number of FEDs
      std::cout << "HGCalRecHitESProducersTest::produce: config=" << config << std::endl;
      std::cout << "HGCalRecHitESProducersTest::produce: nfeds=" << nfeds << ", config=" << config << std::endl;
      std::cout << "HGCalRecHitESProducersTest::produce: configuration:" << std::endl;
      for (std::size_t fedid = 0; fedid < nfeds; ++fedid) {
        auto fed = config.feds[fedid];   // HGCalFedConfig
        auto nmods = fed.econds.size();  // number of ECON-Ds for this FED
        std::cout << "  fedid=" << fedid << ", nmods=" << nmods << ", passthroughMode=" << fed.mismatchPassthroughMode
                  << ", cbHeaderMarker=0x" << std::hex << fed.cbHeaderMarker << ", slinkHeaderMarker=0x"
                  << fed.slinkHeaderMarker << std::dec << std::endl;
        std::cout << "  modid  nrocs  headerMarker" << std::endl;
        for (std::size_t modid = 0; modid < nmods; ++modid) {
          auto mod = fed.econds[modid];
          auto nrocs = mod.rocs.size();  // number of ECON-Ds for this FED
          std::cout << std::setw(7) << modid << std::setw(7) << nrocs << std::setw(14) << int2hex(mod.headerMarker)
                    << std::endl;
        }
      }

      // Alpaka ESProducer for SoA with calibration parameters with pedestals, etc. per channel
      std::cout << line << std::endl;
      int size = calibParamDevice.view().metadata().size();
      std::cout << "HGCalRecHitESProducersTest::produce: device size=" << size << std::endl;
      std::cout << "HGCalRecHitESProducersTest::produce: calibration constants per channel:" << std::endl;
      std::cout << "   idx    hex     ADC_ped   CM_slope   CM_ped   EM_scale" << std::endl;
      for (int idx = 0; idx < size; idx++) {
        if (idx >= maxchans_)
          break;
        printCalPars(calibParamDevice.view(), idx);
      }

      // per module
      int imod = 0;
      std::cout << "HGCalRecHitESProducersTest::produce: calibration constants per module:" << std::endl;
      std::cout << "  imod          typecode   idx    hex     ADC_ped   CM_slope   CM_ped   EM_scale" << std::endl;
      for (const auto& [typecode, ids] : moduleIndexer.typecodeMap()) {
        const auto [fedid, modid] = ids;
        if (imod >= maxmods_)
          break;
        const uint32_t offset = moduleIndexer.getIndexForModuleData(fedid, modid, 0, 0);
        uint32_t minoffset = (offset > 0 ? offset - 1 : offset);
        for (uint32_t idx = minoffset; idx <= offset + 1; idx++) {
          const std::string typecode_ = (idx == offset ? typecode : "");
          std::cout << std::setw(6) << imod << std::setw(18) << typecode_;
          printCalPars(calibParamDevice.view(), idx);
        }
        imod += 1;
      }
    }  // end if for configWatcher_::check

    // test JSON parser
    json fed_data;
    std::cout << line << std::endl;
    std::cout << "HGCalRecHitESProducersTest::produce: testing search_fedkey with " << fedjson_ << std::endl;
    if (fedjson_ == "") {
      fed_data = json::parse(R"({
        // numerical range
        "8-20": { },
        // glob pattern
        "[0-9]": { },
        // glob pattern
        "2[0-9]": { },
        // glob wildcard (default)
        "*": { }
      })",
                             nullptr,
                             true,
                             /*ignore_comments*/ true);
    } else {
      edm::FileInPath fedfip(fedjson_);  // e.g. HGCalCommissioning/LocalCalibration/data/config_feds.json
      std::ifstream fedfile(fedjson_);
      fed_data = json::parse(fedfile, nullptr, true, /*ignore_comments*/ true);
    }
    std::vector<std::string> fedkeys;
    for (int fedid = 0; fedid <= maxfeds_; fedid++) {
      const auto fedkey = hgcal::search_fedkey(fedid, fed_data, fedjson_);  // search matching key
      fedkeys.push_back(fedkey);
    }
    std::cout << "HGCalRecHitESProducersTest::produce: results from search_fedkey:" << fedjson_ << std::endl;
    std::cout << "   fedid   matched fedkey" << std::endl;
    for (int fedid = 0; fedid <= maxfeds_; fedid++) {
      std::cout << std::setw(8) << fedid << "   '" << fedkeys[fedid] << "'" << std::endl;
    }

    std::cout << line << std::endl;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define this as a plug-in
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalRecHitESProducersTest);

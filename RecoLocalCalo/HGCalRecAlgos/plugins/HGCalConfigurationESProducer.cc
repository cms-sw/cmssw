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


/**
   @short plugin parses HGCAL electronics configuration from JSON file
 */
class HGCalConfigurationESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  explicit HGCalConfigurationESProducer(const edm::ParameterSet& iConfig)
    : //edm::ESProducer(iConfig),
      filename_(iConfig.getParameter<std::string>("filename")) {
    auto cc = setWhatProduced(this);
    //findingRecord<HGCalModuleConfigurationRcd>();
    indexToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("indexSource"));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::ESInputTag>("indexSource",edm::ESInputTag(""))->setComment("Label for module indexer to set SoA size");
    //desc.add<edm::FileInPath>("filename")->setComment("JSON file with configuration parameters");
    desc.add<std::string>("filename","")->setComment("JSON file with configuration parameters");
    descriptions.addWithDefaultLabel(desc);
  }

  //std::optional<HGCalConfiguration> produce(const HGCalModuleConfigurationRcd& iRecord) {
  std::shared_ptr<HGCalConfiguration> produce(const HGCalModuleConfigurationRcd& iRecord) {
    std::cout << "HGCalConfigurationESProducer::produce" << std::endl;
    //// fill SoA with default placeholders
    //// TODO: retrieve values from custom JSON format (see HGCalCalibrationESProducer)
    //for (uint32_t imod = 0; imod < nmod; imod++) {
    //  uint8_t gain = gain_; // allow manual override
    //  product.view()[imod].gain() = gain;
    //}
    config_ = HGCalConfiguration();
    return std::shared_ptr<HGCalConfiguration>(&config_, edm::do_nothing_deleter());
  }  // end of produce()

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval& oValidity) override {
    oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
  }

  edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> indexToken_;
  const std::string filename_;
  HGCalConfiguration config_;
};


DEFINE_FWK_EVENTSETUP_SOURCE(HGCalConfigurationESProducer);

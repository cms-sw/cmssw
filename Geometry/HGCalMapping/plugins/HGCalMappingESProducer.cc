#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHostCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"

/**
   @short plugin parses the module/cell locator files to produce the indexer records
 */
class HGCalMappingESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  explicit HGCalMappingESProducer(const edm::ParameterSet& iConfig) {
    //parse the files and hold the list of entities in memory
    for (auto v : {"modules", "si", "sipm"}) {
      edm::FileInPath fip = iConfig.getParameter<edm::FileInPath>(v);
      hgcal::mappingtools::HGCalEntityList pmap;
      pmap.buildFrom(fip.fullPath());
      parsedMaps_[v] = pmap;
    }

    setWhatProduced(this, &HGCalMappingESProducer::produceCellMapIndexer);
    setWhatProduced(this, &HGCalMappingESProducer::produceModuleMapIndexer);

    findingRecord<HGCalElectronicsMappingRcd>();

    prepareCellMapperIndexer();
    prepareModuleMapperIndexer();
  }

  std::shared_ptr<HGCalMappingModuleIndexer> produceModuleMapIndexer(const HGCalElectronicsMappingRcd&) {
    return std::shared_ptr<HGCalMappingModuleIndexer>(&modIndexer_, edm::do_nothing_deleter());
  }

  std::shared_ptr<HGCalMappingCellIndexer> produceCellMapIndexer(const HGCalElectronicsMappingRcd&) {
    return std::shared_ptr<HGCalMappingCellIndexer>(&cellIndexer_, edm::do_nothing_deleter());
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::FileInPath>("modules")->setComment("module locator file");
    desc.add<edm::FileInPath>("si")->setComment("file containing the mapping of the readout cells in Si modules");
    desc.add<edm::FileInPath>("sipm")->setComment(
        "file containing the mapping of the readout cells in SiPM-on-tile modules");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval& oValidity) override {
    oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
  }

  void prepareCellMapperIndexer();
  void prepareModuleMapperIndexer();

  std::map<std::string, hgcal::mappingtools::HGCalEntityList> parsedMaps_;
  HGCalMappingCellIndexer cellIndexer_;
  HGCalMappingModuleIndexer modIndexer_;
};

//
void HGCalMappingESProducer::prepareCellMapperIndexer() {
  for (auto v : {"si", "sipm"}) {
    auto& pmap = parsedMaps_[v];
    const auto& entities = pmap.getEntries();
    for (auto row : entities) {
      std::string typecode = pmap.getAttr("Typecode", row);
      int chip = pmap.getIntAttr("ROC", row);
      int half = pmap.getIntAttr("HalfROC", row);
      cellIndexer_.processNewCell(typecode, chip, half);
    }
  }

  // all {hex,tile}board types are loaded finalize the mapping indexer
  cellIndexer_.update();
}

//
void HGCalMappingESProducer::prepareModuleMapperIndexer() {
  //default values to assign in case module type has not yet been mapped
  //a high density module (max possible) will be assigned so that the mapping doesn't block
  auto defaultTypeCodeIdx = cellIndexer_.getEnumFromTypecode("MH-F");
  auto typecodeidx = defaultTypeCodeIdx;
  auto defaultNerx = cellIndexer_.getNErxExpectedFor(defaultTypeCodeIdx);
  auto nerx = defaultNerx;
  auto defaultTypeNWords = cellIndexer_.getNWordsExpectedFor(defaultTypeCodeIdx);
  auto nwords = defaultTypeNWords;

  auto& pmap = parsedMaps_["modules"];
  auto& entities = pmap.getEntries();
  for (auto row : entities) {
    std::string typecode = pmap.getAttr("typecode", row);

    if (typecode.find('M') == 0 && typecode.size() > 4)
      typecode = typecode.substr(0, 4);

    try {
      typecodeidx = cellIndexer_.getEnumFromTypecode(typecode);
      nwords = cellIndexer_.getNWordsExpectedFor(typecode);
      nerx = cellIndexer_.getNErxExpectedFor(typecode);
    } catch (cms::Exception& e) {
      int plane = pmap.getIntAttr("plane", row);
      int u = pmap.getIntAttr("u", row);
      int v = pmap.getIntAttr("v", row);
      edm::LogWarning("HGCalMappingESProducer") << "Exception caught decoding index for typecode=" << typecode
                                                << " @ plane=" << plane << " u=" << u << " v=" << v << "\n"
                                                << e.what() << "\n"
                                                << "===> will assign default (MH-F) which may be inefficient";
      typecodeidx = defaultTypeCodeIdx;
      nwords = defaultTypeNWords;
      nerx = defaultNerx;
    }

    int fedid = pmap.getIntAttr("fedid", row);
    int captureblockidx = pmap.getIntAttr("captureblockidx", row);
    int econdidx = pmap.getIntAttr("econdidx", row);
    modIndexer_.processNewModule(fedid, captureblockidx, econdidx, typecodeidx, nerx, nwords);
  }

  modIndexer_.finalize();
}

DEFINE_FWK_EVENTSETUP_SOURCE(HGCalMappingESProducer);

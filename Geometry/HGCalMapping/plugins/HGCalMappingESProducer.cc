#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"
#include <regex>  // regular expression

/**
   @short plugin parses the module/cell locator files to produce the indexer records
 */
class HGCalMappingESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  explicit HGCalMappingESProducer(const edm::ParameterSet& iConfig) {
    //parse the files and hold the list of entities in memory
    modulesMap_.buildFrom(iConfig.getParameter<edm::FileInPath>("modules").fullPath());
    sicellsMap_.buildFrom(iConfig.getParameter<edm::FileInPath>("si").fullPath());
    sipmCellsMap_.buildFrom(iConfig.getParameter<edm::FileInPath>("sipm").fullPath());

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

  hgcal::mappingtools::HGCalEntityList modulesMap_, sicellsMap_, sipmCellsMap_;
  HGCalMappingCellIndexer cellIndexer_;
  HGCalMappingModuleIndexer modIndexer_;
};

//
void HGCalMappingESProducer::prepareCellMapperIndexer() {
  for (size_t i = 0; i < 2; i++) {
    const auto& pmap = i == 0 ? sicellsMap_ : sipmCellsMap_;
    const auto& entities = pmap.getEntries();
    for (const auto& row : entities) {
      const std::string& typecode = pmap.getAttr("Typecode", row);
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

  const auto& entities = modulesMap_.getEntries();
  for (const auto& row : entities) {
    const std::string& typecode = modulesMap_.getAttr("typecode", row);  // module type code
    std::string wtypecode;                                               // wafer type code

    // match module type code to regular expression pattern (MM-TTTT-LL-NNNN)
    // see https://edms.cern.ch/ui/#!master/navigator/document?D:101059405:101148061:subDocs
    //const std::regex typecode_regex("([MX])([LH])-([FTBLR5])([123])([WPC])-([A-Z]{2})-([0-9]{3,4})"); // MM-TTTT-LL-NNNN
    const std::regex typecode_regex("(([MX])([LH])-([FTBLR5])).*");  // MM-T*
    std::smatch typecode_match;                                      // match object for string objects
    bool matched = std::regex_match(typecode, typecode_match, typecode_regex);
    if (matched) {
      wtypecode = typecode_match[1].str();  // wafer type following MM-T pattern, e.g. "MH-F"
    } else {
      const std::regex sipm_typecode_regex(R"(^T(.*)-L([0-9]+)S([0-9]+)(?:-(.*))?$)");
      std::smatch sipm_typecode_match;  // match object for string objects
      matched = std::regex_match(typecode, sipm_typecode_match, sipm_typecode_regex);
      if (matched) {
        wtypecode = sipm_typecode_match[0].str();  // assign sipm typecode as wafer type
      } else {
        throw cms::Exception("HGCalMappingIndexESSource")
            << "Could not match module type code to expected pattern: " << typecode;
      }
    }

    typecodeidx = cellIndexer_.getEnumFromTypecode(wtypecode);
    nwords = cellIndexer_.getNWordsExpectedFor(wtypecode);
    nerx = cellIndexer_.getNErxExpectedFor(wtypecode);

    int fedid = modulesMap_.getIntAttr("fedid", row);
    int captureblockidx = modulesMap_.getIntAttr("captureblockidx", row);
    int econdidx = modulesMap_.getIntAttr("econdidx", row);
    modIndexer_.processNewModule(fedid, captureblockidx, econdidx, typecodeidx, nerx, nwords, typecode);
  }

  modIndexer_.finalize();
}

DEFINE_FWK_EVENTSETUP_SOURCE(HGCalMappingESProducer);

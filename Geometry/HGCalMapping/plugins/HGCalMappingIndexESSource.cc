#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"
#include "CondFormats/DataRecord/interface/HGCalMappingCellIndexerRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex> // regular expression

/**
   @short plugin parses the module/cell locator files to produce the indexer records
 */
class HGCalMappingIndexESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  explicit HGCalMappingIndexESSource(const edm::ParameterSet& iConfig)
      : module_filename_(iConfig.getParameter<edm::FileInPath>("modules")),
        si_filename_(iConfig.getParameter<edm::FileInPath>("si")),
        sipm_filename_(iConfig.getParameter<edm::FileInPath>("sipm")) {
    setWhatProduced(this, &HGCalMappingIndexESSource::produceCellMapIndexer);
    setWhatProduced(this, &HGCalMappingIndexESSource::produceModuleMapIndexer);

    findingRecord<HGCalMappingModuleIndexerRcd>();
    findingRecord<HGCalMappingCellIndexerRcd>();

    buildCellMapperIndexer();
    buildModuleMapperIndexer();
  }

  std::shared_ptr<HGCalMappingModuleIndexer> produceModuleMapIndexer(const HGCalMappingModuleIndexerRcd&) {
    return std::shared_ptr<HGCalMappingModuleIndexer>(&modIndexer_, edm::do_nothing_deleter());
  }

  std::shared_ptr<HGCalMappingCellIndexer> produceCellMapIndexer(const HGCalMappingCellIndexerRcd&) {
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

  void buildCellMapperIndexer();
  void buildModuleMapperIndexer();

  HGCalMappingCellIndexer cellIndexer_;
  HGCalMappingModuleIndexer modIndexer_;

  const edm::FileInPath module_filename_, si_filename_, sipm_filename_;
};

//
void HGCalMappingIndexESSource::buildCellMapperIndexer() {
  // load Si cell specific module mapping parameters
  std::ifstream file(si_filename_.fullPath());

  size_t iline(0);
  std::string line, typecode;
  uint16_t chip, half;
  while (std::getline(file, line)) {
    iline++;
    if (iline == 1)
      continue;
    std::istringstream stream(line);
    stream >> typecode;
    stream >> chip >> half;
    cellIndexer_.processNewCell(typecode, chip, half);
  }

  // load SiPM cell specific module mapping parameters
  file = std::ifstream(sipm_filename_.fullPath());
  iline = 0;
  while (std::getline(file, line)) {
    iline++;
    if (iline == 1)
      continue;
    std::istringstream stream(line);

    stream >> typecode;
    stream >> chip >> half;
    cellIndexer_.processNewCell(typecode, chip, half);
  }

  // all {hex,tile}board types are loaded finalize the mapping
  cellIndexer_.update();
}

//
void HGCalMappingIndexESSource::buildModuleMapperIndexer() {
  //default values to assign in case module type has not yet been mapped
  //a high density module (max possible) will be assigned so that the mapping doesn't block
  auto defaultTypeCodeIdx = cellIndexer_.getEnumFromTypecode("MH-F");
  auto typecodeidx = defaultTypeCodeIdx;
  auto defaultNerx = cellIndexer_.getNErxExpectedFor(defaultTypeCodeIdx);
  auto nerx = defaultNerx;
  auto defaultTypeNWords = cellIndexer_.getNWordsExpectedFor(defaultTypeCodeIdx);
  auto nwords = defaultTypeNWords;
  //const std::regex typecode_regex("([MX])([LH])-([FTBLR5])([123])([WPC])-([A-Z]{2})-([0-9]{3,4})"); // MM-TTTT-LL-NNNN
  const std::regex typecode_regex("(([MX])([LH])-([FTBLR5])).*"); // MM-T

  // load module mapping parameters and find ranges
  std::ifstream file(module_filename_.fullPath());
  std::string line;
  size_t iline(0);
  while (std::getline(file, line)) {
    iline++;
    if (iline == 1)
      continue;
    if (line.length()>=1 and line[0]=='#')
      continue; // ignore line

    std::string typecode, wtypecode; // module and wafer type code
    int plane, u, v, zside;
    uint16_t fedid, slinkidx, captureblock, econdidx, captureblockidx;
    std::istringstream stream(line);
    stream >> plane >> u >> v >> typecode >> econdidx >> captureblock >> captureblockidx >> slinkidx >> fedid >> zside;

    // match module type code to regular expression pattern (MM-TTTT-LL-NNNN)
    std::smatch typecode_match; // match object for string objects
    bool matched = std::regex_match(typecode,typecode_match,typecode_regex);
    if (matched){
      wtypecode = typecode_match[1].str(); // wafer type following MM-T pattern, e.g. "MH-F"
    }else{
      // https://edms.cern.ch/ui/#!master/navigator/document?D:101059405:101148061:subDocs
      edm::LogWarning("HGCalMappingIndexESSource") << "Could not match module type code to expected pattern: " << typecode;
    }

    try {
      typecodeidx = cellIndexer_.getEnumFromTypecode(wtypecode);
      nwords = cellIndexer_.getNWordsExpectedFor(wtypecode);
      nerx = cellIndexer_.getNErxExpectedFor(wtypecode);
    } catch (cms::Exception& e) {
      edm::LogWarning("HGCalMappingIndexESSource") << "Exception caught decoding index for typecode=" << typecode
                                                   << " (wtypecode=" << wtypecode << ")"
                                                   << " @ plane=" << plane << " u=" << u << " v=" << v << "\n"
                                                   << e.what() << "\n"
                                                   << "===> will assign default (MH-F) which may be inefficient";
      typecodeidx = defaultTypeCodeIdx;
      nwords = defaultTypeNWords;
      nerx = defaultNerx;
    }

    modIndexer_.processNewModule(fedid, captureblockidx, econdidx, typecodeidx, nerx, nwords, typecode);
  }

  modIndexer_.finalize();
}

DEFINE_FWK_EVENTSETUP_SOURCE(HGCalMappingIndexESSource);

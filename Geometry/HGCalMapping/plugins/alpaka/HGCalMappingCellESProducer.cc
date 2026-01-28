#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <charconv>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcal {

    class HGCalMappingCellESProducer : public ESProducer {
    public:
      //
      HGCalMappingCellESProducer(const edm::ParameterSet& iConfig)
          : ESProducer(iConfig),
            filelist_(iConfig.getParameter<std::vector<std::string> >("filelist")),
            offsetfile_(iConfig.getParameter<edm::FileInPath>("offsetfile")) {
        auto cc = setWhatProduced(this);
        cellIndexTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("cellindexer"));
        moduleIndexTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("moduleindexer"));
      }

      //
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<std::vector<std::string> >("filelist", std::vector<std::string>({}))
            ->setComment("list of files with the readout cells of each module");
        desc.add<edm::ESInputTag>("cellindexer", edm::ESInputTag(""))->setComment("Dense cell index tool");
        desc.add<edm::ESInputTag>("moduleindexer", edm::ESInputTag(""))->setComment("Module index tool");
        desc.add<edm::FileInPath>(
                "offsetfile",
                edm::FileInPath("Geometry/HGCalMapping/data/CellMaps/calibration_to_surrounding_offsetMap.txt"))
            ->setComment("file containing the offsets between calibration and surrounding cells");
        descriptions.addWithDefaultLabel(desc);
      }

      std::map<int, int> makeOffsetMap(edm::FileInPath input_offsetfile,
                                       const HGCalMappingCellIndexer& cellIndexer,
                                       const HGCalMappingModuleIndexer& moduleIndexer) {
        std::map<int, int> offsetMap;
        const auto& offsetfile = input_offsetfile.fullPath();
        ::hgcal::mappingtools::HGCalEntityList omap;
        edm::FileInPath fip(offsetfile);
        omap.buildFrom(fip.fullPath());

        const auto& mapEntries = omap.getEntries();
        for (const auto& row : mapEntries) {
          std::string typecode = omap.getAttr("Typecode", row);
          const auto& allTypecodes = moduleIndexer.typecodeMap();
          // Skip if typecode is not in the module indexer
          bool typecodeFound = false;
          for (const auto& key : allTypecodes) {
            if (key.first.find(typecode) != std::string::npos) {
              typecodeFound = true;
              break;
            }
          }
          if (!typecodeFound)
            continue;
          int roc = omap.getIntAttr("ROC", row);
          int halfroc = omap.getIntAttr("HalfROC", row);
          int readoutsequence = omap.getIntAttr("Seq", row);
          int offset = omap.getIntAttr("Offset", row);
          int idx = cellIndexer.denseIndex(typecode, roc, halfroc, readoutsequence);
          offsetMap[idx] = offset;
        }
        return offsetMap;
      }

      //
      std::optional<HGCalMappingCellParamHost> produce(const HGCalElectronicsMappingRcd& iRecord) {
        //get cell and module indexers
        const HGCalMappingCellIndexer& cellIndexer = iRecord.get(cellIndexTkn_);
        const HGCalMappingModuleIndexer& moduleIndexer = iRecord.get(moduleIndexTkn_);
        const uint32_t size = cellIndexer.maxDenseIndex();  // channel-level size
        HGCalMappingCellParamHost cellParams(cms::alpakatools::host(), size);
        for (uint32_t i = 0; i < size; i++)
          cellParams.view()[i].valid() = false;

        auto offsetMap = makeOffsetMap(offsetfile_, cellIndexer, moduleIndexer);

        //loop over cell types and then over cells
        for (const auto& url : filelist_) {
          ::hgcal::mappingtools::HGCalEntityList pmap;
          edm::FileInPath fip(url);
          pmap.buildFrom(fip.fullPath());
          const auto& entities = pmap.getEntries();
          for (const auto& row : entities) {
            //identify special cases (Si vs SiPM, calib vs normal)
            const std::string& typecode = pmap.getAttr("Typecode", row);
            auto typeidx = cellIndexer.getEnumFromTypecode(typecode);
            bool isSiPM = (typecode[0] == 'T');
            int rocpin = pmap.getIntAttr("ROCpin", row);
            int celltype = pmap.getIntAttr("t", row);
            int i1(0), i2(0), sensorcell(0);
            bool isHD(false), iscalib(false);
            uint32_t detid(0);
            if (isSiPM) {
              i1 = pmap.getIntAttr("iring", row);
              i2 = pmap.getIntAttr("iphi", row);
              isHD = {typecode.find("TH") != std::string::npos ? true : false};
            } else {
              i1 = pmap.getIntAttr("iu", row);
              i2 = pmap.getIntAttr("iv", row);
              isHD = {typecode.find("MH") != std::string::npos ? true : false};
              sensorcell = pmap.getIntAttr("SiCell", row);
              if (celltype == 0) {
                iscalib = true;
                std::string rocpinstr = pmap.getAttr("ROCpin", row);
                rocpin = uint16_t(rocpinstr[rocpinstr.size() - 1]);
              }

              //detector id is initiated for a random sub-detector with Si wafers
              //we only care about the first 10bits where the cell u-v are stored
              DetId::Detector det(DetId::Detector::HGCalEE);
              detid = 0x3ff & HGCSiliconDetId(det, 0, 0, 0, 0, 0, i1, i2).rawId();
            }

            //fill cell info in the appopriate dense index
            int chip = pmap.getIntAttr("ROC", row);
            int half = pmap.getIntAttr("HalfROC", row);
            int seq = pmap.getIntAttr("Seq", row);
            int idx = cellIndexer.denseIndex(typecode, chip, half, seq);
            auto cell = cellParams.view()[idx];
            cell.valid() = true;
            cell.isHD() = isHD;
            cell.iscalib() = iscalib;
            cell.isSiPM() = isSiPM;
            cell.typeidx() = typeidx;
            cell.chip() = chip;
            cell.half() = half;
            cell.seq() = seq;
            cell.rocpin() = rocpin;
            cell.sensorcell() = sensorcell;
            cell.triglink() = pmap.getIntAttr("TrLink", row);
            cell.trigcell() = pmap.getIntAttr("TrCell", row);
            cell.i1() = i1;
            cell.i2() = i2;
            cell.t() = celltype;
            cell.trace() = pmap.getFloatAttr("trace", row);
            cell.eleid() = HGCalElectronicsId(false, 0, 0, 0, chip * 2 + half, seq).raw();
            cell.detid() = detid;

            int offset = (iscalib && offsetMap.find(idx) != offsetMap.end()) ? offsetMap[idx] : 0;
            cell.caliboffset() = offset;

          }  //end loop over entities
        }  //end loop over cell types

        return cellParams;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingCellIndexer, HGCalElectronicsMappingRcd> cellIndexTkn_;
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> moduleIndexTkn_;
      const std::vector<std::string> filelist_;
      edm::FileInPath offsetfile_;
    };

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcal::HGCalMappingCellESProducer);

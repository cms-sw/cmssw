#include "FWCore/ParameterSet/interface/FileInPath.h"
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
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcal {

    class HGCalMappingCellESProducer : public ESProducer {
    public:
      //
      HGCalMappingCellESProducer(const edm::ParameterSet& iConfig)
          : ESProducer(iConfig), filelist_(iConfig.getParameter<std::vector<std::string> >("filelist")) {
        auto cc = setWhatProduced(this);
        cellIndexTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("cellindexer"));
      }

      //
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<std::vector<std::string> >("filelist", std::vector<std::string>({}))
            ->setComment("list of files with the readout cells of each module");
        desc.add<edm::ESInputTag>("cellindexer", edm::ESInputTag(""))->setComment("Dense cell index tool");
        descriptions.addWithDefaultLabel(desc);
      }

      //
      std::optional<HGCalMappingCellParamHost> produce(const HGCalElectronicsMappingRcd& iRecord) {
        //get cell indexer
        const HGCalMappingCellIndexer& cellIndexer = iRecord.get(cellIndexTkn_);
        const uint32_t size = cellIndexer.maxDenseIndex();  // channel-level size
        HGCalMappingCellParamHost cellParams(size, cms::alpakatools::host());
        for (uint32_t i = 0; i < size; i++)
          cellParams.view()[i].valid() = false;

        //loop over cell types and then over cells
        for (auto url : filelist_) {
          ::hgcal::mappingtools::HGCalEntityList pmap;
          edm::FileInPath fip(url);
          pmap.buildFrom(fip.fullPath());
          auto& entities = pmap.getEntries();
          for (auto row : entities) {
            //identify special cases (Si vs SiPM, calib vs normal)
            std::string typecode = pmap.getAttr("Typecode", row);
            auto typeidx = cellIndexer.getEnumFromTypecode(typecode);
            bool isSiPM = typecode.find("TM") != std::string::npos;
            int rocpin = pmap.getIntAttr("ROCpin", row);
            int celltype = pmap.getIntAttr("t", row);
            int i1(0), i2(0), sensorcell(0);
            bool isHD(false), iscalib(false);
            uint32_t detid(0);
            if (isSiPM) {
              i1 = pmap.getIntAttr("iring", row);
              i2 = pmap.getIntAttr("iphi", row);
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
          }  //end loop over entities
        }    //end loop over cell types

        return cellParams;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingCellIndexer, HGCalElectronicsMappingRcd> cellIndexTkn_;
      const std::vector<std::string> filelist_;
    };

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcal::HGCalMappingCellESProducer);

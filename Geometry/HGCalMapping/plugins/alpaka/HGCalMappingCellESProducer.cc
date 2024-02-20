#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "CondFormats/DataRecord/interface/HGCalMappingCellIndexerRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHostCollection.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDeviceCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

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
        desc.add<std::vector<std::string> >("filelist", {});
        desc.add<edm::ESInputTag>("cellindexer", edm::ESInputTag(""))->setComment("Dense cell index tool");
        descriptions.addWithDefaultLabel(desc);
      }

      //
      std::optional<HGCalMappingCellParamHostCollection> produce(const HGCalMappingCellIndexerRcd& iRecord) {
        //get cell indexer
        const HGCalMappingCellIndexer& cellIndexer = iRecord.get(cellIndexTkn_);

        const uint32_t size = cellIndexer.maxDenseIndex();  // channel-level size
        HGCalMappingCellParamHostCollection cellParams(size, cms::alpakatools::host());
        for (uint32_t i = 0; i < size; i++)
          cellParams.view()[i].valid() = false;

        for (const auto& filename : filelist_) {
          //open file and read the first line to identify which type it is
          edm::FileInPath fip(filename);
          std::ifstream file(fip.fullPath());

          //parse file and fill the SoA with the cell info
          std::string line, typecode, rocpincol;
          size_t iline(0);
          bool isHD(false), iscalib(false), isSiPM(false);
          uint16_t typeidx, chip, half;
          uint16_t seq, rocpin;
          int cellidx, triglink, trigcell, i1, i2, t;
          float trace(0);
          uint32_t eleid, detid(0);

          while (std::getline(file, line)) {
            iline++;
            if (iline == 1)
              continue;
            if (iline == 2) {
              std::istringstream stream0(line);
              stream0 >> typecode;
              isSiPM = {typecode.find("TM") != std::string::npos ? true : false};
            }

            std::istringstream stream(line);

            //SiPM version
            if (isSiPM) {
              stream >> typecode >> chip >> half >> cellidx >> seq >> i1 >> i2 >> trigcell >> triglink >> t;
              rocpin = cellidx;
              detid = 0;
            }

            //Si version
            else {
              stream >> typecode >> chip >> half >> seq >> rocpincol >> cellidx >> triglink >> trigcell >> i1 >> i2 >>
                  trace >> t;

              isHD = {typecode.find("MH") != std::string::npos ? true : false};

              if (rocpincol.find("CALIB") != std::string::npos) {
                iscalib = true;
                rocpin = uint16_t(rocpincol[rocpincol.size() - 1]);
              } else {
                iscalib = false;
                rocpin = std::stoi(rocpincol);
              }

              //detector id is initiated for a random sub-detector with Si wafers
              //we only care about the first 10bits where the cell u-v are stored
              DetId::Detector det(DetId::Detector::HGCalEE);
              detid = 0x3ff & HGCSiliconDetId(det, 0, 0, 0, 0, 0, i1, i2).rawId();
            }

            typeidx = cellIndexer.getEnumFromTypecode(typecode);

            uint16_t econderx = chip * 2 + half;
            eleid = HGCalElectronicsId(false, 0, 0, 0, econderx, seq).raw();

            //get dense index and fill the values
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
            cell.cellidx() = cellidx;
            cell.triglink() = triglink;
            cell.trigcell() = trigcell;
            cell.i1() = i1;
            cell.i2() = i2;
            cell.t() = t;
            cell.trace() = trace;
            cell.eleid() = eleid;
            cell.detid() = detid;
          }
        }

        return cellParams;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingCellIndexer, HGCalMappingCellIndexerRcd> cellIndexTkn_;
      const std::vector<std::string> filelist_;
    };

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcal::HGCalMappingCellESProducer);

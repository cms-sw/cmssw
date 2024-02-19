#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
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

    class HGCalMappingModuleESProducer : public ESProducer {
    public:
      //
      HGCalMappingModuleESProducer(const edm::ParameterSet& iConfig)
          : ESProducer(iConfig), filename_(iConfig.getParameter<std::string>("filename")) {
        auto cc = setWhatProduced(this);
        moduleIndexTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("moduleindexer"));
      }

      //
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<std::string>("filename", "Geometry/HGCalMapping/ModuleMaps/modulelocator.txt");
        desc.add<edm::ESInputTag>("moduleindexer", edm::ESInputTag(""))->setComment("Dense module index tool");
        descriptions.addWithDefaultLabel(desc);
      }

      //
      std::optional<HGCalMappingModuleParamHostCollection> produce(const HGCalMappingModuleIndexerRcd& iRecord) {
        //get cell and module indexer
        auto modIndexer = iRecord.get(moduleIndexTkn_);

        // load dense indexing
        const uint32_t size = modIndexer.maxModulesIdx_;
        HGCalMappingModuleParamHostCollection moduleParams(size, cms::alpakatools::host());
        for (size_t i = 0; i < size; i++)
          moduleParams.view()[i].valid() = false;

        // load module mapping parameters
        edm::FileInPath fip(filename_);
        std::ifstream file(fip.fullPath());
        std::string line, typecode;
        size_t iline(0);
        int plane, i1, i2, zside, celltype;
        uint16_t fedid, slinkidx, captureblock, econdidx, captureblockidx, typeidx;
        bool isSiPM;
        uint32_t idx, eleid, detid(0);

        while (std::getline(file, line)) {
          iline++;
          if (iline == 1)
            continue;

          std::istringstream stream(line);
          stream >> plane >> i1 >> i2 >> typecode >> econdidx >> captureblock >> captureblockidx >> slinkidx >> fedid >>
              zside;

          idx = modIndexer.getIndexForModule(fedid, captureblockidx, econdidx);

          typeidx = modIndexer.getTypeForModule(fedid, captureblockidx, econdidx);

          auto celltypes = modIndexer.convertTypeCode(typecode);
          isSiPM = celltypes.first;
          celltype = celltypes.second;

          eleid = HGCalElectronicsId((zside > 0), fedid, captureblock, econdidx, 0, 0).raw();

          if (!isSiPM) {
            int zp(zside > 0 ? 1 : -1);
            DetId::Detector det = plane <= 26 ? DetId::Detector::HGCalEE : DetId::Detector::HGCalHSi;
            detid = HGCSiliconDetId(det, zp, celltype, plane, i1, i2, 0, 0).rawId();
          }

          moduleParams.view()[idx].valid() = true;
          moduleParams.view()[idx].zside() = (zside > 0);
          moduleParams.view()[idx].isSiPM() = isSiPM;
          moduleParams.view()[idx].celltype() = celltype;
          moduleParams.view()[idx].plane() = plane;
          moduleParams.view()[idx].i1() = i1;
          moduleParams.view()[idx].i2() = i2;
          moduleParams.view()[idx].typeidx() = typeidx;
          moduleParams.view()[idx].fedid() = fedid;
          moduleParams.view()[idx].slinkidx() = slinkidx;
          moduleParams.view()[idx].captureblock() = captureblock;
          moduleParams.view()[idx].econdidx() = econdidx;
          moduleParams.view()[idx].captureblockidx() = captureblockidx;
          moduleParams.view()[idx].eleid() = eleid;
          moduleParams.view()[idx].detid() = detid;
        }

        return moduleParams;

      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalMappingModuleIndexerRcd> moduleIndexTkn_;
      const std::string filename_;
    };

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcal::HGCalMappingModuleESProducer);

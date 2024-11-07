#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
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

    class HGCalMappingModuleESProducer : public ESProducer {
    public:
      //
      HGCalMappingModuleESProducer(const edm::ParameterSet& iConfig)
          : ESProducer(iConfig), filename_(iConfig.getParameter<edm::FileInPath>("filename")) {
        auto cc = setWhatProduced(this);
        moduleIndexTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("moduleindexer"));
      }

      //
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::FileInPath>("filename")->setComment("module locator file");
        desc.add<edm::ESInputTag>("moduleindexer", edm::ESInputTag(""))->setComment("Dense module index tool");
        descriptions.addWithDefaultLabel(desc);
      }

      //
      std::optional<HGCalMappingModuleParamHost> produce(const HGCalElectronicsMappingRcd& iRecord) {
        //get cell and module indexer
        auto modIndexer = iRecord.get(moduleIndexTkn_);

        // load dense indexing
        const uint32_t size = modIndexer.maxModulesIndex();
        HGCalMappingModuleParamHost moduleParams(size, cms::alpakatools::host());
        for (size_t i = 0; i < size; i++)
          moduleParams.view()[i].valid() = false;

        ::hgcal::mappingtools::HGCalEntityList pmap;
        pmap.buildFrom(filename_.fullPath());
        auto& entities = pmap.getEntries();
        for (auto row : entities) {
          int fedid = pmap.getIntAttr("fedid", row);
          int captureblockidx = pmap.getIntAttr("captureblockidx", row);
          int econdidx = pmap.getIntAttr("econdidx", row);
          int idx = modIndexer.getIndexForModule(fedid, captureblockidx, econdidx);
          int typeidx = modIndexer.getTypeForModule(fedid, captureblockidx, econdidx);
          std::string typecode = pmap.getAttr("typecode", row);

          auto celltypes = modIndexer.convertTypeCode(typecode);
          bool isSiPM = celltypes.first;
          int celltype = celltypes.second;
          int zside = pmap.getIntAttr("zside", row);
          int plane = pmap.getIntAttr("plane", row);
          int i1 = pmap.getIntAttr("u", row);
          int i2 = pmap.getIntAttr("v", row);
          uint8_t irot = (uint8_t)(pmap.hasColumn("irot") ? pmap.getIntAttr("irot", row) : 0);
          uint32_t eleid = HGCalElectronicsId((zside > 0), fedid, captureblockidx, econdidx, 0, 0).raw();
          uint32_t detid(0);

          if (!isSiPM) {
            int zp(zside > 0 ? 1 : -1);
            DetId::Detector det = plane <= 26 ? DetId::Detector::HGCalEE : DetId::Detector::HGCalHSi;
            detid = HGCSiliconDetId(det, zp, celltype, plane, i1, i2, 0, 0).rawId();
          }

          auto module = moduleParams.view()[idx];
          module.valid() = true;
          module.zside() = (zside > 0);
          module.isSiPM() = isSiPM;
          module.celltype() = celltype;
          module.plane() = plane;
          module.i1() = i1;
          module.i2() = i2;
          module.irot() = irot;
          module.typeidx() = typeidx;
          module.fedid() = fedid;
          module.slinkidx() = pmap.getIntAttr("slinkidx", row);
          module.captureblock() = pmap.getIntAttr("captureblock", row);
          module.econdidx() = econdidx;
          module.captureblockidx() = captureblockidx;
          module.eleid() = eleid;
          module.detid() = detid;
        }

        return moduleParams;

      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> moduleIndexTkn_;
      const edm::FileInPath filename_;
    };

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcal::HGCalMappingModuleESProducer);

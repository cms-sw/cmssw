#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexerTrigger.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcal {

    class HGCalMappingTriggerModuleESProducer : public ESProducer {
    public:
      //
      HGCalMappingTriggerModuleESProducer(const edm::ParameterSet& iConfig)
          : ESProducer(iConfig), filename_(iConfig.getParameter<edm::FileInPath>("filename")) {
        auto cc = setWhatProduced(this);
        moduleIndexTriggerTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("moduleindexer"));
      }

      //
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::FileInPath>("filename")->setComment("module locator file");
        desc.add<edm::ESInputTag>("moduleindexer", edm::ESInputTag(""))->setComment("Dense module index tool");
        descriptions.addWithDefaultLabel(desc);
      }

      //
      std::optional<HGCalMappingModuleTriggerParamHost> produce(const HGCalElectronicsMappingRcd& iRecord) {
        //get cell and module indexer
        const auto& modIndexer = iRecord.get(moduleIndexTriggerTkn_);

        // load dense indexing
        const uint32_t size = modIndexer.maxModulesIndex();
        HGCalMappingModuleTriggerParamHost moduleParams(cms::alpakatools::host(), size);
        for (size_t i = 0; i < size; i++)
          moduleParams.view()[i].valid() = false;

        ::hgcal::mappingtools::HGCalEntityList pmap;
        pmap.buildFrom(filename_.fullPath());
        const auto& entities = pmap.getEntries();
        for (const auto& row : entities) {
          int cassette = pmap.hasColumn("cassette") ? pmap.getIntAttr("cassette", row) : 1;
          int fedid = pmap.getIntAttr("trig_fedid", row);
          int econtidx = pmap.getIntAttr("econtidx", row);
          int idx = modIndexer.getIndexForModule(fedid, static_cast<uint16_t>(econtidx));
          int typeidx = modIndexer.getTypeForModule(fedid, static_cast<uint16_t>(econtidx));
          const std::string& typecode = pmap.getAttr("typecode", row);

          auto celltypes = modIndexer.getCellType(typecode);
          bool isSiPM = celltypes.first;
          int celltype = celltypes.second;
          int zside = pmap.getIntAttr("zside", row);
          int plane = pmap.getIntAttr("plane", row);
          int i1 = pmap.getIntAttr("u", row);
          int i2 = pmap.getIntAttr("v", row);
          uint8_t irot = (uint8_t)(pmap.hasColumn("irot") ? pmap.getIntAttr("irot", row) : 0);
          // TODO : muxid needs to be discussed with BE and will be assigned correctly in the next iteration
          uint32_t muxid(0);
          uint32_t trigdetid(0);
          if (!isSiPM) {
            int zp(zside > 0 ? 1 : -1);
            int subdet = ForwardSubdetector::ForwardEmpty;
            trigdetid = HGCalTriggerDetId(subdet, zp, celltype, plane, i1, i2, 0, 0).rawId();
          }

          auto module = moduleParams.view()[idx];
          module.valid() = true;
          module.zside() = (zside > 0);
          module.isSiPM() = isSiPM;
          module.plane() = plane;
          module.i1() = i1;
          module.i2() = i2;
          module.irot() = irot;
          module.celltype() = celltype;
          module.typeidx() = typeidx;
          module.fedid() = fedid;
          module.slinkidx() = pmap.getIntAttr("slinkidx", row);
          module.econtidx() = econtidx;
          module.muxid() = muxid;
          module.trigdetid() = trigdetid;
          module.cassette() = cassette;
        }

        return moduleParams;

      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexerTrigger, HGCalElectronicsMappingRcd> moduleIndexTriggerTkn_;
      const edm::FileInPath filename_;
    };

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcal::HGCalMappingTriggerModuleESProducer);

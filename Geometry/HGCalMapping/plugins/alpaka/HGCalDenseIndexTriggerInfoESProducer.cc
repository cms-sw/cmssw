#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexerTrigger.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"

#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexerTrigger.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcal {

    class HGCalDenseIndexTriggerInfoESProducer : public ESProducer {
    public:
      //
      HGCalDenseIndexTriggerInfoESProducer(const edm::ParameterSet& iConfig) : ESProducer(iConfig) {
        auto cc = setWhatProduced(this);
        moduleIndexTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("moduleindexer"));
        cellIndexTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("cellindexer"));
        moduleInfoTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("moduleinfo"));
        cellInfoTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("cellinfo"));
      }

      //
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<edm::ESInputTag>("moduleindexer", edm::ESInputTag(""))->setComment("Module index tool");
        desc.add<edm::ESInputTag>("cellindexer", edm::ESInputTag(""))->setComment("Dense cell index tool");
        desc.add<edm::ESInputTag>("moduleinfo", edm::ESInputTag(""))->setComment("Module info table");
        desc.add<edm::ESInputTag>("cellinfo", edm::ESInputTag(""))->setComment("Cell info table");
        descriptions.addWithDefaultLabel(desc);
      }

      //
      std::optional<HGCalDenseIndexTriggerInfoHost> produce(const HGCalDenseIndexInfoRcd& iRecord) {
        //get cell and module indexer
        auto const& modIndexer = iRecord.get(moduleIndexTkn_);
        auto const& cellIndexer = iRecord.get(cellIndexTkn_);

        //get cell and module info
        auto const& moduleInfo = iRecord.get(moduleInfoTkn_);
        auto const& cellInfo = iRecord.get(cellInfoTkn_);

        //declare the dense index info collection to be produced
        //the size is determined by the module indexer
        const uint32_t nIndices = modIndexer.maxDataSize();
        HGCalDenseIndexTriggerInfoHost denseIdxInfo(cms::alpakatools::host(), nIndices);
        for (const auto& fedRS : modIndexer.fedReadoutSequences()) {
          uint32_t fedId = fedRS.id;
          for (size_t imod = 0; imod < fedRS.readoutTypes_.size(); imod++) {
            //the number of words expected, the first channel dense index
            int modTypeIdx = fedRS.readoutTypes_[imod];
            uint32_t nch = modIndexer.globalTypesNTCs()[modTypeIdx];
            int off = fedRS.TCOffsets_[imod];

            //get additional necessary module info
            int modIdx = modIndexer.getIndexForModule(fedId, static_cast<uint32_t>(imod));
            const auto& module_row = moduleInfo.view()[modIdx];
            bool isSiPM = module_row.isSiPM();
            uint16_t typeidx = module_row.typeidx();
            uint32_t muxid = module_row.muxid();

            //get the offset to start reading the cell info from sequential
            uint32_t cellInfoOffset = cellIndexer.offsets_[typeidx];

            //now fill the information sequentially on the cells of this module
            for (uint32_t ich = 0; ich < nch; ich++) {
              //finalize assigning the dense index
              uint32_t denseIdx = off + ich;

              auto row = denseIdxInfo.view()[denseIdx];

              //fill the fields
              row.fedId() = fedId;
              row.fedReadoutSeq() = imod;
              row.TCNumber() = ich;
              row.modInfoIdx() = modIdx;
              uint32_t cellIdx = cellInfoOffset + ich;
              row.cellInfoIdx() = cellIdx;

              const auto& cell_row = cellInfo.view()[cellIdx];
              row.muxid() = muxid + ich;

              //assign det id only for full and calibration cells
              row.trigdetid() = 0;
              if (cell_row.t() == 1 || cell_row.t() == 0) {
                if (isSiPM) {
                  row.trigdetid() = ::hgcal::mappingtools::getSiPMDetId(module_row.zside(),
                                                                        module_row.plane(),
                                                                        module_row.i2(),
                                                                        module_row.celltype(),
                                                                        cell_row.i1(),
                                                                        cell_row.i2());
                } else {
                  row.trigdetid() = module_row.trigdetid() + cell_row.detid();
                }

                //TODO: assign position from geometry, for the moment no position is assigned
                row.x() = 0;
                row.y() = 0;
                row.z() = 0;
              }
            }  // end cell loop

          }  // end module loop

        }  // end fed readout sequence loop

        return denseIdxInfo;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexerTrigger, HGCalElectronicsMappingRcd> moduleIndexTkn_;
      edm::ESGetToken<HGCalMappingCellIndexerTrigger, HGCalElectronicsMappingRcd> cellIndexTkn_;
      edm::ESGetToken<hgcal::HGCalMappingModuleTriggerParamHost, HGCalElectronicsMappingRcd> moduleInfoTkn_;
      edm::ESGetToken<hgcal::HGCalMappingCellParamHost, HGCalElectronicsMappingRcd> cellInfoTkn_;
    };

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcal::HGCalDenseIndexTriggerInfoESProducer);

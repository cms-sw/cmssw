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
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcal {

    class HGCalDenseIndexInfoESProducer : public ESProducer {
    public:
      //
      HGCalDenseIndexInfoESProducer(const edm::ParameterSet& iConfig) : ESProducer(iConfig) {
        auto cc = setWhatProduced(this);
        moduleIndexTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("moduleindexer"));
        cellIndexTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("cellindexer"));
        moduleInfoTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("moduleinfo"));
        cellInfoTkn_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("cellinfo"));
        caloGeomToken_ = cc.consumes();
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
      std::optional<HGCalDenseIndexInfoHost> produce(const HGCalDenseIndexInfoRcd& iRecord) {
        //geometry
        auto const& geo = iRecord.get(caloGeomToken_);

        //get cell and module indexer
        auto const& modIndexer = iRecord.get(moduleIndexTkn_);
        auto const& cellIndexer = iRecord.get(cellIndexTkn_);

        //get cell and module info
        auto const& moduleInfo = iRecord.get(moduleInfoTkn_);
        auto const& cellInfo = iRecord.get(cellInfoTkn_);

        //declare the dense index info collection to be produced
        //the size is determined by the module indexer
        const uint32_t nIndices = modIndexer.getMaxDataSize();
        HGCalDenseIndexInfoHost denseIdxInfo(nIndices, cms::alpakatools::host());
        for (auto fedRS : modIndexer.getFEDReadoutSequences()) {
          uint32_t fedId = fedRS.id;
          for (size_t imod = 0; imod < fedRS.readoutTypes_.size(); imod++) {
            //the number of words expected, the first channel dense index
            int modTypeIdx = fedRS.readoutTypes_[imod];
            uint32_t nch = modIndexer.getGlobalTypesNWords()[modTypeIdx];
            int off = fedRS.chDataOffsets_[imod];

            //get additional necessary module info
            int modIdx = modIndexer.getIndexForModule(fedId, imod);
            auto module_row = moduleInfo.view()[modIdx];
            bool isSiPM = module_row.isSiPM();
            uint16_t typeidx = module_row.typeidx();
            uint32_t eleid = module_row.eleid();
            uint32_t detid = module_row.detid();

            //get the appropriate geometry
            DetId::Detector det = DetId(detid).det();
            int subdet = ForwardSubdetector::ForwardEmpty;
            const HGCalGeometry* hgcal_geom =
                static_cast<const HGCalGeometry*>(geo.getSubdetectorGeometry(det, subdet));

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
              row.chNumber() = ich;
              row.modInfoIdx() = modIdx;
              uint32_t cellIdx = cellInfoOffset + ich;
              row.cellInfoIdx() = cellIdx;

              auto cell_row = cellInfo.view()[cellIdx];
              row.eleid() = eleid + cell_row.eleid();

              //assign det id only for full and calibration cells
              row.detid() = 0;
              if (cell_row.t() == 1 || cell_row.t() == 0) {
                if (isSiPM) {
                  row.detid() = ::hgcal::mappingtools::getSiPMDetId(module_row.zside(),
                                                                    module_row.plane(),
                                                                    module_row.i2(),
                                                                    module_row.celltype(),
                                                                    cell_row.i1(),
                                                                    cell_row.i2());
                } else {
                  row.detid() = module_row.detid() + cell_row.detid();
                }

                //assign position from geometry
                row.x() = 0;
                row.y() = 0;
                row.z() = 0;
                if (hgcal_geom != nullptr) {
                  GlobalPoint position = hgcal_geom->getPosition(row.detid());
                  row.x() = position.x();
                  row.y() = position.y();
                  row.z() = position.z();
                }
              }
            }  // end cell loop

          }  // end module loop

        }  // end fed readout sequence loop

        return denseIdxInfo;
      }  // end of produce()

    private:
      edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> moduleIndexTkn_;
      edm::ESGetToken<HGCalMappingCellIndexer, HGCalElectronicsMappingRcd> cellIndexTkn_;
      edm::ESGetToken<hgcal::HGCalMappingModuleParamHost, HGCalElectronicsMappingRcd> moduleInfoTkn_;
      edm::ESGetToken<hgcal::HGCalMappingCellParamHost, HGCalElectronicsMappingRcd> cellInfoTkn_;
      edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
    };

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(hgcal::HGCalDenseIndexInfoESProducer);

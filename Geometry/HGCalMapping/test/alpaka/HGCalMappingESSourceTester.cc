#include <cstdio>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"
#include "CondFormats/DataRecord/interface/HGCalMappingModuleRcd.h"
#include "CondFormats/DataRecord/interface/HGCalMappingCellIndexerRcd.h"
#include "CondFormats/DataRecord/interface/HGCalMappingCellRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class HGCalMappingESSourceTester : public stream::EDProducer<> {
  public:
    explicit HGCalMappingESSourceTester(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);
    std::map<uint32_t, uint32_t> mapGeoToElectronics(const hgcal::HGCalMappingModuleParamDevice& modules,
                                                     const hgcal::HGCalMappingCellParamDevice& cells,
                                                     bool geo2ele,
                                                     bool sipm);

  private:
    void produce(device::Event&, device::EventSetup const&) override;

    edm::ESWatcher<HGCalMappingModuleIndexerRcd> cfgWatcher_;
    edm::ESGetToken<HGCalMappingCellIndexer, HGCalMappingCellIndexerRcd> cellIndexTkn_;
    device::ESGetToken<hgcal::HGCalMappingCellParamDevice, HGCalMappingCellRcd> cellTkn_;
    edm::ESGetToken<HGCalMappingModuleIndexer, HGCalMappingModuleIndexerRcd> moduleIndexTkn_;
    device::ESGetToken<hgcal::HGCalMappingModuleParamDevice, HGCalMappingModuleRcd> moduleTkn_;
  };

  //
  HGCalMappingESSourceTester::HGCalMappingESSourceTester(const edm::ParameterSet& iConfig)
      : cellIndexTkn_(esConsumes()), cellTkn_(esConsumes()), moduleIndexTkn_(esConsumes()), moduleTkn_(esConsumes()) {}

  //
  void HGCalMappingESSourceTester::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    // if the cfg didn't change there's nothing else to do
    if (!cfgWatcher_.check(iSetup))
      return;

    //get cell indexers and SoA
    auto cellIdx = iSetup.getData(cellIndexTkn_);
    auto const& cells = iSetup.getData(cellTkn_);
    printf("[HGCalMappingIndexESSourceTester][produce] Cell dense indexers and associated SoA retrieved for HGCAL\n");
    int nmodtypes = cellIdx.typeCodeIndexer_.size();
    printf("[HGCalMappingIndexESSourceTester][produce] module cell indexer has %d module types\n", nmodtypes);

    //printout and test the indexer contents for cells
    printf("[HGCalMappingIndexESSourceTester][produce] Test indexer\n");
    uint32_t totOffset(0);
    for (size_t idx = 0; idx < cellIdx.di_.size(); idx++) {
      //check typecode exists
      auto typecode = cellIdx.getTypecodeFromEnum(idx);
      assert(cellIdx.typeCodeIndexer_.count(typecode) == 1);

      //check that the current offset is consistent with the increment from cells from the previous module
      if (idx > 0) {
        uint32_t nch_prev = cellIdx.maxErx_[idx - 1] * cellIdx.maxChPerErx_;
        uint32_t delta_offset = cellIdx.offsets_[idx] - cellIdx.offsets_[idx - 1];
        assert(delta_offset == nch_prev);
      }

      //assert offset is consistent with the accumulation
      auto off = cellIdx.offsets_[idx];
      assert(off == totOffset);

      totOffset += cellIdx.maxErx_[idx] * cellIdx.maxChPerErx_;

      //print
      printf("[HGCalMappingIndexESSourceTester][produce][%s] has index(internal)=%ld #eRx=%d #cells=%d offset=%d\n",
             typecode.c_str(),
             idx,
             cellIdx.maxErx_[idx],
             cellIdx.di_[idx].getMaxIndex(),
             cellIdx.offsets_[idx]);
    }

    assert(totOffset == cellIdx.maxDenseIndex());
    printf("[HGCalMappingIndexESSourceTester][produce] SoA size for module cell mapping will be %d\n", totOffset);

    //printout and test module cells SoA contents
    uint32_t ncells = cells.view().metadata().size();
    uint32_t validCells = 0;
    assert(ncells == cellIdx.maxDenseIndex());  //check for consistent size
    printf("[HGCalMappingIndexESSourceTester][produce] Module cell mapping contents\n");
    for (uint32_t i = 0; i < ncells; i++) {
      auto icell = cells.view()[i];
      if (!cells.view()[i].valid())
        continue;
      validCells++;
      printf(
          "\t idx=%d isHD=%d iscalib=%d isSiPM=%d typeidx=%d chip=%d half=%d seq=%d rocpin=%d cellidx=%d triglink=%d "
          "trigcell=%d i1=%d i2=%d t=%d trace=%f eleid=0x%x detid=0x%x\n",
          i,
          icell.isHD(),
          icell.iscalib(),
          icell.isSiPM(),
          icell.typeidx(),
          icell.chip(),
          icell.half(),
          icell.seq(),
          icell.rocpin(),
          icell.cellidx(),
          icell.triglink(),
          icell.trigcell(),
          icell.i1(),
          icell.i2(),
          icell.t(),
          icell.trace(),
          icell.eleid(),
          icell.detid());
    }

    //module mapping
    auto modulesIdx = iSetup.getData(moduleIndexTkn_);
    printf("[HGCalMappingIndexESSourceTester][produce] Module indexer has FEDs=%d Types in sequences=%ld max idx=%d\n",
           modulesIdx.nfeds_,
           modulesIdx.globalTypesCounter_.size(),
           modulesIdx.maxModulesIdx_);
    printf("[HGCalMappingIndexESSourceTester][produce] FED Readout sequence\n");
    std::unordered_set<uint32_t> unique_modOffsets, unique_erxOffsets, unique_chDataOffsets;
    uint32_t totalmods(0);
    for (const auto& frs : modulesIdx.fedReadoutSequences_) {
      std::copy(
          frs.modOffsets_.begin(), frs.modOffsets_.end(), std::inserter(unique_modOffsets, unique_modOffsets.end()));
      std::copy(
          frs.erxOffsets_.begin(), frs.erxOffsets_.end(), std::inserter(unique_erxOffsets, unique_erxOffsets.end()));
      std::copy(frs.chDataOffsets_.begin(),
                frs.chDataOffsets_.end(),
                std::inserter(unique_chDataOffsets, unique_chDataOffsets.end()));

      size_t nmods = frs.readoutTypes_.size();
      totalmods += nmods;
      printf("\t[FED %d] packs data from %ld ECON-Ds - readout types -> (offsets) :", frs.id, nmods);
      for (size_t i = 0; i < nmods; i++) {
        printf(
            "\t%d -> (%d;%d;%d)", frs.readoutTypes_[i], frs.modOffsets_[i], frs.erxOffsets_[i], frs.chDataOffsets_[i]);
      }
      printf("\n");
    }
    //check that there are unique offsets per modules in the full system
    assert(unique_modOffsets.size() == totalmods);
    assert(unique_erxOffsets.size() == totalmods);
    assert(unique_chDataOffsets.size() == totalmods);

    //get the module mapper SoA
    auto const& modules = iSetup.getData(moduleTkn_);
    int nmodules = modulesIdx.maxModulesIdx_;
    int validModules = 0;
    assert(nmodules == modules.view().metadata().size());  //check for consistent size
    printf("[HGCalMappingIndexESSourceTester][produce] Module mapping contents\n");
    for (int i = 0; i < nmodules; i++) {
      auto imod = modules.view()[i];
      if (!modules.view()[i].valid())
        continue;
      validModules++;
      printf(
          "\t idx=%d zside=%d isSiPM=%d plane=%d i1=%d i2=%d celltype=%d typeidx=%d fedid=%d localfedid=%d "
          "captureblock=%d capturesblockidx=%d econdidx=%d eleid=0x%x detid=0x%d\n",
          i,
          imod.zside(),
          imod.isSiPM(),
          imod.plane(),
          imod.i1(),
          imod.i2(),
          imod.celltype(),
          imod.typeidx(),
          imod.fedid(),
          imod.slinkidx(),
          imod.captureblock(),
          imod.captureblockidx(),
          imod.econdidx(),
          imod.eleid(),
          imod.detid());
    }

    printf(
        "[HGCalMappingIndexESSourceTester][produce] Module and cells maps retrieved for HGCAL %d/%d valid modules "
        "%d/%d valid cells",
        validModules,
        modules.view().metadata().size(),
        validCells,
        cells.view().metadata().size());

    //test DetId to ElectronicsId mapping
    auto tmap = [](auto geo2ele, auto ele2geo) {
      //sizes must match
      assert(geo2ele.size() == ele2geo.size());

      //test uniqueness of keys
      for (auto it : geo2ele) {
        assert(ele2geo.count(it.second) == 1);
        assert(ele2geo[it.second] == it.first);
      }
      for (auto it : ele2geo) {
        assert(geo2ele.count(it.second) == 1);
        assert(geo2ele[it.second] == it.first);
      }

      return true;
    };

    //apply test to Si
    std::map<uint32_t, uint32_t> sigeo2ele = this->mapGeoToElectronics(modules, cells, true, false);
    std::map<uint32_t, uint32_t> siele2geo = this->mapGeoToElectronics(modules, cells, false, false);
    printf("[HGCalMappingIndexESSourceTester][produce] Silicon electronics<->geometry map\n");
    printf("\tID maps ele2geo=%ld ID maps geo2ele=%ld\n", siele2geo.size(), sigeo2ele.size());
    tmap(sigeo2ele, siele2geo);

    //apply test to SiPMs
    std::map<uint32_t, uint32_t> sipmgeo2ele = this->mapGeoToElectronics(modules, cells, true, true);
    std::map<uint32_t, uint32_t> sipmele2geo = this->mapGeoToElectronics(modules, cells, false, true);
    printf("[HGCalMappingIndexESSourceTester][produce] SiPM-on-tile electronics<->geometry map\n");
    printf("\tID maps ele2geo=%ld ID maps geo2ele=%ld\n", sipmele2geo.size(), sipmgeo2ele.size());
    tmap(sipmgeo2ele, sipmele2geo);

    // Timing studies
    uint16_t fedid(175), econdidx(2), captureblockidx(0), chip(0), half(1), seq(12), rocpin(48);
    int zside(0), n_i(6000000);
    printf("[HGCalMappingIndexESSourceTester][produce]  Creating %d number of raw ElectronicsIds\n", n_i);
    uint32_t elecid(0);
    auto start = now();
    for (int i = 0; i < n_i; i++) {
      elecid = ::hgcal::mappingtools::getElectronicsId(zside, fedid, captureblockidx, econdidx, chip, half, seq);
    }
    auto stop = now();
    std::chrono::duration<float> elapsed = stop - start;
    printf("\tTime: %f seconds\n", elapsed.count());

    HGCalElectronicsId eid(elecid);
    assert(eid.localFEDId() == fedid);
    assert((uint32_t)eid.captureBlock() == captureblockidx);
    assert((uint32_t)eid.econdIdx() == econdidx);
    assert((uint32_t)eid.econdeRx() == (uint32_t)(2 * chip + half));
    assert((uint32_t)eid.halfrocChannel() == seq);
    float eidrocpin = (uint32_t)eid.halfrocChannel() + 36 * ((uint32_t)eid.econdeRx() % 2) -
                      ((uint32_t)eid.halfrocChannel() > 17) * ((uint32_t)eid.econdeRx() % 2 + 1);
    assert(eidrocpin == rocpin);

    int plane(1), u(-9), v(-6), celltype(2), celliu(3), celliv(7);
    printf("[HGCalMappingIndexESSourceTester][produce]  Creating %d number of raw  HGCSiliconDetIds\n", n_i);
    uint32_t geoid(0);
    start = now();
    for (int i = 0; i < n_i; i++) {
      geoid = ::hgcal::mappingtools::getSiDetId(zside, plane, u, v, celltype, celliu, celliv);
    }
    stop = now();
    elapsed = stop - start;
    printf("\tTime: %f seconds\n", elapsed.count());
    HGCSiliconDetId gid(geoid);
    assert(gid.type() == celltype);
    assert(gid.layer() == plane);
    assert(gid.cellU() == celliu);
    assert(gid.cellV() == celliv);

    int modidx(0), cellidx(1);
    printf(
        "[HGCalMappingIndexESSourceTester][produce]  Creating %d number of raw ElectronicsIds from SoAs module idx: "
        "%d, cell idx: %d\n",
        n_i,
        modidx,
        cellidx);
    elecid = 0;
    start = now();
    for (int i = 0; i < n_i; i++) {
      elecid = modules.view()[modidx].eleid() + cells.view()[cellidx].eleid();
    }
    stop = now();
    elapsed = stop - start;
    printf("\tTime: %f seconds\n", elapsed.count());
    eid = HGCalElectronicsId(elecid);
    assert(eid.localFEDId() == modules.view()[modidx].fedid());
    assert((uint32_t)eid.captureBlock() == modules.view()[modidx].captureblockidx());
    assert((uint32_t)eid.econdIdx() == modules.view()[modidx].econdidx());
    assert((uint32_t)eid.halfrocChannel() == cells.view()[cellidx].seq());
    uint16_t econderx = cells.view()[cellidx].chip() * 2 + cells.view()[cellidx].half();
    assert((uint32_t)eid.econdeRx() == econderx);

    printf(
        "[HGCalMappingIndexESSourceTester][produce] Creating %d number of raw HGCSiliconDetId from SoAs module idx: "
        "%d, cell idx: %d\n",
        n_i,
        modidx,
        cellidx);
    uint32_t detid(0);
    start = now();
    for (int i = 0; i < n_i; i++) {
      detid = modules.view()[modidx].detid() + cells.view()[cellidx].detid();
    }
    stop = now();
    elapsed = stop - start;
    printf("\tTime: %f seconds\n", elapsed.count());
    HGCSiliconDetId did(detid);
    assert(did.type() == modules.view()[modidx].celltype());
    assert(did.layer() == modules.view()[modidx].plane());
    assert(did.cellU() == cells.view()[cellidx].i1());
    assert(did.cellV() == cells.view()[cellidx].i2());
  }

  //
  void HGCalMappingESSourceTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addWithDefaultLabel(desc);
  }

  //
  std::map<uint32_t, uint32_t> HGCalMappingESSourceTester::mapGeoToElectronics(
      const hgcal::HGCalMappingModuleParamDevice& modules,
      const hgcal::HGCalMappingCellParamDevice& cells,
      bool geo2ele,
      bool sipm) {
    //loop over different modules
    std::map<uint32_t, uint32_t> idmap;
    uint32_t ndups(0);
    printf("\n");
    for (int i = 0; i < modules.view().metadata().size(); i++) {
      auto imod = modules.view()[i];

      if (!imod.valid())
        continue;

      //require match to si or SiPM
      if (sipm != imod.isSiPM())
        continue;

      if (imod.plane() == 0) {
        printf("WARNING: found plane=0 for i1=%d i2=%d siPM=%d @ index=%i\n", imod.i1(), imod.i2(), imod.isSiPM(), i);
        continue;
      }

      //loop over cells in the module
      for (int j = 0; j < cells.view().metadata().size(); j++) {
        auto jcell = cells.view()[j];

        //use only the information for cells which match the module type index
        if (jcell.typeidx() != imod.typeidx())
          continue;

        //require that it's a valid cell
        if (!jcell.valid())
          continue;

        //assert type of sensor
        assert(imod.isSiPM() == jcell.isSiPM());

        // make sure the cell is part of the module and it's not a calibration cell
        if (jcell.t() != 1)
          continue;

        // uint32_t elecid = ::hgcal::mappingtools::getElectronicsId(imod.zside(),
        //                                                         imod.fedid(),
        //                                                         imod.captureblockidx(),
        //                                                         imod.econdidx(),
        //                                                         jcell.chip(),
        //                                                         jcell.half(),
        //                                                         jcell.seq());

        uint32_t elecid = imod.eleid() + jcell.eleid();

        uint32_t geoid(0);

        if (sipm) {
          geoid = ::hgcal::mappingtools::getSiPMDetId(
              imod.zside(), imod.plane(), imod.i2(), imod.celltype(), jcell.i1(), jcell.i2());
        } else {
          // geoid = ::hgcal::mappingtools::getSiDetId(imod.zside(),
          //                                         imod.plane(),
          //                                         imod.i1(),
          //                                         imod.i2(),
          //                                         imod.celltype(),
          //                                         jcell.i1(),
          //                                         jcell.i2());

          geoid = imod.detid() + jcell.detid();
        }

        if (geo2ele) {
          auto it = idmap.find(geoid);
          ndups += (it != idmap.end());
          if (!sipm && it != idmap.end() && imod.plane() <= 26) {
            HGCSiliconDetId detid(geoid);
            printf("WARNING duplicate found for plane=%d u=%d v=%d cellU=%d cellV=%d valid=%d -> detid=0x%x\n",
                   imod.plane(),
                   imod.i1(),
                   imod.i2(),
                   jcell.i1(),
                   jcell.i2(),
                   jcell.valid(),
                   detid.rawId());
          }
        }
        if (!geo2ele) {
          auto it = idmap.find(elecid);
          ndups += (it != idmap.end());
        }

        //map
        idmap[geo2ele ? geoid : elecid] = geo2ele ? elecid : geoid;
      }
    }

    if (ndups > 0) {
      printf("[HGCalMappingESSourceTester][mapGeoToElectronics] found %d duplicates with geo2ele=%d for sipm=%d\n",
             ndups,
             geo2ele,
             sipm);
    }

    return idmap;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define this as a plug-in
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalMappingESSourceTester);

#include <iomanip>  // for std::setw
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"

#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"
#include "CondFormats/DataRecord/interface/HGCalMappingModuleRcd.h"
#include "CondFormats/DataRecord/interface/HGCalMappingCellIndexerRcd.h"
#include "CondFormats/DataRecord/interface/HGCalMappingCellRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDeviceCollection.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"

namespace {
  template <class T>
  double duration(T t0, T t1) {
    auto elapsed_secs = t1 - t0;
    typedef std::chrono::duration<float> float_seconds;
    auto secs = std::chrono::duration_cast<float_seconds>(elapsed_secs);
    return secs.count();
  }

  inline std::chrono::time_point<std::chrono::steady_clock> now() { return std::chrono::steady_clock::now(); }
}  // namespace

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class HGCalMappingESSourceTester : public stream::EDProducer<> {
  public:
    explicit HGCalMappingESSourceTester(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);
    std::map<uint32_t, uint32_t> mapGeoToElectronics(const hgcal::HGCalMappingModuleParamDeviceCollection& modules,
                                                     const hgcal::HGCalMappingCellParamDeviceCollection& cells,
                                                     bool geo2ele,
                                                     bool sipm);

  private:
    void produce(device::Event&, device::EventSetup const&) override;

    edm::ESWatcher<HGCalMappingModuleIndexerRcd> cfgWatcher_;
    edm::ESGetToken<HGCalMappingCellIndexer, HGCalMappingCellIndexerRcd> cellIndexTkn_;
    device::ESGetToken<hgcal::HGCalMappingCellParamDeviceCollection, HGCalMappingCellRcd> cellTkn_;
    edm::ESGetToken<HGCalMappingModuleIndexer, HGCalMappingModuleIndexerRcd> moduleIndexTkn_;
    device::ESGetToken<hgcal::HGCalMappingModuleParamDeviceCollection, HGCalMappingModuleRcd> moduleTkn_;
    const device::EDPutToken<portabletest::TestDeviceCollection> testCollToken_;
  };

  //
  HGCalMappingESSourceTester::HGCalMappingESSourceTester(const edm::ParameterSet& iConfig)
      : cellIndexTkn_(esConsumes()),
        cellTkn_(esConsumes()),
        moduleIndexTkn_(esConsumes()),
        moduleTkn_(esConsumes()),
        testCollToken_{produces()} {}

  //
  void HGCalMappingESSourceTester::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    //put a dummy collection to the event
    portabletest::TestDeviceCollection testColl{0, iEvent.queue()};
    iEvent.emplace(testCollToken_, std::move(testColl));

    // if the cfg didn't change there's nothing else to do
    if (!cfgWatcher_.check(iSetup))
      return;

    //get cell indexers and SoA
    auto cellIdx = iSetup.getData(cellIndexTkn_);
    auto const& cells = iSetup.getData(cellTkn_);
    edm::LogInfo("HGCalMappingIndexESSourceTester") << "Cell dense indexers and associated SoA retrieved for HGCAL";
    int nmodtypes = cellIdx.typeCodeIndexer_.size();
    edm::LogInfo("HGCalMappingIndexESSourceTester")
        << "[Module cell indexer] has " << nmodtypes << " module types" << std::endl;

    //printout and test the indexer contents for cells
    edm::LogInfo("HGCalMappingIndexESSourceTester").log([&](auto& log) {
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
        log << "\t [" << typecode << "] has "
            << " index(internal)=" << idx << " #eRx = " << cellIdx.maxErx_[idx]
            << " #cells = " << cellIdx.di_[idx].getMaxIndex() << " offset @ " << cellIdx.offsets_[idx] << "\n";
      }

      assert(totOffset == cellIdx.maxDenseIndex());
      log << "SoA size for module cell mapping will be " << totOffset << "\n";
    });

    //printout and test module cells SoA contents
    uint32_t ncells = cells.view().metadata().size();
    assert(ncells == cellIdx.maxDenseIndex());  //check for consistent size
    LogDebug("HGCalMappingIndexESSourceTester").log([&](auto& log) {
      log << "Module cell mapping contents \n";
      for (uint32_t i = 0; i < ncells; i++) {
        log << "\tidx = " << i << ", "
            << "isHD = " << cells.view()[i].isHD() << ", "
            << "iscalib = " << cells.view()[i].iscalib() << ", "
            << "isSiPM = " << cells.view()[i].isSiPM() << ", "
            << "typeidx = " << cells.view()[i].typeidx() << ", "
            << "chip = " << cells.view()[i].chip() << ", "
            << "half = " << cells.view()[i].half() << ", "
            << "seq = " << cells.view()[i].seq() << ", "
            << "rocpin = " << cells.view()[i].rocpin() << ", "
            << "cellidx = " << cells.view()[i].cellidx() << ", "
            << "triglink = " << cells.view()[i].triglink() << ", "
            << "trigcell = " << cells.view()[i].trigcell() << ", "
            << "i1 = " << cells.view()[i].i1() << ", "
            << "i2 = " << cells.view()[i].i2() << ", "
            << "t = " << cells.view()[i].t() << ", "
            << "trace = " << cells.view()[i].trace() << ", "
            << "eleid = " << cells.view()[i].eleid() << ", "
            << "detid = " << cells.view()[i].detid() << " \n";
      }
    });

    //module mapping
    auto modulesIdx = iSetup.getData(moduleIndexTkn_);
    edm::LogInfo("HGCalMappingIndexESSourceTester")
        << "[Module indexer]"
        << "FEDs=" << modulesIdx.nfeds_ << " Types in sequences=" << modulesIdx.globalTypesCounter_.size()
        << " max idx=" << modulesIdx.maxModulesIdx_;
    // edm::LogInfo("HGCalMappingIndexESSourceTester").log( [&](auto& log) {
    LogDebug("HGCalMappingIndexESSourceTester").log([&](auto& log) {
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
        log << "[FED " << frs.id << "] packs data from " << nmods << " ECON-Ds - readout types -> (offsets) : ";
        for (size_t i = 0; i < nmods; i++) {
          log << frs.readoutTypes_[i] << "->(" << frs.modOffsets_[i] << ";" << frs.erxOffsets_[i] << ";"
              << frs.chDataOffsets_[i] << ")\t";
        }
        log << "\n";
      }

      //check that there are unique offsets per modules in the full system
      assert(unique_modOffsets.size() == totalmods);
      assert(unique_erxOffsets.size() == totalmods);
      assert(unique_chDataOffsets.size() == totalmods);
    });

    //get the module mapper SoA
    auto const& modules = iSetup.getData(moduleTkn_);
    int nmodules = modulesIdx.maxModulesIdx_;
    assert(nmodules == modules.view().metadata().size());  //check for consistent size
    // edm::LogInfo("HGCalMappingIndexESSourceTester").log( [&](auto& log) {
    LogDebug("HGCalMappingIndexESSourceTester").log([&](auto& log) {
      log << "Module mapping contents \n";
      for (int i = 0; i < nmodules; i++) {
        log << "idx = " << i << ", "
            << "zside = " << modules.view()[i].zside() << ", "
            << "isSiPM = " << modules.view()[i].isSiPM() << ", "
            << "plane = " << modules.view()[i].plane() << ", "
            << "i1 = " << modules.view()[i].i1() << ", "
            << "i2 = " << modules.view()[i].i2() << ", "
            << "celltype = " << modules.view()[i].celltype() << ", "
            << "typeidx = " << modules.view()[i].typeidx() << ", "
            << "fedid = " << modules.view()[i].fedid() << ", "
            << "localfedid = " << modules.view()[i].slinkidx() << ", "
            << "captureblock = " << modules.view()[i].captureblock() << ", "
            << "captureblockidx = " << modules.view()[i].captureblockidx() << ", "
            << "econdidx = " << modules.view()[i].econdidx() << ", "
            << "eleid = " << modules.view()[i].eleid() << ", "
            << "detid = " << modules.view()[i].detid() << "\n";
      }
    });

    //count  number of valid cells
    int validModules(0), validCells(0);
    for (int i = 0; i < modules.view().metadata().size(); i++)
      validModules += modules.view()[i].valid();
    for (int i = 0; i < cells.view().metadata().size(); i++)
      validCells += cells.view()[i].valid();
    edm::LogInfo("HGCalMappingIndexESSourceTester")
        << "Module and cells maps retrieved for HGCAL"
        << "\t" << validModules << "/" << modules.view().metadata().size() << " valid modules"
        << "\t" << validCells << "/" << cells.view().metadata().size() << " valid cells";

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
    edm::LogInfo("HGCalMappingIndexESSourceTester")
        << "[Silicon electronics<->geometry map]"
        << "\n\t ID maps ele2geo=" << siele2geo.size() << " ID maps geo2ele=" << sigeo2ele.size() << std::endl;
    tmap(sigeo2ele, siele2geo);

    //apply test to SiPMs
    std::map<uint32_t, uint32_t> sipmgeo2ele = this->mapGeoToElectronics(modules, cells, true, true);
    std::map<uint32_t, uint32_t> sipmele2geo = this->mapGeoToElectronics(modules, cells, false, true);
    edm::LogInfo("HGCalMappingIndexESSourceTester")
        << "[SiPM-on-tile electronics<->geometry map]"
        << "\n\t ID maps ele2geo=" << sipmele2geo.size() << " ID maps geo2ele=" << sipmgeo2ele.size() << std::endl;
    tmap(sipmgeo2ele, sipmele2geo);

    // Timing studies
    uint16_t fedid(175), econdidx(2), captureblockidx(0), chip(0), half(1), seq(12), rocpin(48);
    int zside(0), n_i(6000000);

    edm::LogInfo("HGCalMappingIndexESSourceTester").log([&](auto& log) {
      log << "Creating " << n_i << " number of raw ElectronicsIds\n";
      uint32_t elecid(0);
      auto start = now();
      for (int i = 0; i < n_i; i++) {
        elecid = ::hgcal::mappingtools::getElectronicsId(zside, fedid, captureblockidx, econdidx, chip, half, seq);
      }
      auto stop = now();
      log << "Time: " << duration(start, stop) << " seconds.\n";

      HGCalElectronicsId eid(elecid);
      assert(eid.localFEDId() == fedid);
      assert((uint32_t)eid.captureBlock() == captureblockidx);
      assert((uint32_t)eid.econdIdx() == econdidx);
      assert((uint32_t)eid.econdeRx() == (uint32_t)(2 * chip + half));
      assert((uint32_t)eid.halfrocChannel() == seq);
      float eidrocpin = (uint32_t)eid.halfrocChannel() + 36 * ((uint32_t)eid.econdeRx() % 2) -
                        ((uint32_t)eid.halfrocChannel() > 17) * ((uint32_t)eid.econdeRx() % 2 + 1);
      assert(eidrocpin == rocpin);
    });

    int plane(1), u(-9), v(-6), celltype(2), celliu(3), celliv(7);

    edm::LogInfo("HGCalMappingIndexESSourceTester").log([&](auto& log) {
      log << "Creating " << n_i << " number of raw HGCSiliconDetIds\n";
      uint32_t geoid(0);
      auto start = now();
      for (int i = 0; i < n_i; i++) {
        geoid = ::hgcal::mappingtools::getSiDetId(zside, plane, u, v, celltype, celliu, celliv);
      }
      auto stop = now();
      log << "Time: " << duration(start, stop) << " seconds.";

      HGCSiliconDetId gid(geoid);
      assert(gid.type() == celltype);
      assert(gid.layer() == plane);
      assert(gid.cellU() == celliu);
      assert(gid.cellV() == celliv);
    });

    edm::LogInfo("HGCalMappingIndexESSourceTester").log([&](auto& log) {
      int modidx(0), cellidx(1);
      log << "Creating " << n_i << " number of raw ElectronicsIds from SoAs module idx: " << modidx
          << ", cell idx: " << cellidx << ".\n";
      uint32_t elecid(0);
      auto start = now();
      for (int i = 0; i < n_i; i++) {
        elecid = modules.view()[modidx].eleid() + cells.view()[cellidx].eleid();
      }
      auto stop = now();
      log << "Time: " << duration(start, stop) << " seconds.\n";

      HGCalElectronicsId eid(elecid);
      assert(eid.localFEDId() == modules.view()[modidx].fedid());
      assert((uint32_t)eid.captureBlock() == modules.view()[modidx].captureblockidx());
      assert((uint32_t)eid.econdIdx() == modules.view()[modidx].econdidx());

      assert((uint32_t)eid.halfrocChannel() == cells.view()[cellidx].seq());
      uint16_t econderx = cells.view()[cellidx].chip() * 2 + cells.view()[cellidx].half();
      assert((uint32_t)eid.econdeRx() == econderx);
      // assert((uint32_t)eid.rocChannel()==cells.view()[cellidx].rocpin());

      log << "Creating " << n_i << " number of raw HGCSiliconDetId from SoAs module idx: " << modidx
          << ", cell idx: " << cellidx << ".\n";
      uint32_t detid(0);
      start = now();
      for (int i = 0; i < n_i; i++) {
        detid = modules.view()[modidx].detid() + cells.view()[cellidx].detid();
      }
      stop = now();
      log << "Time: " << duration(start, stop) << " seconds.\n";

      HGCSiliconDetId did(detid);
      assert(did.type() == modules.view()[modidx].celltype());
      assert(did.layer() == modules.view()[modidx].plane());
      assert(did.cellU() == cells.view()[cellidx].i1());
      assert(did.cellV() == cells.view()[cellidx].i2());
    });
  }

  //
  void HGCalMappingESSourceTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addWithDefaultLabel(desc);
  }

  //
  std::map<uint32_t, uint32_t> HGCalMappingESSourceTester::mapGeoToElectronics(
      const hgcal::HGCalMappingModuleParamDeviceCollection& modules,
      const hgcal::HGCalMappingCellParamDeviceCollection& cells,
      bool geo2ele,
      bool sipm) {
    //loop over different modules
    std::map<uint32_t, uint32_t> idmap;
    uint32_t ndups(0);
    std::cout << std::endl;
    for (int i = 0; i < modules.view().metadata().size(); i++) {
      if (!modules.view()[i].valid())
        continue;

      //require match to si or SiPM
      if (sipm != modules.view()[i].isSiPM())
        continue;

      if (modules.view()[i].plane() == 0) {
        std::cout << "Check plane=" << modules.view()[i].plane() << " u=" << modules.view()[i].i1()
                  << " v=" << modules.view()[i].i2() << " " << modules.view()[i].isSiPM() << " @ index=" << i
                  << std::endl;
        continue;
      }

      //loop over cells in the module
      for (int j = 0; j < cells.view().metadata().size(); j++) {
        //use only the information for cells which match the module type index
        if (cells.view()[j].typeidx() != modules.view()[i].typeidx())
          continue;

        //require that it's a valid cell
        if (!cells.view()[j].valid())
          continue;

        //assert type of sensor
        assert(modules.view()[i].isSiPM() == cells.view()[j].isSiPM());

        // make sure the cell is part of the module and it's not a calibration cell
        if (cells.view()[j].t() != 1)
          continue;

        // uint32_t elecid = ::hgcal::mappingtools::getElectronicsId(modules.view()[i].zside(),
        //                                                         modules.view()[i].fedid(),
        //                                                         modules.view()[i].captureblockidx(),
        //                                                         modules.view()[i].econdidx(),
        //                                                         cells.view()[j].chip(),
        //                                                         cells.view()[j].half(),
        //                                                         cells.view()[j].seq());

        uint32_t elecid = modules.view()[i].eleid() + cells.view()[j].eleid();

        uint32_t geoid(0);

        if (sipm) {
          geoid = ::hgcal::mappingtools::getSiPMDetId(modules.view()[i].zside(),
                                                      modules.view()[i].plane(),
                                                      modules.view()[i].i2(),
                                                      modules.view()[i].celltype(),
                                                      cells.view()[j].i1(),
                                                      cells.view()[j].i2());
        } else {
          // geoid = ::hgcal::mappingtools::getSiDetId(modules.view()[i].zside(),
          //                                         modules.view()[i].plane(),
          //                                         modules.view()[i].i1(),
          //                                         modules.view()[i].i2(),
          //                                         modules.view()[i].celltype(),
          //                                         cells.view()[j].i1(),
          //                                         cells.view()[j].i2());

          geoid = modules.view()[i].detid() + cells.view()[j].detid();
        }

        if (geo2ele) {
          auto it = idmap.find(geoid);
          ndups += (it != idmap.end());
          if (!sipm && it != idmap.end() && modules.view()[i].plane() <= 26) {
            std::cout << "plane=" << modules.view()[i].plane() << " u=" << modules.view()[i].i1()
                      << " v=" << modules.view()[i].i2() << " cellU=" << cells.view()[j].i1()
                      << " cellV=" << cells.view()[j].i2() << " " << cells.view()[j].valid() << std::endl;
            HGCSiliconDetId detid(geoid);
            std::cout << detid << std::endl;
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
      edm::LogInfo("HGCalMappingIndexESSourceTester")
          << "mapGeoToElectronics found " << ndups << " duplicates with geo2ele=" << geo2ele << " for sipm=" << sipm;
    }

    return idmap;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define this as a plug-in
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalMappingESSourceTester);

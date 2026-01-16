#include <cstdio>
#include <chrono>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexerTrigger.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexerTrigger.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"

class HGCalMappingTriggerESSourceTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalMappingTriggerESSourceTester(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  std::map<uint32_t, uint32_t> mapGeoToElectronics(const hgcal::HGCalMappingModuleTriggerParamHost& modules,
                                                   const hgcal::HGCalMappingCellParamHost& cells,
                                                   bool geo2ele,
                                                   bool sipm);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::ESWatcher<HGCalElectronicsMappingRcd> cfgWatcher_;
  edm::ESGetToken<HGCalMappingCellIndexerTrigger, HGCalElectronicsMappingRcd> cellIndexTkn_;
  edm::ESGetToken<hgcal::HGCalMappingCellParamHost, HGCalElectronicsMappingRcd> cellTkn_;
  edm::ESGetToken<HGCalMappingModuleIndexerTrigger, HGCalElectronicsMappingRcd> moduleIndexTkn_;
  edm::ESGetToken<hgcal::HGCalMappingModuleTriggerParamHost, HGCalElectronicsMappingRcd> moduleTkn_;
  edm::ESGetToken<hgcal::HGCalDenseIndexTriggerInfoHost, HGCalDenseIndexInfoRcd> denseIndexTkn_;
};

//
HGCalMappingTriggerESSourceTester::HGCalMappingTriggerESSourceTester(const edm::ParameterSet& iConfig)
    : cellIndexTkn_(esConsumes()),
      cellTkn_(esConsumes()),
      moduleIndexTkn_(esConsumes()),
      moduleTkn_(esConsumes()),
      denseIndexTkn_(esConsumes()) {}

//
void HGCalMappingTriggerESSourceTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // if the cfg didn't change there's nothing else to do
  if (!cfgWatcher_.check(iSetup))
    return;

  //get cell indexers and SoA
  auto const& cellIdx = iSetup.getData(cellIndexTkn_);
  auto const& cells = iSetup.getData(cellTkn_);
  printf("[HGCalMappingIndexESSourceTester][analyze] Cell dense indexers and associated SoA retrieved for HGCAL\n");
  int nmodtypes = cellIdx.typeCodeIndexer_.size();
  printf("[HGCalMappingIndexESSourceTester][analyze] module cell indexer has %d module types\n", nmodtypes);

  //printout and test the indexer contents for cells
  printf("[HGCalMappingIndexESSourceTester][analyze] Test indexer\n");
  uint32_t totOffset(0);
  for (size_t idx = 0; idx < cellIdx.di_.size(); idx++) {
    //check typecode exists
    const auto& typecode = cellIdx.getTypecodeFromEnum(idx);
    assert(cellIdx.typeCodeIndexer_.count(typecode) == 1);

    //check that the current offset is consistent with the increment from cells from the previous module
    if (idx > 0) {
      uint32_t nch_prev = cellIdx.maxROC_[idx - 1] * cellIdx.maxTrLink_[idx - 1] * cellIdx.maxTCPerLink_[idx - 1];
      uint32_t delta_offset = cellIdx.offsets_[idx] - cellIdx.offsets_[idx - 1];
      assert(delta_offset == nch_prev);
    }

    //assert offset is consistent with the accumulation
    uint32_t off = cellIdx.offsets_[idx];
    assert(off == totOffset);

    totOffset += cellIdx.maxROC_[idx] * cellIdx.maxTrLink_[idx] * cellIdx.maxTCPerLink_[idx];

    //print
    printf(
        "[HGCalMappingIndexESSourceTester][analyze][%s] has index(internal)=%ld #ROCs=%d #TLinks=%d #TCells=%d "
        "offset=%d-%d\n",
        typecode.c_str(),
        idx,
        cellIdx.maxROC_[idx],
        cellIdx.maxTrLink_[idx],
        cellIdx.di_[idx].maxIndex(),
        cellIdx.offsets_[idx],
        totOffset);
  }
  assert(totOffset == cellIdx.maxDenseIndex());
  printf("[HGCalMappingIndexESSourceTester][analyze] SoA size for module cell mapping will be %d\n", totOffset);

  //printout and test module cells SoA contents
  uint32_t ncells = cells.view().metadata().size();
  uint32_t validCells = 0;
  printf("[HGCalMappingIndexESSourceTester][analyze] Module cell mapping contents\n");
  for (uint32_t i = 0; i < ncells; i++) {
    auto icell = cells.view()[i];
    if (!cells.view()[i].valid())
      continue;
    validCells++;
    printf(
        "\t idx=%d isHD=%d iscalib=%d isSiPM=%d typeidx=%d chip=%d half=%d seq=%d rocpin=%d sensorcell=%d triglink=%d "
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
        icell.sensorcell(),
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
  auto const& modulesIdx = iSetup.getData(moduleIndexTkn_);
  printf("[HGCalMappingIndexESSourceTester][analyze] Module indexer has FEDs=%d Types in sequences=%ld max idx=%d\n",
         modulesIdx.fedCount(),
         modulesIdx.globalTypesCounter().size(),
         modulesIdx.maxModulesIndex());
  printf("[HGCalMappingIndexESSourceTester][analyze] FED Readout sequence\n");
  std::unordered_set<uint32_t> unique_modOffsets, unique_TrLinkOffsets, unique_TCOffsets;
  uint32_t totalmods(0);
  for (const auto& frs : modulesIdx.fedReadoutSequences()) {
    std::copy(
        frs.modOffsets_.begin(), frs.modOffsets_.end(), std::inserter(unique_modOffsets, unique_modOffsets.end()));
    std::copy(frs.TrLinkOffsets_.begin(),
              frs.TrLinkOffsets_.end(),
              std::inserter(unique_TrLinkOffsets, unique_TrLinkOffsets.end()));
    std::copy(frs.TCOffsets_.begin(), frs.TCOffsets_.end(), std::inserter(unique_TCOffsets, unique_TCOffsets.end()));

    size_t nmods = frs.readoutTypes_.size();
    totalmods += nmods;
    printf("\t[FED %d] packs data from %ld ECON-Ts - readout types -> (offsets) :", frs.id, nmods);
    for (size_t i = 0; i < nmods; i++) {
      printf("\t%d -> (%d;%d;%d)", frs.readoutTypes_[i], frs.modOffsets_[i], frs.TrLinkOffsets_[i], frs.TCOffsets_[i]);
    }
    printf("\n");
  }
  //check that there are unique offsets per modules in the full system
  assert(unique_modOffsets.size() == totalmods);
  assert(unique_TrLinkOffsets.size() == totalmods);
  assert(unique_TCOffsets.size() == totalmods);

  //get the module mapper SoA
  auto const& modules = iSetup.getData(moduleTkn_);
  int nmodules = modulesIdx.maxModulesIndex();
  int validModules = 0;
  assert(nmodules == modules.view().metadata().size());  //check for consistent size
  printf("[HGCalMappingIndexESSourceTester][analyze] Module mapping contents\n");
  for (int i = 0; i < nmodules; i++) {
    auto imod = modules.view()[i];
    if (!modules.view()[i].valid())
      continue;
    validModules++;
    printf(
        "\t idx=%d zside=%d isSiPM=%d plane=%d i1=%d i2=%d irot=%d celltype=%d typeidx=%d fedid=%d localfedid=%d "
        "econtidx=%d muxid=0x%x trigid=0x%d cassette=0x%d\n",
        i,
        imod.zside(),
        imod.isSiPM(),
        imod.plane(),
        imod.i1(),
        imod.i2(),
        imod.irot(),
        imod.celltype(),
        imod.typeidx(),
        imod.fedid(),
        imod.slinkidx(),
        imod.econtidx(),
        imod.muxid(),
        imod.trigdetid(),
        imod.cassette());
  }

  printf(
      "[HGCalMappingIndexESSourceTester][analyze] Module and cells maps retrieved for HGCAL %d/%d valid modules "
      "%d/%d valid cells",
      validModules,
      modules.view().metadata().size(),
      validCells,
      cells.view().metadata().size());

  //test TrigDetId to MuxId mapping
  //FIXME: this is for the moment disabled as extra input is needed from the backend regarding the MUX id - there is degeneracy
  /*
  const auto& tmap = [](auto geo2ele, auto ele2geo) {
    //sizes must match
    assert(geo2ele.size() == ele2geo.size());

    //test uniqueness of keys
    for (const auto& it : geo2ele) {
      assert(ele2geo.count(it.second) == 1);
      assert(ele2geo[it.second] == it.first);
    }

    for (const auto& it : ele2geo) {
      assert(geo2ele.count(it.second) == 1);
      assert(geo2ele[it.second] == it.first);
    }

    return true;
  };

  //apply test to Si
  const auto& sigeo2ele = this->mapGeoToElectronics(modules, cells, true, false);
  const auto& siele2geo = this->mapGeoToElectronics(modules, cells, false, false);
  printf("[HGCalMappingIndexESSourceTester][produce] Silicon electronics<->geometry map\n");
  printf("\tID maps ele2geo=%ld ID maps geo2ele=%ld\n", siele2geo.size(), sigeo2ele.size());
  tmap(sigeo2ele, siele2geo);

  //apply test to SiPMs
  const auto& sipmgeo2ele = this->mapGeoToElectronics(modules, cells, true, true);
  const auto& sipmele2geo = this->mapGeoToElectronics(modules, cells, false, true);
  printf("[HGCalMappingIndexESSourceTester][produce] SiPM-on-tile electronics<->geometry map\n");
  printf("\tID maps ele2geo=%ld ID maps geo2ele=%ld\n", sipmele2geo.size(), sipmgeo2ele.size());
  tmap(sipmgeo2ele, sipmele2geo);
  */

  //test dense index token
  auto const& denseIndexInfo = iSetup.getData(denseIndexTkn_);
  printf("Retrieved %d dense index info\n", denseIndexInfo.view().metadata().size());
  int nindices = denseIndexInfo.view().metadata().size();
  printf("fedId fedReadoutSeq detId muxid modix cellidx channel x y z\n");
  for (int i = 0; i < nindices; i++) {
    auto row = denseIndexInfo.view()[i];
    printf("%d %d 0x%x 0x%x %d %d %d %f %f %f\n",
           row.fedId(),
           row.fedReadoutSeq(),
           row.trigdetid(),
           row.muxid(),
           row.modInfoIdx(),
           row.cellInfoIdx(),
           row.TCNumber(),
           row.x(),
           row.y(),
           row.z());
  }
}

//
void HGCalMappingTriggerESSourceTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

//
std::map<uint32_t, uint32_t> HGCalMappingTriggerESSourceTester::mapGeoToElectronics(
    const hgcal::HGCalMappingModuleTriggerParamHost& modules,
    const hgcal::HGCalMappingCellParamHost& cells,
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

      uint32_t elecid = imod.muxid();

      uint32_t geoid(0);

      if (sipm) {
        geoid = ::hgcal::mappingtools::getSiPMDetId(
            imod.zside(), imod.plane(), imod.i2(), imod.celltype(), jcell.i1(), jcell.i2());
      } else {
        geoid = imod.trigdetid() + jcell.detid();
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
    printf("[HGCalMappingTriggerESSourceTester][mapGeoToElectronics] found %d duplicates with geo2ele=%d for sipm=%d\n",
           ndups,
           geo2ele,
           sipm);
  }

  return idmap;
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalMappingTriggerESSourceTester);

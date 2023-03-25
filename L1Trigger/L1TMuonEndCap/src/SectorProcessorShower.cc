#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessorShower.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"

void SectorProcessorShower::configure(const edm::ParameterSet& pset, int endcap, int sector) {
  emtf_assert(emtf::MIN_ENDCAP <= endcap && endcap <= emtf::MAX_ENDCAP);
  emtf_assert(emtf::MIN_TRIGSECTOR <= sector && sector <= emtf::MAX_TRIGSECTOR);

  endcap_ = endcap;
  sector_ = sector;

  enableOneLooseShower_ = pset.getParameter<bool>("enableOneLooseShower");
  enableTwoLooseShowers_ = pset.getParameter<bool>("enableTwoLooseShowers");
  enableOneNominalShower_ = pset.getParameter<bool>("enableOneNominalShowers");
  enableOneTightShower_ = pset.getParameter<bool>("enableOneTightShowers");
}

void SectorProcessorShower::process(const CSCShowerDigiCollection& in_showers,
                                    l1t::RegionalMuonShowerBxCollection& out_showers) const {
  // reset
  std::vector<CSCShowerDigi> selected_showers;

  // shower selection
  auto chamber = in_showers.begin();
  auto chend = in_showers.end();
  for (; chamber != chend; ++chamber) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;
    for (; digi != dend; ++digi) {
      // Returns CSC "link" index (0 - 45)
      int selected_shower = select_shower((*chamber).first, *digi);

      // index is valid
      if (selected_shower >= 0) {
        // 18 in ME1; 9x3 in ME2,3,4
        emtf_assert(selected_shower < CSCConstants::MAX_CSCS_PER_EMTF_SP_NO_OVERLAP);

        // shower is valid
        if (digi->isValid()) {
          selected_showers.emplace_back(*digi);
        }
      }
    }
  }

  // Shower recognition logic: at least one nominal shower (see DN-20-033, section 5.2)
  // Updated shower recognition logic: at least one loose shower (starting April 2023)
  const unsigned nLooseInTime(std::count_if(
      selected_showers.begin(), selected_showers.end(), [](CSCShowerDigi p) { return p.isLooseInTime(); }));
  const unsigned nNominalInTime(std::count_if(
      selected_showers.begin(), selected_showers.end(), [](CSCShowerDigi p) { return p.isNominalInTime(); }));
  const unsigned nTightInTime(std::count_if(
      selected_showers.begin(), selected_showers.end(), [](CSCShowerDigi p) { return p.isTightInTime(); }));

  const bool hasOneLooseInTime(nLooseInTime >= 1);
  const bool hasTwoLooseInTime(nLooseInTime >= 2);
  const bool hasOneNominalInTime(nNominalInTime >= 1);
  const bool hasOneTightInTime(nTightInTime >= 1);

  // for startup Run-3 we're not considering out of time triggers
  const bool acceptLoose(enableOneLooseShower_ and hasOneLooseInTime);
  const bool acceptTwoLoose(enableTwoLooseShowers_ and hasTwoLooseInTime);
  const bool acceptNominal(enableOneNominalShower_ and hasOneNominalInTime);
  const bool acceptTight(enableOneTightShower_ and hasOneTightInTime);

  // trigger condition
  const bool accept(acceptLoose or acceptTwoLoose or acceptNominal or acceptTight);

  if (accept) {
    // shower output
    l1t::RegionalMuonShower out_shower(
        hasOneNominalInTime, false, hasTwoLooseInTime, false, hasOneLooseInTime, hasOneTightInTime, false);
    l1t::tftype tftype = (endcap_ == 1) ? l1t::tftype::emtf_pos : l1t::tftype::emtf_neg;
    out_shower.setTFIdentifiers(sector_ - 1, tftype);
    out_showers.push_back(0, out_shower);
  }
}

// shower selection
int SectorProcessorShower::select_shower(const CSCDetId& tp_detId, const CSCShowerDigi& shower) const {
  int selected = -1;

  int tp_endcap = tp_detId.endcap();
  int tp_sector = tp_detId.triggerSector();
  int tp_station = tp_detId.station();
  int tp_chamber = tp_detId.chamber();
  int tp_csc_ID = shower.getCSCID();

  // station 1 --> subsector 1 or 2
  // station 2,3,4 --> subsector 0
  int tp_subsector = (tp_station != 1) ? 0 : ((tp_chamber % 6 > 2) ? 1 : 2);

  // Check if the chamber belongs to this sector processor at this BX.
  selected = get_index_shower(tp_endcap, tp_sector, tp_subsector, tp_station, tp_csc_ID);
  return selected;
}

int SectorProcessorShower::get_index_shower(
    int tp_endcap, int tp_sector, int tp_subsector, int tp_station, int tp_csc_ID) const {
  int selected = -1;

  // shower trigger does not considers overlaps
  if (is_in_sector_csc(tp_endcap, tp_sector)) {
    if (tp_station == 1) {  // ME1: 0 - 8, 9 - 17
      selected = (tp_subsector - 1) * 9 + (tp_csc_ID - 1);
    } else {  // ME2,3,4: 18 - 26, 27 - 35, 36 - 44
      selected = (tp_station)*9 + (tp_csc_ID - 1);
    }
  }

  emtf_assert(selected != -1);
  return selected;
}

bool SectorProcessorShower::is_in_sector_csc(int tp_endcap, int tp_sector) const {
  return ((endcap_ == tp_endcap) && (sector_ == tp_sector));
}

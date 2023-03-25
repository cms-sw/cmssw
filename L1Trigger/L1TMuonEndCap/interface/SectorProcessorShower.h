#ifndef L1Trigger_L1TMuonEndCap_SectorProcessorShower_h
#define L1Trigger_L1TMuonEndCap_SectorProcessorShower_h

/*
  This class executes the trigger logic for the EMTF shower trigger.
  In the basic mode, the EMTF shower sector processor will find any valid
  CSC shower and send a trigger to the uGMT. In a possible extension, the
  EMTF shower sector processor can also trigger on two loose showers - to
  enhance the sensitivity to long-lived particles that produce multiple
  showers, instead of a single showers.
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "L1Trigger/L1TMuonEndCap/interface/Common.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include <vector>

class SectorProcessorShower {
public:
  explicit SectorProcessorShower() {}
  ~SectorProcessorShower() {}

  void configure(const edm::ParameterSet&, int endcap, int sector);

  void process(const CSCShowerDigiCollection& showers, l1t::RegionalMuonShowerBxCollection& out_showers) const;

private:
  int select_shower(const CSCDetId&, const CSCShowerDigi&) const;
  int get_index_shower(int tp_endcap, int tp_sector, int tp_subsector, int tp_station, int tp_csc_ID) const;
  bool is_in_sector_csc(int tp_endcap, int tp_sector) const;

  int verbose_, endcap_, sector_;
  // loose shower trigger for physics
  bool enableOneLooseShower_;
  // nominal trigger for physics
  bool enableOneNominalShower_;
  // backup trigger
  bool enableOneTightShower_;
  // trigger to extend the physics reach
  bool enableTwoLooseShowers_;
};

#endif

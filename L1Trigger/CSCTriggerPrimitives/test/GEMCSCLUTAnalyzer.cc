#ifndef L1Trigger_CSCTriggerPrimitives_GEMCSCLUTAnalyzer_h
#define L1Trigger_CSCTriggerPrimitives_GEMCSCLUTAnalyzer_h

/** \class GEMCSCLUTAnalyzer
 *
 * Makes the lookup tables for the GEM-CSC integrated local trigger
 * in simulation and firmware
 *
 * authors: Sven Dildick (Rice University)
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include <fstream>
#include <iostream>
#include <vector>

class GEMCSCLUTAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit GEMCSCLUTAnalyzer(const edm::ParameterSet&);
  ~GEMCSCLUTAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  /// generate and print LUT
  void generateLUTs(const CSCDetId& id) const;
  void generateLUTsME11(const CSCDetId& id) const;
  void generateLUTsME21(const CSCDetId& id) const;
  int assignRoll(const std::vector<std::pair<double, double>>&, double eta) const;

  void gemRollToEtaLimitsLUT(const GEMChamber* gemChamber, std::vector<std::pair<double, double>>& lut) const;

  // create LUT: WG->(rollMin,rollMax)
  void cscWgToRollLUT(const std::vector<std::pair<double, double>>&,
                      const std::vector<std::pair<double, double>>&,
                      std::vector<std::pair<int, int>>&) const;

  // create LUT: WG->(etaMin,etaMax)
  void cscWgToEtaLimitsLUT(const CSCLayer*, std::vector<std::pair<double, double>>&) const;

  // create LUT: ES->pad
  void cscEsToGemPadLUT(
      const CSCLayer*, const GEMEtaPartition*, int minH, int maxH, std::vector<std::pair<int, int>>&) const;

  // create LUT: pad->ES
  void gemPadToCscEsLUT(const CSCLayer*, const GEMEtaPartition*, std::vector<int>&) const;

  // create LUT: roll-> center wg
  void gemRollToCscWgLUT(const CSCLayer*, const GEMChamber*, std::vector<std::pair<int, int>>&) const;

  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemToken_;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscToken_;

  const GEMGeometry* gemGeometry_;
  const CSCGeometry* cscGeometry_;
};

#endif

GEMCSCLUTAnalyzer::GEMCSCLUTAnalyzer(const edm::ParameterSet& conf) {
  gemToken_ = esConsumes<GEMGeometry, MuonGeometryRecord>();
  cscToken_ = esConsumes<CSCGeometry, MuonGeometryRecord>();
}

GEMCSCLUTAnalyzer::~GEMCSCLUTAnalyzer() {}

void GEMCSCLUTAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup& setup) {
  edm::ESHandle<GEMGeometry> h_gem = setup.getHandle(gemToken_);
  edm::ESHandle<CSCGeometry> h_csc = setup.getHandle(cscToken_);

  gemGeometry_ = &*h_gem;
  cscGeometry_ = &*h_csc;

  // LUTs are made for ME1/1 and ME2/1, for even/odd

  // ME+1/1/1 (odd)
  generateLUTs(CSCDetId(1, 1, 1, 1));

  // ME+1/1/2 (even)
  generateLUTs(CSCDetId(1, 1, 1, 2));

  // ME+2/1/1 (odd)
  generateLUTs(CSCDetId(1, 2, 1, 1));

  // ME+2/1/2 (even)
  generateLUTs(CSCDetId(1, 2, 1, 2));
}

void GEMCSCLUTAnalyzer::generateLUTs(const CSCDetId& id) const {
  if (id.station() == 1)
    generateLUTsME11(id);
  if (id.station() == 2)
    generateLUTsME21(id);
}

void GEMCSCLUTAnalyzer::generateLUTsME11(const CSCDetId& id) const {
  // CSC trigger geometry
  const CSCDetId me1bId(id);
  const CSCDetId me1aId(id.endcap(), 1, 4, id.chamber());
  const CSCChamber* cscChamberME1b(cscGeometry_->chamber(me1bId));
  const CSCChamber* cscChamberME1a(cscGeometry_->chamber(me1aId));
  const CSCLayer* keyLayerME1b(cscChamberME1b->layer(3));
  const CSCLayer* keyLayerME1a(cscChamberME1a->layer(3));

  // GEM trigger geometry
  const GEMDetId gem_id_l1(id.zendcap(), 1, 1, 1, me1bId.chamber(), 0);
  const GEMDetId gem_id_l2(id.zendcap(), 1, 1, 2, me1bId.chamber(), 0);
  const GEMChamber* gemChamber_l1(gemGeometry_->chamber(gem_id_l1));
  const GEMChamber* gemChamber_l2(gemGeometry_->chamber(gem_id_l2));
  const GEMEtaPartition* randRoll(gemChamber_l1->etaPartition(4));

  // LUTs
  std::vector<std::pair<double, double>> gemRollEtaLimits_l1;
  std::vector<std::pair<double, double>> gemRollEtaLimits_l2;
  std::vector<std::pair<double, double>> cscWGToEtaLimits;
  std::vector<std::pair<int, int>> cscWgToGemRoll_l1;
  std::vector<std::pair<int, int>> cscWgToGemRoll_l2;
  std::vector<std::pair<int, int>> cscEsToGemPadME1a;
  std::vector<std::pair<int, int>> cscEsToGemPadME1b;
  std::vector<int> gemPadToCscEsME1a;
  std::vector<int> gemPadToCscEsME1b;
  std::vector<std::pair<int, int>> gemRollToCscWg;

  gemRollToEtaLimitsLUT(gemChamber_l1, gemRollEtaLimits_l1);
  gemRollToEtaLimitsLUT(gemChamber_l2, gemRollEtaLimits_l2);
  cscWgToEtaLimitsLUT(keyLayerME1b, cscWGToEtaLimits);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l1, cscWgToGemRoll_l1);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l2, cscWgToGemRoll_l2);
  cscEsToGemPadLUT(keyLayerME1a, randRoll, 2, 94, cscEsToGemPadME1a);
  cscEsToGemPadLUT(keyLayerME1b, randRoll, 4, 124, cscEsToGemPadME1b);
  gemPadToCscEsLUT(keyLayerME1a, randRoll, gemPadToCscEsME1a);
  gemPadToCscEsLUT(keyLayerME1b, randRoll, gemPadToCscEsME1b);
  gemRollToCscWgLUT(keyLayerME1b, gemChamber_l1, gemRollToCscWg);

  const std::string oddeven(id.chamber() % 2 == 0 ? "_even" : "_odd");

  std::ofstream ofos;
  // simulation LUTs
  ofos.open("GEMCSCLUT_pad_es_ME1a" + oddeven + ".txt");
  for (const auto& p : gemPadToCscEsME1a)
    ofos << p << std::endl;
  ofos.close();

  ofos.open("GEMCSCLUT_pad_es_ME1b" + oddeven + ".txt");
  for (const auto& p : gemPadToCscEsME1b)
    ofos << p << std::endl;
  ofos.close();

  ofos.open("GEMCSCLUT_roll_min_wg_ME11" + oddeven + ".txt");
  for (const auto& p : gemRollToCscWg)
    ofos << p.first << std::endl;
  ofos.close();

  ofos.open("GEMCSCLUT_roll_max_wg_ME11" + oddeven + ".txt");
  for (const auto& p : gemRollToCscWg)
    ofos << p.second << std::endl;
  ofos.close();

  // firmware LUTs
  ofos.open("GEMCSCLUT_pad_es_ME1a" + oddeven + ".mem");
  for (const auto& p : gemPadToCscEsME1a)
    ofos << std::hex << p << std::endl;
  ofos.close();

  ofos.open("GEMCSCLUT_pad_es_ME1b" + oddeven + ".mem");
  for (const auto& p : gemPadToCscEsME1b)
    ofos << std::hex << p << std::endl;
  ofos.close();

  ofos.open("GEMCSCLUT_roll_min_wg_ME11" + oddeven + ".mem");
  for (const auto& p : gemRollToCscWg)
    ofos << std::hex << p.first << std::endl;
  ofos.close();

  ofos.open("GEMCSCLUT_roll_max_wg_ME11" + oddeven + ".mem");
  for (const auto& p : gemRollToCscWg)
    ofos << std::hex << p.second << std::endl;
  ofos.close();
}

void GEMCSCLUTAnalyzer::generateLUTsME21(const CSCDetId& csc_id) const {
  const CSCChamber* cscChamber(cscGeometry_->chamber(csc_id));
  const CSCLayer* keyLayer(cscChamber->layer(3));

  // GEM trigger geometry
  const GEMDetId gem_id_l1(csc_id.zendcap(), 1, 2, 1, csc_id.chamber(), 0);
  const GEMDetId gem_id_l2(csc_id.zendcap(), 1, 2, 2, csc_id.chamber(), 0);
  const GEMChamber* gemChamber_l1(gemGeometry_->chamber(gem_id_l1));
  const GEMChamber* gemChamber_l2(gemGeometry_->chamber(gem_id_l2));
  const GEMEtaPartition* randRoll(gemChamber_l1->etaPartition(4));

  // LUTs
  std::vector<std::pair<double, double>> gemRollEtaLimits_l1;
  std::vector<std::pair<double, double>> gemRollEtaLimits_l2;
  std::vector<std::pair<double, double>> cscWGToEtaLimits;
  std::vector<std::pair<int, int>> cscWgToGemRoll_l1;
  std::vector<std::pair<int, int>> cscWgToGemRoll_l2;
  std::vector<std::pair<int, int>> cscEsToGemPad;
  std::vector<int> gemPadToCscEs;
  std::vector<std::pair<int, int>> gemRollToCscWg;

  gemRollToEtaLimitsLUT(gemChamber_l1, gemRollEtaLimits_l1);
  gemRollToEtaLimitsLUT(gemChamber_l2, gemRollEtaLimits_l2);
  cscWgToEtaLimitsLUT(keyLayer, cscWGToEtaLimits);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l1, cscWgToGemRoll_l1);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l2, cscWgToGemRoll_l2);
  cscEsToGemPadLUT(keyLayer, randRoll, 4, 155, cscEsToGemPad);
  gemPadToCscEsLUT(keyLayer, randRoll, gemPadToCscEs);
  gemRollToCscWgLUT(keyLayer, gemChamber_l1, gemRollToCscWg);

  const std::string oddeven(csc_id.chamber() % 2 == 0 ? "_even" : "_odd");

  std::ofstream ofos;
  // simulation LUTs
  ofos.open("GEMCSCLUT_pad_es_ME21" + oddeven + ".txt");
  for (const auto& p : gemPadToCscEs)
    ofos << p << std::endl;
  ofos.close();

  ofos.open("GEMCSCLUT_roll_min_wg_ME21" + oddeven + ".txt");
  for (const auto& p : gemRollToCscWg)
    ofos << p.first << std::endl;
  ofos.close();

  ofos.open("GEMCSCLUT_roll_max_wg_ME21" + oddeven + ".txt");
  for (const auto& p : gemRollToCscWg)
    ofos << p.second << std::endl;
  ofos.close();

  // firmware LUTs
  ofos.open("GEMCSCLUT_pad_es_ME21" + oddeven + ".mem");
  for (const auto& p : gemPadToCscEs)
    ofos << std::hex << p << std::endl;
  ofos.close();

  ofos.open("GEMCSCLUT_roll_min_wg_ME21" + oddeven + ".mem");
  for (const auto& p : gemRollToCscWg)
    ofos << std::hex << p.first << std::endl;
  ofos.close();

  ofos.open("GEMCSCLUT_roll_max_wg_ME21" + oddeven + ".mem");
  for (const auto& p : gemRollToCscWg)
    ofos << std::hex << p.second << std::endl;
  ofos.close();
}

int GEMCSCLUTAnalyzer::assignRoll(const std::vector<std::pair<double, double>>& lut, double eta) const {
  int result = -99;
  int iRoll = 0;
  for (const auto& p : lut) {
    iRoll++;
    const float minEta(p.first);
    const float maxEta(p.second);
    if (minEta <= std::abs(eta) and std::abs(eta) < maxEta) {
      result = iRoll;
      break;
    }
  }
  return result;
}

void GEMCSCLUTAnalyzer::gemRollToEtaLimitsLUT(const GEMChamber* gemChamber,
                                              std::vector<std::pair<double, double>>& lut) const {
  for (const auto& roll : gemChamber->etaPartitions()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength() / 2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    const double bottom_eta(std::abs(gp_bottom.eta()));
    const double top_eta(std::abs(gp_top.eta()));
    lut.emplace_back(std::min(bottom_eta, top_eta), std::max(bottom_eta, top_eta));
  }
}

void GEMCSCLUTAnalyzer::cscWgToRollLUT(const std::vector<std::pair<double, double>>& inLUT1,
                                       const std::vector<std::pair<double, double>>& inLUT2,
                                       std::vector<std::pair<int, int>>& outLUT) const {
  for (const auto& p : inLUT1) {
    double etaMin(p.first);
    double etaMax(p.second);
    outLUT.emplace_back(assignRoll(inLUT2, etaMin), assignRoll(inLUT2, etaMax));
  }
}

void GEMCSCLUTAnalyzer::cscWgToEtaLimitsLUT(const CSCLayer* keyLayer,
                                            std::vector<std::pair<double, double>>& lut) const {
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  const int numberOfWG(keyLayerGeometry->numberOfWireGroups());
  for (int i = 0; i < numberOfWG; ++i) {
    const float middle_wire(keyLayerGeometry->middleWireOfGroup(i));
    const std::pair<LocalPoint, LocalPoint> wire_ends(keyLayerGeometry->wireTopology()->wireEnds(middle_wire));
    const GlobalPoint gp_top(keyLayer->toGlobal(wire_ends.first));
    const GlobalPoint gp_bottom(keyLayer->toGlobal(wire_ends.second));
    const double bottom_eta(std::abs(gp_bottom.eta()));
    const double top_eta(std::abs(gp_top.eta()));
    lut.emplace_back(std::min(bottom_eta, top_eta), std::max(bottom_eta, top_eta));
  }
}

void GEMCSCLUTAnalyzer::cscEsToGemPadLUT(const CSCLayer* keyLayer,
                                         const GEMEtaPartition* randRoll,
                                         int minH,
                                         int maxH,
                                         std::vector<std::pair<int, int>>& lut) const {
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  auto nStrips(keyLayerGeometry->numberOfStrips());
  for (float i = 0; i < nStrips; i = i + 0.125) {
    const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
    const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
    const LocalPoint lpGEM(randRoll->toLocal(gp));
    const float pad(randRoll->pad(lpGEM));
    lut.emplace_back(std::floor(pad), std::ceil(pad));
  }
}

void GEMCSCLUTAnalyzer::gemPadToCscEsLUT(const CSCLayer* keyLayer,
                                         const GEMEtaPartition* randRoll,
                                         std::vector<int>& lut) const {
  int offset(0);
  if (keyLayer->id().ring() == 4)
    offset = 64;
  const int nGEMPads(randRoll->npads());
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  for (int i = 0; i < nGEMPads; ++i) {
    const LocalPoint lpGEM(randRoll->centreOfPad(i));
    const GlobalPoint gp(randRoll->toGlobal(lpGEM));
    const LocalPoint lpCSC(keyLayer->toLocal(gp));
    const float strip(keyLayerGeometry->strip(lpCSC));
    lut.push_back(int((strip + offset) * 8));
  }
}

void GEMCSCLUTAnalyzer::gemRollToCscWgLUT(const CSCLayer* keyLayer,
                                          const GEMChamber* gemChamber,
                                          std::vector<std::pair<int, int>>& lut) const {
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  for (const auto& roll : gemChamber->etaPartitions()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength() / 2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);

    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));

    const LocalPoint lp_csc_top(keyLayer->toLocal(gp_top));
    const LocalPoint lp_csc_bottom(keyLayer->toLocal(gp_bottom));

    const int wire_top(keyLayerGeometry->nearestWire(lp_csc_top));
    const int wg_top(keyLayerGeometry->wireGroup(wire_top));

    const int wire_bottom(keyLayerGeometry->nearestWire(lp_csc_bottom));
    const int wg_bottom(keyLayerGeometry->wireGroup(wire_bottom));

    lut.emplace_back(wg_bottom, wg_top);
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMCSCLUTAnalyzer);

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCUpgradeMotherboardLUTGenerator.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

void CSCUpgradeMotherboardLUTGenerator::generateLUTs(
    unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber) const {
  if (theStation == 1)
    generateLUTsME11(theEndcap, theSector, theSubsector, theTrigChamber);
  if (theStation == 2)
    generateLUTsME21(theEndcap, theSector, theSubsector, theTrigChamber);
}

void CSCUpgradeMotherboardLUTGenerator::generateLUTsME11(unsigned theEndcap,
                                                         unsigned theSector,
                                                         unsigned theSubsector,
                                                         unsigned theTrigChamber) const {
  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    LogTrace("CSCUpgradeMotherboardLUTGenerator") << "+++ generateLUTsME11() called for ME11 chamber! +++ \n";
    gemGeometryAvailable = true;
  }

  // check for GEM geometry
  if (not gemGeometryAvailable) {
    LogTrace("CSCUpgradeMotherboardLUTGenerator")
        << "+++ generateLUTsME11() called for ME11 chamber without valid GEM geometry! +++ \n";
    return;
  }

  // CSC trigger geometry
  const int chid = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, 1, theTrigChamber);
  const CSCDetId me1bId(theEndcap, 1, 1, chid, 0);
  const CSCDetId me1aId(theEndcap, 1, 4, chid, 0);
  const CSCChamber* cscChamberME1b(csc_g->chamber(me1bId));
  const CSCChamber* cscChamberME1a(csc_g->chamber(me1aId));
  const CSCLayer* keyLayerME1b(cscChamberME1b->layer(3));
  const CSCLayer* keyLayerME1a(cscChamberME1a->layer(3));

  // GEM trigger geometry
  const int region((theEndcap == 1) ? 1 : -1);
  const GEMDetId gem_id_l1(region, 1, 1, 1, me1bId.chamber(), 0);
  const GEMDetId gem_id_l2(region, 1, 1, 2, me1bId.chamber(), 0);
  const GEMChamber* gemChamber_l1(gem_g->chamber(gem_id_l1));
  const GEMChamber* gemChamber_l2(gem_g->chamber(gem_id_l2));
  const GEMEtaPartition* randRoll(gemChamber_l1->etaPartition(4));

  // LUTs
  std::vector<std::pair<double, double> > gemRollEtaLimits_l1;
  std::vector<std::pair<double, double> > gemRollEtaLimits_l2;
  std::vector<std::pair<double, double> > cscWGToEtaLimits;
  std::vector<std::pair<int, int> > cscWgToGemRoll_l1;
  std::vector<std::pair<int, int> > cscWgToGemRoll_l2;
  std::vector<std::pair<int, int> > cscHsToGemPadME1a;
  std::vector<std::pair<int, int> > cscHsToGemPadME1b;
  std::vector<int> gemPadToCscHsME1a;
  std::vector<int> gemPadToCscHsME1b;
  std::vector<int> gemRollToCscWg1b;

  gemRollToEtaLimitsLUT(gemChamber_l1, gemRollEtaLimits_l1);
  gemRollToEtaLimitsLUT(gemChamber_l2, gemRollEtaLimits_l2);
  cscWgToEtaLimitsLUT(keyLayerME1b, cscWGToEtaLimits);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l1, cscWgToGemRoll_l1);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l2, cscWgToGemRoll_l2);
  cscHsToGemPadLUT(keyLayerME1a, randRoll, 2, 94, cscHsToGemPadME1a);
  cscHsToGemPadLUT(keyLayerME1b, randRoll, 4, 124, cscHsToGemPadME1b);
  gemPadToCscHsLUT(keyLayerME1a, randRoll, gemPadToCscHsME1a);
  gemPadToCscHsLUT(keyLayerME1b, randRoll, gemPadToCscHsME1b);
  gemRollToCscWgLUT(keyLayerME1b, gemChamber_l1, gemRollToCscWg1b);

  // print LUTs
  std::stringstream os;
  os << "ME11 " << me1bId << std::endl;

  os << "GEM L1 roll to eta limits" << std::endl;
  os << gemRollEtaLimits_l1;

  os << "GEM L2 roll to eta limits" << std::endl;
  os << gemRollEtaLimits_l2;

  os << "ME1b " << me1bId << std::endl;
  os << "WG roll to eta limits" << std::endl;
  os << cscWGToEtaLimits;

  os << "WG to Roll L1" << std::endl;
  os << cscWgToGemRoll_l1;

  os << "WG to Roll L2" << std::endl;
  os << cscWgToGemRoll_l2;

  os << "CSC HS to GEM pad LUT in ME1a" << std::endl;
  os << cscHsToGemPadME1a;

  os << "CSC HS to GEM pad LUT in ME1b" << std::endl;
  os << cscHsToGemPadME1b;

  os << "GEM pad to CSC HS LUT in ME1a" << std::endl;
  os << gemPadToCscHsME1a;

  os << "GEM pad to CSC HS LUT in ME1b" << std::endl;
  os << gemPadToCscHsME1b;

  os << "GEM roll to CSC WG" << std::endl;
  os << gemRollToCscWg1b;

  // print LUTs
  LogTrace("CSCUpgradeMotherboardLUTGenerator") << os.str();
}

void CSCUpgradeMotherboardLUTGenerator::generateLUTsME21(unsigned theEndcap,
                                                         unsigned theSector,
                                                         unsigned theSubsector,
                                                         unsigned theTrigChamber) const {
  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    LogTrace("CSCUpgradeMotherboardLUTGenerator") << "+++ generateLUTsME11() called for ME21 chamber! +++ \n";
    gemGeometryAvailable = true;
  }

  // check for GEM geometry
  if (not gemGeometryAvailable) {
    LogTrace("CSCUpgradeMotherboardLUTGenerator")
        << "+++ generateLUTsME21() called for ME21 chamber without valid GEM geometry! +++ \n";
    return;
  }

  // CSC trigger geometry
  const int chid = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, 2, theTrigChamber);
  const CSCDetId csc_id(theEndcap, 2, 1, chid, 0);
  const CSCChamber* cscChamber(csc_g->chamber(csc_id));
  const CSCLayer* keyLayer(cscChamber->layer(3));

  // GEM trigger geometry
  const int region((theEndcap == 1) ? 1 : -1);
  const GEMDetId gem_id_l1(region, 1, 2, 1, csc_id.chamber(), 0);
  const GEMDetId gem_id_l2(region, 1, 2, 2, csc_id.chamber(), 0);
  const GEMChamber* gemChamber_l1(gem_g->chamber(gem_id_l1));
  const GEMChamber* gemChamber_l2(gem_g->chamber(gem_id_l2));
  const GEMEtaPartition* randRoll(gemChamber_l1->etaPartition(4));

  // LUTs
  std::vector<std::pair<double, double> > gemRollEtaLimits_l1;
  std::vector<std::pair<double, double> > gemRollEtaLimits_l2;
  std::vector<std::pair<double, double> > cscWGToEtaLimits;
  std::vector<std::pair<int, int> > cscWgToGemRoll_l1;
  std::vector<std::pair<int, int> > cscWgToGemRoll_l2;
  std::vector<std::pair<int, int> > cscHsToGemPad;
  std::vector<int> gemPadToCscHs;
  std::vector<int> gemRollToCscWg;

  gemRollToEtaLimitsLUT(gemChamber_l1, gemRollEtaLimits_l1);
  gemRollToEtaLimitsLUT(gemChamber_l2, gemRollEtaLimits_l2);
  cscWgToEtaLimitsLUT(keyLayer, cscWGToEtaLimits);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l1, cscWgToGemRoll_l1);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l2, cscWgToGemRoll_l2);
  cscHsToGemPadLUT(keyLayer, randRoll, 4, 155, cscHsToGemPad);
  gemPadToCscHsLUT(keyLayer, randRoll, gemPadToCscHs);
  gemRollToCscWgLUT(keyLayer, gemChamber_l1, gemRollToCscWg);

  std::stringstream os;
  os << "ME21 " << csc_id << std::endl;

  os << "GEM roll to eta limits L1" << std::endl;
  os << gemRollEtaLimits_l1;

  os << "GEM roll to eta limits L2" << std::endl;
  os << gemRollEtaLimits_l2;

  os << "WG to eta limits" << std::endl;
  os << cscWGToEtaLimits;

  os << "WG to Roll L1" << std::endl;
  os << cscWgToGemRoll_l1;

  os << "WG to Roll L2" << std::endl;
  os << cscWgToGemRoll_l2;

  os << "CSC HS to GEM pad LUT in ME21" << std::endl;
  os << cscHsToGemPad;

  os << "GEM pad to CSC HS LUT in ME21" << std::endl;
  os << gemPadToCscHs;

  os << "GEM roll to CSC WG" << std::endl;
  os << gemRollToCscWg;

  // print LUTs
  LogTrace("CSCUpgradeMotherboardLUTGenerator") << os.str();
}

int CSCUpgradeMotherboardLUTGenerator::assignRoll(const std::vector<std::pair<double, double> >& lut,
                                                  double eta) const {
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

void CSCUpgradeMotherboardLUTGenerator::gemRollToEtaLimitsLUT(const GEMChamber* gemChamber,
                                                              std::vector<std::pair<double, double> >& lut) const {
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

void CSCUpgradeMotherboardLUTGenerator::cscWgToRollLUT(const std::vector<std::pair<double, double> >& inLUT1,
                                                       const std::vector<std::pair<double, double> >& inLUT2,
                                                       std::vector<std::pair<int, int> >& outLUT) const {
  for (const auto& p : inLUT1) {
    double etaMin(p.first);
    double etaMax(p.second);
    outLUT.emplace_back(assignRoll(inLUT2, etaMin), assignRoll(inLUT2, etaMax));
  }
}

void CSCUpgradeMotherboardLUTGenerator::cscWgToEtaLimitsLUT(const CSCLayer* keyLayer,
                                                            std::vector<std::pair<double, double> >& lut) const {
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

void CSCUpgradeMotherboardLUTGenerator::cscHsToGemPadLUT(const CSCLayer* keyLayer,
                                                         const GEMEtaPartition* randRoll,
                                                         int minH,
                                                         int maxH,
                                                         std::vector<std::pair<int, int> >& lut) const {
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  auto nStrips(keyLayerGeometry->numberOfStrips());
  for (float i = 0; i < nStrips; i = i + 0.5) {
    const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
    const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
    const LocalPoint lpGEM(randRoll->toLocal(gp));
    const float pad(randRoll->pad(lpGEM));
    lut.emplace_back(std::floor(pad), std::ceil(pad));
  }
}

void CSCUpgradeMotherboardLUTGenerator::gemPadToCscHsLUT(const CSCLayer* keyLayer,
                                                         const GEMEtaPartition* randRoll,
                                                         std::vector<int>& lut) const {
  const int nGEMPads(randRoll->npads());
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  for (int i = 0; i < nGEMPads; ++i) {
    const LocalPoint lpGEM(randRoll->centreOfPad(i));
    const GlobalPoint gp(randRoll->toGlobal(lpGEM));
    const LocalPoint lpCSC(keyLayer->toLocal(gp));
    const float strip(keyLayerGeometry->strip(lpCSC));
    lut.push_back(int(strip * 2));
  }
}

void CSCUpgradeMotherboardLUTGenerator::gemRollToCscWgLUT(const CSCLayer* keyLayer,
                                                          const GEMChamber* gemChamber,
                                                          std::vector<int>& lut) const {
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  for (const auto& roll : gemChamber->etaPartitions()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength() / 2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    float x, y, z;
    x = (gp_top.x() + gp_bottom.x()) / 2.;
    y = (gp_top.y() + gp_bottom.y()) / 2.;
    z = (gp_top.z() + gp_bottom.z()) / 2.;
    const GlobalPoint gp_ave(x, y, z);
    const LocalPoint lp_ave(keyLayer->toLocal(gp_ave));
    const int wire(keyLayerGeometry->nearestWire(lp_ave));
    const int wg(keyLayerGeometry->wireGroup(wire));
    lut.emplace_back(wg);
  }
}

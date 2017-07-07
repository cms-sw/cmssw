#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUTGenerator.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream> 

RPCDetId CSCUpgradeMotherboardLUTGenerator::Helpers::getRPCfromCSC(const CSCDetId& csc_id)
{
  const int region(csc_id.zendcap());
  const int csc_trig_sect(CSCTriggerNumbering::triggerSectorFromLabels(csc_id));
  const int csc_trig_id( CSCTriggerNumbering::triggerCscIdFromLabels(csc_id));
  const int csc_trig_chid((3*(csc_trig_sect-1)+csc_trig_id)%18 +1);
  const int rpc_trig_sect((csc_trig_chid-1)/3+1);
  const int rpc_trig_subsect((csc_trig_chid-1)%3+1);
  return RPCDetId(region,1,csc_id.station(),rpc_trig_sect,1,rpc_trig_subsect,0);
}

void CSCUpgradeMotherboardLUTGenerator::generateLUTs(unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber) const
{
  if (theStation==1) generateLUTsME11(theEndcap, theSector, theSubsector, theTrigChamber);
  if (theStation==2) generateLUTsME21(theEndcap, theSector, theSubsector, theTrigChamber);
  if (theStation==3) generateLUTsME3141(theEndcap, 3, theSector, theSubsector, theTrigChamber);
  if (theStation==4) generateLUTsME3141(theEndcap, 4, theSector, theSubsector, theTrigChamber);
}

void CSCUpgradeMotherboardLUTGenerator::generateLUTsME11(unsigned theEndcap, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber) const
{
  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    LogTrace("CSCUpgradeMotherboardLUTGenerator")
      << "+++ generateLUTsME11() called for ME11 chamber! +++ \n";
    gemGeometryAvailable = true;
  }
  
  // check for GEM geometry
  if (not gemGeometryAvailable){
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
  const int region((theEndcap == 1) ? 1: -1);
  const GEMDetId gem_id_l1(region, 1, 1, 1, me1bId.chamber(), 0);
  const GEMDetId gem_id_l2(region, 1, 1, 2, me1bId.chamber(), 0);
  const GEMChamber* gemChamber_l1(gem_g->chamber(gem_id_l1));
  const GEMChamber* gemChamber_l2(gem_g->chamber(gem_id_l2));
  const GEMEtaPartition* randRoll(gemChamber_l1->etaPartition(2));
  
  // LUTs
  std::vector<std::pair<double,double> > gemRollEtaLimits_l1;
  std::vector<std::pair<double,double> > gemRollEtaLimits_l2;
  std::vector<std::pair<double,double> > cscWGToEtaLimits;
  std::vector<std::pair<int,int> > cscWgToGemRoll_l1;
  std::vector<std::pair<int,int> > cscWgToGemRoll_l2;
  std::vector<std::pair<int,int> > cscHsToGemPadME1a;
  std::vector<std::pair<int,int> > cscHsToGemPadME1b;
  std::vector<int> gemPadToCscHsME1a;
  std::vector<int> gemPadToCscHsME1b;

  gemRollToEtaLimitsLUT(gemChamber_l1, gemRollEtaLimits_l1);
  gemRollToEtaLimitsLUT(gemChamber_l2, gemRollEtaLimits_l2);
  cscWgToEtaLimitsLUT(keyLayerME1b, cscWGToEtaLimits);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l1, cscWgToGemRoll_l1);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l2, cscWgToGemRoll_l2);
  cscHsToGemPadLUT(keyLayerME1a, randRoll, 4, 93, cscHsToGemPadME1a);
  cscHsToGemPadLUT(keyLayerME1b, randRoll, 5, 124, cscHsToGemPadME1b);
  gemPadToCscHsLUT(keyLayerME1a, randRoll, gemPadToCscHsME1a);
  gemPadToCscHsLUT(keyLayerME1b, randRoll, gemPadToCscHsME1b);

  // print LUTs
  std::stringstream os;
  os << "ME11 "<< me1bId <<std::endl;

  os << "GEM L1 roll to eta limits" << std::endl;
  os << gemRollEtaLimits_l1;

  os << "GEM L2 roll to eta limits" << std::endl;
  os << gemRollEtaLimits_l2;
  
  os << "ME1b "<< me1bId <<std::endl;
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

  // print LUTs
  LogTrace("CSCUpgradeMotherboardLUTGenerator") << os.str();
}

void CSCUpgradeMotherboardLUTGenerator::generateLUTsME21(unsigned theEndcap, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber) const
{
  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    LogTrace("CSCUpgradeMotherboardLUTGenerator") << "+++ generateLUTsME11() called for ME21 chamber! +++ \n";
    gemGeometryAvailable = true;
  }

  // check for GEM geometry
  if (not gemGeometryAvailable){
    LogTrace("CSCUpgradeMotherboardLUTGenerator") << "+++ generateLUTsME21() called for ME21 chamber without valid GEM geometry! +++ \n";
    return;
  }
  
  // CSC trigger geometry
  const int chid = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, 2, theTrigChamber);
  const CSCDetId csc_id(theEndcap, 2, 1, chid, 0);
  const CSCChamber* cscChamber(csc_g->chamber(csc_id));
  const CSCLayer* keyLayer(cscChamber->layer(3));  
    
  // GEM trigger geometry
  const int region((theEndcap == 1) ? 1: -1);
  const GEMDetId gem_id_l1(region, 1, 2, 1, csc_id.chamber(), 0);
  const GEMDetId gem_id_l2(region, 1, 2, 2, csc_id.chamber(), 0);
  const GEMChamber* gemChamber_l1(gem_g->chamber(gem_id_l1));
  const GEMChamber* gemChamber_l2(gem_g->chamber(gem_id_l2));
  const GEMEtaPartition* randRoll(gemChamber_l1->etaPartition(2));
  
  // LUTs
  std::vector<std::pair<double,double> > gemRollEtaLimits_l1;
  std::vector<std::pair<double,double> > gemRollEtaLimits_l2;
  std::vector<std::pair<double,double> > cscWGToEtaLimits;
  std::vector<std::pair<int,int> > cscWgToGemRoll_l1;
  std::vector<std::pair<int,int> > cscWgToGemRoll_l2;
  std::vector<std::pair<int,int> > cscHsToGemPad;
  std::vector<int> gemPadToCscHs;

  gemRollToEtaLimitsLUT(gemChamber_l1, gemRollEtaLimits_l1);
  gemRollToEtaLimitsLUT(gemChamber_l2, gemRollEtaLimits_l2);
  cscWgToEtaLimitsLUT(keyLayer, cscWGToEtaLimits);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l1, cscWgToGemRoll_l1);
  cscWgToRollLUT(cscWGToEtaLimits, gemRollEtaLimits_l2, cscWgToGemRoll_l2);
  cscHsToGemPadLUT(keyLayer, randRoll, 4, 155, cscHsToGemPad);
  gemPadToCscHsLUT(keyLayer, randRoll, gemPadToCscHs);

  std::stringstream os;
  os << "ME21 "<< csc_id <<std::endl;

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

  // print LUTs
  LogTrace("CSCUpgradeMotherboardLUTGenerator") << os.str();
}

void CSCUpgradeMotherboardLUTGenerator::generateLUTsME3141(unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber) const
{
  bool rpcGeometryAvailable(false);
  if (rpc_g != nullptr) {
    LogTrace("CSCUpgradeMotherboardLUTGenerator")<< "+++ generateLUTsME3141() called for ME3141 chamber! +++ \n";
    rpcGeometryAvailable = true;
  }
  
  if (not rpcGeometryAvailable){
    LogTrace("CSCUpgradeMotherboardLUTGenerator") << "+++ generateLUTsME3141() called for ME3141 chamber without valid RPC geometry! +++ \n";
    return;
  }

  // CSC trigger geometry
  const int chid = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, theStation, theTrigChamber);
  const CSCDetId csc_id(theEndcap, theStation, 1, chid, 0);
  const CSCChamber* cscChamber(csc_g->chamber(csc_id));
  const CSCLayer* keyLayer(cscChamber->layer(3));

  // RPC trigger geometry
  const RPCDetId& rpc_id(CSCUpgradeMotherboardLUTGenerator::Helpers::getRPCfromCSC(csc_id));
  const RPCChamber* rpcChamber(rpc_g->chamber(rpc_id));
  const RPCRoll* randRoll(rpcChamber->roll(2));

  // LUTs
  std::vector<std::pair<double,double> > rpcRollToEtaLimits;
  std::vector<std::pair<double,double> > cscWGToEtaLimits;
  std::vector<std::pair<int,int> > cscWgToRpcRoll;
  std::vector<std::pair<int,int> > cscHsToRpcStrip;
  std::vector<int> rpcStripToCscHs;

  rpcRollToEtaLimitsLUT(rpcChamber, rpcRollToEtaLimits);
  cscWgToEtaLimitsLUT(keyLayer, cscWGToEtaLimits);
  cscWgToRollLUT(cscWGToEtaLimits, rpcRollToEtaLimits, cscWgToRpcRoll);
  cscHsToRpcStripLUT(keyLayer, randRoll, 5, 155, cscHsToRpcStrip);
  rpcStripToCscHsLUT(keyLayer, randRoll, rpcStripToCscHs);

  std::stringstream os;
  os << "ME31/41 "<< csc_id <<std::endl;

  os << "RPC roll to eta limits" << std::endl;
  os << rpcRollToEtaLimits;

  os << "WG to eta limits" << std::endl;
  os << cscWGToEtaLimits;

  os << "WG to Roll" << std::endl;
  os << cscWgToRpcRoll;

  os << "CSC HS to RPC strip LUT in ME3141" << std::endl;
  os << cscHsToRpcStrip;
  
  os << "RPC strip to CSC HS LUT in ME3141" << std::endl;
  os << rpcStripToCscHs;

  // print LUTs
  LogTrace("CSCUpgradeMotherboardLUTGenerator") << os.str();
}

int CSCUpgradeMotherboardLUTGenerator::assignRoll(const std::vector<std::pair<double,double> >& lut_, double eta) const
{
  int result = -99;
  for(const auto& p : lut_) {
    const float minEta(p.first);
    const float maxEta(p.second);
    if (minEta <= std::abs(eta) and std::abs(eta) < maxEta) {
      result = p.first;
      break;
    }
  }
  return result;
}

void CSCUpgradeMotherboardLUTGenerator::gemRollToEtaLimitsLUT(const GEMChamber* gemChamber, std::vector<std::pair<double,double> >& lut) const
{
  for(const auto& roll : gemChamber->etaPartitions()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    lut.emplace_back(std::abs(gp_top.eta()), std::abs(gp_bottom.eta()));
  }
}


void CSCUpgradeMotherboardLUTGenerator::rpcRollToEtaLimitsLUT(const RPCChamber* rpcChamber, std::vector<std::pair<double,double> >& lut) const
{
  for(const auto& roll : rpcChamber->rolls()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    lut.push_back(std::make_pair(std::abs(gp_top.eta()), std::abs(gp_bottom.eta())));
  }
}

void CSCUpgradeMotherboardLUTGenerator::cscWgToRollLUT(const std::vector<std::pair<double,double> >& inLUT1,
						       const std::vector<std::pair<double,double> >& inLUT2,
						       std::vector<std::pair<int,int> >& outLUT) const
{
  for (const auto& p: inLUT1){
    double etaMin(p.first);
    double etaMax(p.second);
    outLUT.emplace_back(assignRoll(inLUT2, etaMin), assignRoll(inLUT2, etaMax));
  }
}

void CSCUpgradeMotherboardLUTGenerator::cscWgToEtaLimitsLUT(const CSCLayer* keyLayer, std::vector<std::pair<double, double> >& lut) const
{
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  const int numberOfWG(keyLayerGeometry->numberOfWireGroups());
  for (int i = 0; i< numberOfWG; ++i){
    const float middle_wire(keyLayerGeometry->middleWireOfGroup(i));
    const std::pair<LocalPoint, LocalPoint> wire_ends(keyLayerGeometry->wireTopology()->wireEnds(middle_wire));
    const GlobalPoint gp_top(keyLayer->toGlobal(wire_ends.first));
    const GlobalPoint gp_bottom(keyLayer->toGlobal(wire_ends.first));
    lut.emplace_back(gp_top.eta(), gp_bottom.eta());
  }
}

void CSCUpgradeMotherboardLUTGenerator::cscHsToGemPadLUT(const CSCLayer* keyLayer, 
						    const GEMEtaPartition* randRoll, 
						    int minH, int maxH, std::vector<std::pair<int,int> >& lut) const
{
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  auto nStrips(keyLayerGeometry->numberOfStrips());
  for (float i = 0; i< nStrips; i = i+0.5){
    const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
    const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
    const LocalPoint lpGEM(randRoll->toLocal(gp));
    const int HS(i/0.5);
    const bool edge(HS < minH or HS > maxH);
    const float pad(edge ? -99 : randRoll->pad(lpGEM));
    lut.emplace_back(std::floor(pad),std::ceil(pad));
  }
}

void
CSCUpgradeMotherboardLUTGenerator::gemPadToCscHsLUT(const CSCLayer* keyLayer, 
						    const GEMEtaPartition* randRoll,
						    std::vector<int>& lut) const
{
  const int nGEMPads(randRoll->npads());
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  for (int i = 1; i<= nGEMPads; ++i){
    const LocalPoint lpGEM(randRoll->centreOfPad(i));
    const GlobalPoint gp(randRoll->toGlobal(lpGEM));
    const LocalPoint lpCSC(keyLayer->toLocal(gp));
    const float strip(keyLayerGeometry->strip(lpCSC));
    lut.push_back( (int) (strip)/0.5 );
  }
}

void
CSCUpgradeMotherboardLUTGenerator::cscHsToRpcStripLUT(const CSCLayer* keyLayer, 
						      const RPCRoll* randRoll, 
						      int minH, int maxH, std::vector<std::pair<int,int> >& lut) const
{
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  auto nStrips(keyLayerGeometry->numberOfStrips());
  for (float i = 0; i< nStrips; i = i+0.5){
    const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
    const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
    const LocalPoint lpRPC(randRoll->toLocal(gp));
    const int HS(i/0.5);
    const bool edge(HS < minH or HS > maxH);
    const float strip(edge ? -99 : randRoll->strip(lpRPC));
    lut.emplace_back(std::floor(strip),std::ceil(strip));
  }
}

void
CSCUpgradeMotherboardLUTGenerator::rpcStripToCscHsLUT(const CSCLayer* keyLayer, 
						      const RPCRoll* randRoll, std::vector<int>& lut) const
{
  const int nRPCStrips(randRoll->nstrips());
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  for (int i = 1; i<= nRPCStrips; ++i){
    const LocalPoint lpRPC(randRoll->centreOfStrip(i));
    const GlobalPoint gp(randRoll->toGlobal(lpRPC));
    const LocalPoint lpCSC(keyLayer->toLocal(gp));
    const float strip(keyLayerGeometry->strip(lpCSC));
    lut.push_back( (int) (strip-0.25)/0.5 );
  }
}

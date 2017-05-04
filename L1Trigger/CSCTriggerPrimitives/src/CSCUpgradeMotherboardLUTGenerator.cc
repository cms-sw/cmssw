#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUTGenerator.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream> 

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
  
  // CSC geometry
  int chid = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, 1, theTrigChamber);
  const CSCDetId me1bId(theEndcap, 1, 1, chid, 0);
  const CSCDetId me1aId(theEndcap, 1, 4, chid, 0);
  const CSCChamber* cscChamberME1b(csc_g->chamber(me1bId));
  const CSCChamber* cscChamberME1a(csc_g->chamber(me1aId));
  
  // check for GEM geometry
  if (not gemGeometryAvailable){
    LogTrace("CSCUpgradeMotherboardLUTGenerator")
      << "+++ generateLUTsME11() called for ME11 chamber without valid GEM geometry! +++ \n";
    return;
  }
  
  // CSC trigger geometry
  const CSCLayer* keyLayerME1b(cscChamberME1b->layer(3));
  const CSCLayer* keyLayerME1a(cscChamberME1a->layer(3));
  
  const int region((theEndcap == 1) ? 1: -1);
  const GEMDetId gem_id_l1(region, 1, 1, 1, me1bId.chamber(), 0);
  const GEMDetId gem_id_l2(region, 1, 1, 2, me1bId.chamber(), 0);
  const GEMChamber* gemChamber_l1(gem_g->chamber(gem_id_l1));
  const GEMChamber* gemChamber_l2(gem_g->chamber(gem_id_l2));
  
  // LUT<roll,<etaMin,etaMax> >    
  std::vector<std::pair<double,double> > gem_roll_eta_limits_l1 = gemRollToEtaLimitsLUT(gemChamber_l1);
  std::vector<std::pair<double,double> > gem_roll_eta_limits_l2 = gemRollToEtaLimitsLUT(gemChamber_l2);
  
  // LUT<WG,<etaMin,etaMax> >
  std::vector<std::pair<double,double> > cscWGToEtaLimits_ = cscWgToEtaLimitsLUT(keyLayerME1b);

  // LUT <WG,rollMin,rollMax>
  std::vector<std::pair<int,int> > cscWgToGemRoll_l1 = cscWgToRollLUT(cscWGToEtaLimits_, gem_roll_eta_limits_l1);
  std::vector<std::pair<int,int> > cscWgToGemRoll_l2 = cscWgToRollLUT(cscWGToEtaLimits_, gem_roll_eta_limits_l2);

  // pick any roll
  auto randRoll(gemChamber_l1->etaPartition(2));

  // map of HS to pad
  std::vector<std::pair<int,int> > cscHsToGemPadME1a_ = cscHsToGemPadLUT(keyLayerME1a, randRoll, 4, 93);
  std::vector<std::pair<int,int> > cscHsToGemPadME1b_ = cscHsToGemPadLUT(keyLayerME1b, randRoll, 5, 124);

  // map pad to HS
  std::vector<int> gemPadToCscHsME1a_ = gemPadToCscHsLUT(keyLayerME1a, randRoll);
  std::vector<int> gemPadToCscHsME1b_ = gemPadToCscHsLUT(keyLayerME1b, randRoll);

  // print LUTs
  std::stringstream os;
  os << "GEM L1 roll to eta limits" << std::endl;
  for(auto p : gem_roll_eta_limits_l1) {
    os << "{" << p.first << ", " << p.second << "}, " << std::endl;
  }
  os << "GEM L2 roll to eta limits" << std::endl;
  for(auto p : gem_roll_eta_limits_l2) {
    os << "{" << p.first << ", " << p.second << "}, " << std::endl;
  }
  

  os << "ME1b "<< me1bId <<std::endl;
  os << "WG roll to eta limits" << std::endl;
  for(auto p : cscWGToEtaLimits_) {
    os << "{" << p.first << ", " << p.second << "}, " << std::endl;
  }
  

  int i = 0;
  os << "WG to Roll L1" << std::endl;
  for(auto p : cscWgToGemRoll_l1) {
    os << "{" << p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) os << std::endl;
    i++;
  }
  os << "WG to Roll L2" << std::endl;
  for(auto p : cscWgToGemRoll_l2) {
    os << "{" << p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) os << std::endl;
    i++;
  }
  
  os << "CSC HS to GEM pad LUT in ME1a";
  i = 1;
  for(auto p : cscHsToGemPadME1a_) {
    os << "{" << p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) os << std::endl;
    i++;
  }

  os << "CSC HS to GEM pad LUT in ME1b";
  i = 1;
  for(auto p : cscHsToGemPadME1b_) {
    os << "{" << p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) os << std::endl;
    i++;
  }
  
  
  os << "GEM pad to CSC HS LUT in ME1a";
  i = 1;
  for(auto p : gemPadToCscHsME1a_) {
    os << p;
    if (i%8==0) os << std::endl;
    i++;
  }

  os << "GEM pad to CSC HS LUT in ME1b";
  i = 1;
  for(auto p : gemPadToCscHsME1b_) {
    os << p;
    if (i%8==0) os << std::endl;
    i++;
  }

  // print LUTs
  LogTrace("CSCUpgradeMotherboardLUTGenerator") << os;
  std::cout << os.str();
}

void CSCUpgradeMotherboardLUTGenerator::generateLUTsME21(unsigned theEndcap, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber) const
{
  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    LogTrace("CSCUpgradeMotherboardLUTGenerator") << "+++ generateLUTsME11() called for ME21 chamber! +++ \n";
    gemGeometryAvailable = true;
  }

  // retrieve CSCChamber geometry
  int chid = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, 2, theTrigChamber);
  const CSCDetId csc_id(theEndcap, 2, 1, chid, 0);
  const CSCChamber* cscChamber(csc_g->chamber(csc_id));
    
  // check for GEM geometry
  if (not gemGeometryAvailable){
    LogTrace("CSCUpgradeMotherboardLUTGenerator") << "+++ generateLUTsME11() called for ME21 chamber without valid GEM geometry! +++ \n";
    return;
  }
  
  // trigger geometry
  const CSCLayer* keyLayer(cscChamber->layer(3));  
  const int region((theEndcap == 1) ? 1: -1);
  const GEMDetId gem_id_l1(region, 1, 2, 1, csc_id.chamber(), 0);
  const GEMDetId gem_id_l2(region, 1, 2, 2, csc_id.chamber(), 0);
  const GEMChamber* gemChamber_l1(gem_g->chamber(gem_id_l1));
  const GEMChamber* gemChamber_l2(gem_g->chamber(gem_id_l2));
  
  // LUT: roll->(etaMin,etaMax)    
  std::vector<std::pair<double,double> > gem_roll_eta_limits_l1 = gemRollToEtaLimitsLUT(gemChamber_l1);
  std::vector<std::pair<double,double> > gem_roll_eta_limits_l2 = gemRollToEtaLimitsLUT(gemChamber_l2);

  // LUT: WG->(etaMin,etaMax)
  std::vector<std::pair<double, double> > cscWGToEtaLimits_ = cscWgToEtaLimitsLUT(keyLayer);
  
  // LUT: WG->(rollMin,rollMax)
  std::vector<std::pair<int,int> > cscWgToGemRoll_l1 = cscWgToRollLUT(cscWGToEtaLimits_, gem_roll_eta_limits_l1);
  std::vector<std::pair<int,int> > cscWgToGemRoll_l2 = cscWgToRollLUT(cscWGToEtaLimits_, gem_roll_eta_limits_l2);

  auto randRoll(gemChamber_l1->etaPartition(2));

  // LUT: HS->pad
  std::vector<std::pair<int,int> > cscHsToGemPad_ = cscHsToGemPadLUT(keyLayer, randRoll, 5, 155);

  // LUT: pad->HS
  std::vector<int> gemPadToCscHs_ = gemPadToCscHsLUT(keyLayer, randRoll);
  
 
  LogTrace("CSCUpgradeMotherboardLUTGenerator") << "ME21 "<< csc_id <<std::endl;

  std::stringstream os;
  os << "GEM roll to eta limits" << std::endl;
  for(auto p : gem_roll_eta_limits_l1) {
    os << "{"<< p.first << ", " << p.second << "}, " << std::endl;
  }


  os << "WG to eta limits" << std::endl;
  for(auto p : cscWGToEtaLimits_) {
    os << "{"<< p.first << ", " << p.second << "}, " << std::endl;
  }


  int i = 0;
  os << "WG to Roll L1" << std::endl;
  for(auto p : cscWgToGemRoll_l1) {
    os << "{" << p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) os << std::endl;
    i++;
  }
  os << "WG to Roll L2" << std::endl;
  for(auto p : cscWgToGemRoll_l2) {
    os << "{" << p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) os << std::endl;
    i++;
  }
  

  os << "CSC HS to GEM pad LUT in ME21";
  i = 1;
  for(auto p : cscHsToGemPad_) {
    os << "{"<< p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) os << std::endl;
    i++;
  }
  
  
  os << "GEM pad to CSC HS LUT in ME21";
  i = 1;
  for(auto p : gemPadToCscHs_) {
    os << p;
    if (i%8==0) os << std::endl;
    i++;
  }

  // print LUTs
  LogTrace("CSCUpgradeMotherboardLUTGenerator") << os;
  std::cout << os.str();
}

void CSCUpgradeMotherboardLUTGenerator::generateLUTsME3141(unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber) const
{
  bool rpcGeometryAvailable(false);
  if (rpc_g != nullptr) {
   LogTrace("CSCUpgradeMotherboardLUTGenerator")<< "+++ generateLUTsME3141() called for ME3141 chamber! +++ \n";
    rpcGeometryAvailable = true;
  }

  // retrieve CSCChamber geometry
  int chid = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, theStation, theTrigChamber);
  const CSCDetId csc_id(theEndcap, theStation, 1, chid, 0);
  const CSCChamber* cscChamber(csc_g->chamber(csc_id));

  // trigger geometry
  const CSCLayer* keyLayer(cscChamber->layer(3));
  const int region((theEndcap == 1) ? 1: -1);
  const int csc_trig_sect(CSCTriggerNumbering::triggerSectorFromLabels(csc_id));
  const int csc_trig_id( CSCTriggerNumbering::triggerCscIdFromLabels(csc_id));
  const int csc_trig_chid((3*(csc_trig_sect-1)+csc_trig_id)%18 +1);
  const int rpc_trig_sect((csc_trig_chid-1)/3+1);
  const int rpc_trig_subsect((csc_trig_chid-1)%3+1);
  const RPCDetId rpc_id(region,1,theStation,rpc_trig_sect,1,rpc_trig_subsect,0);
  const RPCChamber* rpcChamber(rpc_g->chamber(rpc_id));

  if (not rpcGeometryAvailable){
    LogTrace("CSCUpgradeMotherboardLUTGenerator") << "+++ generateLUTsME11() called for ME3141 chamber without valid RPC geometry! +++ \n";
    return;
  }

  // LUT<roll,<etaMin,etaMax> >    
  std::vector<std::pair<double,double> > rpcRollToEtaLimits_ = rpcRollToEtaLimitsLUT(rpcChamber);
  
  // LUT<WG,<etaMin,etaMax> >
  std::vector<std::pair<double, double> > cscWGToEtaLimits_ = cscWgToEtaLimitsLUT(keyLayer);

  // LUT <WG,rollMin,rollMax>
  std::vector<std::pair<int,int> > cscWgToRpcRoll_ = cscWgToRollLUT(cscWGToEtaLimits_, rpcRollToEtaLimits_);

  // pick any roll
  auto randRoll(rpcChamber->roll(2));

  // map of HS to pad
  std::vector<std::pair<int,int> > cscHsToRpcStrip_ = cscHsToRpcStripLUT(keyLayer,randRoll,5,155);

  std::vector<int> rpcStripToCscHs_ = rpcStripToCscHsLUT(keyLayer, randRoll);


  std::stringstream os;
  os << "RPC roll to eta limits" << std::endl;
  for(auto p : rpcRollToEtaLimits_) {
    os << "{"<< p.first << ", " << p.second << "}, " << std::endl;
  }

  os << "ME3141 "<< csc_id <<std::endl;
  os << "WG roll to eta limits" << std::endl;
  for(auto p : cscWGToEtaLimits_) {
    os << "{"<< p.first << ", " << p.second << "}, " << std::endl;
  }

  os << "WG to ROLL" << std::endl;
  int i = 1;
  for(auto p : cscWgToRpcRoll_) {
    os << "{"<< p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0)os << std::endl;
    i++;
  }

  
  
  os << "CSC HS to RPC strip LUT" << std::endl;
  i = 1;
  for(auto p : cscHsToRpcStrip_) {
    os << "{"<< p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0)os << std::endl;
    i++;
  }
  
  
  os << "RPC strip to CSC HS LUT" << std::endl;
  i = 1;
  for(auto p : rpcStripToCscHs_) {
   os << p << ", ";
    if (i%8==0)os << std::endl;
    i++;
  }

  // print LUTs
  LogTrace("CSCUpgradeMotherboardLUTGenerator") << os;
  std::cout << os.str();
}

int CSCUpgradeMotherboardLUTGenerator::assignRoll(const std::vector<std::pair<double,double> >& lut_, double eta) const
{
  int result = -99;
  for(auto p : lut_) {
    const float minEta(p.first);
    const float maxEta(p.second);
    if (minEta <= std::abs(eta) and std::abs(eta) < maxEta) {
      result = p.first;
      break;
    }
  }
  return result;
}

std::vector<std::pair<double,double> >
CSCUpgradeMotherboardLUTGenerator::gemRollToEtaLimitsLUT(const GEMChamber* gemChamber) const
{
  std::vector<std::pair<double,double> > lut;
  for(auto roll : gemChamber->etaPartitions()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    lut.push_back(std::make_pair(std::abs(gp_top.eta()), std::abs(gp_bottom.eta())));
  }
  return lut;
}

std::vector<std::pair<double,double> >
CSCUpgradeMotherboardLUTGenerator::rpcRollToEtaLimitsLUT(const RPCChamber* rpcChamber) const
{
  std::vector<std::pair<double,double> > lut;
  for(auto roll : rpcChamber->rolls()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    lut.push_back(std::make_pair(std::abs(gp_top.eta()), std::abs(gp_bottom.eta())));
  }
  return lut;
}

std::vector<std::pair<int,int> > 
CSCUpgradeMotherboardLUTGenerator::cscWgToRollLUT(const std::vector<std::pair<double,double> >& inLUT1,
						  const std::vector<std::pair<double,double> >& inLUT2) const
{
  std::vector<std::pair<int,int> > outLUT;
  for (const auto& p: inLUT1){
    double etaMin(p.first);
    double etaMax(p.second);
    outLUT.push_back(std::make_pair(assignRoll(inLUT2, etaMin), assignRoll(inLUT2, etaMax)));
  }
  return outLUT;
}

std::vector<std::pair<double, double> > 
CSCUpgradeMotherboardLUTGenerator::cscWgToEtaLimitsLUT(const CSCLayer* keyLayer) const
{
  std::vector<std::pair<double, double> > lut;
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  const int numberOfWG(keyLayerGeometry->numberOfWireGroups());
  for (int i = 0; i< numberOfWG; ++i){
    const float middle_wire(keyLayerGeometry->middleWireOfGroup(i));
    const std::pair<LocalPoint, LocalPoint> wire_ends(keyLayerGeometry->wireTopology()->wireEnds(middle_wire));
    const GlobalPoint gp_top(keyLayer->toGlobal(wire_ends.first));
    const GlobalPoint gp_bottom(keyLayer->toGlobal(wire_ends.first));
    lut.push_back(std::make_pair(gp_top.eta(), gp_bottom.eta()));
  }
  return lut;
}

std::vector<std::pair<int,int> >
CSCUpgradeMotherboardLUTGenerator::cscHsToGemPadLUT(const CSCLayer* keyLayer, 
						    const GEMEtaPartition* randRoll, 
						    int minH, int maxH) const
{
  std::vector<std::pair<int,int> > lut;
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  auto nStrips(keyLayerGeometry->numberOfStrips());
  for (float i = 0; i< nStrips; i = i+0.5){
    const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
    const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
    const LocalPoint lpGEM(randRoll->toLocal(gp));
    const int HS(i/0.5);
    const bool edge(HS < minH or HS > maxH);
    const float pad(edge ? -99 : randRoll->pad(lpGEM));
    lut.push_back(std::make_pair(std::floor(pad),std::ceil(pad)));
  }
  return lut;
}

std::vector<int>
CSCUpgradeMotherboardLUTGenerator::gemPadToCscHsLUT(const CSCLayer* keyLayer, 
						    const GEMEtaPartition* randRoll) const
{
  std::vector<int> lut;
  // pick any roll 
  const int nGEMPads(randRoll->npads());
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  for (int i = 1; i<= nGEMPads; ++i){
    const LocalPoint lpGEM(randRoll->centreOfPad(i));
    const GlobalPoint gp(randRoll->toGlobal(lpGEM));
    const LocalPoint lpCSC(keyLayer->toLocal(gp));
    const float strip(keyLayerGeometry->strip(lpCSC));
    lut.push_back( (int) (strip)/0.5 );
  }
  return lut;
}

std::vector<std::pair<int,int> >
CSCUpgradeMotherboardLUTGenerator::cscHsToRpcStripLUT(const CSCLayer* keyLayer, 
						      const RPCRoll* randRoll, 
						      int minH, int maxH) const
{
  std::vector<std::pair<int,int> > lut;
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  auto nStrips(keyLayerGeometry->numberOfStrips());
  for (float i = 0; i< nStrips; i = i+0.5){
    const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
    const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
    const LocalPoint lpRPC(randRoll->toLocal(gp));
    const int HS(i/0.5);
    const bool edge(HS < minH or HS > maxH);
    const float strip(edge ? -99 : randRoll->strip(lpRPC));
    lut.push_back(std::make_pair(std::floor(strip),std::ceil(strip)));
  }
  return lut;
}

std::vector<int>
CSCUpgradeMotherboardLUTGenerator::rpcStripToCscHsLUT(const CSCLayer* keyLayer, 
						      const RPCRoll* randRoll) const
{
  std::vector<int> lut;
  // pick any roll 
  const int nRPCStrips(randRoll->nstrips());
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  for (int i = 1; i<= nRPCStrips; ++i){
    const LocalPoint lpRPC(randRoll->centreOfStrip(i));
    const GlobalPoint gp(randRoll->toGlobal(lpRPC));
    const LocalPoint lpCSC(keyLayer->toLocal(gp));
    const float strip(keyLayerGeometry->strip(lpCSC));
    lut.push_back( (int) (strip-0.25)/0.5 );
  }
  return lut;
}

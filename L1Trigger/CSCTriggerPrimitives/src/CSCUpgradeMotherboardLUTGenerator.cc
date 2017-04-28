#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUTGenerator.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include <Geometry/RPCGeometry/interface/RPCRollSpecs.h>

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
    std::cout<< "+++ generateLUTsME11() called for ME11 chamber! +++ \n";
    gemGeometryAvailable = true;
  }
  
  // CSC geometry
  CSCTriggerGeomManager* geo_manager(CSCTriggerGeometry::get());
  std::cout << "theEndcap " << theEndcap << std::endl
	    << "theSector " << theSector<< std::endl
	    << "theSubsector " << theSubsector<< std::endl
	    << "theTrigChamber " << theTrigChamber << std::endl;
  
  const CSCChamber* cscChamberME1b(geo_manager->chamber(theEndcap, 1, theSector, theSubsector, theTrigChamber));
  const CSCDetId me1bId(cscChamberME1b->id());
  const CSCDetId me1aId(me1bId.endcap(), 1, 4, me1bId.chamber());
  const CSCChamber* cscChamberME1a(csc_g->chamber(me1aId));
  
  // check for GEM geometry
  if (not gemGeometryAvailable){
    std::cout << "+++ generateLUTsME11() called for ME11 chamber without valid GEM geometry! +++ \n";
    return;
  }
  
  // CSC trigger geometry
  const CSCLayer* keyLayerME1b(cscChamberME1b->layer(3));
  const CSCLayerGeometry* keyLayerGeometryME1b(keyLayerME1b->geometry());
  const CSCLayer* keyLayerME1a(cscChamberME1a->layer(3));
  const CSCLayerGeometry* keyLayerGeometryME1a(keyLayerME1a->geometry());
  
  // GEM trigger geometry
  const int region((theEndcap == 1) ? 1: -1);
  const GEMDetId gem_id(region, 1, 1, 1, me1bId.chamber(), 0);
  const GEMChamber* gemChamber(gem_g->chamber(gem_id));
  
  // LUT<roll,<etaMin,etaMax> >    
  std::map<int,std::pair<double,double> > gemRollToEtaLimits_;

  int n = 1;
  for(auto roll : gemChamber->etaPartitions()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    const float mini = std::min(std::abs(gp_top.eta()), std::abs(gp_bottom.eta()));
    const float maxi = std::max(std::abs(gp_top.eta()), std::abs(gp_bottom.eta()));
    gemRollToEtaLimits_[n] = std::make_pair(mini, maxi);
    ++n;
  }

  std::cout << "GEM roll to eta limits" << std::endl;
  for(auto p : gemRollToEtaLimits_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, " << std::endl;
  }
  
  // LUT<WG,<etaMin,etaMax> >
  std::map<int,std::pair<double,double> > cscWGToEtaLimits_;
 
  const int numberOfWG(keyLayerGeometryME1b->numberOfWireGroups());
  for (int i = 1; i<=numberOfWG; ++i){
    const float middle_wire(keyLayerGeometryME1b->middleWireOfGroup(i));
    const std::pair<LocalPoint, LocalPoint> wire_ends(keyLayerGeometryME1b->wireTopology()->wireEnds(middle_wire));
    const GlobalPoint gp_top(keyLayerME1b->toGlobal(wire_ends.first));
    const GlobalPoint gp_bottom(keyLayerME1b->toGlobal(wire_ends.second));
    const float mini = std::min(std::abs(gp_top.eta()), std::abs(gp_bottom.eta()));
    const float maxi = std::max(std::abs(gp_top.eta()), std::abs(gp_bottom.eta()));
    cscWGToEtaLimits_[i] = std::make_pair(mini, maxi);
  }

  std::cout << "ME1b "<< me1bId <<std::endl;
  std::cout << "WG roll to eta limits" << std::endl;
  for(auto p : cscWGToEtaLimits_) {
    std::cout << "{"<< p.first << ", " << (p.second).first << ", " << (p.second).second << "}, " << std::endl;
  }
  
  // LUT <WG,rollMin,rollMax>
  std::map<int,std::pair<int,int>> cscWgToGemRoll_;

  for (int i = 1; i<= numberOfWG; ++i){
    auto etaMin(cscWGToEtaLimits_[i].first);
    auto etaMax(cscWGToEtaLimits_[i].second);
    cscWgToGemRoll_[i] = std::make_pair(assignRoll(gemRollToEtaLimits_, etaMin), assignRoll(gemRollToEtaLimits_, etaMax));
  }
  std::cout << "WG to ROLL" << std::endl;
  int i = 1;
  for(auto p : cscWgToGemRoll_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }
  

  // map of HS to pad
  std::map<int,std::pair<int,int>> cscHsToGemPadME1a_;
  std::map<int,std::pair<int,int>> cscHsToGemPadME1b_;
  
  // pick any roll
  auto randRoll(gemChamber->etaPartition(2));
  
  // ME1a
  auto nStripsME1a(keyLayerGeometryME1a->numberOfStrips());
  for (float i = 0; i< nStripsME1a; i = i+0.5){
    const LocalPoint lpCSC(keyLayerGeometryME1a->topology()->localPosition(i));
    const GlobalPoint gp(keyLayerME1a->toGlobal(lpCSC));
    const LocalPoint lpGEM(randRoll->toLocal(gp));
    const int HS(i/0.5);
    const bool edge(HS < 3 or HS > 93); 
    const float pad(edge ? -99 :randRoll->pad(lpGEM));
    cscHsToGemPadME1a_[HS] = std::make_pair(std::floor(pad),std::ceil(pad));
  }
 
  // ME1b
  auto nStripsME1b(keyLayerGeometryME1b->numberOfStrips());
  for (float i = 0; i< nStripsME1b; i = i+0.5){
    const LocalPoint lpCSC(keyLayerGeometryME1b->topology()->localPosition(i));
    const GlobalPoint gp(keyLayerME1b->toGlobal(lpCSC));
    const LocalPoint lpGEM(randRoll->toLocal(gp));
    const int HS(i/0.5);
    const bool edge(HS < 5 or HS > 124);
    const float pad(edge ? -99 : randRoll->pad(lpGEM)); 
    cscHsToGemPadME1b_[HS] = std::make_pair(std::floor(pad),std::ceil(pad));
  }

  std::cout << "CSC HS to GEM pad LUT in ME1a" << std::endl;
  i = 1;
  for(auto p : cscHsToGemPadME1a_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, "; 
    if (i%8==0) std::cout << std::endl;
    i++;
  }
  std::cout << "CSC HS to GEM pad LUT in ME1b" << std::endl;
  i = 1;
  for(auto p : cscHsToGemPadME1b_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }
  
  // map pad to HS
  std::map<int,int> gemPadToCscHsME1a_;
  std::map<int,int> gemPadToCscHsME1b_;
  
  const int nGEMPads(randRoll->npads());
  for (int i = 1; i<= nGEMPads; ++i){
    const LocalPoint lpGEM(randRoll->centreOfPad(i));
    const GlobalPoint gp(randRoll->toGlobal(lpGEM));
    const LocalPoint lpCSCME1a(keyLayerME1a->toLocal(gp));
    const LocalPoint lpCSCME1b(keyLayerME1b->toLocal(gp));
    const float stripME1a(keyLayerGeometryME1a->strip(lpCSCME1a));
    const float stripME1b(keyLayerGeometryME1b->strip(lpCSCME1b));
    gemPadToCscHsME1a_[i] = int(2*stripME1a);
    gemPadToCscHsME1b_[i] = int(2*stripME1b);
  }

  std::cout << "GEM pad to CSC HS LUT in ME1a" << std::endl;
  i = 1;
  for(auto p : gemPadToCscHsME1a_) {
    std::cout << "{"<< p.first << ", " << p.second << "}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }
  std::cout << "GEM pad to CSC HS LUT in ME1b" << std::endl;
  i = 1;
  for(auto p : gemPadToCscHsME1b_) {
    std::cout << "{"<< p.first << ", " << p.second << "}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }
}

void CSCUpgradeMotherboardLUTGenerator::generateLUTsME21(unsigned theEndcap, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber) const
{
  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    std::cout<< "+++ generateLUTsME11() called for ME21 chamber! +++ \n";
    gemGeometryAvailable = true;
  }

  // retrieve CSCChamber geometry                                                                                                                                       
  CSCTriggerGeomManager* geo_manager(CSCTriggerGeometry::get());
  const CSCChamber* cscChamber(geo_manager->chamber(theEndcap, 2, theSector, theSubsector, theTrigChamber));
  const CSCDetId csc_id(cscChamber->id());
    
  // check for GEM geometry
  if (not gemGeometryAvailable){
    std::cout << "+++ generateLUTsME11() called for ME21 chamber without valid GEM geometry! +++ \n";
    return;
  }
  
  // trigger geometry
  const CSCLayer* keyLayer(cscChamber->layer(3));
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  
  const int region((theEndcap == 1) ? 1: -1);
  const GEMDetId gem_id(region, 1, 2, 1, csc_id.chamber(), 0);
  const GEMChamber* gemChamber(gem_g->chamber(gem_id));
  
  std::cout << "ME21 "<< csc_id <<std::endl;

  // LUT<roll,<etaMin,etaMax> >    
  std::map<int,std::pair<double,double> > gemRollToEtaLimits_;
  int n = 1;
  for(auto roll : gemChamber->etaPartitions()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    gemRollToEtaLimits_[n] = std::make_pair(gp_top.eta(), gp_bottom.eta());
    ++n;
  }

  std::cout << "GEM roll to eta limits" << std::endl;
  for(auto p : gemRollToEtaLimits_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, " << std::endl;
  }

  // LUT<WG,<etaMin,etaMax> >
  std::map<int,double> cscWGToEtaLimits_;
  const int numberOfWG(keyLayerGeometry->numberOfWireGroups());
  n = 1;
  for (int i = 0; i< numberOfWG; ++i){
    const LocalPoint wire_loc(keyLayerGeometry->localCenterOfWireGroup(i));
    const GlobalPoint gp(keyLayer->toGlobal(wire_loc));
    cscWGToEtaLimits_[n] = gp.eta();
    ++n;
  }

  std::cout << "WG to eta limits" << std::endl;
  for(auto p : cscWGToEtaLimits_) {
    std::cout << "{"<< p.first << ", " << p.second << "}, " << std::endl;
  }

  // LUT <WG,rollMin,rollMax>
  std::map<int,int> cscWgToGemRoll_;
  for (int i = 0; i< numberOfWG; ++i){
    auto eta(cscWGToEtaLimits_[i]);
    cscWgToGemRoll_[i] = assignRoll(gemRollToEtaLimits_, eta);
  }

  std::cout << "WG to roll" << std::endl;
  int i = 1;
  for(auto p : cscWgToGemRoll_) {
    std::cout << "{"<< p.first << ", " << p.second << "}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }
  
  // map of pad to HS
  std::map<int,int> gemPadToCscHs_;
  std::map<int,std::pair<int,int>> cscHsToGemPad_;

  auto randRoll(gemChamber->etaPartition(2));
  auto nStrips(keyLayerGeometry->numberOfStrips());
  for (float i = 0; i< nStrips; i = i+0.5){
    const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
    const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
    const LocalPoint lpGEM(randRoll->toLocal(gp));
    const int HS(i/0.5);
    //      const bool edge(HS < 5 or HS > 155);
    //const float pad(edge ? -99 : randRoll->pad(lpGEM));
    const float pad(randRoll->pad(lpGEM));
    // HS are wrapped-around
    cscHsToGemPad_[HS] = std::make_pair(std::floor(pad),std::ceil(pad));
  }

  std::cout << "CSC HS to GEM pad LUT in ME21" << std::endl;
  i = 1;
  for(auto p : cscHsToGemPad_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, "; 
    if (i%8==0) std::cout << std::endl;
    i++;
  }
  
  // pick any roll 
  const int nGEMPads(randRoll->npads());
  for (int i = 1; i<= nGEMPads; ++i){
    const LocalPoint lpGEM(randRoll->centreOfPad(i));
    const GlobalPoint gp(randRoll->toGlobal(lpGEM));
    const LocalPoint lpCSC(keyLayer->toLocal(gp));
    const float strip(keyLayerGeometry->strip(lpCSC));
    // HS are wrapped-around
    gemPadToCscHs_[i] = int(2*strip);
  }
  
  std::cout << "GEM pad to CSC HS LUT in ME21" << std::endl;
  i = 1;
  for(auto p : gemPadToCscHs_) {
    std::cout << "{"<< p.first << ", " << p.second << "}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }
}

void CSCUpgradeMotherboardLUTGenerator::generateLUTsME3141(unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubSector, unsigned theTrigChamber) const
{
  bool rpcGeometryAvailable(false);
  if (rpc_g != nullptr) {
    std::cout<< "+++ generateLUTsME3141() called for ME3141 chamber! +++ \n";
    rpcGeometryAvailable = true;
  }

  // retrieve CSCChamber geometry                                                                                                                                       
  CSCTriggerGeomManager* geo_manager(CSCTriggerGeometry::get());
  const CSCChamber* cscChamber(geo_manager->chamber(theEndcap, theStation, theSector, theSubSector, theTrigChamber));
  const CSCDetId csc_id(cscChamber->id());

  // trigger geometry
  const CSCLayer* keyLayer(cscChamber->layer(3));
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  const int region((theEndcap == 1) ? 1: -1);
  const int csc_trig_sect(CSCTriggerNumbering::triggerSectorFromLabels(csc_id));
  const int csc_trig_id( CSCTriggerNumbering::triggerCscIdFromLabels(csc_id));
  const int csc_trig_chid((3*(csc_trig_sect-1)+csc_trig_id)%18 +1);
  const int rpc_trig_sect((csc_trig_chid-1)/3+1);
  const int rpc_trig_subsect((csc_trig_chid-1)%3+1);
  const RPCDetId rpc_id(region,1,theStation,rpc_trig_sect,1,rpc_trig_subsect,0);
  const RPCChamber* rpcChamber(rpc_g->chamber(rpc_id));

  if (not rpcGeometryAvailable){
    std::cout << "+++ generateLUTsME11() called for ME3141 chamber without valid RPC geometry! +++ \n";
    return;
  }

  // LUT<roll,<etaMin,etaMax> >    
  std::map<int,std::pair<double,double> > rpcRollToEtaLimits_;

  for(int i = 1; i<= rpcChamber->nrolls(); ++i){
    auto roll(rpcChamber->roll(i));
    if (roll==nullptr) continue;
    
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    //result[i] = std::make_pair(floorf(gp_top.eta() * 100) / 100, ceilf(gp_bottom.eta() * 100) / 100);
    rpcRollToEtaLimits_[i] = std::make_pair(std::abs(gp_top.eta()), std::abs(gp_bottom.eta()));
  }
  
  std::cout << "RPC roll to eta limits" << std::endl;
  for(auto p : rpcRollToEtaLimits_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, " << std::endl;
  }

  // LUT<WG,<etaMin,etaMax> >
  std::map<int,std::pair<double,double> > cscWGToEtaLimits_;

  const int numberOfWG(keyLayerGeometry->numberOfWireGroups());
  for (int i = 0; i< numberOfWG; ++i){
    const float middle_wire(keyLayerGeometry->middleWireOfGroup(i));
    const std::pair<LocalPoint, LocalPoint> wire_ends(keyLayerGeometry->wireTopology()->wireEnds(middle_wire));

    const GlobalPoint gp_top(keyLayer->toGlobal(wire_ends.first));
    const GlobalPoint gp_bottom(keyLayer->toGlobal(wire_ends.first));
    cscWGToEtaLimits_[i] = std::make_pair(gp_top.eta(), gp_bottom.eta());
  }

  std::cout << "ME3141 "<< csc_id <<std::endl;
  std::cout << "WG roll to eta limits" << std::endl;
  for(auto p : cscWGToEtaLimits_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, " << std::endl;
  }

  // LUT <WG,rollMin,rollMax>
  std::map<int,std::pair<int,int>> cscWgToRpcRoll_;

  for (int i = 0; i< numberOfWG; ++i){
    auto etaMin(cscWGToEtaLimits_[i].first);
    auto etaMax(cscWGToEtaLimits_[i].second);
    cscWgToRpcRoll_[i] = std::make_pair(assignRoll(rpcRollToEtaLimits_, etaMin), assignRoll(rpcRollToEtaLimits_, etaMax));
  }
  std::cout << "WG to ROLL" << std::endl;
  int i = 1;
  for(auto p : cscWgToRpcRoll_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }

  // map of HS to pad
  std::map<int,std::pair<int,int>> cscHsToRpcStrip_;
  
  // pick any roll
  auto randRoll(rpcChamber->roll(2));

  auto nStrips(keyLayerGeometry->numberOfStrips());
  for (float i = 0; i< nStrips; i = i+0.5){
    const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
    const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
    const LocalPoint lpRPC(randRoll->toLocal(gp));
    const int HS(i/0.5);
    const bool edge(HS < 5 or HS > 155);
    const float strip(edge ? -99 : randRoll->strip(lpRPC));
    // HS are wrapped-around
    cscHsToRpcStrip_[HS] = std::make_pair(std::floor(strip),std::ceil(strip));
  }

  std::cout << "CSC HS to RPC strip LUT" << std::endl;
  i = 1;
  for(auto p : cscHsToRpcStrip_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, "; 
    if (i%8==0) std::cout << std::endl;
    i++;
  }

  std::map<int,int> rpcStripToCscHs_;
  
  const int nRPCStrips(randRoll->nstrips());
  for (int i = 0; i< nRPCStrips; ++i){
    const LocalPoint lpRPC(randRoll->centreOfStrip(i));
    const GlobalPoint gp(randRoll->toGlobal(lpRPC));
    const LocalPoint lpCSC(keyLayer->toLocal(gp));
    const float strip(keyLayerGeometry->strip(lpCSC));
    // HS are wrapped-around
    rpcStripToCscHs_[i] = int((strip - 0.25)/0.5);
  }
  
  std::cout << "RPC strip to CSC HS LUT" << std::endl;
  i = 1;
  for(auto p : rpcStripToCscHs_) {
    std::cout << "{"<< p.first << ", " << p.second << "}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }
}

int CSCUpgradeMotherboardLUTGenerator::assignRoll(const std::map<int,std::pair<double,double> >& lut_, double eta) const
{
  int result = -99;
  for(auto p : lut_) {
    const float minEta((p.second).first);
    const float maxEta((p.second).second);
    if (minEta <= std::abs(eta) and std::abs(eta) < maxEta) {
      result = p.first;
      break;
    }
  }
  return result;
}

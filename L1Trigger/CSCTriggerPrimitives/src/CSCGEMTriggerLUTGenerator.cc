#include "L1Trigger/CSCTriggerPrimitives/src/CSCGEMTriggerLUTGenerator.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

void CSCGEMTriggerLUTGenerator::generateLUTsME11(unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber)
{
  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    std::cout<< "+++ generateLUTsME11() called for ME11 chamber! +++ \n";
    gemGeometryAvailable = true;
  }
  
  // CSC geometry
  CSCTriggerGeomManager* geo_manager(CSCTriggerGeometry::get());
  const CSCChamber* cscChamberME1b(geo_manager->chamber(theEndcap, theStation, theSector, theSubsector, theTrigChamber));
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
  
  const int region((theEndcap == 1) ? 1: -1);
  const GEMDetId gem_id(region, 1, theStation, 1, me1bId.chamber(), 0);
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
    gemRollToEtaLimits_[n] = std::make_pair(gp_top.eta(), gp_bottom.eta());
    ++n;
  }

  std::cout << "GEM roll to eta limits" << std::endl;
  for(auto p : gemRollToEtaLimits_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, " << std::endl;
  }
  
  // LUT<WG,<etaMin,etaMax> >
  std::map<int,std::pair<double,double> > cscWGToEtaLimits_;
 
  const int numberOfWG(keyLayerGeometryME1b->numberOfWireGroups());
  n = 1;
  for (int i = 0; i< numberOfWG; ++i){
    const float middle_wire(keyLayerGeometryME1b->middleWireOfGroup(i));
    const std::pair<LocalPoint, LocalPoint> wire_ends(keyLayerGeometryME1b->wireTopology()->wireEnds(middle_wire));

    const GlobalPoint gp_top(keyLayerME1b->toGlobal(wire_ends.first));
    const GlobalPoint gp_bottom(keyLayerME1b->toGlobal(wire_ends.first));
    cscWGToEtaLimits_[n] = std::make_pair(gp_top.eta(), gp_bottom.eta());
    ++n;
  }

  std::cout << "ME1b "<< me1bId <<std::endl;
  std::cout << "WG roll to eta limits" << std::endl;
  for(auto p : cscWGToEtaLimits_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, " << std::endl;
  }
  
  // LUT <WG,rollMin,rollMax>
  std::map<int,std::pair<int,int>> cscWgToGemRoll_;

  for (int i = 0; i< numberOfWG; ++i){
    auto etaMin(cscWGToEtaLimits_[i].first);
    auto etaMax(cscWGToEtaLimits_[i].second);
    cscWgToGemRoll_[i] = std::make_pair(assignGEMRoll(gemRollToEtaLimits_, etaMin), assignGEMRoll(gemRollToEtaLimits_, etaMax));
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
    const bool edge(HS < 4 or HS > 93);
    const float pad(edge ? -99 : randRoll->pad(lpGEM));
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

  std::cout << "CSC HS to GEM pad LUT in ME1a";
  i = 1;
  for(auto p : cscHsToGemPadME1a_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, "; 
    if (i%8==0) std::cout << std::endl;
    i++;
  }
  std::cout << "CSC HS to GEM pad LUT in ME1b";
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
  for (int i = 0; i< nGEMPads; ++i){
    const LocalPoint lpGEM(randRoll->centreOfPad(i));
    const GlobalPoint gp(randRoll->toGlobal(lpGEM));
    const LocalPoint lpCSCME1a(keyLayerME1a->toLocal(gp));
    const LocalPoint lpCSCME1b(keyLayerME1b->toLocal(gp));
    const float stripME1a(keyLayerGeometryME1a->strip(lpCSCME1a));
    const float stripME1b(keyLayerGeometryME1b->strip(lpCSCME1b));
    gemPadToCscHsME1a_[i] = (int) (stripME1a)/0.5;
    gemPadToCscHsME1b_[i] = (int) (stripME1b)/0.5;
  }

  std::cout << "GEM pad to CSC HS LUT in ME1a";
  i = 1;
  for(auto p : gemPadToCscHsME1a_) {
    std::cout << "{"<< p.first << ", " << p.second << "}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }
  std::cout << "GEM pad to CSC HS LUT in ME1b";
  i = 1;
  for(auto p : gemPadToCscHsME1b_) {
    std::cout << "{"<< p.first << ", " << p.second << "}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }
}

void CSCGEMTriggerLUTGenerator::generateLUTsME21(unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber)
{
  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    std::cout<< "+++ generateLUTsME11() called for ME21 chamber! +++ \n";
    gemGeometryAvailable = true;
  }

  // retrieve CSCChamber geometry                                                                                                                                       
  CSCTriggerGeomManager* geo_manager(CSCTriggerGeometry::get());
  const CSCChamber* cscChamber(geo_manager->chamber(theEndcap, theStation, theSector, theSubsector, theTrigChamber));
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
    cscWgToGemRoll_[i] = assignGEMRoll(gemRollToEtaLimits_, eta);
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

  std::cout << "CSC HS to GEM pad LUT in ME21";
  i = 1;
  for(auto p : cscHsToGemPad_) {
    std::cout << "{"<< p.first << ", {" << (p.second).first << ", " << (p.second).second << "}}, "; 
    if (i%8==0) std::cout << std::endl;
    i++;
  }
  
  // pick any roll 
  const int nGEMPads(randRoll->npads());
  for (int i = 0; i< nGEMPads; ++i){
    const LocalPoint lpGEM(randRoll->centreOfPad(i));
    const GlobalPoint gp(randRoll->toGlobal(lpGEM));
    const LocalPoint lpCSC(keyLayer->toLocal(gp));
    const float strip(keyLayerGeometry->strip(lpCSC));
    // HS are wrapped-around
    gemPadToCscHs_[i] = (int) (strip)/0.5;
  }
  
  std::cout << "GEM pad to CSC HS LUT in ME21";
  i = 1;
  for(auto p : gemPadToCscHs_) {
    std::cout << "{"<< p.first << ", " << p.second << "}, ";
    if (i%8==0) std::cout << std::endl;
    i++;
  }
}


int CSCGEMTriggerLUTGenerator::assignGEMRoll(const std::map<int,std::pair<double,double> >& gemRollToEtaLimits_, double eta)
{
  int result = -99;
  for(auto p : gemRollToEtaLimits_) {
    const float minEta((p.second).first);
    const float maxEta((p.second).second);
    if (minEta <= eta and eta <= maxEta) {
      result = p.first;
      break;
    }
  }
  return result;
}

#include "L1Trigger/CSCTriggerPrimitives/src/CSCGEMTriggerLUTGenerator.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void CSCGEMTriggerLUTGenerator::generateLUTsME11(unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber)
{
  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    LogTrace("CSCGEMTriggerLUTGenerator")
      << "+++ generateLUTsME11() called for ME11 chamber! +++ \n";
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
    LogTrace("CSCGEMTriggerLUTGenerator")
      << "+++ generateLUTsME11() called for ME11 chamber without valid GEM geometry! +++ \n";
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
  std::vector<std::pair<double,double> > gemRollToEtaLimits_;

  for(auto roll : gemChamber->etaPartitions()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    gemRollToEtaLimits_.push_back(std::make_pair(gp_top.eta(), gp_bottom.eta()));
  }

  LogTrace("CSCGEMTriggerLUTGenerator") << "GEM roll to eta limits" << std::endl;
  for(auto p : gemRollToEtaLimits_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << "{" << p.first << ", " << p.second << "}, " << std::endl;
  }
  
  // LUT<WG,<etaMin,etaMax> >
  std::vector<std::pair<double,double> > cscWGToEtaLimits_;
 
  const int numberOfWG(keyLayerGeometryME1b->numberOfWireGroups());
  for (int i = 0; i< numberOfWG; ++i){
    const float middle_wire(keyLayerGeometryME1b->middleWireOfGroup(i));
    const std::pair<LocalPoint, LocalPoint> wire_ends(keyLayerGeometryME1b->wireTopology()->wireEnds(middle_wire));

    const GlobalPoint gp_top(keyLayerME1b->toGlobal(wire_ends.first));
    const GlobalPoint gp_bottom(keyLayerME1b->toGlobal(wire_ends.first));
    cscWGToEtaLimits_.push_back(std::make_pair(gp_top.eta(), gp_bottom.eta()));
  }

  LogTrace("CSCGEMTriggerLUTGenerator") << "ME1b "<< me1bId <<std::endl;
  LogTrace("CSCGEMTriggerLUTGenerator") << "WG roll to eta limits" << std::endl;
  for(auto p : cscWGToEtaLimits_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << "{" << p.first << ", " << p.second << "}, " << std::endl;
  }
  
  // LUT <WG,rollMin,rollMax>
  std::vector<std::pair<int,int> > cscWgToGemRoll_;

  for (int i = 0; i< numberOfWG; ++i){
    auto etaMin(cscWGToEtaLimits_[i].first);
    auto etaMax(cscWGToEtaLimits_[i].second);
    cscWgToGemRoll_.push_back(std::make_pair(assignGEMRoll(gemRollToEtaLimits_, etaMin), assignGEMRoll(gemRollToEtaLimits_, etaMax)));
  }
  int i = 0;
  LogTrace("CSCGEMTriggerLUTGenerator") << "WG to ROLL" << std::endl;
  for(auto p : cscWgToGemRoll_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << "{" << p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) LogTrace("CSCGEMTriggerLUTGenerator") << std::endl;
    i++;
  }
  

  // map of HS to pad
  std::vector<std::pair<int,int> > cscHsToGemPadME1a_;
  std::vector<std::pair<int,int> > cscHsToGemPadME1b_;
  
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
    cscHsToGemPadME1a_.push_back(std::make_pair(std::floor(pad),std::ceil(pad)));
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
    cscHsToGemPadME1b_.push_back(std::make_pair(std::floor(pad),std::ceil(pad)));
  }

  LogTrace("CSCGEMTriggerLUTGenerator") << "CSC HS to GEM pad LUT in ME1a";
  i = 1;
  for(auto p : cscHsToGemPadME1a_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << "{" << p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) LogTrace("CSCGEMTriggerLUTGenerator") << std::endl;
    i++;
  }
  LogTrace("CSCGEMTriggerLUTGenerator") << "CSC HS to GEM pad LUT in ME1b";
  i = 1;
  for(auto p : cscHsToGemPadME1b_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << "{" << p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) LogTrace("CSCGEMTriggerLUTGenerator") << std::endl;
    i++;
  }
  
  // map pad to HS
  std::vector<int> gemPadToCscHsME1a_;
  std::vector<int> gemPadToCscHsME1b_;
  
  const int nGEMPads(randRoll->npads());
  for (int i = 0; i< nGEMPads; ++i){
    const LocalPoint lpGEM(randRoll->centreOfPad(i));
    const GlobalPoint gp(randRoll->toGlobal(lpGEM));
    const LocalPoint lpCSCME1a(keyLayerME1a->toLocal(gp));
    const LocalPoint lpCSCME1b(keyLayerME1b->toLocal(gp));
    const float stripME1a(keyLayerGeometryME1a->strip(lpCSCME1a));
    const float stripME1b(keyLayerGeometryME1b->strip(lpCSCME1b));
    gemPadToCscHsME1a_.push_back( (int) (stripME1a)/0.5);
    gemPadToCscHsME1b_.push_back( (int) (stripME1b)/0.5);
  }

  LogTrace("CSCGEMTriggerLUTGenerator") << "GEM pad to CSC HS LUT in ME1a";
  i = 1;
  for(auto p : gemPadToCscHsME1a_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << p;
    if (i%8==0) LogTrace("CSCGEMTriggerLUTGenerator") << std::endl;
    i++;
  }
  LogTrace("CSCGEMTriggerLUTGenerator") << "GEM pad to CSC HS LUT in ME1b";
  i = 1;
  for(auto p : gemPadToCscHsME1b_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << p;
    if (i%8==0) LogTrace("CSCGEMTriggerLUTGenerator") << std::endl;
    i++;
  }
}

void CSCGEMTriggerLUTGenerator::generateLUTsME21(unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubsector, unsigned theTrigChamber)
{
  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    LogTrace("CSCGEMTriggerLUTGenerator") << "+++ generateLUTsME11() called for ME21 chamber! +++ \n";
    gemGeometryAvailable = true;
  }

  // retrieve CSCChamber geometry                                                                                                                                       
  CSCTriggerGeomManager* geo_manager(CSCTriggerGeometry::get());
  const CSCChamber* cscChamber(geo_manager->chamber(theEndcap, theStation, theSector, theSubsector, theTrigChamber));
  const CSCDetId csc_id(cscChamber->id());
    
  // check for GEM geometry
  if (not gemGeometryAvailable){
    LogTrace("CSCGEMTriggerLUTGenerator") << "+++ generateLUTsME11() called for ME21 chamber without valid GEM geometry! +++ \n";
    return;
  }
  
  // trigger geometry
  const CSCLayer* keyLayer(cscChamber->layer(3));
  const CSCLayerGeometry* keyLayerGeometry(keyLayer->geometry());
  
  const int region((theEndcap == 1) ? 1: -1);
  const GEMDetId gem_id(region, 1, 2, 1, csc_id.chamber(), 0);
  const GEMChamber* gemChamber(gem_g->chamber(gem_id));
  
  LogTrace("CSCGEMTriggerLUTGenerator") << "ME21 "<< csc_id <<std::endl;

  // LUT<roll,<etaMin,etaMax> >    
  std::vector<std::pair<double,double> > gemRollToEtaLimits_;
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

  LogTrace("CSCGEMTriggerLUTGenerator") << "GEM roll to eta limits" << std::endl;
  for(auto p : gemRollToEtaLimits_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << "{"<< p.first << ", " << p.second << "}, " << std::endl;
  }

  // LUT<WG,<etaMin,etaMax> >
  std::vector<double> cscWGToEtaLimits_;
  const int numberOfWG(keyLayerGeometry->numberOfWireGroups());
  n = 1;
  for (int i = 0; i< numberOfWG; ++i){
    const LocalPoint wire_loc(keyLayerGeometry->localCenterOfWireGroup(i));
    const GlobalPoint gp(keyLayer->toGlobal(wire_loc));
    cscWGToEtaLimits_.push_back( gp.eta() );
    ++n;
  }

  LogTrace("CSCGEMTriggerLUTGenerator") << "WG to eta limits" << std::endl;
  for(auto p : cscWGToEtaLimits_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << p << std::endl;
  }

  // LUT <WG,rollMin,rollMax>
  std::vector<int> cscWgToGemRoll_;
  for (int i = 0; i< numberOfWG; ++i){
    auto eta(cscWGToEtaLimits_[i]);
    cscWgToGemRoll_.push_back( assignGEMRoll(gemRollToEtaLimits_, eta) );
  }

  LogTrace("CSCGEMTriggerLUTGenerator") << "WG to roll" << std::endl;
  int i = 1;
  for(auto p : cscWgToGemRoll_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << p;
    if (i%8==0) LogTrace("CSCGEMTriggerLUTGenerator") << std::endl;
    i++;
  }
  
  // vector of pad to HS
  std::vector<int> gemPadToCscHs_;
  std::vector<std::pair<int,int>> cscHsToGemPad_;

  auto randRoll(gemChamber->etaPartition(2));
  auto nStrips(keyLayerGeometry->numberOfStrips());
  for (float i = 0; i< nStrips; i = i+0.5){
    const LocalPoint lpCSC(keyLayerGeometry->topology()->localPosition(i));
    const GlobalPoint gp(keyLayer->toGlobal(lpCSC));
    const LocalPoint lpGEM(randRoll->toLocal(gp));
    const int HS(i/0.5);
    const bool edge(HS < 5 or HS > 155);
    const float pad(edge ? -99 : randRoll->pad(lpGEM));
    // HS are wrapped-around
    cscHsToGemPad_.push_back( std::make_pair(std::floor(pad),std::ceil(pad)) );
  }

  LogTrace("CSCGEMTriggerLUTGenerator") << "CSC HS to GEM pad LUT in ME21";
  i = 1;
  for(auto p : cscHsToGemPad_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << "{"<< p.first << ", " << p.second << "}, " << std::endl;
    if (i%8==0) LogTrace("CSCGEMTriggerLUTGenerator") << std::endl;
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
    gemPadToCscHs_.push_back( (int) (strip)/0.5 );
  }
  
  LogTrace("CSCGEMTriggerLUTGenerator") << "GEM pad to CSC HS LUT in ME21";
  i = 1;
  for(auto p : gemPadToCscHs_) {
    LogTrace("CSCGEMTriggerLUTGenerator") << p;
    if (i%8==0) LogTrace("CSCGEMTriggerLUTGenerator") << std::endl;
    i++;
  }
}


int CSCGEMTriggerLUTGenerator::assignGEMRoll(const std::vector<std::pair<double,double> >& gemRollToEtaLimits_, double eta)
{
  int roll = 1;
  for(auto p : gemRollToEtaLimits_) {
    roll++;
    const float minEta(p.first);
    const float maxEta(p.second);
    if (minEta <= eta and eta <= maxEta) {
      return roll;
    }
  }
  return -99;
}

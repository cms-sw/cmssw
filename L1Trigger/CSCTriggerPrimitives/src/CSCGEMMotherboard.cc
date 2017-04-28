#include <L1Trigger/CSCTriggerPrimitives/src/CSCGEMMotherboard.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>

CSCGEMMotherboard::CSCGEMMotherboard(unsigned endcap, unsigned station,
                                     unsigned sector, unsigned subsector,
                                     unsigned chamber,
                                     const edm::ParameterSet& conf) :
  CSCIntegratedMotherboard(endcap, station, sector, subsector, chamber, conf)
{
  gemId = GEMDetId(theRegion, 1, theStation, 1, theChamber, 0).rawId();
  
  const edm::ParameterSet coPadParams(conf.getParameter<edm::ParameterSet>("copadParam"));
  coPadProcessor.reset( new GEMCoPadProcessor(endcap, station, 1, chamber, coPadParams) );

  // use "old" or "new" dataformat for integrated LCTs?
  useOldLCTDataFormat_ = tmbParams_.getParameter<bool>("useOldLCTDataFormat");

  // LCT ghostbusting
  doLCTGhostBustingWithGEMs_ = tmbParams_.getParameter<bool>("doLCTGhostBustingWithGEMs");

  // promote ALCT-GEM pattern
  promoteALCTGEMpattern_ = tmbParams_.getParameter<bool>("promoteALCTGEMpattern");
  promoteALCTGEMquality_ = tmbParams_.getParameter<bool>("promoteALCTGEMquality");

  //  deltas used to match to GEM pads
  maxDeltaBXPad_ = tmbParams_.getParameter<int>("maxDeltaBXPad");
  maxDeltaBXCoPad_ = tmbParams_.getParameter<int>("maxDeltaBXCoPad");

  maxDeltaPadL1_ = (isEven ? tmbParams_.getParameter<int>("maxDeltaPadL1Even") :
		    tmbParams_.getParameter<int>("maxDeltaPadL1Odd") );
  maxDeltaPadL2_ = (isEven ? tmbParams_.getParameter<int>("maxDeltaPadL2Even") :
		    tmbParams_.getParameter<int>("maxDeltaPadL2Odd") );
}

CSCGEMMotherboard::CSCGEMMotherboard() : CSCIntegratedMotherboard()
{
}

CSCGEMMotherboard::~CSCGEMMotherboard()
{
}

void CSCGEMMotherboard::clear()
{
  pads_.clear();
  coPads_.clear();    
}  


void CSCGEMMotherboard::run(const CSCWireDigiCollection* wiredc,
			    const CSCComparatorDigiCollection* compdc,
			    const GEMPadDigiClusterCollection* gemClusters)
{
  std::unique_ptr<GEMPadDigiCollection> gemPads(new GEMPadDigiCollection());
  coPadProcessor->declusterize(gemClusters, *gemPads);
  run(wiredc, compdc, gemPads.get());
}


void CSCGEMMotherboard::retrieveGEMPads(const GEMPadDigiCollection* gemPads, unsigned id)
{
  auto superChamber(gem_g->superChamber(id));
  for (auto ch : superChamber->chambers()) {
    for (auto roll : ch->etaPartitions()) {
      GEMDetId roll_id(roll->id());
      auto pads_in_det = gemPads->get(roll_id);
      for (auto pad = pads_in_det.first; pad != pads_in_det.second; ++pad) {
        auto id_pad = std::make_pair(roll_id, *pad);
        const int bx_shifted(lct_central_bx + pad->bx());
        for (int bx = bx_shifted - maxDeltaBXPad_;bx <= bx_shifted + maxDeltaBXPad_; ++bx) {
          pads_[bx].push_back(id_pad);  
        }
      }
    }
  }
}

void CSCGEMMotherboard::retrieveGEMCoPads()
{
  for (auto copad: gemCoPadV){
    if (copad.first().bx() != lct_central_bx) continue;
    coPads_[copad.bx(1)].push_back(std::make_pair(copad.roll(), copad));  
  }
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
							 const GEMCoPadDigi& gem,
 							 enum CSCPart part,
							 int trknmb) 
{
  auto mymap = (*getLUT()->get_gem_pad_to_csc_hs(isEven, part));

  if (useOldLCTDataFormat_){
    // CLCT pattern number - set it to a highest value
    // hack to get LCTs in the CSCTF
    unsigned int pattern = promoteALCTGEMpattern_ ? 10 : 0;
    
    // LCT quality number - set it to a very high value 
    // hack to get LCTs in the CSCTF
    unsigned int quality = promoteALCTGEMquality_ ? 15 : 11;
    
    // Bunch crossing
    int bx = alct.getBX();
    
    // get keyStrip from LUT
    int keyStrip = mymap[gem.pad(2)];

    // get wiregroup from ALCT
    int wg = alct.getKeyWG();

    // if (keyStrip>wgvshs[wg][0] && keyStrip<wgvshs[wg][1])
    // { // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(trknmb, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
    // }
    // return CSCCorrelatedLCTDigi(0,0,0,0,0,0,0,0,0,0,0,0);
   } 
  else {
    
    // CLCT pattern number - no pattern
    unsigned int pattern = 0;

    // LCT quality number
    unsigned int quality = 1;
    
    // Bunch crossing
    int bx = gem.bx(1) + lct_central_bx;
    
    // get keyStrip from LUT
    int keyStrip = mymap[gem.pad(2)];

    // get wiregroup from ALCT
    int wg = alct.getKeyWG();
    
    // if (keyStrip>wgvshs[wg][0] && keyStrip<wgvshs[wg][1])
    // { // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(trknmb, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
    // }
    // else return CSCCorrelatedLCTDigi(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
   }
}


CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCCLCTDigi& clct,
							 const GEMCoPadDigi& gem,
							 enum CSCPart part,
							 int trknmb) 
{
  if (useOldLCTDataFormat_){
    // CLCT pattern number - no pattern
    unsigned int pattern = encodePattern(clct.getPattern(), clct.getStripType());
    
    // LCT quality number -  dummy quality
    // const bool promoteCLCTGEMquality(ME == ME1A ? promoteCLCTGEMquality_ME1a_:promoteCLCTGEMquality_ME1b_);
    unsigned int quality = 15;//promoteCLCTGEMquality ? 14 : 11;
    
    // Bunch crossing: get it from cathode LCT if anode LCT is not there.
    int bx = gem.bx(1) + lct_central_bx;;
    
   // pick a random WG in the roll range    
    int wg = 5;
    
    // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(trknmb, 1, quality, wg, clct.getKeyStrip(), pattern, clct.getBend(), bx, 0, 0, 0, theTrigChamber);
  }
  else {
    // CLCT pattern number - no pattern
    unsigned int pattern = encodePattern(clct.getPattern(), clct.getStripType());
    
    // LCT quality number -  dummy quality
    unsigned int quality = 15;//findQualityGEM(alct, gem);
    
    // Bunch crossing: get it from cathode LCT if anode LCT is not there.
    int bx = gem.bx(1) + lct_central_bx;
    
    // ALCT WG
    int wg = 2;
    
    // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(trknmb, 1, quality, wg, 0, pattern, 0, bx, 0, 0, 0, theTrigChamber);
  }
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
							 const CSCCLCTDigi& clct,
							 const GEMCoPadDigi& gem,
							 enum CSCPart p,
							 int trknmb) 
{
  if (useOldLCTDataFormat_){
    // CLCT pattern number - set it to a highest value
    // hack to get LCTs in the CSCTF
    unsigned int pattern = encodePattern(clct.getPattern(), clct.getStripType());
    
    // LCT quality number 
    unsigned int quality = findQualityGEM<GEMCoPadDigi>(alct, clct);
    
    // Bunch crossing
    int bx = alct.getBX();
    
    // get keyStrip from LUT
    int keyStrip = clct.getKeyStrip();

    // get wiregroup from ALCT
    int wg = alct.getKeyWG();

    // if (keyStrip>wgvshs[wg][0] && keyStrip<wgvshs[wg][1])
    // { // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(trknmb, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
    // }
    // return CSCCorrelatedLCTDigi(0,0,0,0,0,0,0,0,0,0,0,0);
   } 
  else {
    
    // CLCT pattern number - no pattern
    unsigned int pattern = encodePattern(clct.getPattern(), clct.getStripType());

    // LCT quality number
    unsigned int quality = 1;
    
    // Bunch crossing
    int bx = gem.bx(1) + lct_central_bx;
    
    // get keyStrip from LUT
    int keyStrip = clct.getKeyStrip();

    // get wiregroup from ALCT
    int wg = alct.getKeyWG();
    
    // if (keyStrip>wgvshs[wg][0] && keyStrip<wgvshs[wg][1])
    // { // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(trknmb, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
    // }
    // else return CSCCorrelatedLCTDigi(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
   }
}


CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
							 const CSCCLCTDigi& clct,
							 const GEMPadDigi& gem,
							 enum CSCPart p,
							 int trknmb) 
{
  if (useOldLCTDataFormat_){
    // CLCT pattern number - set it to a highest value
    // hack to get LCTs in the CSCTF
    unsigned int pattern = encodePattern(clct.getPattern(), clct.getStripType());
    
    // LCT quality number
    unsigned int quality = findQualityGEM<GEMPadDigi>(alct, clct);
    
    // Bunch crossing
    int bx = alct.getBX();
    
    // get keyStrip from LUT
    int keyStrip = clct.getKeyStrip();

    // get wiregroup from ALCT
    int wg = alct.getKeyWG();

    // if (keyStrip>wgvshs[wg][0] && keyStrip<wgvshs[wg][1])
    // { // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(trknmb, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
    // }
    // return CSCCorrelatedLCTDigi(0,0,0,0,0,0,0,0,0,0,0,0);
   } 
  else {
    
    // CLCT pattern number - no pattern
    unsigned int pattern = encodePattern(clct.getPattern(), clct.getStripType());

    // LCT quality number
    unsigned int quality = 1;
    
    // Bunch crossing
    int bx = alct.getBX();
    
    // get keyStrip from LUT
    int keyStrip = clct.getKeyStrip();

    // get wiregroup from ALCT
    int wg = alct.getKeyWG();
    
    // if (keyStrip>wgvshs[wg][0] && keyStrip<wgvshs[wg][1])
    // { // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(trknmb, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
    // }
    // else return CSCCorrelatedLCTDigi(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
   }
}

bool CSCGEMMotherboard::isPadInOverlap(int roll)
{
  // this only works for ME1A!
  auto mymap = (*getLUT()->get_csc_wg_to_gem_roll(isEven));
  for (auto& p : mymap) {
    // overlap region are WGs 10-15
    if ((p.first < 10) or (p.first > 15)) continue;
    if (((p.second).first <= roll) and (roll <= (p.second).second)) return true;
  }
  return false;
}

int CSCGEMMotherboard::getBX(const GEMPadDigi& p)
{
  return p.bx();
}

int CSCGEMMotherboard::getBX(const GEMCoPadDigi& p)
{
  return p.bx(1);
}

int CSCGEMMotherboard::getRoll(const GEMPadDigiId& p)
{
  return GEMDetId(p.first).roll();
}

int CSCGEMMotherboard::getRoll(const GEMCoPadDigiId& p)
{
  return p.second.roll();
}

int CSCGEMMotherboard::getRoll(const CSCALCTDigi& alct)
{
  return (*getLUT()->get_csc_wg_to_gem_roll(isEven))[alct.getKeyWG()].first;
}

float CSCGEMMotherboard::getAvePad(const GEMPadDigi& p)
{
  return p.pad();
}

float CSCGEMMotherboard::getAvePad(const GEMCoPadDigi& p)
{
  return 0.5*(p.pad(1) + p.pad(2));
}

float CSCGEMMotherboard::getAvePad(const CSCCLCTDigi& clct, enum CSCPart part)
{
  auto mymap = (*getLUT()->get_csc_hs_to_gem_pad(isEven, part));
  return 0.5*(mymap[clct.getKeyStrip()].first + mymap[clct.getKeyStrip()].second);
}

void CSCGEMMotherboard::setupGeometry()
{
  CSCIntegratedMotherboard::setupGeometry();
  generator_->setGEMGeometry(gem_g);
}

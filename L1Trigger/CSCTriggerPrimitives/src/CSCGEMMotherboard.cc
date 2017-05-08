#include "L1Trigger/CSCTriggerPrimitives/src/CSCGEMMotherboard.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"

CSCGEMMotherboard::CSCGEMMotherboard(unsigned endcap, unsigned station,
                                     unsigned sector, unsigned subsector,
                                     unsigned chamber,
                                     const edm::ParameterSet& conf) :
  CSCUpgradeMotherboard(endcap, station, sector, subsector, chamber, conf)
  , maxDeltaBXPad_(tmbParams_.getParameter<int>("maxDeltaBXPad"))
  , maxDeltaBXCoPad_(tmbParams_.getParameter<int>("maxDeltaBXCoPad"))
  , useOldLCTDataFormat_(tmbParams_.getParameter<bool>("useOldLCTDataFormat"))
  , promoteALCTGEMpattern_(tmbParams_.getParameter<bool>("promoteALCTGEMpattern"))
  , promoteALCTGEMquality_(tmbParams_.getParameter<bool>("promoteALCTGEMquality"))
  , doLCTGhostBustingWithGEMs_(tmbParams_.getParameter<bool>("doLCTGhostBustingWithGEMs"))
{
  gemId = GEMDetId(theRegion, 1, theStation, 1, theChamber, 0).rawId();
  
  const edm::ParameterSet coPadParams(conf.getParameter<edm::ParameterSet>("copadParam"));
  coPadProcessor.reset( new GEMCoPadProcessor(endcap, station, 1, chamber, coPadParams) );

  maxDeltaPadL1_ = (par ? tmbParams_.getParameter<int>("maxDeltaPadL1Even") :
		    tmbParams_.getParameter<int>("maxDeltaPadL1Odd") );
  maxDeltaPadL2_ = (par ? tmbParams_.getParameter<int>("maxDeltaPadL2Even") :
		    tmbParams_.getParameter<int>("maxDeltaPadL2Odd") );
}

CSCGEMMotherboard::CSCGEMMotherboard() : CSCUpgradeMotherboard()
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
  for (const auto& ch : superChamber->chambers()) {
    for (const auto& roll : ch->etaPartitions()) {
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
  for (const auto& copad: gemCoPadV){
    if (copad.first().bx() != lct_central_bx) continue;
    coPads_[copad.bx(1)].push_back(std::make_pair(copad.roll(), copad));  
  }
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
							 const GEMCoPadDigi& gem,
 							 enum CSCPart part,
							 int trknmb) 
{
  return constructLCTsGEM(alct, CSCCLCTDigi(), GEMPadDigi(), gem, part, trknmb);
}


CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCCLCTDigi& clct,
							 const GEMCoPadDigi& gem,
							 enum CSCPart part,
							 int trknmb) 
{
  return constructLCTsGEM(CSCALCTDigi(), clct, GEMPadDigi(), gem, part, trknmb);
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
							 const CSCCLCTDigi& clct,
							 const GEMCoPadDigi& gem,
							 enum CSCPart part,
							 int trknmb) 
{
  return constructLCTsGEM(alct, clct, GEMPadDigi(), gem, part, trknmb);
}


CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
							 const CSCCLCTDigi& clct,
							 const GEMPadDigi& gem,
							 enum CSCPart part,
							 int trknmb) 
{
  return constructLCTsGEM(alct, clct, gem, GEMCoPadDigi(), part, trknmb);
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
							 const CSCCLCTDigi& clct,
							 const GEMPadDigi& gem1,
							 const GEMCoPadDigi& gem2,
							 enum CSCPart p, int trknmb)
{
  // step 1: determine the case
  int lctCase = 0;
  if (alct.isValid() and clct.isValid() and gem1 != GEMPadDigi() and gem2 == GEMCoPadDigi())      lctCase = 1;
  else if (alct.isValid() and clct.isValid() and gem1 == GEMPadDigi() and gem2 != GEMCoPadDigi()) lctCase = 2;
  else if (alct.isValid() and gem2 != GEMCoPadDigi()) lctCase = 3;
  else if (clct.isValid() and gem2 != GEMCoPadDigi()) lctCase = 4;

  // step 2: assign properties depending on the LCT dataformat (old/new)
  int pattern = 0, quality = 0, bx = 0, keyStrip = 0, keyWG = 0;
  switch(lctCase){
  case 1: {
    pattern = encodePattern(clct.getPattern(), clct.getStripType());
    quality = findQualityGEM<GEMPadDigi>(alct, clct);
    bx = alct.getBX();
    keyStrip = clct.getKeyStrip();
    keyWG = alct.getKeyWG();
    break;
  }
  case 2: {
    pattern = encodePattern(clct.getPattern(), clct.getStripType());
    quality = findQualityGEM<GEMCoPadDigi>(alct, clct);
    bx = alct.getBX();
    keyStrip = clct.getKeyStrip();
    keyWG = alct.getKeyWG();
    break;
  }
  case 3: {
    const auto& mymap1 = getLUT()->get_gem_pad_to_csc_hs(par, p);
    pattern = promoteALCTGEMpattern_ ? 10 : 0;
    quality = promoteALCTGEMquality_ ? 15 : 11;
    bx = alct.getBX();
    keyStrip = mymap1[gem2.pad(2)];
    keyWG = alct.getKeyWG();
    break;
  }
  case 4: {
    const auto& mymap2 = getLUT()->get_gem_roll_to_csc_wg(par, p);
    pattern = encodePattern(clct.getPattern(), clct.getStripType());
    quality = promoteCLCTGEMquality_ ? 15 : 11;
    bx = gem2.bx(1) + lct_central_bx;;
    keyStrip = clct.getKeyStrip();
    // choose the corresponding wire-group in the middle of the partition
    keyWG = mymap2[gem2.roll()];
    break;
  }
  };

  // future work: add a section that produces LCTs according
  // to the new LCT dataformat (not yet defined)

  // step 3: make new LCT with afore assigned properties}
  return CSCCorrelatedLCTDigi(trknmb, 1, quality, keyWG, keyStrip,
			      pattern, 0, bx, 0, 0, 0, theTrigChamber);
}


bool CSCGEMMotherboard::isPadInOverlap(int roll)
{
  // this only works for ME1A!
  const auto& mymap = (getLUT()->get_csc_wg_to_gem_roll(par));
  for (unsigned i=0; i<mymap.size(); i++) {
    // overlap region are WGs 10-15
    if ((i < 10) or (i > 15)) continue;
    if ((mymap[i].first <= roll) and (roll <= mymap[i].second)) return true;
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
  return (getLUT()->get_csc_wg_to_gem_roll(par))[alct.getKeyWG()].first;
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
  const auto& mymap = (getLUT()->get_csc_hs_to_gem_pad(par, part));
  return 0.5*(mymap[clct.getKeyStrip()].first + mymap[clct.getKeyStrip()].second);
}

void CSCGEMMotherboard::setupGeometry()
{
  CSCUpgradeMotherboard::setupGeometry();
  generator_->setGEMGeometry(gem_g);
}

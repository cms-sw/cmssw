#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboard.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"

CSCUpgradeMotherboard::LCTContainer::LCTContainer(unsigned int trig_window_size)
  : match_trig_window_size_(trig_window_size)
{
}

CSCCorrelatedLCTDigi&
CSCUpgradeMotherboard::LCTContainer::operator()(int bx, int match_bx, int lct)
{
  return data[bx][match_bx][lct];
}

void
CSCUpgradeMotherboard::LCTContainer::getTimeMatched(const int bx,
                                                    std::vector<CSCCorrelatedLCTDigi>& lcts) const
{
  for (unsigned int mbx = 0; mbx < match_trig_window_size_; mbx++) {
    for (int i=0;i<2;i++) {
      // consider only valid LCTs
      if (not data[bx][mbx][i].isValid()) continue;

      // remove duplicated LCTs
      if (std::find(lcts.begin(), lcts.end(), data[bx][mbx][i]) != lcts.end()) continue;

      lcts.push_back(data[bx][mbx][i]);
    }
  }
}

void
CSCUpgradeMotherboard::LCTContainer::getMatched(std::vector<CSCCorrelatedLCTDigi>& lcts) const
{
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++){
    std::vector<CSCCorrelatedLCTDigi> temp_lcts;
    CSCUpgradeMotherboard::LCTContainer::getTimeMatched(bx,temp_lcts);
    lcts.insert(std::end(lcts), std::begin(temp_lcts), std::end(temp_lcts));
  }
}

void
CSCUpgradeMotherboard::LCTContainer::clear()
{
  // Loop over all time windows
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    // Loop over all matched trigger windows
    for (unsigned int mbx = 0; mbx < match_trig_window_size_; mbx++) {
      // Loop over all stubs
      for (int i=0;i<CSCConstants::MAX_LCTS_PER_CSC;i++) {
        data[bx][mbx][i].clear();
      }
    }
  }
}

CSCUpgradeMotherboard::CSCUpgradeMotherboard(unsigned endcap, unsigned station,
                                             unsigned sector, unsigned subsector,
                                             unsigned chamber,
                                             const edm::ParameterSet& conf) :
  // special configuration parameters for ME11 treatment
  CSCMotherboard(endcap, station, sector, subsector, chamber, conf)
  , allLCTs(match_trig_window_size)
  // special configuration parameters for ME11 treatment
  , disableME1a(commonParams_.getParameter<bool>("disableME1a"))
  , gangedME1a(commonParams_.getParameter<bool>("gangedME1a"))
{
  if (!isSLHC_) edm::LogError("CSCUpgradeMotherboard|ConfigError")
    << "+++ Upgrade CSCUpgradeMotherboard constructed while isSLHC_ is not set! +++\n";

  theRegion = (theEndcap == 1) ? 1: -1;
  theChamber = CSCTriggerNumbering::chamberFromTriggerLabels(theSector,theSubsector,theStation,theTrigChamber);
  par = theChamber%2==0 ? Parity::Even : Parity::Odd;

  // generate the LUTs
  generator_.reset(new CSCUpgradeMotherboardLUTGenerator());

  match_earliest_alct_only = tmbParams_.getParameter<bool>("matchEarliestAlctOnly");
  match_earliest_clct_only = tmbParams_.getParameter<bool>("matchEarliestClctOnly");
  clct_to_alct = tmbParams_.getParameter<bool>("clctToAlct");
  drop_used_clcts = tmbParams_.getParameter<bool>("tmbDropUsedClcts");
  tmb_cross_bx_algo = tmbParams_.getParameter<unsigned int>("tmbCrossBxAlgorithm");
  max_lcts = tmbParams_.getParameter<unsigned int>("maxLCTs");
  debug_matching = tmbParams_.getParameter<bool>("debugMatching");
  debug_luts = tmbParams_.getParameter<bool>("debugLUTs");

  setPrefIndex();
}

CSCUpgradeMotherboard::CSCUpgradeMotherboard()
  : CSCMotherboard()
  , allLCTs(match_trig_window_size)
{
  if (!isSLHC_) edm::LogError("CSCUpgradeMotherboard|ConfigError")
    << "+++ Upgrade CSCUpgradeMotherboard constructed while isSLHC_ is not set! +++\n";

  setPrefIndex();
}

CSCUpgradeMotherboard::~CSCUpgradeMotherboard()
{
}

enum CSCPart CSCUpgradeMotherboard::getCSCPart(int keystrip) const
{
  if (theStation == 1 and (theRing ==1 or theRing == 4)){
    if (keystrip > CSCConstants::MAX_HALF_STRIP_ME1B){
      if ( !gangedME1a )
        return CSCPart::ME1Ag;
      else
        return CSCPart::ME1A;
    }else if (keystrip <= CSCConstants::MAX_HALF_STRIP_ME1B and keystrip >= 0)
      return CSCPart::ME1B;
    else
      return CSCPart::ME11;
  }else if (theStation == 2 and theRing == 1 )
    return CSCPart::ME21;
  else if  (theStation == 3 and theRing == 1 )
    return CSCPart::ME31;
  else if (theStation == 4 and theRing == 1 )
    return CSCPart::ME41;
  else{
    edm::LogError("CSCUpgradeMotherboard|Error") <<" ++ getCSCPart() failed to find the CSC chamber for in case ";
    return  CSCPart::ME11;// return ME11 by default
  }
}

void CSCUpgradeMotherboard::debugLUTs()
{
  if (debug_luts) generator_->generateLUTs(theEndcap, theStation, theSector, theSubsector, theTrigChamber);
}

bool CSCUpgradeMotherboard::sortLCTsByQuality(const CSCCorrelatedLCTDigi& lct1, const CSCCorrelatedLCTDigi& lct2)
{
  return lct1.getQuality() > lct2.getQuality();
}

bool CSCUpgradeMotherboard::sortLCTsByGEMDphi(const CSCCorrelatedLCTDigi& lct1, const CSCCorrelatedLCTDigi& lct2)
{
  return true;
}

void CSCUpgradeMotherboard::sortLCTs(std::vector<CSCCorrelatedLCTDigi>& lcts,
				     bool (*sorter)(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&)) const
{
  std::sort(lcts.begin(), lcts.end(), *sorter);
  if (lcts.size() > max_lcts) lcts.erase(lcts.begin()+max_lcts, lcts.end());
}


void CSCUpgradeMotherboard::setupGeometry()
{
  // check whether chamber is even or odd
  const int chid(CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, theStation, theTrigChamber));
  const CSCDetId csc_id(theEndcap, theStation, theStation, chid, 0);
  cscChamber = csc_g->chamber(csc_id);
  generator_->setCSCGeometry(csc_g);
}


void CSCUpgradeMotherboard::setPrefIndex()
{
  pref[0] = match_trig_window_size/2;
  for (unsigned int m=2; m<match_trig_window_size; m+=2)
  {
    pref[m-1] = pref[0] - m/2;
    pref[m]   = pref[0] + m/2;
  }
}


void CSCUpgradeMotherboard::clear()
{
  CSCMotherboard::clear();
  allLCTs.clear();
}

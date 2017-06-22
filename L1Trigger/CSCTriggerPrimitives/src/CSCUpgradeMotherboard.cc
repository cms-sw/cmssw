#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboard.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"

void CSCUpgradeMotherboard::LCTContainer::getTimeMatched(const int bx, std::vector<CSCCorrelatedLCTDigi>& lcts) const 
{
  for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
    for (int i=0;i<2;i++)
      if (data[bx][mbx][i].isValid())
        lcts.push_back(data[bx][mbx][i]);
}

void CSCUpgradeMotherboard::LCTContainer::getMatched(std::vector<CSCCorrelatedLCTDigi>& lcts) const 
{
  for (int bx = 0; bx < MAX_LCT_BINS; bx++){
    std::vector<CSCCorrelatedLCTDigi> temp_lcts;
    CSCUpgradeMotherboard::LCTContainer::getTimeMatched(bx,temp_lcts);
    lcts.insert(std::end(lcts), std::begin(temp_lcts), std::end(temp_lcts));
  }
}

CSCUpgradeMotherboard::CSCUpgradeMotherboard(unsigned endcap, unsigned station,
						   unsigned sector, unsigned subsector,
						   unsigned chamber,
						   const edm::ParameterSet& conf) :
  CSCMotherboard(endcap, station, sector, subsector, chamber, conf)
{
  if (!isSLHC) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ Upgrade CSCUpgradeMotherboard constructed while isSLHC is not set! +++\n";

  theRegion = (theEndcap == 1) ? 1: -1;
  theChamber = CSCTriggerNumbering::chamberFromTriggerLabels(theSector,theSubsector,theStation,theTrigChamber);
  par = theChamber%2==0 ? Parity::Even : Parity::Odd;

  commonParams_ = conf.getParameter<edm::ParameterSet>("commonParam");
  if (theStation==1) tmbParams_ = conf.getParameter<edm::ParameterSet>("me11tmbSLHCGEM");
  else if (theStation==2) tmbParams_ = conf.getParameter<edm::ParameterSet>("me21tmbSLHCGEM");
  else if (theStation==3 or theStation==4) tmbParams_ = conf.getParameter<edm::ParameterSet>("me3141tmbSLHCRPC");
  
  generator_.reset(new CSCUpgradeMotherboardLUTGenerator());

  match_earliest_alct_only = tmbParams_.getParameter<bool>("matchEarliestAlctOnly");
  match_earliest_clct_only = tmbParams_.getParameter<bool>("matchEarliestClctOnly");
  clct_to_alct = tmbParams_.getParameter<bool>("clctToAlct");
  drop_used_clcts = tmbParams_.getParameter<bool>("tmbDropUsedClcts");
  tmb_cross_bx_algo = tmbParams_.getParameter<unsigned int>("tmbCrossBxAlgorithm");
  max_lcts = tmbParams_.getParameter<unsigned int>("maxLCTs");
  debug_matching = tmbParams_.getParameter<bool>("debugMatching");
  debug_luts = tmbParams_.getParameter<bool>("debugLUTs");

  pref[0] = match_trig_window_size/2;
  for (unsigned int m=2; m<match_trig_window_size; m+=2)
  {
    pref[m-1] = pref[0] - m/2;
    pref[m]   = pref[0] + m/2;
  }
}

CSCUpgradeMotherboard::CSCUpgradeMotherboard() : CSCMotherboard()
{
  pref[0] = match_trig_window_size/2;
  for (unsigned int m=2; m<match_trig_window_size; m+=2)
  {
    pref[m-1] = pref[0] - m/2;
    pref[m]   = pref[0] + m/2;
  }
}

CSCUpgradeMotherboard::~CSCUpgradeMotherboard()
{
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

void CSCUpgradeMotherboard::sortLCTs(std::vector<CSCCorrelatedLCTDigi>& lcts, bool (*sorter)(const CSCCorrelatedLCTDigi&,const CSCCorrelatedLCTDigi&)){
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

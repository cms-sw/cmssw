#include <L1Trigger/ME0Trigger/src/ME0Motherboard.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

ME0Motherboard::ME0Motherboard(unsigned endcap, unsigned chamber,
                               const edm::ParameterSet& conf) :
  theEndcap(endcap), theChamber(chamber) 
{
  edm::ParameterSet tmbParams  =  conf.getParameter<edm::ParameterSet>("tmbParam");
  infoV = tmbParams.getParameter<int>("verbosity");
}

ME0Motherboard::ME0Motherboard() :
  theEndcap(1), theChamber(1) 
{
  infoV = 2;
}

ME0Motherboard::~ME0Motherboard() {
}

void ME0Motherboard::clear() 
{
  for (int bx = 0; bx < MAX_LCT_BINS; bx++) {
    for (int i = 0; i < MAX_LCTS; i++) {
      LCTs[bx][i].clear();
    }
  }
}

void
ME0Motherboard::run(const ME0PadDigiCollection*) 
{
  clear();
}

// Returns vector of read-out correlated LCTs, if any.  Starts with
// the vector of all found LCTs and selects the ones in the read-out
// time window.
std::vector<ME0TriggerDigi> ME0Motherboard::readoutLCTs() 
{
  std::vector<ME0TriggerDigi> tmpV;
  
  std::vector<ME0TriggerDigi> all_lcts = getLCTs();
  for (auto plct = all_lcts.begin(); plct != all_lcts.end(); plct++) {
    tmpV.push_back(*plct);
  }
  return tmpV;
}

// Returns vector of all found correlated LCTs, if any.
std::vector<ME0TriggerDigi> ME0Motherboard::getLCTs() 
{
  std::vector<ME0TriggerDigi> tmpV;
  
  // Do not report LCTs found in ME1/A if mpc_block_me1/a is set.
  for (int bx = 0; bx < MAX_LCT_BINS; bx++) {
    for (int i = 0; i < MAX_LCTS; i++) {
      tmpV.push_back(LCTs[bx][i]);
    }
  }
  return tmpV;
}

// compare LCTs by quality
bool ME0Motherboard::sortByQuality(const ME0TriggerDigi& lct1, const ME0TriggerDigi& lct2) 
{ 
  return lct1.getQuality() > lct2.getQuality();
}

// compare LCTs by GEM bending angle
bool ME0Motherboard::sortByME0Dphi(const ME0TriggerDigi& lct1, const ME0TriggerDigi& lct2) 
{ 
  return true;
}

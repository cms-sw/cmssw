#ifndef _CSCStubMatcher_h_
#define _CSCStubMatcher_h_

/**\class CSCStubMatcher

 Description: Matching of CSC L1 trigger stubs to SimTrack

 Original Author:  "Vadim Khotilovich"
 $Id: $
*/

#include "CSCDigiMatcher.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"


#include <vector>
#include <map>
#include <set>

class SimHitMatcher;
class CSCDigiMatcher;

class CSCStubMatcher : public DigiMatcher
{
public:

  CSCStubMatcher(SimHitMatcher& sh, CSCDigiMatcher& dg);
  
  ~CSCStubMatcher();

  // chamber detIds with matching stubs
  std::set<unsigned int> chamberIdsCLCT() const;
  std::set<unsigned int> chamberIdsALCT() const;
  std::set<unsigned int> chamberIdsLCT() const;

  // single matched stubs from a particular chamber
  Digi clctInChamber(unsigned int) const;
  Digi alctInChamber(unsigned int) const;
  Digi lctInChamber(unsigned int) const;


  // crossed chamber detIds with not necessarily matching stubs
  std::set<unsigned int> chamberIdsAllCLCT() const;
  std::set<unsigned int> chamberIdsAllALCT() const;
  std::set<unsigned int> chamberIdsAllLCT() const;

  // all stubs (not necessarily matching) from a particular crossed chamber
  const DigiContainer& allCLCTsInChamber(unsigned int) const;
  const DigiContainer& allALCTsInChamber(unsigned int) const;
  const DigiContainer& allLCTsInChamber(unsigned int) const;


  // How many CSC chambers with matching stubs of some minimal quality did this SimTrack hit?
  int nChambersWithCLCT(int min_quality = 0) const;
  int nChambersWithALCT(int min_quality = 0) const;
  int nChambersWithLCT(int min_quality = 0) const;

private:

  void init();

  void matchCLCTsToSimTrack(const CSCCLCTDigiCollection& clcts);
  void matchALCTsToSimTrack(const CSCALCTDigiCollection& alcts);
  void matchLCTsToSimTrack(const CSCCorrelatedLCTDigiCollection& lcts);

  const CSCDigiMatcher* digi_matcher_;

  edm::InputTag clctInput_;
  edm::InputTag alctInput_;
  edm::InputTag lctInput_;

  int minBXCLCT_, maxBXCLCT_;
  int minBXALCT_, maxBXALCT_;
  int minBXLCT_, maxBXLCT_;

  // matched stubs in crossed chambers
  std::map<unsigned int, Digi> chamber_to_clct_;
  std::map<unsigned int, Digi> chamber_to_alct_;
  std::map<unsigned int, Digi> chamber_to_lct_;

  // all stubs (not necessarily matching) in crossed chambers with digis
  std::map<unsigned int, DigiContainer> chamber_to_clcts_;
  std::map<unsigned int, DigiContainer> chamber_to_alcts_;
  std::map<unsigned int, DigiContainer> chamber_to_lcts_;
};

#endif

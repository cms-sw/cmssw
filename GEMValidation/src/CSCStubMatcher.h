#ifndef GEMValidation_CSCStubMatcher_h
#define GEMValidation_CSCStubMatcher_h

/**\class CSCStubMatcher

 Description: Matching of CSC L1 trigger stubs to SimTrack

 Original Author:  "Vadim Khotilovich"
 $Id: $
*/

#include "GEMCode/GEMValidation/src/CSCDigiMatcher.h"

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

  /// chamber detIds with matching stubs
  /// by default, only returns those from ME1b; use al chambers if csc_type=0
  std::set<unsigned int> chamberIdsCLCT(int csc_type = CSC_ME1b) const;
  std::set<unsigned int> chamberIdsALCT(int csc_type = CSC_ME1b) const;
  std::set<unsigned int> chamberIdsLCT(int csc_type = CSC_ME1b) const;
  std::set<unsigned int> chamberIdsMPLCT(int csc_type = CSC_ME1b) const;

  /// single matched stubs from a particular chamber
  Digi clctInChamber(unsigned int) const;
  Digi alctInChamber(unsigned int) const;
  Digi lctInChamber(unsigned int) const;
  Digi mplctInChamber(unsigned int) const;

  /// crossed chamber detIds with not necessarily matching stubs
  std::set<unsigned int> chamberIdsAllCLCT(int csc_type = CSC_ME1b) const;
  std::set<unsigned int> chamberIdsAllALCT(int csc_type = CSC_ME1b) const;
  std::set<unsigned int> chamberIdsAllLCT(int csc_type = CSC_ME1b) const;
  std::set<unsigned int> chamberIdsAllMPLCT(int csc_type = CSC_ME1b) const;

  /// all stubs (not necessarily matching) from a particular crossed chamber
  const DigiContainer& allCLCTsInChamber(unsigned int) const;
  const DigiContainer& allALCTsInChamber(unsigned int) const;
  const DigiContainer& allLCTsInChamber(unsigned int) const;
  const DigiContainer& allMPLCTsInChamber(unsigned int) const;

  /// How many CSC chambers with matching stubs of some minimal quality did this SimTrack hit?
  int nChambersWithCLCT(int min_quality = 0) const;
  int nChambersWithALCT(int min_quality = 0) const;
  int nChambersWithLCT(int min_quality = 0) const;
  int nChambersWithMPLCT(int min_quality = 0) const;

private:

  void init();

  void matchCLCTsToSimTrack(const CSCCLCTDigiCollection& clcts);
  void matchALCTsToSimTrack(const CSCALCTDigiCollection& alcts);
  void matchLCTsToSimTrack(const CSCCorrelatedLCTDigiCollection& lcts);
  void matchMPLCTsToSimTrack(const CSCCorrelatedLCTDigiCollection& mplcts);

  const CSCDigiMatcher* digi_matcher_;

  edm::InputTag clctInput_;
  edm::InputTag alctInput_;
  edm::InputTag lctInput_;
  edm::InputTag mplctInput_;

  int minBXCLCT_, maxBXCLCT_;
  int minBXALCT_, maxBXALCT_;
  int minBXLCT_, maxBXLCT_;
  int minBXMPLCT_, maxBXMPLCT_;

  // matched stubs in crossed chambers
  typedef std::map<unsigned int, Digi> Id2Digi;
  Id2Digi chamber_to_clct_;
  Id2Digi chamber_to_alct_;
  Id2Digi chamber_to_lct_;
  Id2Digi chamber_to_mplct_;

  // all stubs (not necessarily matching) in crossed chambers with digis
  typedef std::map<unsigned int, DigiContainer> Id2DigiContainer;
  Id2DigiContainer chamber_to_clcts_;
  Id2DigiContainer chamber_to_alcts_;
  Id2DigiContainer chamber_to_lcts_;
  Id2DigiContainer chamber_to_mplcts_;

  template<class D>
  std::set<unsigned int> selectDetIds(D &digis, int csc_type) const;

  bool addGhostLCTs_;
  bool addGhostMPLCTs_;

  int minNHitsChamber_;
  int minNHitsChamberALCT_;
  int minNHitsChamberCLCT_;
  int minNHitsChamberLCT_;
  int minNHitsChamberMPLCT_;
  
  bool verboseALCT_;
  bool verboseCLCT_;
  bool verboseLCT_;
  bool verboseMPLCT_;
};


template<class D>
std::set<unsigned int>
CSCStubMatcher::selectDetIds(D &digis, int csc_type) const
{
  std::set<unsigned int> result;
  for (auto& p: digis)
  {
    auto id = p.first;
    if (csc_type > 0)
    {
      CSCDetId detId(id);
      if (detId.iChamberType() != csc_type) continue;
    }
    result.insert(p.first);
  }
  return result;
}

#endif

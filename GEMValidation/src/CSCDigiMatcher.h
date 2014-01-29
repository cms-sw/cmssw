#ifndef GEMValidation_CSCDigiMatcher_h
#define GEMValidation_CSCDigiMatcher_h

/**\class CSCDigiMatcher

 Description: Matching of Digis to SimTrack in CSC

 Original Author:  "Vadim Khotilovich"
*/

#include "GEMCode/GEMValidation/src/DigiMatcher.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"

#include <vector>
#include <map>
#include <set>
#include <tuple>

class SimHitMatcher;


class CSCDigiMatcher : public DigiMatcher
{
public:

  CSCDigiMatcher(SimHitMatcher& sh);
  
  ~CSCDigiMatcher();

  /// layer detIds with digis
  /// by default, only returns those from ME1b; use al chambers if csc_type=0
  std::set<unsigned int> detIdsStrip(int csc_type = CSC_ME1b) const;
  std::set<unsigned int> detIdsWire(int csc_type = CSC_ME1b) const;

  /// chamber detIds with digis
  std::set<unsigned int> chamberIdsStrip(int csc_type = CSC_ME1b) const;
  std::set<unsigned int> chamberIdsWire(int csc_type = CSC_ME1b) const;

  /// CSC strip digis from a particular layer or chamber
  const DigiContainer& stripDigisInDetId(unsigned int) const;
  const DigiContainer& stripDigisInChamber(unsigned int) const;

  /// CSC wire digis from a particular layer or chamber
  const DigiContainer& wireDigisInDetId(unsigned int) const;
  const DigiContainer& wireDigisInChamber(unsigned int) const;

  // #layers with hits
  int nLayersWithStripInChamber(unsigned int) const;
  int nLayersWithWireInChamber(unsigned int) const;

  /// How many CSC chambers with minimum number of layer with digis did this simtrack get?
  int nCoincidenceStripChambers(int min_n_layers = 4) const;
  int nCoincidenceWireChambers(int min_n_layers = 4) const;

  std::set<int> stripsInDetId(unsigned int) const;
  std::set<int> wiregroupsInDetId(unsigned int) const;

  // A non-zero max_gap_to_fill parameter would insert extra half-strips or wiregroups
  // so that gaps of that size or smaller would be filled.
  // E.g., if max_gap_to_fill = 1, and there are digis with strips 4 and 6 in a chamber,
  // the resulting set of digis would be 4,5,6
  std::set<int> stripsInChamber(unsigned int, int max_gap_to_fill = 0) const;
  std::set<int> wiregroupsInChamber(unsigned int, int max_gap_to_fill = 0) const;

private:

  void init();

  void matchTriggerDigisToSimTrack(const CSCComparatorDigiCollection& comparators, const CSCWireDigiCollection& wires);

  edm::InputTag cscComparatorDigiInput_;
  edm::InputTag cscWireDigiInput_;

  int minBXCSCComp_, maxBXCSCComp_;
  int minBXCSCWire_, maxBXCSCWire_;

  int matchDeltaStrip_;
  int matchDeltaWG_;

  typedef std::map<unsigned int, DigiContainer> Id2DigiContainer;

  Id2DigiContainer detid_to_halfstrips_;
  Id2DigiContainer chamber_to_halfstrips_;

  Id2DigiContainer detid_to_wires_;
  Id2DigiContainer chamber_to_wires_;

  std::set<unsigned int> selectDetIds(const Id2DigiContainer &digis, int csc_type) const;
  
  int verboseStrip_;
  int verboseWG_;
};

#endif

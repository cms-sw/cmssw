#ifndef _CSCDigiMatcher_h_
#define _CSCDigiMatcher_h_

/**\class CSCDigiMatcher

 Description: Matching of Digis to SimTrack in CSC

 Original Author:  "Vadim Khotilovich"
 $Id$
*/

#include "DigiMatcher.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"

#include <vector>
#include <map>
#include <set>
#include <tuple>

class SimHitMatcher;
class CSCGeometry;
class GEMGeometry;

class CSCDigiMatcher : public DigiMatcher
{
public:

  CSCDigiMatcher(SimHitMatcher& sh);
  
  ~CSCDigiMatcher();

  // layer detIds with digis
  std::set<unsigned int> detIdsStrip();
  std::set<unsigned int> detIdsWire();

  // chamber detIds with digis
  std::set<unsigned int> chamberIdsStrip();
  std::set<unsigned int> chamberIdsWire();

  // CSC strip digis from a particular layer or chamber
  DigiContainer stripDigisInDetId(unsigned int);
  DigiContainer stripDigisInChamber(unsigned int);

  // CSC wire digis from a particular layer or chamber
  DigiContainer wireDigisInDetId(unsigned int);
  DigiContainer wireDigisInChamber(unsigned int);

  // #layers with hits
  int nLayersWithStripInChamber(unsigned int);
  int nLayersWithWireInChamber(unsigned int);

  /// How many CSC chambers with minimum number of layer with digis did this simtrack get?
  int nCoincidenceStripChambers(int min_n_layers = 4);
  int nCoincidenceWireChambers(int min_n_layers = 4);

  std::set<int> stripsInDetId(unsigned int);
  std::set<int> wiregroupsInDetId(unsigned int);

private:

  void init();

  void matchTriggerDigisToSimTrack(const CSCComparatorDigiCollection& comparators, const CSCWireDigiCollection& wires);

  edm::InputTag cscComparatorDigiInput_;
  edm::InputTag cscWireDigiInput_;

  int minBXCSCComp_, maxBXCSCComp_;
  int minBXCSCWire_, maxBXCSCWire_;

  int matchDeltaStrip_;

  std::map<unsigned int, DigiContainer> detid_to_halfstrips_;
  std::map<unsigned int, DigiContainer> chamber_to_halfstrips_;

  std::map<unsigned int, DigiContainer> detid_to_wires_;
  std::map<unsigned int, DigiContainer> chamber_to_wires_;
};

#endif

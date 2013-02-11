#ifndef _GEMDigiMatcher_h_
#define _GEMDigiMatcher_h_

/**\class DigiMatcher

 Description: Matching of Digis for SimTrack in GEM

 Original Author:  "Vadim Khotilovich"
 $Id$
*/

#include "DigiMatcher.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"

#include <vector>
#include <map>
#include <set>

class SimHitMatcher;

class GEMDigiMatcher : public DigiMatcher
{
public:

  GEMDigiMatcher(SimHitMatcher& sh);
  
  ~GEMDigiMatcher();

  // partition GEM detIds with digis
  std::set<unsigned int> detIds();

  // chamber detIds with digis
  std::set<unsigned int> chamberIds();

  // superchamber detIds with digis
  std::set<unsigned int> superChamberIds();

  // partition detIds with coincidence pads
  std::set<unsigned int> detIdsWithCoPads();

  // superchamber detIds with coincidence pads
  std::set<unsigned int> superChamberIdsWithCoPads();


  // GEM digis from a particular partition, chamber or superchamber
  DigiContainer digisInDetId(unsigned int);
  DigiContainer digisInChamber(unsigned int);
  DigiContainer digisInSuperChamber(unsigned int);

  // GEM pads from a particular partition, chamber or superchamber
  DigiContainer padsInDetId(unsigned int);
  DigiContainer padsInChamber(unsigned int);
  DigiContainer padsInSuperChamber(unsigned int);

  // GEM co-pads from a particular partition or superchamber
  DigiContainer coPadsInDetId(unsigned int);
  DigiContainer coPadsInSuperChamber(unsigned int);

  // #layers with digis from this simtrack
  int nLayersWithDigisInSuperChamber(unsigned int);

  /// How many pads in GEM did this simtrack get in total?
  int nPads();

  /// How many coincidence pads in GEM did this simtrack get in total?
  int nCoPads();

  std::set<int> stripNumbersInDetId(unsigned int);
  std::set<int> padNumbersInDetId(unsigned int);
  std::set<int> coPadNumbersInDetId(unsigned int);

  // what unique partitions numbers with digis from this simtrack?
  std::set<int> partitionNumbers();
  std::set<int> partitionNumbersWithCoPads();

private:

  void init();

  void matchDigisToSimTrack(const GEMDigiCollection& digis);
  void matchPadsToSimTrack(const GEMCSCPadDigiCollection& pads);
  void matchCoPadsToSimTrack(const GEMCSCPadDigiCollection& co_pads);

  edm::InputTag gemDigiInput_;
  edm::InputTag gemPadDigiInput_;
  edm::InputTag gemCoPadDigiInput_;

  int minBXGEM_, maxBXGEM_;

  int matchDeltaStrip_;

  std::map<unsigned int, DigiContainer> detid_to_digis_;
  std::map<unsigned int, DigiContainer> chamber_to_digis_;
  std::map<unsigned int, DigiContainer> superchamber_to_digis_;

  std::map<unsigned int, DigiContainer> detid_to_pads_;
  std::map<unsigned int, DigiContainer> chamber_to_pads_;
  std::map<unsigned int, DigiContainer> superchamber_to_pads_;

  std::map<unsigned int, DigiContainer> detid_to_copads_;
  std::map<unsigned int, DigiContainer> chamber_to_copads_;
  std::map<unsigned int, DigiContainer> superchamber_to_copads_;
};

#endif

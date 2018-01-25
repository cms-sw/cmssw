#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

namespace l1t {

  void RegionalMuonCand::setTFIdentifiers(int processor, tftype trackFinder) {
    m_trackFinder = trackFinder;
    m_processor = processor;

    switch (m_trackFinder) {
    case tftype::emtf_pos:
      m_link = m_processor + 36;  // range 36...41
      break;
    case tftype::omtf_pos:
      m_link = m_processor + 42;  // range 42...47
      break;
    case tftype::bmtf:
      m_link = m_processor + 48;  // range 48...59
      break;
    case tftype::omtf_neg:
      m_link = m_processor + 60;  // range 60...65
      break;
    case tftype::emtf_neg:
      m_link = m_processor + 66;  // range 66...71
    }
  }

  void SortCandsEMTF(RegionalMuonCandBxCollection& cands) {
    
    int minBX = cands.getFirstBX();
    int maxBX = cands.getLastBX();
    int emtfMinProc =  0; // ME+ sector 1
    int emtfMaxProc = 11; // ME- sector 6
    
    // New collection, sorted by processor to match uGMT unpacked order
    RegionalMuonCandBxCollection* sortedCands = new RegionalMuonCandBxCollection();
    sortedCands->clear();
    sortedCands->setBXRange(minBX, maxBX);
    for (int iBX = minBX; iBX <= maxBX; ++iBX) {
      for (int proc = emtfMinProc; proc <= emtfMaxProc; proc++) {
        for (RegionalMuonCandBxCollection::const_iterator cand = cands.begin(iBX); cand != cands.end(iBX); ++cand) {
          if (cand->processor() != proc) continue;
          sortedCands->push_back(iBX, *cand);
        }
      }
    }
    
    // Return sorted collection
    cands.clear();
    cands = (*sortedCands);
  }

} // namespace l1t

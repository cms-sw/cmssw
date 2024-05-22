#ifndef DataFormats_Luminosity_PixelClusterCounts_h
#define DataFormats_Luminosity_PixelClusterCounts_h
/** \class reco::PixelClusterCounts
 *  
 * Reconstructed PixelClusterCounts object that will contain the moduleID, BX, and counts.
 *
 * \authors: Sam Higginbotham shigginb@cern.ch and Chris Palmer capalmer@cern.ch
 * 
 *
 */
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

#include "DataFormats/Luminosity/interface/LumiConstants.h"
#include "DataFormats/Luminosity/interface/PixelClusterCountsInEvent.h"

namespace reco {
  class PixelClusterCounts {
  public:
    PixelClusterCounts() : m_events(LumiConstants::numBX) {}

    void increment(int mD, unsigned int bxID, int count) {
      size_t modIndex = std::distance(m_ModID.begin(), std::find(m_ModID.begin(), m_ModID.end(), mD));
      if (modIndex == m_ModID.size()) {
        m_ModID.push_back(mD);
        m_counts.resize(m_counts.size() + LumiConstants::numBX, 0);
      }
      m_counts.at(LumiConstants::numBX * modIndex + bxID - 1) += count;
    }

    void incrementRoc(int rD, unsigned int bxID, int count) {
      size_t rocIndex = std::distance(m_RocID.begin(), std::find(m_RocID.begin(), m_RocID.end(), rD));
      if (rocIndex == m_RocID.size()) {
        m_RocID.push_back(rD);
        m_countsRoc.resize(m_countsRoc.size() + LumiConstants::numBX, 0);
      }
      m_countsRoc.at(LumiConstants::numBX * rocIndex + bxID - 1) += count;
    }

    void eventCounter(unsigned int bxID) { m_events.at(bxID - 1)++; }

    void add(reco::PixelClusterCountsInEvent const& pccInEvent) {
      std::vector<int> const& countsInEvent = pccInEvent.counts();
      std::vector<int> const& rocCountsInEvent = pccInEvent.countsRoc();
      std::vector<int> const& modIDInEvent = pccInEvent.modID();
      std::vector<int> const& rocIDInEvent = pccInEvent.rocID();
      int bxIDInEvent = pccInEvent.bxID();
      for (unsigned int i = 0; i < modIDInEvent.size(); i++) {
        increment(modIDInEvent[i], bxIDInEvent, countsInEvent.at(i));
      }
      for (unsigned int i = 0; i < rocIDInEvent.size(); i++) {
        incrementRoc(rocIDInEvent[i], bxIDInEvent, rocCountsInEvent.at(i));
      }
    }

    void merge(reco::PixelClusterCounts const& pcc) {
      std::vector<int> const& counts = pcc.readCounts();
      std::vector<int> const& countsRoc = pcc.readRocCounts();
      std::vector<int> const& modIDs = pcc.readModID();
      std::vector<int> const& rocIDs = pcc.readRocID();
      std::vector<int> const& events = pcc.readEvents();
      for (unsigned int i = 0; i < modIDs.size(); i++) {
        for (unsigned int bxID = 0; bxID < LumiConstants::numBX; ++bxID) {
          increment(modIDs[i], bxID + 1, counts.at(i * LumiConstants::numBX + bxID));
        }
      }
      for (unsigned int i = 0; i < rocIDs.size(); i++) {
        for (unsigned int bxID = 0; bxID < LumiConstants::numBX; ++bxID) {
          incrementRoc(rocIDs[i], bxID + 1, countsRoc.at(i * LumiConstants::numBX + bxID));
        }
      }
      for (unsigned int i = 0; i < LumiConstants::numBX; ++i) {
        m_events[i] += events[i];
      }
    }

    void reset() {
      m_counts.clear();
      m_countsRoc.clear();
      m_ModID.clear();
      m_RocID.clear();
      m_events.clear();
      m_events.resize(LumiConstants::numBX, 0);
    }

    std::vector<int> const& readCounts() const { return (m_counts); }
    std::vector<int> const& readRocCounts() const { return (m_countsRoc); }
    std::vector<int> const& readEvents() const { return (m_events); }
    std::vector<int> const& readModID() const { return (m_ModID); }
    std::vector<int> const& readRocID() const { return (m_RocID); }

  private:
    std::vector<int> m_counts;
    std::vector<int> m_countsRoc;
    std::vector<int> m_events;
    std::vector<int> m_ModID;
    std::vector<int> m_RocID;
  };

}  // namespace reco
#endif

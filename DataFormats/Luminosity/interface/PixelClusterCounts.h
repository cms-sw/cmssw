#ifndef DataFormats_Luminosity_PixelClusterCounts_h
#define DataFormats_Luminosity_PixelClusterCounts_h
/** \class reco::PixelClusterCounts
 *  
 * Reconstructed PixelClusterCounts object that will contain the moduleID, BX, and counts.
 *
 * \authors: Sam Higginbotham shiggib@cern.ch and Chris Palmer capalmer@cern.ch
 * 
 *
 */
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Luminosity/interface/LumiConstants.h"

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

    void eventCounter(unsigned int bxID) { m_events.at(bxID - 1)++; }

    std::vector<int> const& readCounts() const { return (m_counts); }
    std::vector<int> const& readEvents() const { return (m_events); }
    std::vector<int> const& readModID() const { return (m_ModID); }

  private:
    std::vector<int> m_counts;
    std::vector<int> m_events;
    std::vector<int> m_ModID;
  };

}  // namespace reco
#endif

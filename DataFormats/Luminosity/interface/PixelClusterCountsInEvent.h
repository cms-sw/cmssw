#ifndef DataFormats_Luminosity_PixelClusterCountsInEvent_h
#define DataFormats_Luminosity_PixelClusterCountsInEvent_h
/** \class reco::PixelClusterCountsInEvent
 *  
 * Reconstructed PixelClusterCountsInEvent object that will contain the moduleID, bxID, and counts per event.
 *
 * \authors: Sam Higginbotham (shigginb@cern.ch), Chris Palmer (capalmer@cern.ch), Attila Radl (attila.radl@cern.ch) 
 * 
 *
 */
#include <algorithm>
#include <vector>

namespace reco {
  class PixelClusterCountsInEvent {
  public:
    PixelClusterCountsInEvent() : m_bxID() {}

    void increment(int mD, int count) {
      size_t modIndex = std::distance(m_ModID.begin(), std::find(m_ModID.begin(), m_ModID.end(), mD));
      if (modIndex == m_ModID.size()) {
        m_ModID.push_back(mD);
        m_counts.push_back(0);
      }
      m_counts[modIndex] += count;
    }

    void incrementRoc(int rD, int count) {
      size_t rocIndex = std::distance(m_RocID.begin(), std::find(m_RocID.begin(), m_RocID.end(), rD));
      if (rocIndex == m_RocID.size()) {
        m_RocID.push_back(rD);
        m_countsRoc.push_back(0);
      }
      m_countsRoc[rocIndex] += count;
    }

    void setbxID(unsigned int inputbxID) { m_bxID = inputbxID; }

    std::vector<int> const& counts() const { return (m_counts); }

    std::vector<int> const& countsRoc() const { return (m_countsRoc); }

    std::vector<int> const& modID() const { return (m_ModID); }

    std::vector<int> const& rocID() const { return (m_RocID); }

    unsigned int const& bxID() const { return m_bxID; }

  private:
    std::vector<int> m_counts;
    std::vector<int> m_countsRoc;
    std::vector<int> m_ModID;
    std::vector<int> m_RocID;
    unsigned int m_bxID;
  };

}  // namespace reco
#endif

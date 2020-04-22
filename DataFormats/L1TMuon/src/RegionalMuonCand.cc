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

  bool RegionalMuonCand::operator==(const RegionalMuonCand& rhs) const {
    return m_hwPt == rhs.hwPt() && m_hwPtUnconstrained == rhs.hwPtUnconstrained() && m_hwDXY == rhs.hwDXY() &&
           m_hwPhi == rhs.hwPhi() && m_hwEta == rhs.hwEta() && m_hwHF == (bool)rhs.hwHF() && m_hwSign == rhs.hwSign() &&
           m_hwSignValid == rhs.hwSignValid() && m_hwQuality == rhs.hwQual() && m_link == rhs.link() &&
           m_processor == rhs.processor() && m_trackFinder == rhs.trackFinderType() &&
           m_trackAddress == rhs.trackAddress();
  }

}  // namespace l1t

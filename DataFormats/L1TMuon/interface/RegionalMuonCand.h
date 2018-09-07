#ifndef __l1t_regional_muon_candidate_h__
#define __l1t_regional_muon_candidate_h__

#include "RegionalMuonCandFwd.h"
#include <map>

namespace l1t {
class RegionalMuonCand {
  public:
    /// Enum to identify the individual parts of the BMTF track address
    /// Update kNumBmtfSubAddr if you add additional enums
    enum bmtfAddress {
        kWheelSide=0, kWheelNum=1, kStat1=2, kStat2=3, kStat3=4, kStat4=5, kSegSelStat1=6, kSegSelStat2=7, kSegSelStat3=8, kSegSelStat4=9, kNumBmtfSubAddr=10
    };
    /// Enum to identify the individual parts of the OMTF track address
    /// Update kNumOmtfSubAddr if you add additional enums
    enum omtfAddress {
	kLayers=0, kZero=1, kWeight=2, kNumOmtfSubAddr=3
    };
    /// Enum to identify the individual parts of the EMTF track address
    /// Update kNumEmtfSubAddr if you add additional enums
    enum emtfAddress {
	kME1Seg=0, kME1Ch=1, kME2Seg=2, kME2Ch=3, kME3Seg=4, kME3Ch=5, kME4Seg=6, kME4Ch=7, kTrkNum=8, kBX=9, kNumEmtfSubAddr=10
    };


    explicit RegionalMuonCand(uint64_t dataword);

    RegionalMuonCand() :
    m_hwPt(0),m_hwPt2(0),m_hwDXY(0),m_hwPhi(0), m_hwEta(0), m_hwHF(false), m_hwSign(0), m_hwSignValid(0), m_hwQuality(0),
      m_dataword(0)
      {
        setTFIdentifiers(0, bmtf);
      };

    RegionalMuonCand(int pt, int phi, int eta, int sign, int signvalid, int quality, int processor, tftype trackFinder) :
    m_hwPt(pt),m_hwPt2(0),m_hwDXY(0),m_hwPhi(phi), m_hwEta(eta), m_hwHF(false), m_hwSign(sign), m_hwSignValid(signvalid), m_hwQuality(quality),
	m_dataword(0)
      {
        setTFIdentifiers(processor, trackFinder);
        // set default track addresses
        if (trackFinder == tftype::bmtf) {
          m_trackAddress = {{kWheelSide, 0}, {kWheelNum, 0}, {kStat1, 0}, {kStat2, 0}, {kStat3, 0}, {kStat4, 0}, {kSegSelStat1, 0}, {kSegSelStat2, 0}, {kSegSelStat3, 0}, {kSegSelStat4, 0}};
        } else if (trackFinder == tftype::omtf_pos || trackFinder == tftype::omtf_neg) {
          m_trackAddress = {{kLayers, 0}, {kZero, 0}, {kWeight, 0}};
        } else if (trackFinder == tftype::emtf_pos || trackFinder == tftype::emtf_neg) {
          m_trackAddress = {{kME1Seg, 0}, {kME1Ch, 0}, {kME2Seg, 0}, {kME2Ch, 0}, {kME3Seg, 0}, {kME3Ch, 0}, {kME4Seg, 0}, {kME4Ch, 0}, {kTrkNum, 0}, {kBX, 0}};
        }
      };

    RegionalMuonCand(int pt, int phi, int eta, int sign, int signvalid, int quality, int processor, tftype trackFinder, std::map<int, int> trackAddress) :
    m_hwPt(pt) ,m_hwPt2(0),m_hwDXY(0),m_hwPhi(phi), m_hwEta(eta), m_hwHF(false), m_hwSign(sign), m_hwSignValid(signvalid), m_hwQuality(quality), m_trackAddress(trackAddress),
	m_dataword(0)
      {
        setTFIdentifiers(processor, trackFinder);
      };

    virtual ~RegionalMuonCand() {};

    /// Set compressed pT as transmitted by hardware LSB = 0.5 (9 bits)
    void setHwPt(int bits) { m_hwPt = bits; };
    /// Set compressed second displaced  pT as transmitted by hardware LSB = 1.0 (8 bits)
    void setHwPt2(int bits) { m_hwPt2 = bits; };
    /// Set compressed impact parameter with respect to beamspot (4 bits)
    void setHwDXY(int bits) { m_hwDXY = bits; };
    /// Set compressed relative phi as transmitted by hardware LSB = 2*pi/576 (8 bits)
    void setHwPhi(int bits) { m_hwPhi = bits; };
    /// Set compressed eta as transmitted by hardware LSB = 0.010875 (9 bits)
    void setHwEta(int bits) { m_hwEta = bits; };
    /// Set charge sign bit (charge = (-1)^(sign))
    void setHwSign(int bits) { m_hwSign = bits; };
    /// Set whether charge measurement is valid (0 for high pT muons)
    void setHwSignValid(int bits) { m_hwSignValid = bits; };
    /// Set compressed quality code as transmitted by hardware (4 bits)
    void setHwQual(int bits) { m_hwQuality = bits; };
    /// Set HF (halo / fine eta) bit (EMTF: halo -> 1; BMTF: fine eta -> 1)
    void setHwHF(bool bit) { m_hwHF = bit; };
    /// Set the processor ID, track-finder type. From these two, the link is set
    void setTFIdentifiers(int processor, tftype trackFinder);
    // this is left to still be compatible with OMTF
    void setLink(int link) {m_link = link; };
    // Set the 64 bit word from two 32 words. bits 0-31->lsbs, bits 32-63->msbs
    void setDataword(uint32_t msbs, uint32_t lsbs) { m_dataword = (((uint64_t)msbs) << 32) + lsbs; };
    // Set the 64 bit word coming from HW directly
    void setDataword(uint64_t bits) { m_dataword = bits; };
    /// Set a part of the muon candidates track address; specialised for BMTF
    void setTrackSubAddress(bmtfAddress subAddress, int value) {
        m_trackAddress[subAddress] = value;
    }
    /// Set a part of the muon candidates track address; specialised for OMTF
    void setTrackSubAddress(omtfAddress subAddress, int value) {
        m_trackAddress[subAddress] = value;
    }
    /// Set a part of the muon candidates track address; specialised for EMTF
    void setTrackSubAddress(emtfAddress subAddress, int value) {
        m_trackAddress[subAddress] = value;
    }
    /// Set the whole track address
    void setTrackAddress(const std::map<int, int>& address) {
        m_trackAddress = address;
    }


    /// Get compressed pT (returned int * 0.5 = pT (GeV))
    const int hwPt() const { return m_hwPt; };
    /// Get second compressed pT (returned int * 1.0 = pT (GeV))
    const int hwPt2() const { return m_hwPt2; };
    /// Get compressed impact parameter (4 bits)
    const int hwDXY() const { return m_hwDXY; };
    /// Get compressed local phi (returned int * 2*pi/576 = local phi in rad)
    const int hwPhi() const { return m_hwPhi; };
    /// Get compressed eta (returned int * 0.010875 = eta)
    const int hwEta() const { return m_hwEta; };
    /// Get charge sign bit (charge = (-1)^(sign))
    const int hwSign() const { return m_hwSign; };
    /// Get charge sign valid bit (0 - not valid (high pT muon); 1 - valid)
    const int hwSignValid() const { return m_hwSignValid; };
    /// Get quality code
    const int hwQual() const { return m_hwQuality; };
    /// Get link on which the MicroGMT receives the candidate
    const int link() const { return m_link; };
    /// Get processor ID on which the candidate was found (0..5 for OMTF/EMTF; 0..11 for BMTF)
    const int processor() const { return m_processor; };
    /// Get track-finder which found the muon (bmtf, emtf_pos/emtf_neg or omtf_pos/omtf_neg)
    const tftype trackFinderType() const { return m_trackFinder; };
    /// Get HF (halo / fine eta) bit (EMTF: halo -> 1; BMTF: fine eta -> 1)
    const int hwHF() const { return m_hwHF; };
    /// Get 64 bit data word
    const uint64_t dataword() const { return m_dataword; };
    /// Get 32 MSBs of data word
    const uint32_t dataword32Msb() const { return (uint32_t)((m_dataword >> 32) & 0xFFFFFFFF); };
    /// Get 32 LSBs of data word
    const uint32_t dataword32Lsb() const { return (uint32_t)(m_dataword & 0xFFFFFFFF); };
    /// Get the track address (identifies track primitives used for reconstruction)
    const std::map<int, int>& trackAddress() const {
        return m_trackAddress;
    }
    /// Get part of track address (identifies track primitives used for reconstruction)
    int trackSubAddress(bmtfAddress subAddress) const {
        return m_trackAddress.at(subAddress);
    }
    /// Get part of track address (identifies track primitives used for reconstruction)
    int trackSubAddress(omtfAddress subAddress) const {
        return m_trackAddress.at(subAddress);
    }
    /// Get part of track address (identifies track primitives used for reconstruction)
    int trackSubAddress(emtfAddress subAddress) const {
        return m_trackAddress.at(subAddress);
    }

  bool operator==(const RegionalMuonCand& rhs) const;
  inline bool operator!=(const RegionalMuonCand& rhs) const { return !(operator==(rhs)); };


  private:
    int m_hwPt;
    int m_hwPt2;
    int m_hwDXY;
    int m_hwPhi;
    int m_hwEta;
    bool m_hwHF;
    int m_hwSign;
    int m_hwSignValid;
    int m_hwQuality;
    int m_link;
    int m_processor;
    tftype m_trackFinder;
    std::map<int, int> m_trackAddress;

    /// This is the 64 bit word as transmitted in HW
    uint64_t m_dataword;

};

}

#endif /* define __l1t_regional_muon_candidate_h__ */

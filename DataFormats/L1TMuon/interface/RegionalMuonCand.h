#ifndef __l1t_regional_muon_candidate_h__
#define __l1t_regional_muon_candidate_h__

#include <iostream>
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  enum tftype {
    bmtf, omtf_neg, omtf_pos, emtf_neg, emtf_pos
  };
  class RegionalMuonCand;
  typedef BXVector<RegionalMuonCand> RegionalMuonCandBxCollection;


class RegionalMuonCand {
  public:
    explicit RegionalMuonCand(uint64_t dataword);

    RegionalMuonCand() :
      m_hwPt(0), m_hwPhi(0), m_hwEta(0), m_hwHF(false), m_hwSign(0), m_hwSignValid(0), m_hwQuality(0),
      m_hwTrackAddress(0), m_link(0), m_processor(0), m_trackFinder(bmtf), m_dataword(0)
      {};

    RegionalMuonCand(int pt, int phi, int eta, int sign, int signvalid, int quality, int processor, tftype trackFinder) :
      m_hwPt(pt), m_hwPhi(phi), m_hwEta(eta), m_hwHF(false), m_hwSign(sign), m_hwSignValid(signvalid), m_hwQuality(quality),
      m_hwTrackAddress(0), m_link(0), m_dataword(0)
      {
        setTFIdentifiers(processor, trackFinder);
      };

    virtual ~RegionalMuonCand() {};

    /// Set compressed pT as transmitted by hardware LSB = 0.5 (9 bits)
    void setHwPt(int bits) { m_hwPt = bits; };
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
    /// Set compressed track address as transmitted by hardware. Identifies trigger primitives.
    void setHwTrackAddress(int bits) { m_hwTrackAddress = bits; };
    /// Set the processor ID, track-finder type. From these two, the link is set
    void setTFIdentifiers(int processor, tftype trackFinder);
    // this is left to still be compatible with OMTF
    void setLink(int link);
    // Set the 64 bit word from two 32 words. bits 0-31->lsbs, bits 32-63->msbs
    void setDataword(int msbs, int lsbs) { m_dataword = (((uint64_t)msbs) << 32) + lsbs; };
    // Set the 64 bit word coming from HW directly
    void setDataword(uint64_t bits) { m_dataword = bits; };


    /// Get compressed pT (returned int * 0.5 = pT (GeV))
    const int hwPt() const { return m_hwPt; };
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
    /// Get track address identifying trigger primitives
    const int hwTrackAddress() const { return m_hwTrackAddress; };
    /// Get link on which the MicroGMT receives the candidate
    const int link() const { return m_link; };
    /// Get processor ID on which the candidate was found (1..6 for OMTF/EMTF; 1..12 for BMTF)
    const int processor() const { return m_processor; };
    /// Get track-finder which found the muon (bmtf, emtf_pos/emtf_neg or omtf_pos/omtf_neg)
    const tftype trackFinderType() const { return m_trackFinder; };
    /// Get HF (halo / fine eta) bit (EMTF: halo -> 1; BMTF: fine eta -> 1)
    const int hwHF() const { return m_hwHF; };
    /// Get 64 bit data word
    const uint64_t dataword() const { return m_dataword; };
    /// Get 32 MSBs of data word
    const int dataword32Msb() const { return (int)((m_dataword >> 32) & 0xFFFFFFFF); };
    /// Get 32 LSBs of data word
    const int dataword32Lsb() const { return (int)(m_dataword & 0xFFFFFFFF); };

  private:
    int m_hwPt;
    int m_hwPhi;
    int m_hwEta;
    bool m_hwHF;
    int m_hwSign;
    int m_hwSignValid;
    int m_hwQuality;
    int m_hwTrackAddress;
    int m_link;
    int m_processor;
    tftype m_trackFinder;

    /// This is the 64 bit word as transmitted in HW
    uint64_t m_dataword;

};

}

#endif /* define __l1t_regional_muon_candidate_h__ */


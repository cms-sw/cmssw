#ifndef L1GCTJET_H_
#define L1GCTJET_H_

#include <functional>
#include <vector>
#include <ostream>
#include <memory>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include <cstdint>

/*!
 * \author Jim Brooke & Robert Frazier
 * \date April 2006
 */

/*! \class L1GctJet
 * \brief A Level-1 jet candidate, used within GCT emulation
 * 
 *  Move this to DataFormats/L1GlobalCaloTrigger if possible
 */

class L1GctJetCand;
class L1GctJetEtCalibrationLut;

class L1GctJet {
public:
  //Statics
  enum numberOfBits { kRawsumBitWidth = 10, kRawsumMaxValue = (1 << kRawsumBitWidth) - 1 };

  //Typedefs
  typedef std::shared_ptr<L1GctJetEtCalibrationLut> lutPtr;

  //Constructors/destructors
  L1GctJet(const uint16_t rawsum = 0,
           const unsigned eta = 0,
           const unsigned phi = 0,
           const bool overFlow = false,
           const bool forwardJet = true,
           const bool tauVeto = true,
           const int16_t bx = 0);
  ~L1GctJet();

  // set rawsum and position bits
  void setRawsum(const uint16_t rawsum) {
    m_rawsum = rawsum & kRawsumMaxValue;
    m_overFlow |= (rawsum > kRawsumMaxValue);
  }
  void setDetId(const L1CaloRegionDetId detId) { m_id = detId; }
  void setOverFlow(const bool overFlow) { m_overFlow = overFlow; }
  void setTauVeto(const bool tauVeto) { m_tauVeto = tauVeto; }
  void setForward(const bool forward) { m_forwardJet = forward; }
  void setBx(const int16_t bx) { m_bx = bx; }

  // get rawsum and position bits
  uint16_t rawsum() const { return m_rawsum; }
  L1CaloRegionDetId id() const { return m_id(); }
  bool tauVeto() const { return m_tauVeto; }

  /// get overflow
  bool overFlow() const { return m_overFlow; }

  /// test whether this jet candidate is a valid tau jet
  bool isTauJet() const { return (!m_forwardJet && !m_tauVeto); }

  /// test whether this jet candidate is a (non-tau) central jet
  bool isCentralJet() const { return (!m_forwardJet && m_tauVeto); }

  /// test whether this jet candidate is a forward jet
  bool isForwardJet() const { return m_forwardJet; }

  /// test whether this jet candidate has been filled
  bool isNullJet() const { return ((m_rawsum == 0) && (globalEta() == 0) && (globalPhi() == 0)); }

  friend std::ostream& operator<<(std::ostream& os, const L1GctJet& cand);

  /// test whether two jets are the same
  bool operator==(const L1GctJet& cand) const;

  /// test whether two jets are different
  bool operator!=(const L1GctJet& cand) const;

  ///Setup an existing jet all in one go
  void setupJet(const uint16_t rawsum,
                const unsigned eta,
                const unsigned phi,
                const bool overFlow,
                const bool forwardJet,
                const bool tauVeto = true,
                const int16_t bx = 0);

  /// eta value in global CMS coordinates
  unsigned globalEta() const { return m_id.ieta(); }

  /// phi value in global CMS coordinates
  unsigned globalPhi() const { return m_id.iphi(); }

  /// eta value in global CMS coordinates
  unsigned rctEta() const { return m_id.rctEta(); }

  /// phi value in global CMS coordinates
  unsigned rctPhi() const { return m_id.rctPhi(); }

  /// eta value as encoded in hardware at the GCT output
  unsigned hwEta() const;

  /// phi value as encoded in hardware at the GCT output
  unsigned hwPhi() const;

  /// the bunch crossing number
  int16_t bx() const { return m_bx; }

  /// Functions to convert from internal format to external jet candidates at the output of the jetFinder
  L1GctJetCand jetCand(const lutPtr lut) const;
  L1GctJetCand jetCand(const std::vector<lutPtr>& luts) const;

  /// The two separate Lut outputs
  uint16_t rank(const lutPtr lut) const;
  unsigned calibratedEt(const lutPtr lut) const;

private:
  uint16_t m_rawsum;
  /// region id, encodes eta and phi
  L1CaloRegionDetId m_id;
  bool m_overFlow;
  bool m_forwardJet;
  bool m_tauVeto;
  int16_t m_bx;

  uint16_t lutValue(const lutPtr lut) const;
};

std::ostream& operator<<(std::ostream& os, const L1GctJet& cand);

#endif /*L1GCTJET_H_*/

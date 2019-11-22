#ifndef CTPPSDigi_CTPPSDiamondDigi_h
#define CTPPSDigi_CTPPSDiamondDigi_h

/** \class CTPPSDiamondDigi
 *
 * Digi Class for CTPPS Timing Detector
 *
 *
 * \author Seyed Mohsen Etesami
 * March 2016
 */

#include <cstdint>
#include "DataFormats/CTPPSDigi/interface/HPTDCErrorFlags.h"

class CTPPSDiamondDigi {
public:
  CTPPSDiamondDigi(
      unsigned int ledgt_, unsigned int tedgt_, unsigned int threvolt, bool mhit_, unsigned short hptdcerror_);
  CTPPSDiamondDigi();
  ~CTPPSDiamondDigi(){};

  /// Digis are equal if they are have same  ledt and tedt, threshold voltage, multihit flag, hptdcerror flags
  bool operator==(const CTPPSDiamondDigi& digi) const;

  /// Return digi values number

  unsigned int leadingEdge() const { return ledgt; }

  unsigned int trailingEdge() const { return tedgt; }

  unsigned int thresholdVoltage() const { return threvolt; }

  bool multipleHit() const { return mhit; }

  HPTDCErrorFlags hptdcErrorFlags() const { return hptdcerror; }

  /// Set digi values
  inline void setLeadingEdge(unsigned int ledgt_) { ledgt = ledgt_; }
  inline void setTrailingEdge(unsigned int tedgt_) { tedgt = tedgt_; }
  inline void setThresholdVoltage(unsigned int threvolt_) { threvolt = threvolt_; }
  inline void setMultipleHit(bool mhit_) { mhit = mhit_; }
  inline void setHPTDCErrorFlags(const HPTDCErrorFlags& hptdcerror_) { hptdcerror = hptdcerror_; }

private:
  // variable represents leading edge time
  unsigned int ledgt;
  // variable	represents trailing edge time
  unsigned int tedgt;
  // variable represents threshold voltage
  unsigned int threvolt;
  // variable represents multi-hit
  bool mhit;
  HPTDCErrorFlags hptdcerror;
};

#include <iostream>

inline bool operator<(const CTPPSDiamondDigi& one, const CTPPSDiamondDigi& other) {
  if (one.leadingEdge() < other.leadingEdge())
    return true;
  if (one.leadingEdge() > other.leadingEdge())
    return false;
  if (one.trailingEdge() < other.trailingEdge())
    return true;
  if (one.trailingEdge() > other.trailingEdge())
    return false;
  if (one.multipleHit() < other.multipleHit())
    return true;
  if (one.multipleHit() > other.multipleHit())
    return false;
  if (one.hptdcErrorFlags().errorFlag() < other.hptdcErrorFlags().errorFlag())
    return true;
  if (one.hptdcErrorFlags().errorFlag() > other.hptdcErrorFlags().errorFlag())
    return false;
  if (one.thresholdVoltage() < other.thresholdVoltage())
    return true;
  if (one.thresholdVoltage() > other.thresholdVoltage())
    return false;
  return false;
}

inline std::ostream& operator<<(std::ostream& o, const CTPPSDiamondDigi& digi) {
  return o << " " << digi.leadingEdge() << " " << digi.trailingEdge() << " " << digi.thresholdVoltage() << " "
           << digi.multipleHit() << " " << digi.hptdcErrorFlags().errorFlag();
}

#endif

/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSReco_CTPPSDiamondRecHit
#define DataFormats_CTPPSReco_CTPPSDiamondRecHit

#include "DataFormats/CTPPSDigi/interface/HPTDCErrorFlags.h"
#include "DataFormats/CTPPSReco/interface/CTPPSTimingRecHit.h"

/// Reconstructed hit in diamond detectors.
class CTPPSDiamondRecHit : public CTPPSTimingRecHit {
public:
  CTPPSDiamondRecHit() : CTPPSTimingRecHit(), tot_(0), tPrecision_(0), tsIndex_(0), hptdcErr_(0), mh_(false) {}
  CTPPSDiamondRecHit(float x,
                     float xWidth,
                     float y,
                     float yWidth,
                     float z,
                     float zWidth,
                     float t,
                     float tot,
                     float tPrecision,
                     int ootIdx,
                     const HPTDCErrorFlags &hptdcErr,
                     const bool mh)
      : CTPPSTimingRecHit(x, xWidth, y, yWidth, z, zWidth, t),
        tot_(tot),
        tPrecision_(tPrecision),
        tsIndex_(ootIdx),
        hptdcErr_(hptdcErr),
        mh_(mh) {}

  static constexpr int TIMESLICE_WITHOUT_LEADING = -10;

  inline void setToT(float tot) { tot_ = tot; }
  inline float toT() const { return tot_; }

  inline void setTPrecision(float tPrecision) { tPrecision_ = tPrecision; }
  inline float tPrecision() const { return tPrecision_; }

  inline void setOOTIndex(int i) { tsIndex_ = i; }
  inline int ootIndex() const { return tsIndex_; }

  inline void setMultipleHits(bool mh) { mh_ = mh; }
  inline bool multipleHits() const { return mh_; }

  inline void setHPTDCErrorFlags(const HPTDCErrorFlags &err) { hptdcErr_ = err; }
  inline HPTDCErrorFlags hptdcErrorFlags() const { return hptdcErr_; }

private:
  /// Time over threshold
  float tot_;
  /// Expected timing precision
  float tPrecision_;
  /// Time slice index
  int tsIndex_;
  /// List of error flags encountered by the HPTDC
  HPTDCErrorFlags hptdcErr_;
  /// Multiple hits detected
  bool mh_;
};

#endif

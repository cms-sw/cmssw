#ifndef DataFormats_L1DTTrackFinder_L1Phase2MuDTExtPhiThetaPair_h
#define DataFormats_L1DTTrackFinder_L1Phase2MuDTExtPhiThetaPair_h

/** \class L1Phase2MuDTExtPhiThetaPair
 *
 *  Data container for a matched pair of Phase-2 DT Phi and Theta digis.
 *  Provides utilities to retrieve the closest-N time-position pairs per chamber ordered by Phi quality.
 *
 *  \author J. Fernandez
 *  \date   2025-11-19
 */

#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtThDigi.h"
#include <vector>
#include <algorithm>

class L1Phase2MuDTExtPhiThetaPair {
public:
  /// Default constructor
  L1Phase2MuDTExtPhiThetaPair() = default;

  /// Constructor with digis and quality
  L1Phase2MuDTExtPhiThetaPair(const L1Phase2MuDTExtPhDigi& phi, const L1Phase2MuDTExtThDigi& theta, int quality);

  // Explicitly allow copy/move
  L1Phase2MuDTExtPhiThetaPair(const L1Phase2MuDTExtPhiThetaPair&) = default;
  L1Phase2MuDTExtPhiThetaPair& operator=(const L1Phase2MuDTExtPhiThetaPair&) = default;
  L1Phase2MuDTExtPhiThetaPair(L1Phase2MuDTExtPhiThetaPair&&) = default;
  L1Phase2MuDTExtPhiThetaPair& operator=(L1Phase2MuDTExtPhiThetaPair&&) = default;

  /// Accessors
  const L1Phase2MuDTExtPhDigi& phiDigi() const { return phi_; }
  const L1Phase2MuDTExtThDigi& thetaDigi() const { return theta_; }
  int quality() const { return quality_; }

private:
  L1Phase2MuDTExtPhDigi phi_;
  L1Phase2MuDTExtThDigi theta_;
  int quality_;
};

#endif

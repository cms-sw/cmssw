#ifndef DataFormats_CTPPSReco_interface_CTPPSPixelRecHit_h
#define DataFormats_CTPPSReco_interface_CTPPSPixelRecHit_h

/*
 *
 * This is a part of CTPPS offline software.
 * Author:
 *   Fabrizio Ferro (ferro@ge.infn.it)
 *
 */

#include <cassert>

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "FWCore/Utilities/interface/isFinite.h"

// Reconstructed hits in CTPPS Pixel detector

class CTPPSPixelRecHit {
public:
  CTPPSPixelRecHit(LocalPoint lp = LocalPoint(0., 0., 0.),
                   LocalError le = LocalError(0., 0., 0.),
                   bool edge = false,
                   bool bad = false,
                   bool rocs = false,
                   int minrow = 0,
                   int mincol = 0,
                   int size = 0,
                   int rowsize = 0,
                   int colsize = 0)
      : thePoint_(lp),
        theError_(le),
        isOnEdge_(edge),
        hasBadPixels_(bad),
        spanTwoRocs_(rocs),
        minPixelRow_(minrow),
        minPixelCol_(mincol),
        clusterSize_(size),
        clusterSizeRow_(rowsize),
        clusterSizeCol_(colsize) {}

  LocalPoint point() const { return thePoint_; }
  LocalError error() const { return theError_; }

  bool isOnEdge() const { return isOnEdge_; }
  bool hasBadPixels() const { return hasBadPixels_; }
  bool spanTwoRocs() const { return spanTwoRocs_; }

  unsigned int minPixelRow() const { return minPixelRow_; }
  unsigned int minPixelCol() const { return minPixelCol_; }

  unsigned int clusterSize() const { return clusterSize_; }
  unsigned int clusterSizeRow() const { return clusterSizeRow_; }
  unsigned int clusterSizeCol() const { return clusterSizeCol_; }

  float sort_key() const { return thePoint_.mag2(); }

private:
  LocalPoint thePoint_;
  LocalError theError_;

  bool isOnEdge_;
  bool hasBadPixels_;
  bool spanTwoRocs_;

  unsigned int minPixelRow_;
  unsigned int minPixelCol_;

  unsigned int clusterSize_;
  unsigned int clusterSizeRow_;
  unsigned int clusterSizeCol_;
};

inline bool operator<(CTPPSPixelRecHit const& a, CTPPSPixelRecHit const& b) {
  float a_key = a.sort_key();
  float b_key = b.sort_key();
  assert(edm::isFinite(a_key));
  assert(edm::isFinite(b_key));
  return a_key < b_key;
}

#endif  // DataFormats_CTPPSReco_interface_CTPPSPixelRecHit_h

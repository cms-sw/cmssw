/*
 *
* This is a part of CTPPS offline software.
* Author:
*   Fabrizio Ferro (ferro@ge.infn.it)
*
*/

#ifndef DataFormats_CTPPSReco_CTPPSPixelRecHit_H
#define DataFormats_CTPPSReco_CTPPSPixelRecHit_H

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

// Reconstructed hits in CTPPS Pixel detector

class CTPPSPixelRecHit {
public:
  CTPPSPixelRecHit() {}

  CTPPSPixelRecHit(LocalPoint lp,
                   LocalError le,
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

  inline LocalPoint point() const { return thePoint_; }
  inline LocalError error() const { return theError_; }

  inline bool isOnEdge() const { return isOnEdge_; }
  inline bool hasBadPixels() const { return hasBadPixels_; }
  inline bool spanTwoRocs() const { return spanTwoRocs_; }

  inline unsigned int minPixelRow() const { return minPixelRow_; }
  inline unsigned int minPixelCol() const { return minPixelCol_; }

  inline unsigned int clusterSize() const { return clusterSize_; }
  inline unsigned int clusterSizeRow() const { return clusterSizeRow_; }
  inline unsigned int clusterSizeCol() const { return clusterSizeCol_; }

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

inline bool operator<(CTPPSPixelRecHit& a, CTPPSPixelRecHit& b) { return (a.point().mag() < b.point().mag()); };

#endif

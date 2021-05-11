#ifndef DATAFORMATS_SISTRIPDETSETCOMPRESSEDCLUSTER_H
#define DATAFORMATS_SISTRIPDETSETCOMPRESSEDCLUSTER_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <numeric>

class SiStripDetSetCompressedCluster {
public:
  static const uint16_t stripIndexMask = 0x7FFF;   // The first strip index is in the low 15 bits of firstStrip_
  static const uint16_t mergedValueMask = 0x8000;  // The merged state is given by the high bit of firstStrip_

  /** Construct from a range of digis that form a cluster and from 
   *  a DetID. The range is assumed to be non-empty.
   */

  SiStripDetSetCompressedCluster();

  SiStripDetSetCompressedCluster(std::vector<std::pair<uint16_t, bool>>&, std::vector<float>&, std::vector<uint8_t>&);

  void push_back_supportInfo(uint16_t firstStrip, bool merged = false, float errx = -99999.9);
  void loadCompressedAmplitudes(std::vector<uint8_t>& inVect);
  void addCompressedAmplitudes(std::vector<uint8_t>& inVect);

  /** The number of the first strip in the cluster.
   *  The high bit of firstStrip_ indicates whether the cluster is a candidate for being merged.
   */
  const std::vector<uint16_t>& firstStrip() const { return firstStrip_; }
  const std::vector<uint8_t>& compressedAmplitudes() const { return compressedAmplitudes_; }
  //const std::vector<float>& plitClusterError() const { return error_x_;}

  bool isMerged(uint16_t firstStrip) const { return (firstStrip & mergedValueMask) != 0; }

private:
  void push_back_firstStip(uint16_t firstStrip, bool merged = false) {
    firstStrip_.push_back(merged ? firstStrip |= mergedValueMask : firstStrip &= stripIndexMask);
  }
  void push_back_splitClusterError(float errx = -99999.9) {}  //error_x_.push_back(errx);}

  std::vector<uint8_t> compressedAmplitudes_;
  std::vector<uint16_t> firstStrip_;
  //std::vector<float> error_x_;
};

#endif  // DATAFORMATS_SISTRIPDETSETCOMPRESSEDCLUSTER_H

#include "DataFormats/SiStripDetSetCompressedCluster/interface/SiStripDetSetCompressedCluster.h"

SiStripDetSetCompressedCluster::SiStripDetSetCompressedCluster(std::vector<std::pair<uint16_t, bool>>& firstStripMerged,
                                                               std::vector<uint8_t>& inVect) {
  for (auto itFs : firstStripMerged)
    push_back_firstStip(itFs.first, itFs.second);
  
  loadCompressedAmplitudes(inVect);
}

void SiStripDetSetCompressedCluster::push_back_supportInfo(uint16_t firstStrip, bool merged) {
  push_back_firstStip(firstStrip, merged);
}

void SiStripDetSetCompressedCluster::loadCompressedAmplitudes(std::vector<uint8_t>& inVect) {
  compressedAmplitudes_.clear();
  compressedAmplitudes_.assign(inVect.begin(), inVect.end());
}

void SiStripDetSetCompressedCluster::addCompressedAmplitudes(std::vector<uint8_t>& inVect) {
  compressedAmplitudes_.insert(compressedAmplitudes_.end(), inVect.begin(), inVect.end());
}
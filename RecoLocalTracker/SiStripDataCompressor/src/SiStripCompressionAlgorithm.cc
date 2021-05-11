#include "RecoLocalTracker/SiStripDataCompressor/interface/SiStripCompressionAlgorithm.h"
#include "RecoLocalTracker/SiStripDataCompressor/include/anlz4cmssw.h"
#include <iostream>

SiStripCompressionAlgorithm::SiStripCompressionAlgorithm() { this->LoadRealModelDataFromFile(); }

void SiStripCompressionAlgorithm::compress(vclusters_t const& ncColl, vcomp_clusters_t& compColl) {
  for (vclusters_t::const_iterator itInColl = ncColl.begin(); itInColl != ncColl.end(); itInColl++) {
    vcomp_clusters_t::TSFastFiller ff(compColl, itInColl->detId());
    commpressDetModule(*itInColl, ff);
    if (ff.empty())
      ff.abort();
  }
}

void SiStripCompressionAlgorithm::commpressDetModule(const clusters_t& ncClusters,
                                                     vcomp_clusters_t::TSFastFiller& compressedClusters) {
  std::vector<std::vector<uint8_t>> toBeCompressed;
  std::vector<std::uint8_t> compAmplitudes;
  SiStripDetSetCompressedCluster compCluster;

  for (clusters_t::const_iterator itNcClusters = ncClusters.begin(); itNcClusters != ncClusters.end(); itNcClusters++) {
    compCluster.push_back_supportInfo(
        itNcClusters->firstStrip(), itNcClusters->isMerged(), itNcClusters->getSplitClusterError());
    std::vector<uint8_t> clsuterToBeCompressed(itNcClusters->begin(), itNcClusters->end());
    toBeCompressed.push_back(clsuterToBeCompressed);
  }

  anlz4cmssw_compress(toBeCompressed, compAmplitudes);

  compCluster.loadCompressedAmplitudes(compAmplitudes);
  compressedClusters.push_back(compCluster);
}

void SiStripCompressionAlgorithm::LoadRealModelDataFromFile() {
  std::uint8_t modelBuf[65536];
  std::vector<std::uint8_t> model;

  char* globalPath = getenv("CMSSW_BASE");
  std::string fullName =
      std::string(globalPath) + std::string("src/RecoLocalTracker/SiStripDataCompressor/data/model.dat");
  LogDebug("Output") << " Path: " << fullName;
  FILE* fTrained = std::fopen(&fullName[0], "rb");
  while (true) {
    std::size_t rdSz = std::fread(modelBuf, 1, sizeof modelBuf, fTrained);
    if (rdSz <= 0)
      break;
    model.insert(model.end(), modelBuf, &modelBuf[rdSz]);
  }
  std::fclose(fTrained);

  anlz4cmssw_load_trained_model(model);
}

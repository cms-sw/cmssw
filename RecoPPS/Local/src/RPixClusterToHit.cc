#include "RecoPPS/Local/interface/RPixClusterToHit.h"

RPixClusterToHit::RPixClusterToHit(edm::ParameterSet const &conf)
    : verbosity_(conf.getUntrackedParameter<int>("RPixVerbosity")) {}

void RPixClusterToHit::buildHits(unsigned int detId,
                                 const std::vector<CTPPSPixelCluster> &clusters,
                                 std::vector<CTPPSPixelRecHit> &hits,
                                 const PPSPixelTopology &ppt) const {
  if (verbosity_)
    edm::LogInfo("PPS") << " RPixClusterToHit " << detId << " received cluster array of size = " << clusters.size();

  for (unsigned int i = 0; i < clusters.size(); i++) {
    makeHit(clusters[i], hits, ppt);
  }
}

void RPixClusterToHit::makeHit(CTPPSPixelCluster cluster,
                               std::vector<CTPPSPixelRecHit> &hits,
                               const PPSPixelTopology &ppt) const {
  // take a cluster, generate a rec hit and push it in the rec hit vector

  //call the numbering inside the ROC
  CTPPSPixelIndices pxlInd;
  // get information from the cluster
  // get the whole cluster size and row/col size
  unsigned int thisClusterSize = cluster.size();
  unsigned int thisClusterRowSize = cluster.sizeRow();
  unsigned int thisClusterColSize = cluster.sizeCol();

  // get the minimum pixel row/col
  unsigned int thisClusterMinRow = cluster.minPixelRow();
  unsigned int thisClusterMinCol = cluster.minPixelCol();

  // calculate "on edge" flag
  bool anEdgePixel = false;
  if (cluster.minPixelRow() == 0 || cluster.minPixelCol() == 0 ||
      int(cluster.minPixelRow() + cluster.rowSpan()) == (pxlInd.getDefaultRowDetSize() - 1) ||
      int(cluster.minPixelCol() + cluster.colSpan()) == (pxlInd.getDefaultColDetSize() - 1))
    anEdgePixel = true;

  // check for bad (ADC=0) pixels in cluster
  bool aBadPixel = false;
  for (unsigned int i = 0; i < thisClusterSize; i++) {
    if (cluster.pixelADC(i) == 0)
      aBadPixel = true;
  }

  // check for spanning two ROCs
  bool twoRocs = false;
  int currROCId = pxlInd.getROCId(cluster.pixelCol(0), cluster.pixelRow(0));

  for (unsigned int i = 1; i < thisClusterSize; i++) {
    if (pxlInd.getROCId(cluster.pixelCol(i), cluster.pixelRow(i)) != currROCId) {
      twoRocs = true;
      break;
    }
  }

  // estimate position and error of the hit
  double avgWLocalX = 0;
  double avgWLocalY = 0;
  double weights = 0;
  double weightedVarianceX = 0.;
  double weightedVarianceY = 0.;

  if (verbosity_)
    edm::LogInfo("PPS") << "RPixClusterToHit "
                        << " hit pixels: ";

  for (unsigned int i = 0; i < thisClusterSize; i++) {
    if (verbosity_)
      edm::LogInfo("PPS") << "RPixClusterToHit " << cluster.pixelRow(i) << " " << cluster.pixelCol(i) << " "
                          << cluster.pixelADC(i);

    double minPxlX = 0;
    double minPxlY = 0;
    double maxPxlX = 0;
    double maxPxlY = 0;

    ppt.pixelRange(cluster.pixelRow(i), cluster.pixelCol(i), minPxlX, maxPxlX, minPxlY, maxPxlY);
    double halfSizeX = (maxPxlX - minPxlX) / 2.;
    double halfSizeY = (maxPxlY - minPxlY) / 2.;
    double avgPxlX = minPxlX + halfSizeX;
    double avgPxlY = minPxlY + halfSizeY;

    // error propagation
    weightedVarianceX += cluster.pixelADC(i) * cluster.pixelADC(i) * halfSizeX * halfSizeX / 3.;
    weightedVarianceY += cluster.pixelADC(i) * cluster.pixelADC(i) * halfSizeY * halfSizeY / 3.;

    avgWLocalX += avgPxlX * cluster.pixelADC(i);
    avgWLocalY += avgPxlY * cluster.pixelADC(i);
    weights += cluster.pixelADC(i);
  }

  if (weights == 0) {
    edm::LogError("RPixClusterToHit") << " unexpected weights = 0 for cluster (Row_min, Row_max, Col_min, Col_max) = ("
                                      << cluster.minPixelRow() << "," << cluster.minPixelRow() + cluster.rowSpan()
                                      << "," << cluster.minPixelCol() << ","
                                      << cluster.minPixelCol() + cluster.colSpan() << ")";
    return;
  }

  double invWeights = 1. / weights;
  double avgLocalX = avgWLocalX * invWeights;
  double avgLocalY = avgWLocalY * invWeights;

  double varianceX = weightedVarianceX * invWeights * invWeights;
  double varianceY = weightedVarianceY * invWeights * invWeights;

  LocalPoint lp(avgLocalX, avgLocalY, 0);
  LocalError le(varianceX, 0, varianceY);
  if (verbosity_)
    edm::LogInfo("PPS") << "RPixClusterToHit " << lp << " with error " << le;

  hits.emplace_back(lp,
                    le,
                    anEdgePixel,
                    aBadPixel,
                    twoRocs,
                    thisClusterMinRow,
                    thisClusterMinCol,
                    thisClusterSize,
                    thisClusterRowSize,
                    thisClusterColSize);
}

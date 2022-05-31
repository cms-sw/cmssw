//----------------------------------------------------------------------------
//! \class PixelThresholdClusterizerForBricked
//! \brief A specific threshold-based pixel clustering algorithm
//!
//! Same logic as the base class PixelThresholdClusterizer but specialized for bricked pixels topology
//----------------------------------------------------------------------------

// Our own includes
#include "PixelThresholdClusterizerForBricked.h"
#include "SiPixelArrayBuffer.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
// Geometry
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
//#include "Geometry/CommonTopologies/RectangularPixelTopology.h"

// STL
#include <stack>
#include <vector>
#include <iostream>
#include <atomic>
#include <algorithm>
#include <limits>

//----------------------------------------------------------------------------
//! Constructor:
//!  Initilize the buffer to hold pixels from a detector module.
//!  This is a vector of 44k ints, stays valid all the time.
//----------------------------------------------------------------------------
PixelThresholdClusterizerForBricked::PixelThresholdClusterizerForBricked(edm::ParameterSet const& conf)
    : PixelThresholdClusterizer(conf) {}
/////////////////////////////////////////////////////////////////////////////
PixelThresholdClusterizerForBricked::~PixelThresholdClusterizerForBricked() {}

//----------------------------------------------------------------------------
//!  \brief Cluster pixels.
//!  This method operates on a matrix of pixels
//!  and finds the largest contiguous cluster around
//!  each seed pixel.
//!  Input and output data stored in DetSet
//----------------------------------------------------------------------------
template <typename T>
void PixelThresholdClusterizerForBricked::clusterizeDetUnitT(const T& input,
                                                             const PixelGeomDetUnit* pixDet,
                                                             const TrackerTopology* tTopo,
                                                             const std::vector<short>& badChannels,
                                                             edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) {
  typename T::const_iterator begin = input.begin();
  typename T::const_iterator end = input.end();

  edm::LogInfo("PixelThresholdClusterizerForBricked::clusterizeDetUnitT()");

  // this should never happen and the raw2digi does not create empty detsets
  if (begin == end) {
    edm::LogError("PixelThresholdClusterizerForBricked")
        << "@SUB=PixelThresholdClusterizerForBricked::clusterizeDetUnitT()"
        << " No digis to clusterize";
  }

  //  Set up the clusterization on this DetId.
  if (!setup(pixDet))
    return;

  theDetid = input.detId();

  bool isBarrel = (DetId(theDetid).subdetId() == PixelSubdetector::PixelBarrel);
  // Set separate cluster threshold for L1 (needed for phase1)
  auto clusterThreshold = theClusterThreshold;
  theLayer = (DetId(theDetid).subdetId() == 1) ? tTopo->pxbLayer(theDetid) : 0;
  if (theLayer == 1)
    clusterThreshold = theClusterThreshold_L1;

  //  Copy PixelDigis to the buffer array; select the seed pixels
  //  on the way, and store them in theSeeds.
  if (end > begin)
    copy_to_buffer(begin, end);

  assert(output.empty());
  //  Loop over all seeds.  TO DO: wouldn't using iterators be faster?
  for (unsigned int i = 0; i < theSeeds.size(); i++) {
    // Gavril : The charge of seeds that were already inlcuded in clusters is set to 1 electron
    // so we don't want to call "make_cluster" for these cases
    if (theBuffer(theSeeds[i]) >= theSeedThreshold) {  // Is this seed still valid?
      //  Make a cluster around this seed
      SiPixelCluster cluster;
      if ((&pixDet->specificTopology())->isBricked()) {
        cluster = make_cluster_bricked(theSeeds[i], output, isBarrel);
      } else {
        cluster = make_cluster(theSeeds[i], output);
      }

      //  Check if the cluster is above threshold
      // (TO DO: one is signed, other unsigned, gcc warns...)
      if (cluster.charge() >= clusterThreshold) {
        // sort by row (x)
        output.push_back(std::move(cluster));
        std::push_heap(output.begin(), output.end(), [](SiPixelCluster const& cl1, SiPixelCluster const& cl2) {
          return cl1.minPixelRow() < cl2.minPixelRow();
        });
      }
    }
  }
  // sort by row (x)   maybe sorting the seed would suffice....
  std::sort_heap(output.begin(), output.end(), [](SiPixelCluster const& cl1, SiPixelCluster const& cl2) {
    return cl1.minPixelRow() < cl2.minPixelRow();
  });

  // Erase the seeds.
  theSeeds.clear();

  //  Need to clean unused pixels from the buffer array.
  clear_buffer(begin, end);

  theFakePixels.clear();
}

//----------------------------------------------------------------------------
//!  \brief The actual clustering algorithm: group the neighboring pixels around the seed.
//----------------------------------------------------------------------------
SiPixelCluster PixelThresholdClusterizerForBricked::make_cluster_bricked(
    const SiPixelCluster::PixelPos& pix, edmNew::DetSetVector<SiPixelCluster>::FastFiller& output, bool isbarrel) {
  //First we acquire the seeds for the clusters
  std::stack<SiPixelCluster::PixelPos, std::vector<SiPixelCluster::PixelPos> > dead_pixel_stack;

  //The individual modules have been loaded into a buffer.
  //After each pixel has been considered by the clusterizer, we set the adc count to 1
  //to mark that we have already considered it.
  //The only difference between dead/noisy pixels and standard ones is that for dead/noisy pixels,
  //We consider the charge of the pixel to always be zero.

  // Note: each ADC value is limited here to 65535 (std::numeric_limits<uint16_t>::max),
  //       as it is later stored as uint16_t in SiPixelCluster and PixelClusterizerBase/AccretionCluster
  //       (reminder: ADC values here may be expressed in number of electrons)
  uint16_t seed_adc = std::min(theBuffer(pix.row(), pix.col()), int(std::numeric_limits<uint16_t>::max()));
  theBuffer.set_adc(pix, 1);

  AccretionCluster acluster;
  acluster.add(pix, seed_adc);

  //Here we search all pixels adjacent to all pixels in the cluster.
  bool dead_flag = false;
  while (!acluster.empty()) {
    //This is the standard algorithm to find and add a pixel
    auto curInd = acluster.top();
    acluster.pop();

    for (auto r = std::max(0, int(acluster.x[curInd]) - 1); r < std::min(int(acluster.x[curInd]) + 2, theBuffer.rows());
         ++r) {
      int LowerAccLimity = 0;
      int UpperAccLimity = 0;

      if (r % 2 == int(acluster.x[curInd]) % 2) {
        LowerAccLimity = std::max(0, int(acluster.y[curInd]) - 1);
        UpperAccLimity = std::min(int(acluster.y[curInd]) + 2, theBuffer.columns());
      }

      else {
        int parity_curr = int(acluster.x[curInd]) % 2;
        int parity_hit = r % 2;

        LowerAccLimity = std::max(0, int(acluster.y[curInd]) - parity_hit);
        UpperAccLimity = std::min(int(acluster.y[curInd]) + parity_curr + 1, theBuffer.columns());
      }

      /*
	for (auto c = std::max(0, int(acluster.y[curInd]) - 1);
	c < std::min(int(acluster.y[curInd]) + 2, theBuffer.columns());
	++c)
      */
      for (auto c = LowerAccLimity; c < UpperAccLimity; ++c) {
        if (theBuffer(r, c) >= thePixelThreshold) {
          SiPixelCluster::PixelPos newpix(r, c);
          auto const newpix_adc = std::min(theBuffer(r, c), int(std::numeric_limits<uint16_t>::max()));
          if (!acluster.add(newpix, newpix_adc))
            goto endClus;
          if (isbarrel)
            edm::LogInfo("make_cluster_bricked()") << "add" << r << c << theBuffer(r, c);
          theBuffer.set_adc(newpix, 1);
          //std::cout<<"col "<<c<<" row "<<r<<std::endl;
        }
      }
    }

  }  // while accretion
endClus:
  SiPixelCluster cluster(acluster.isize, acluster.adc, acluster.x, acluster.y, acluster.xmin, acluster.ymin);
  //Here we split the cluster, if the flag to do so is set and we have found a dead or noisy pixel.

  if (dead_flag && doSplitClusters) {
    // Set separate cluster threshold for L1 (needed for phase1)
    auto clusterThreshold = theClusterThreshold;
    if (theLayer == 1)
      clusterThreshold = theClusterThreshold_L1;

    //Set the first cluster equal to the existing cluster.
    SiPixelCluster first_cluster = cluster;
    bool have_second_cluster = false;
    while (!dead_pixel_stack.empty()) {
      //consider each found dead pixel
      SiPixelCluster::PixelPos deadpix = dead_pixel_stack.top();
      dead_pixel_stack.pop();
      theBuffer.set_adc(deadpix, 1);

      //Clusterize the split cluster using the dead pixel as a seed
      SiPixelCluster second_cluster = make_cluster_bricked(deadpix, output, isbarrel);

      //If both clusters would normally have been found by the clusterizer, put them into output
      if (second_cluster.charge() >= clusterThreshold && first_cluster.charge() >= clusterThreshold) {
        output.push_back(second_cluster);
        have_second_cluster = true;
      }

      //We also want to keep the merged cluster in data and let the RecHit algorithm decide which set to keep
      //This loop adds the second cluster to the first.
      const std::vector<SiPixelCluster::Pixel>& branch_pixels = second_cluster.pixels();
      for (unsigned int i = 0; i < branch_pixels.size(); i++) {
        auto const temp_x = branch_pixels[i].x;
        auto const temp_y = branch_pixels[i].y;
        auto const temp_adc = branch_pixels[i].adc;
        SiPixelCluster::PixelPos newpix(temp_x, temp_y);
        cluster.add(newpix, temp_adc);
      }
    }

    //Remember to also add the first cluster if we added the second one.
    if (first_cluster.charge() >= clusterThreshold && have_second_cluster) {
      output.push_back(first_cluster);
      std::push_heap(output.begin(), output.end(), [](SiPixelCluster const& cl1, SiPixelCluster const& cl2) {
        return cl1.minPixelRow() < cl2.minPixelRow();
      });
    }
  }

  return cluster;
}

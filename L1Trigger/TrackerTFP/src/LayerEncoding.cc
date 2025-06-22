#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"

#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>

namespace trackerTFP {

  LayerEncoding::LayerEncoding(const DataFormats* dataFormats)
      : setup_(dataFormats->setup()),
        dataFormats_(dataFormats),
        zT_(&dataFormats->format(Variable::zT, Process::gp)),
        layerEncoding_(std::vector<std::vector<int>>(pow(2, zT_->width()))),
        maybePattern_(std::vector<TTBV>(pow(2, zT_->width()), TTBV(0, setup_->numLayers()))),
        nullLE_(setup_->numLayers(), 0),
        nullMP_(0, setup_->numLayers()) {
    // number of boundaries of fiducial area in r-z plane for a given set of rough r-z track parameter
    static constexpr int boundaries = 2;
    // z at radius chosenRofZ wrt zT of sectorZT of this bin boundaries
    const std::vector<double> z0s = {-setup_->beamWindowZ(), setup_->beamWindowZ()};
    // find unique sensor mouldes in r-z
    // allowed distance in r and z in cm between modules to consider them not unique
    static constexpr double delta = 1.e-3;
    std::vector<const tt::SensorModule*> sensorModules;
    sensorModules.reserve(setup_->sensorModules().size());
    for (const tt::SensorModule& sm : setup_->sensorModules())
      sensorModules.push_back(&sm);
    auto smallerR = [](const tt::SensorModule* lhs, const tt::SensorModule* rhs) { return lhs->r() < rhs->r(); };
    auto smallerZ = [](const tt::SensorModule* lhs, const tt::SensorModule* rhs) { return lhs->z() < rhs->z(); };
    auto equalRZ = [](const tt::SensorModule* lhs, const tt::SensorModule* rhs) {
      return std::abs(lhs->r() - rhs->r()) < delta && std::abs(lhs->z() - rhs->z()) < delta;
    };
    std::stable_sort(sensorModules.begin(), sensorModules.end(), smallerZ);
    std::stable_sort(sensorModules.begin(), sensorModules.end(), smallerR);
    sensorModules.erase(std::unique(sensorModules.begin(), sensorModules.end(), equalRZ), sensorModules.end());
    std::stable_sort(sensorModules.begin(), sensorModules.end(), smallerZ);
    sensorModules.erase(std::unique(sensorModules.begin(), sensorModules.end(), equalRZ), sensorModules.end());
    // find set of moudles for each set of rough r-z track parameter
    // loop over zT bins
    for (int binZT = 0; binZT < std::pow(2, zT_->width()); binZT++) {
      // z at radius chosenRofZ
      const double zT = zT_->floating(zT_->toSigned(binZT));
      // z at radius chosenRofZ wrt zT of sectorZT of this bin boundaries
      const std::vector<double> zTs = {zT - zT_->base() / 2., zT + zT_->base() / 2.};
      std::vector<std::vector<double>> cots(boundaries);
      for (int i = 0; i < boundaries; i++)
        for (double z0 : z0s)
          cots[i].push_back((zTs[i] - z0) / setup_->chosenRofZ());
      // layer ids crossed by left and right rough r-z parameter shape boundaries
      std::vector<std::set<int>> layers(boundaries);
      // loop over all unique modules
      for (const tt::SensorModule* sm : sensorModules) {
        // check if module is crossed by left and right rough r-z parameter shape boundaries
        for (int i = 0; i < boundaries; i++) {
          const double zTi = zTs[i];
          const double coti = sm->r() < setup_->chosenRofZ() ? cots[i][i == 0 ? 0 : 1] : cots[i][i == 0 ? 1 : 0];
          // distance between module and boundary in moudle tilt angle direction
          const double d =
              (zTi - sm->z() + (sm->r() - setup_->chosenRofZ()) * coti) / (sm->cosTilt() - sm->sinTilt() * coti);
          // compare distance with module size and add module layer id to layers if module is crossed
          if (std::abs(d) < sm->numColumns() * sm->pitchCol() / 2.)
            layers[i].insert(sm->layerId());
        }
      }
      // mayber layers are given by layer ids crossed by only one boundary
      std::set<int> maybeLayer;
      std::set_symmetric_difference(layers[0].begin(),
                                    layers[0].end(),
                                    layers[1].begin(),
                                    layers[1].end(),
                                    std::inserter(maybeLayer, maybeLayer.end()));
      // layerEncoding is given by sorted layer ids crossed by any boundary
      std::set<int> layerEncoding;
      std::set_union(layers[0].begin(),
                     layers[0].end(),
                     layers[1].begin(),
                     layers[1].end(),
                     std::inserter(layerEncoding, layerEncoding.end()));
      // fill layerEncoding_
      std::vector<int>& le = layerEncoding_[binZT];
      le = std::vector<int>(layerEncoding.begin(), layerEncoding.end());
      le.resize(setup_->numLayers(), -1);
      // fill maybePattern_
      TTBV& mp = maybePattern_[binZT];
      for (int m : maybeLayer)
        mp.set(std::min(static_cast<int>(std::distance(le.begin(), std::find(le.begin(), le.end(), m))),
                        setup_->numLayers() - 1));
    }
  }

  // Set of layers for given bin in zT
  const std::vector<int>& LayerEncoding::layerEncoding(int zT) const {
    const int binZT = zT_->toUnsigned(zT);
    return zT_->inRange(zT) ? layerEncoding_.at(binZT) : nullLE_;
  }

  // Set of layers for given zT in cm
  const std::vector<int>& LayerEncoding::layerEncoding(double zT) const {
    const int binZT = zT_->integer(zT);
    return layerEncoding(binZT);
  }

  // pattern of maybe layers for given bin in zT
  const TTBV& LayerEncoding::maybePattern(int zT) const {
    const int binZT = zT_->toUnsigned(zT);
    return zT_->inRange(zT) ? maybePattern_[binZT] : nullMP_;
  }

  // pattern of maybe layers for given zT in cm
  const TTBV& LayerEncoding::maybePattern(double zT) const {
    const int binZT = zT_->integer(zT);
    return maybePattern(binZT);
  }

  // fills numPS, num2S, numMissingPS and numMissingPS for given hitPattern and trajectory
  void LayerEncoding::analyze(
      int hitpattern, double cot, double z0, int& numPS, int& num2S, int& numMissingPS, int& numMissing2S) const {
    // look up layer encoding nad maybe pattern
    const double zT = z0 + setup_->chosenRofZ() * cot;
    const std::vector<int>& le = this->layerEncoding(zT);
    const TTBV& mp = this->maybePattern(zT);
    const TTBV hp(hitpattern, setup_->numLayers());
    // loop from innermost layer to outermost hitted layer
    for (int layerIdKF = 0; layerIdKF <= hp.pmEncode(); layerIdKF++) {
      // look up layer Id [1-6 barrel, 11-15 disks]
      const int layerId = le[layerIdKF];
      // identify module type
      bool ps = layerId <= setup_->numBarrelLayerPS();
      const bool barrel = layerId <= setup_->numBarrelLayer();
      if (!barrel) {
        // calc disk id (0 - 4)
        const int diskId = layerId - setup_->offsetLayerDisks() - setup_->offsetLayerId();
        // avergae disk z position
        const double z = setup_->hybridDiskZ(diskId) * (cot < 0. ? -1. : 1.);
        // innermost edge of 2S modules
        const double rLimit = setup_->disk2SR(diskId, 0) - setup_->pitchCol2S();
        // trajectory radius at avergae disk z position
        const double r = (z - z0) / cot;
        // compare with innermost edge of 2S modules to identify PS
        if (r < rLimit)
          ps = true;
      }
      if (hp.test(layerIdKF))  // layer is hit
        ps ? numPS++ : num2S++;
      else if (!mp.test(layerIdKF))  // layer is not hit but should have been hitted (roughly by) trajectory
        ps ? numMissingPS++ : numMissing2S++;
    }
  }

}  // namespace trackerTFP

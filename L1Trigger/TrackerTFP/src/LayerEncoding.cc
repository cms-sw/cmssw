#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "FWCore/Utilities/interface/Exception.h"

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
        layerEncoding_(
            std::vector<std::vector<int>>(std::pow(2, zT_->width()), std::vector<int>(setup_->sysNumLayer(), -1))),
        maybePattern_(std::vector<TTBV>(std::pow(2, zT_->width()), TTBV(0, setup_->sysNumLayer()))),
        maybePS_(std::pow(2, zT_->width()), -1),
        nullLE_(setup_->sysNumLayer(), 0),
        nullMP_(0, setup_->sysNumLayer()) {
    // number of boundaries of fiducial area in r-z plane for a given set of rough r-z track parameter
    static constexpr int boundaries = 2;
    // z at radius chosenRofZ wrt zT of sectorZT of this bin boundaries
    const std::vector<double> z0s = {-setup_->regBeamWindowZ(), setup_->regBeamWindowZ()};
    // find unique sensor mouldes in r-z
    // allowed distance in r and z in cm between modules to consider them not unique
    static constexpr double delta = 1.e-3;
    std::vector<const trackerDTC::SensorModule*> sensorModules;
    sensorModules.reserve(setup_->sensorModules().size());
    for (const trackerDTC::SensorModule& sm : setup_->sensorModules())
      sensorModules.push_back(&sm);
    auto smallerR = [](const trackerDTC::SensorModule* lhs, const trackerDTC::SensorModule* rhs) {
      return lhs->r() < rhs->r();
    };
    auto smallerZ = [](const trackerDTC::SensorModule* lhs, const trackerDTC::SensorModule* rhs) {
      return lhs->z() < rhs->z();
    };
    auto equalRZ = [](const trackerDTC::SensorModule* lhs, const trackerDTC::SensorModule* rhs) {
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
          cots[i].push_back((zTs[i] - z0) / setup_->regChosenRofZ());
      // Sensormodules crossed by left and right rough r-z parameter shape boundaries
      std::vector<std::deque<const trackerDTC::SensorModule*>> sms(boundaries);
      // loop over all unique modules
      for (const trackerDTC::SensorModule* sm : sensorModules) {
        // check if module is crossed by left and right rough r-z parameter shape boundaries
        for (int i = 0; i < boundaries; i++) {
          const double zTi = zTs[i];
          const double coti = sm->r() < setup_->regChosenRofZ() ? cots[i][i == 0 ? 0 : 1] : cots[i][i == 0 ? 1 : 0];
          // distance between module and boundary in moudle tilt angle direction
          const double d =
              (zTi - sm->z() + (sm->r() - setup_->regChosenRofZ()) * coti) / (sm->cosTilt() - sm->sinTilt() * coti);
          // compare distance with module size and add module layer id to layers if module is crossed
          if (std::abs(d) < sm->numColumns() * sm->pitchCol() / 2.)
            sms[i].push_back(sm);
        }
      }
      // crossed layer ids
      std::vector<std::set<int>> layers(boundaries);
      for (int i = 0; i < boundaries; i++) {
        std::set<int>& layer = layers[i];
        for (const trackerDTC::SensorModule* sm : sms[i])
          layer.insert(sm->layerId());
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
      int layerIdKF(0);
      std::vector<int>& le = layerEncoding_[binZT];
      for (int layerId : layerEncoding)
        le[layerIdKF++] = layerId;
      // fill maybePattern_
      TTBV& mp = maybePattern_[binZT];
      for (int m : maybeLayer)
        mp.set(std::min(static_cast<int>(std::distance(le.begin(), std::find(le.begin(), le.end(), m))),
                        setup_->sysNumLayer() - 1));
      // maybe PS
      TTBV hitPattern2S(0, setup_->sysNumLayer());
      TTBV hitPatternPS(0, setup_->sysNumLayer());
      std::stringstream ss;
      for (int i = 0; i < boundaries; i++) {
        for (const trackerDTC::SensorModule* sm : sms[i]) {
          ss << sm->layerId() << " " << sm->psModule() << " ";
          const int layerId = std::distance(le.begin(), std::find(le.begin(), le.end(), sm->layerId()));
          ss << layerId << std::endl;
          sm->psModule() ? hitPatternPS.set(layerId) : hitPattern2S.set(layerId);
        }
      }
      int& maybePS = maybePS_[binZT];
      const TTBV hitPatternAnd = hitPattern2S && hitPatternPS;
      if (hitPatternAnd.ids().size() > 1) {
        cms::Exception exception("LogicError");
        exception << "Found more then one layer with ambigous PS/2S assignment.";
        throw exception;
      }
      if (hitPatternAnd.any())
        maybePS = hitPatternAnd.ids().front();
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

  // encoded layer id which may be PS or 2S for given zT
  int LayerEncoding::maybePS(int zT) const {
    const int binZT = zT_->toUnsigned(zT);
    return maybePS_.at(binZT);
  }

  // encoded layer id which may be PS or 2S for given zT in cm
  int LayerEncoding::maybePS(double zT) const {
    const int binZT = zT_->integer(zT);
    return maybePS(binZT);
  }

}  // namespace trackerTFP

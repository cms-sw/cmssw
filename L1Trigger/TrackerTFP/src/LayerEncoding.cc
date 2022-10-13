#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackTrigger/interface/SensorModule.h"

#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>

using namespace std;
using namespace tt;

namespace trackerTFP {

  LayerEncoding::LayerEncoding(const DataFormats* dataFormats)
      : setup_(dataFormats->setup()),
        dataFormats_(dataFormats),
        zT_(&dataFormats->format(Variable::zT, Process::zht)),
        cot_(&dataFormats->format(Variable::cot, Process::zht)),
        layerEncoding_(setup_->numSectorsEta(),
                       vector<vector<vector<int>>>(pow(2, zT_->width()), vector<vector<int>>(pow(2, cot_->width())))),
        maybeLayer_(setup_->numSectorsEta(),
                    vector<vector<vector<int>>>(pow(2, zT_->width()), vector<vector<int>>(pow(2, cot_->width())))) {
    // number of boundaries of fiducial area in r-z plane for a given set of rough r-z track parameter
    static constexpr int boundaries = 2;
    // find unique sensor mouldes in r-z
    // allowed distance in r and z in cm between modules to consider them not unique
    static constexpr double delta = 1.e-3;
    vector<const SensorModule*> sensorModules;
    sensorModules.reserve(setup_->sensorModules().size());
    for (const SensorModule& sm : setup_->sensorModules())
      sensorModules.push_back(&sm);
    auto smallerR = [](const SensorModule* lhs, const SensorModule* rhs) { return lhs->r() < rhs->r(); };
    auto smallerZ = [](const SensorModule* lhs, const SensorModule* rhs) { return lhs->z() < rhs->z(); };
    auto equalRZ = [](const SensorModule* lhs, const SensorModule* rhs) {
      return abs(lhs->r() - rhs->r()) < delta && abs(lhs->z() - rhs->z()) < delta;
    };
    stable_sort(sensorModules.begin(), sensorModules.end(), smallerR);
    stable_sort(sensorModules.begin(), sensorModules.end(), smallerZ);
    sensorModules.erase(unique(sensorModules.begin(), sensorModules.end(), equalRZ), sensorModules.end());
    // find set of moudles for each set of rough r-z track parameter
    // loop over eta sectors
    for (int binEta = 0; binEta < setup_->numSectorsEta(); binEta++) {
      // cotTheta of eta sector centre
      const double sectorCot = (sinh(setup_->boundarieEta(binEta + 1)) + sinh(setup_->boundarieEta(binEta))) / 2.;
      // z at radius choenRofZ of eta sector centre
      const double sectorZT = setup_->chosenRofZ() * sectorCot;
      // loop over bins in zT
      for (int binZT = 0; binZT < pow(2, zT_->width()); binZT++) {
        // z at radius chosenRofZ wrt zT of sectorZT of this bin centre
        const double zT = zT_->floating(zT_->toSigned(binZT));
        // z at radius chosenRofZ wrt zT of sectorZT of this bin boundaries
        const vector<double> zTs = {sectorZT + zT - zT_->base() / 2., sectorZT + zT + zT_->base() / 2.};
        // loop over bins in cotTheta
        for (int binCot = 0; binCot < pow(2, cot_->width()); binCot++) {
          // cotTheta wrt sectorCot of this bin centre
          const double cot = cot_->floating(cot_->toSigned(binCot));
          // layer ids crossed by left and right rough r-z parameter shape boundaries
          vector<set<int>> layers(boundaries);
          // cotTheta wrt sectorCot of this bin boundaries
          const vector<double> cots = {sectorCot + cot - cot_->base() / 2., sectorCot + cot + cot_->base() / 2.};
          // loop over all unique modules
          for (const SensorModule* sm : sensorModules) {
            // check if module is crossed by left and right rough r-z parameter shape boundaries
            for (int i = 0; i < boundaries; i++) {
              const int j = boundaries - i - 1;
              const double zTi = zTs[sm->r() > setup_->chosenRofZ() ? i : j];
              const double coti = cots[sm->r() > setup_->chosenRofZ() ? j : i];
              // distance between module and boundary in moudle tilt angle direction
              const double d =
                  (zTi - sm->z() + (sm->r() - setup_->chosenRofZ()) * coti) / (sm->cosTilt() - sm->sinTilt() * coti);
              // compare distance with module size and add module layer id to layers if module is crossed
              if (abs(d) < sm->numColumns() * sm->pitchCol() / 2.)
                layers[i].insert(sm->layerId());
            }
          }
          // mayber layers are given by layer ids crossed by only one booundary
          set<int> maybeLayer;
          set_symmetric_difference(layers[0].begin(),
                                   layers[0].end(),
                                   layers[1].begin(),
                                   layers[1].end(),
                                   inserter(maybeLayer, maybeLayer.end()));
          // layerEncoding is given by sorted layer ids crossed by any booundary
          set<int> layerEncoding;
          set_union(layers[0].begin(),
                    layers[0].end(),
                    layers[1].begin(),
                    layers[1].end(),
                    inserter(layerEncoding, layerEncoding.end()));
          vector<int>& le = layerEncoding_[binEta][binZT][binCot];
          le = vector<int>(layerEncoding.begin(), layerEncoding.end());
          vector<int>& ml = maybeLayer_[binEta][binZT][binCot];
          ml.reserve(maybeLayer.size());
          for (int m : maybeLayer) {
            int layer = distance(le.begin(), find(le.begin(), le.end(), m));
            if (layer >= setup_->numLayers())
              layer = setup_->numLayers() - 1;
            ml.push_back(layer);
          }
        }
      }
    }
    const bool print = false;
    if (!print)
      return;
    static constexpr int widthLayer = 3;
    static constexpr auto layerIds = {1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15};
    stringstream ss;
    for (int layer : layerIds) {
      auto encode = [layer, this](const vector<int>& layers, int& l) {
        const auto it = find(layers.begin(), layers.end(), layer);
        if (it == layers.end())
          return false;
        l = distance(layers.begin(), it);
        if (l >= setup_->numLayers())
          l = setup_->numLayers() - 1;
        return true;
      };
      for (int binEta = 0; binEta < setup_->numSectorsEta(); binEta++) {
        for (int binZT = 0; binZT < pow(2, zT_->width()); binZT++) {
          for (int binCot = 0; binCot < pow(2, cot_->width()); binCot++) {
            const int zT =
                binZT < pow(2, zT_->width() - 1) ? binZT + pow(2, zT_->width() - 1) : binZT - pow(2, zT_->width() - 1);
            const int cot = binCot < pow(2, cot_->width() - 1) ? binCot + pow(2, cot_->width() - 1)
                                                               : binCot - pow(2, cot_->width() - 1);
            const vector<int>& layers = layerEncoding_[binEta][zT][cot];
            const vector<int>& maybes = maybeLayer_[binEta][zT][cot];
            int layerKF(-1);
            if (encode(layers, layerKF))
              ss << "1" << TTBV(layerKF, widthLayer) << (encode(maybes, layerKF) ? "1" : "0");
            else
              ss << "00000";
            ss << endl;
          }
        }
      }
    }
    fstream file;
    file.open("layerEncoding.txt", ios::out);
    file << ss.rdbuf();
    file.close();
  }

  // encoded layer id for given eta sector, bin in zT, bin in cotThea and decoed layer id, returns -1 if layer incositent with track
  const int LayerEncoding::layerIdKF(int binEta, int binZT, int binCot, int layerId) const {
    const vector<int>& layers = layerEncoding_[binEta][binZT][binCot];
    const auto it = find(layers.begin(), layers.end(), layerId);
    if (it == layers.end())
      return -1;
    int layer = distance(layers.begin(), it);
    if (layer >= setup_->numLayers())
      layer = setup_->numLayers() - 1;
    return layer;
  }

  // pattern of maybe layers for given eta sector, bin in zT and bin in cotThea
  TTBV LayerEncoding::maybePattern(int binEta, int binZT, int binCot) const {
    TTBV ttBV(0, setup_->numLayers());
    const vector<int>& layers = layerEncoding_[binEta][binZT][binCot];
    const vector<int>& maybes = maybeLayer_[binEta][binZT][binCot];
    for (int m : maybes)
      ttBV.set(distance(layers.begin(), find(layers.begin(), layers.end(), m)));
    return ttBV;
  }

}  // namespace trackerTFP
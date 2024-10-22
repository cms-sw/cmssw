#ifndef L1Trigger_TrackerTFP_LayerEncoding_h
#define L1Trigger_TrackerTFP_LayerEncoding_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncodingRcd.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackTrigger/interface/SensorModule.h"

#include <vector>

namespace trackerTFP {

  /*! \class  trackerTFP::LayerEncoding
   *  \brief  Class to encode layer ids for Kalman Filter
   *          Layers consitent with rough r-z track parameters are counted from 0 onwards.
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class LayerEncoding {
  public:
    LayerEncoding() {}
    LayerEncoding(const DataFormats* dataFormats);
    ~LayerEncoding() {}
    // Set of layers in each (zT,tanL) digi Bin of each eta sector numbered 0->N
    const std::vector<int>& layerEncoding(int binEta, int binZT, int binCot) const {
      return layerEncoding_.at(binEta).at(binZT).at(binCot);
    }
    const std::map<int, const tt::SensorModule*>& layerEncodingMap(int binEta, int binZT, int binCot) const {
      return layerEncodingMap_.at(binEta).at(binZT).at(binCot);
    }
    // maybe layers for given ets sector, bin in zT and bin in cotThea
    const std::vector<int>& maybeLayer(int binEta, int binZT, int binCot) const {
      return maybeLayer_.at(binEta).at(binZT).at(binCot);
    }
    // encoded layer id for given eta sector, bin in zT, bin in cotThea and decoed layer id, returns -1 if layer incositent with track
    const int layerIdKF(int binEta, int binZT, int binCot, int layerId) const;
    // pattern of maybe layers for given eta sector, bin in zT and bin in cotThea
    TTBV maybePattern(int binEta, int binZT, int binCot) const;

  private:
    // helper class providing run-time constants
    const tt::Setup* setup_;
    // helper class providing dataformats
    const DataFormats* dataFormats_;
    // data foramt of variable zT
    const DataFormat* zT_;
    // data foramt of variable cotTheta
    const DataFormat* cot_;
    // outer to inner indices: eta sector, bin in zT, bin in cotTheta, layerId
    std::vector<std::vector<std::vector<std::vector<int>>>> layerEncoding_;
    std::vector<std::vector<std::vector<std::map<int, const tt::SensorModule*>>>> layerEncodingMap_;
    // outer to inner indices: eta sector, bin in zT, bin in cotTheta, layerId of maybe layers
    std::vector<std::vector<std::vector<std::vector<int>>>> maybeLayer_;
  };

}  // namespace trackerTFP

EVENTSETUP_DATA_DEFAULT_RECORD(trackerTFP::LayerEncoding, trackerTFP::LayerEncodingRcd);

#endif

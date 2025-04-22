#ifndef L1Trigger_TrackerTFP_LayerEncoding_h
#define L1Trigger_TrackerTFP_LayerEncoding_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackTrigger/interface/SensorModule.h"

#include <vector>

namespace trackerTFP {

  /*! \class  trackerTFP::LayerEncoding
   *  \brief  Class to encode layer ids for Kalman Filter
   *          Layers (1 to 6 for barrel, 11 to 15 for end caps) consitent with rough r-z track parameters are counted from 0 onwards (0 to 7).
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class LayerEncoding {
  public:
    LayerEncoding() {}
    LayerEncoding(const DataFormats* dataFormats);
    ~LayerEncoding() {}
    // Set of layers for given bin in zT
    const std::vector<int>& layerEncoding(int zT) const;
    // Set of layers for given zT in cm
    const std::vector<int>& layerEncoding(double zT) const;
    // pattern of maybe layers for given bin in zT
    const TTBV& maybePattern(int zT) const;
    // pattern of maybe layers for given zT in cm
    const TTBV& maybePattern(double zT) const;
    // fills numPS, num2S, numMissingPS and numMissingPS for given hitPattern and trajectory
    void analyze(
        int hitpattern, double cot, double z0, int& numPS, int& num2S, int& numMissingPS, int& numMissing2S) const;

  private:
    // helper class providing run-time constants
    const tt::Setup* setup_;
    // helper class providing dataformats
    const DataFormats* dataFormats_;
    // data foramt of variable zT
    const DataFormat* zT_;
    // outer to inner indices: bin in zT, layerId
    std::vector<std::vector<int>> layerEncoding_;
    // outer to inner indices: bin in zT, maybe patterns
    std::vector<TTBV> maybePattern_;
    std::vector<int> nullLE_;
    TTBV nullMP_;
  };

}  // namespace trackerTFP

EVENTSETUP_DATA_DEFAULT_RECORD(trackerTFP::LayerEncoding, trackerTFP::DataFormatsRcd);

#endif

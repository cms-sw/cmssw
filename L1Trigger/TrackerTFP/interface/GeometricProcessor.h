#ifndef L1Trigger_TrackerTFP_GeometricProcessor_h
#define L1Trigger_TrackerTFP_GeometricProcessor_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>
#include <deque>

namespace trackerTFP {

  // Class to route Stubs of one region to one stream per sector
  class GeometricProcessor {
  public:
    GeometricProcessor(const tt::Setup* setup_,
                       const DataFormats* dataFormats,
                       const LayerEncoding* layerEncoding,
                       std::vector<StubGP>& stubs);
    ~GeometricProcessor() {}

    // fill output data
    void produce(const std::vector<std::vector<StubPP*>>& streamsIn, std::vector<std::deque<StubGP*>>& streamsOut);

  private:
    // convert stub
    StubGP* produce(const StubPP& stub, int phiT, int zT);
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides dataformats
    const DataFormats* dataFormats_;
    // provides layer encoding
    const LayerEncoding* layerEncoding_;
    // storage of output stubs
    std::vector<StubGP>& stubs_;
    // number of input channel
    int numChannelIn_;
    // number of output channel
    int numChannelOut_;
  };

}  // namespace trackerTFP

#endif

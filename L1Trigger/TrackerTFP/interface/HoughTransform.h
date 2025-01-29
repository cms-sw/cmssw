#ifndef L1Trigger_TrackerTFP_HoughTransform_h
#define L1Trigger_TrackerTFP_HoughTransform_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>
#include <deque>

namespace trackerTFP {

  // Class to find initial rough candidates in r-phi in a region
  class HoughTransform {
  public:
    HoughTransform(const tt::Setup* setup,
                   const DataFormats* dataFormats,
                   const LayerEncoding* layerEncoding,
                   std::vector<StubHT>& stubs);
    ~HoughTransform() {}
    // fill output products
    void produce(const std::vector<std::vector<StubGP*>>& streamsIn, std::vector<std::deque<StubHT*>>& streamsOut);

  private:
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;
    // associate stubs with phiT bins in this inv2R column
    void fillIn(int inv2R, int sector, const std::vector<StubGP*>& input, std::vector<StubHT*>& output);
    // identify tracks
    void readOut(const std::vector<StubHT*>& input, std::deque<StubHT*>& output) const;
    //
    bool noTrack(const TTBV& pattern, int zT) const;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides dataformats
    const DataFormats* dataFormats_;
    //
    const LayerEncoding* layerEncoding_;
    // data format of inv2R
    const DataFormat* inv2R_;
    // data format of phiT
    const DataFormat* phiT_;
    // data format of zT
    const DataFormat* zT_;
    // data format of phi
    const DataFormat* phi_;
    // data format of z
    const DataFormat* z_;
    // container of stubs
    std::vector<StubHT>& stubs_;
    // number of input channel
    int numChannelIn_;
    // number of output channel
    int numChannelOut_;
    //
    int chan_;
    //
    int mux_;
  };

}  // namespace trackerTFP

#endif

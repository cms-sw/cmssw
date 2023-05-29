#ifndef L1Trigger_TrackerTFP_ZHoughTransform_h
#define L1Trigger_TrackerTFP_ZHoughTransform_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>
#include <deque>

namespace trackerTFP {

  // Class to refine MHT track candidates in r-z
  class ZHoughTransform {
  public:
    ZHoughTransform(const edm::ParameterSet& iConfig,
                    const tt::Setup* setup,
                    const DataFormats* dataFormats,
                    int region);
    ~ZHoughTransform() {}

    // read in and organize input product (fill vector input_)
    void consume(const tt::StreamsStub& streams);
    // fill output products
    void produce(tt::StreamsStub& accepted, tt::StreamsStub& lost);

  private:
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;
    // perform finer pattern recognition per track
    void fill(int channel, const std::deque<StubZHT*>& input, std::vector<std::deque<StubZHT*>>& output);
    // Static load balancing of inputs: mux 4 streams to 1 stream
    void slb(std::vector<std::deque<StubZHT*>>& inputs, std::deque<StubZHT*>& accepted, tt::StreamStub& lost) const;
    //
    void merge(std::deque<StubZHT*>& stubs, tt::StreamStub& stream) const;

    // true if truncation is enbaled
    bool enableTruncation_;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides dataformats
    const DataFormats* dataFormats_;
    // processing region (0 - 8)
    int region_;
    // container of in- and output stubs
    std::vector<StubZHT> stubsZHT_;
    // h/w liked organized pointer to input stubs
    std::vector<std::vector<StubZHT*>> input_;
    //
    int stage_;
  };

}  // namespace trackerTFP

#endif
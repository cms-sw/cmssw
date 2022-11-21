#ifndef L1Trigger_TrackerTFP_MiniHoughTransform_h
#define L1Trigger_TrackerTFP_MiniHoughTransform_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>
#include <set>
#include <deque>

namespace trackerTFP {

  // Class to refine HT track candidates in r-phi, by subdividing each HT cell into a finer granularity array
  class MiniHoughTransform {
  public:
    MiniHoughTransform(const edm::ParameterSet& iConfig,
                       const tt::Setup* setup,
                       const DataFormats* dataFormats,
                       int region);
    ~MiniHoughTransform() {}

    // read in and organize input product (fill vector input_)
    void consume(const tt::StreamsStub& streams);
    // fill output products
    void produce(tt::StreamsStub& accepted, tt::StreamsStub& lost);

  private:
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;
    // perform finer pattern recognition per track
    void fill(int channel, const std::vector<StubHT*>& input, std::vector<std::deque<StubMHT*>>& output);
    // Static load balancing of inputs: mux 4 streams to 1 stream
    void slb(std::vector<std::deque<StubMHT*>>& inputs, std::vector<StubMHT*>& accepted, tt::StreamStub& lost) const;
    // Dynamic load balancing of inputs: swapping parts of streams to balance the amount of tracks per stream
    void dlb(std::vector<std::vector<StubMHT*>>& streams) const;

    // true if truncation is enbaled
    bool enableTruncation_;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides dataformats
    const DataFormats* dataFormats_;
    // dataformat of inv2R
    DataFormat inv2R_;
    // dataformat of phiT
    DataFormat phiT_;
    // processing region (0 - 8)
    int region_;
    // number of inv2R bins used in HT
    int numBinsInv2R_;
    // number of cells used in MHT
    int numCells_;
    // number of dynamic load balancing nodes
    int numNodes_;
    // number of channel per dynamic load balancing node
    int numChannel_;
    // container of input stubs
    std::vector<StubHT> stubsHT_;
    // container of output stubs
    std::vector<StubMHT> stubsMHT_;
    // h/w liked organized pointer to input stubs
    std::vector<std::vector<StubHT*>> input_;
  };

}  // namespace trackerTFP

#endif
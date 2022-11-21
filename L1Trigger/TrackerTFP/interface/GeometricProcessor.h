#ifndef L1Trigger_TrackerTFP_GeometricProcessor_h
#define L1Trigger_TrackerTFP_GeometricProcessor_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"

#include <vector>
#include <deque>

namespace trackerTFP {

  // Class to route Stubs of one region to one stream per sector
  class GeometricProcessor {
  public:
    GeometricProcessor(const edm::ParameterSet& iConfig,
                       const tt::Setup* setup_,
                       const DataFormats* dataFormats,
                       int region);
    ~GeometricProcessor() {}

    // read in and organize input product (fill vector input_)
    void consume(const TTDTC& ttDTC);
    // fill output products
    void produce(tt::StreamsStub& accepted, tt::StreamsStub& lost);

  private:
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;

    // true if truncation is enbaled
    bool enableTruncation_;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides dataformats
    const DataFormats* dataFormats_;
    // processing region (0 - 8)
    const int region_;
    // storage of input stubs
    std::vector<StubPP> stubsPP_;
    // storage of output stubs
    std::vector<StubGP> stubsGP_;
    // h/w liked organized pointer to input stubs
    std::vector<std::vector<std::deque<StubPP*>>> input_;
  };

}  // namespace trackerTFP

#endif
#ifndef L1Trigger_TrackerDTC_DTC_h
#define L1Trigger_TrackerDTC_DTC_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "L1Trigger/TrackerDTC/interface/Stub.h"

#include <vector>
#include <deque>

namespace trackerDTC {

  class Settings;
  class Module;

  // representation of an outer tracker DTC board
  class DTC {
  private:
    typedef std::deque<Stub*> Stubs;
    typedef std::vector<Stubs> Stubss;
    typedef std::vector<Stubss> Stubsss;

  public:
    DTC(Settings* settings, int nStubs);
    ~DTC() {}
    // convert and assign TTStubRef to DTC routing block channel
    void consume(const std::vector<TTStubRef>& ttStubRefStream, Module* module);
    // board level routing in two steps and product filling
    void produce(TTDTC& product, int dtcId);

  private:
    // router step 1: merges stubs of all modules connected to one routing block into one stream
    void merge(Stubss& inputs, Stubs& output);
    // router step 2: merges stubs of all routing blocks and splits stubs into one stream per overlapping region
    void split(Stubss& inputs, Stubss& outputs);
    // pop_front function which additionally returns copy of deleted front
    Stub* pop_front(Stubs& stubs);

    // helper class to store configurations
    Settings* settings_;
    // container of stubs on this DTC
    std::vector<Stub> stubs_;
  };

}  // namespace trackerDTC

#endif
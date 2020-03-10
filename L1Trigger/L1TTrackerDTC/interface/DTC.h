#ifndef __L1TTrackerDTC_DTC_H__
#define __L1TTrackerDTC_DTC_H__

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>
#include <deque>

namespace L1TTrackerDTC {

  class Settings;
  class Module;
  class Stub;

  // representation of an outer tracker DTC board
  class DTC {
  private:
    typedef std::deque<Stub*> Stubs;
    typedef std::vector<Stubs> Stubss;
    typedef std::vector<Stubss> Stubsss;

  public:
    DTC(Settings* settings, const int& dtcId, const std::vector<Module*>& modules, const int& nStubs);

    ~DTC();

    // convert and assign TTStubRef to DTC routing block channel
    void consume(const std::vector<TTStubRef>& ttStubRefStream, const int& channelId);

    // board level routing in two steps and product filling
    void produce(TTDTC& product);

  private:
    // new pop_front function which additionally returns copy of deleted front
    Stub* pop_front(Stubs& stubs);

    // router step 1: merges stubs of all modules connected to one routing block into one stream
    void merge(Stubss& inputs, Stubs& output);

    // router step 2: merges stubs of all routing blocks and splits stubs into one stream per overlapping region
    void split(Stubss& inputs, Stubss& outputs);

  private:
    Settings* settings_;  // helper class to store configurations
    int region_;          // outer tracker region [0-8]
    int board_;           // outer tracker dtc id per region [0-23]

    std::vector<Module*> modules_;  // container of sensor modules connected to this DTC
    std::vector<Stub*> stubs_;      // container of dynamic allocated stubs on this DTC
  };

}  // namespace L1TTrackerDTC

#endif
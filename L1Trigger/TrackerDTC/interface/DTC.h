#ifndef L1Trigger_TrackerDTC_DTC_h
#define L1Trigger_TrackerDTC_DTC_h

#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"
#include "L1Trigger/TrackerDTC/interface/Stub.h"

#include <vector>
#include <deque>

namespace trackerDTC {

  /*! \class  trackerDTC::DTC
   *  \brief  Class to represent an outer tracker DTC board
   *  \author Thomas Schuh
   *  \date   2020, Jan
   */
  class DTC {
  private:
    typedef std::deque<Stub*> Stubs;
    typedef std::vector<Stubs> Stubss;
    typedef std::vector<Stubss> Stubsss;

  public:
    DTC(const edm::ParameterSet& iConfig,
        const tt::Setup* setup,
        const LayerEncoding* layerEncoding,
        int dtcId,
        const std::vector<std::vector<TTStubRef>>& stubsDTC);
    ~DTC() {}
    // board level routing in two steps and products filling
    void produce(TTDTC& accepted, TTDTC& lost);

  private:
    // router step 1: merges stubs of all modules connected to one routing block into one stream
    void merge(Stubss& inputs, Stubs& output, Stubs& lost);
    // router step 2: merges stubs of all routing blocks and splits stubs into one stream per overlapping region
    void split(Stubss& inputs, Stubss& outputs);
    // conversion from Stubss to TTDTC
    void produce(const Stubss& stubss, TTDTC& product);
    // pop_front function which additionally returns copy of deleted front
    Stub* pop_front(Stubs& stubs);
    // helper class to store configurations
    const tt::Setup* setup_;
    // enables emulation of truncation
    bool enableTruncation_;
    // outer tracker detector region [0-8]
    int region_;
    // outer tracker dtc id in region [0-23]
    int board_;
    // container of modules connected to this DTC
    std::vector<tt::SensorModule*> modules_;
    // container of stubs on this DTC
    std::vector<Stub> stubs_;
    // input stubs organised in routing blocks [0..1] and channel [0..35]
    Stubsss input_;
    // lost stubs organised in dtc output channel [0..1]
    Stubss lost_;
  };

}  // namespace trackerDTC

#endif
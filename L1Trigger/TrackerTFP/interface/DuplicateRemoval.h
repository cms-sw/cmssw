#ifndef L1Trigger_TrackerTFP_DuplicateRemoval_h
#define L1Trigger_TrackerTFP_DuplicateRemoval_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>

namespace trackerTFP {

  // Class to do duplicate removal in a region.
  class DuplicateRemoval {
  public:
    DuplicateRemoval(const edm::ParameterSet& iConfig,
                     const tt::Setup* setup,
                     const DataFormats* dataFormats,
                     std::vector<TrackDR>& tracks,
                     std::vector<StubDR>& stubs);
    ~DuplicateRemoval() {}
    // fill output products
    void produce(const std::vector<std::vector<TrackKF*>>& tracksIn,
                 const std::vector<std::vector<StubKF*>>& stubsIn,
                 std::vector<std::vector<TrackDR*>>& tracksOut,
                 std::vector<std::vector<StubDR*>>& stubsOut);

  private:
    struct Track {
      Track(TrackKF* track, const std::vector<StubKF*>& stubs, bool match, int inv2R, int phiT, int zT)
          : track_(track), stubs_(stubs), match_(match), inv2R_(inv2R), phiT_(phiT), zT_(zT) {}
      //
      TrackKF* track_;
      //
      std::vector<StubKF*> stubs_;
      //
      bool match_;
      //
      int inv2R_;
      //
      int phiT_;
      //
      int zT_;
    };
    // true if truncation is enbaled
    bool enableTruncation_;
    // provides run-time constants
    const tt::Setup* setup_;
    // provides dataformats
    const DataFormats* dataFormats_;
    // container of output tracks
    std::vector<TrackDR>& tracks_;
    // container of output stubs
    std::vector<StubDR>& stubs_;
  };

}  // namespace trackerTFP

#endif
#ifndef L1Trigger_TrackerTFP_DistServer_h__
#define L1Trigger_TrackerTFP_DistServer_h__

#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <vector>

namespace trackerTFP {

  class DistServer {
  public:
    DistServer(unsigned int nInputs, unsigned int nOutputs, unsigned int nInterleaving);
    ~DistServer() {}

    TrackKFOutSAPtrCollection clock(TrackKFOutSAPtrCollection& inputs);

    unsigned int nInputs() const { return nInputs_; }
    unsigned int nOutputs() const { return nOutputs_; }
    unsigned int nInterleaving() const { return nInterleaving_; }
    std::vector<std::vector<unsigned int> >& addr() { return addr_; }
    TrackKFOutSAPtrCollections& inputs() { return inputs_; }

  private:
    unsigned int nInputs_;
    unsigned int nOutputs_;
    unsigned int nInterleaving_;

    TrackKFOutSAPtrCollections inputs_;
    std::vector<std::vector<unsigned int> > addr_;
  };
}  // namespace trackerTFP

#endif
#ifndef L1Trigger_Phase2L1ParticleFlow_deregionizer_input_h
#define L1Trigger_Phase2L1ParticleFlow_deregionizer_input_h

#include <vector>
#include <algorithm>
#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  class DeregionizerInput {
  public:
    // struct to represent how each correlator layer 1 board output is structured
    struct BoardInfo {
      uint nOutputFramesPerBX_;
      uint nPuppiFramesPerRegion_;
      uint nLinksPuppi_;
      uint nPuppiPerRegion_;
      uint order_;
      std::vector<uint> regions_;
    };

    // struct to represent how each puppi object is positioned in the input frame
    struct LinkPlacementInfo {
      uint board_;
      uint link_;
      uint clock_cycle_;
      // for sorting
      bool operator<(const LinkPlacementInfo &other) const {
        bool cc_lt = this->clock_cycle_ < other.clock_cycle_;
        bool cc_eq = this->clock_cycle_ == other.clock_cycle_;
        bool board_lt = this->board_ < other.board_;
        bool board_eq = this->board_ == other.board_;
        bool link_lt = this->link_ < other.link_;
        return cc_eq ? (board_eq ? link_lt : board_lt) : cc_lt;
      }
    };
    typedef LinkPlacementInfo LPI;
    typedef std::pair<l1ct::PuppiObjEmu, LPI> PlacedPuppi;

    std::vector<BoardInfo> boardInfos_;

    // note: this one for use in standalone testbench
    DeregionizerInput(std::vector<BoardInfo> boardInfos) : boardInfos_(boardInfos) {}

    // note: this one will work only in CMSSW
    DeregionizerInput(const std::vector<edm::ParameterSet> linkConfigs);

    ~DeregionizerInput(){};

    std::vector<std::pair<l1ct::PuppiObjEmu, LPI>> inputOrderInfo(
        const std::vector<l1ct::OutputRegion> &inputRegions) const;
    std::vector<std::vector<std::vector<l1ct::PuppiObjEmu>>> orderInputs(
        const std::vector<l1ct::OutputRegion> &inputRegions) const;

    void setDebug(bool debug = true) { debug_ = debug; }

  private:
    bool debug_ = false;
    // these are not configurable in current design
    uint nInputFramesPerBX_ = 9;
    uint tmuxFactor_ = 6;
  };

}  // namespace l1ct

#endif

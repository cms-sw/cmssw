#ifndef tdr_regionizer_ref_h
#define tdr_regionizer_ref_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/regionizer_base_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/tdr_regionizer_elements_ref.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {
  class TDRRegionizerEmulator : public RegionizerEmulator {
  public:
    TDRRegionizerEmulator(uint32_t ntk,
                          uint32_t ncalo,
                          uint32_t nem,
                          uint32_t nmu,
                          uint32_t nclocks,
                          std::vector<int32_t> bigRegionEdges,
                          bool dosort);

    // note: this one will work only in CMSSW
    TDRRegionizerEmulator(const edm::ParameterSet& iConfig);

    ~TDRRegionizerEmulator() override;

    static edm::ParameterSetDescription getParameterSetDescription();

    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) override;

  private:
    /// The nubmer of barrel big regions (boards)
    uint32_t nBigRegions_;
    /// The maximum number of objects of each type to output per small region
    uint32_t ntk_, ncalo_, nem_, nmu_;
    /// The number of clocks to receive all data of an event (TMUX18 = 162)
    uint32_t nclocks_;

    /// The phi edges of the big regions (boards); one greater than the number of boards
    std::vector<int32_t> bigRegionEdges_;

    bool dosort_;

    /// The number of eta and phi small regions in a big region (board)
    uint32_t netaInBR_, nphiInBR_;

    bool init_;  // has initialization happened

    std::vector<tdr_regionizer::Regionizer<l1ct::TkObjEmu>> tkRegionizers_;
    std::vector<tdr_regionizer::Regionizer<l1ct::HadCaloObjEmu>> hadCaloRegionizers_;
    std::vector<tdr_regionizer::Regionizer<l1ct::EmCaloObjEmu>> emCaloRegionizers_;
    std::vector<tdr_regionizer::Regionizer<l1ct::MuObjEmu>> muRegionizers_;
  };

}  // namespace l1ct

#endif

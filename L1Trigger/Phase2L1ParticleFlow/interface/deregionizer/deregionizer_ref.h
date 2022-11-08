#ifndef L1Trigger_Phase2L1ParticleFlow_deregionizer_ref_h
#define L1Trigger_Phase2L1ParticleFlow_deregionizer_ref_h

#include <vector>
#include "deregionizer_input.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  class DeregionizerEmulator {
  public:
    DeregionizerEmulator(const unsigned int nPuppiFinalBuffer = 128,
                         const unsigned int nPuppiPerClk = 6,
                         const unsigned int nPuppiFirstBuffers = 12,
                         const unsigned int nPuppiSecondBuffers = 32,
                         const unsigned int nPuppiThirdBuffers = 64);

    // note: this one will work only in CMSSW
    DeregionizerEmulator(const edm::ParameterSet &iConfig);

    ~DeregionizerEmulator(){};

    void setDebug(bool debug = true) { debug_ = debug; }

    void run(std::vector<std::vector<std::vector<l1ct::PuppiObjEmu>>> in,
             std::vector<l1ct::PuppiObjEmu> &out,
             std::vector<l1ct::PuppiObjEmu> &truncated);

  private:
    unsigned int nPuppiFinalBuffer_, nPuppiPerClk_, nPuppiFirstBuffers_, nPuppiSecondBuffers_, nPuppiThirdBuffers_;
    bool debug_;

    static std::vector<l1ct::PuppiObjEmu> mergeXtoY(const unsigned int X,
                                                    const unsigned int Y,
                                                    const std::vector<l1ct::PuppiObjEmu> &inLeft,
                                                    const std::vector<l1ct::PuppiObjEmu> &inRight);

    static std::vector<l1ct::PuppiObjEmu> mergeXtoY(const std::vector<l1ct::PuppiObjEmu> &inLeft,
                                                    const std::vector<l1ct::PuppiObjEmu> &inRight);

    static void accumulateToY(const unsigned int Y,
                              const std::vector<l1ct::PuppiObjEmu> &in,
                              std::vector<l1ct::PuppiObjEmu> &out,
                              std::vector<l1ct::PuppiObjEmu> &truncated);
  };

}  // namespace l1ct

#endif

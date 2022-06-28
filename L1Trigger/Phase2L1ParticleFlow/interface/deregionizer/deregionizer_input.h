#ifndef L1Trigger_Phase2L1ParticleFlow_deregionizer_input_h
#define L1Trigger_Phase2L1ParticleFlow_deregionizer_input_h

#include <vector>
#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

namespace l1ct {

  class DeregionizerInput {
  public:
    static const unsigned int nEtaRegions =
        6 /*7 with HF*/;  // Fold ([-0.5,0.0] and [0.0,+0.5]) and ([+-0.5,+-1.0] and [+-1.0,+-1.5]) eta slices into phi
    static const unsigned int nPhiRegions =
        18;  // 9 phi slices * 2 to account for the barrel having x2 PF regions per eta slice in the barrel

    DeregionizerInput(std::vector<float> &regionEtaCenter,
                      std::vector<float> &regionPhiCenter,
                      const std::vector<l1ct::OutputRegion> &inputRegions);

    void setDebug(bool debug = true) { debug_ = debug; }

    enum regionIndex {
      centralBarl = 0,
      negBarl = 1,
      posBarl = 2,
      negHGCal = 3,
      posHGCal = 4,
      forwardHGCal = 5 /*, HF = 6*/
    };
    void orderRegions(int order[nEtaRegions]);

    const std::vector<std::vector<std::vector<l1ct::PuppiObjEmu> > > &orderedInRegionsPuppis() const {
      return orderedInRegionsPuppis_;
    };

  private:
    std::vector<float> regionEtaCenter_;
    std::vector<float> regionPhiCenter_;
    std::vector<std::vector<std::vector<l1ct::PuppiObjEmu> > > orderedInRegionsPuppis_;

    bool debug_ = false;

    unsigned int orderRegionsInPhi(const float eta, const float phi, const float etaComp) const;
    void initRegions(const std::vector<l1ct::OutputRegion> &inputRegions);
  };

}  // namespace l1ct

#endif

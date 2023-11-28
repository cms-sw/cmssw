#ifndef PFALGODUMMY_REF_H
#define PFALGODUMMY_REF_H

#include "pfalgo_common_ref.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {

  class PFAlgoDummyEmulator : public PFAlgoEmulatorBase {
  public:
    PFAlgoDummyEmulator(unsigned int nCalo, unsigned int nMu) : PFAlgoEmulatorBase(0, nCalo, nMu, 0, 0, 0, 0, 0) {}

    // note: this one will work only in CMSSW
    PFAlgoDummyEmulator(const edm::ParameterSet& iConfig);

    ~PFAlgoDummyEmulator() override {}

    static edm::ParameterSetDescription getParameterSetDescription();

    void run(const PFInputRegion& in, OutputRegion& out) const override;

    /// moves all objects from out.pfphoton to the beginning of out.pfneutral: nothing to do for this algo
    void mergeNeutrals(OutputRegion& out) const override {}
  };

}  // namespace l1ct

#endif

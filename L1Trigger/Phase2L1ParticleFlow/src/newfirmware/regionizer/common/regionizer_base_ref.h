#ifndef REGIONIZER_BASE_REF_H
#define REGIONIZER_BASE_REF_H

#include "../../dataformats/layer1_emulator.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  class RegionizerEmulator {
  public:
    RegionizerEmulator(bool useAlsoVtxCoords = true) : useAlsoVtxCoords_(useAlsoVtxCoords), debug_(false) {}
    RegionizerEmulator(const edm::ParameterSet& iConfig);

    virtual ~RegionizerEmulator();

    void setDebug(bool debug = true) { debug_ = debug; }

    virtual void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) {}
    virtual void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out);

  protected:
    bool useAlsoVtxCoords_;
    bool debug_;
  };

}  // namespace l1ct
#endif

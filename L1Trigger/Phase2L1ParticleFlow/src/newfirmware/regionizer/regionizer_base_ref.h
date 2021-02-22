#ifndef REGIONIZER_BASE_REF_H
#define REGIONIZER_BASE_REF_H

#ifdef CMSSW_GIT_HASH
#include "../dataformats/layer1_emulator.h"
#else
#include "../../dataformats/layer1_emulator.h"
#endif

namespace l1ct {

  class RegionizerEmulator {
  public:
    RegionizerEmulator() : debug_(false) {}

    virtual ~RegionizerEmulator();

    void setDebug(bool debug = true) { debug_ = debug; }

    virtual void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) const;

  protected:
    bool debug_;
  };

}  // namespace l1ct
#endif

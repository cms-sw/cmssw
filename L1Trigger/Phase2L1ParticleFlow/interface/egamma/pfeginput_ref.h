#ifndef PFEGINPUT_REF_H
#define PFEGINPUT_REF_H

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "DataFormats/L1TParticleFlow/interface/egamma.h"
#include "DataFormats/L1TParticleFlow/interface/pf.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  struct EGInputSelectorEmuConfig {
    EGInputSelectorEmuConfig(const edm::ParameterSet &iConfig);
    EGInputSelectorEmuConfig(unsigned int emIdMask, unsigned int nHADCALO_IN, unsigned int nEMCALO_OUT, int debug)
        : idMask(emIdMask), nHADCALO_IN(nHADCALO_IN), nEMCALO_OUT(nEMCALO_OUT), debug(debug) {}

    emid_t idMask;
    unsigned int nHADCALO_IN;
    unsigned int nEMCALO_OUT;

    int debug;
  };

  class EGInputSelectorEmulator {
  public:
    EGInputSelectorEmulator(const EGInputSelectorEmuConfig &config) : cfg(config), debug_(cfg.debug) {}

    virtual ~EGInputSelectorEmulator() {}

    void toFirmware(const PFInputRegion &in, HadCaloObj hadcalo[/*nCALO*/]) const;
    void toFirmware(const std::vector<l1ct::EmCaloObjEmu> &emcalo_sel, l1ct::EmCaloObj emcalo[]) const;

    void select_eginput(const l1ct::HadCaloObjEmu &in, l1ct::EmCaloObjEmu &out, bool &valid_out) const;
    void select_eginputs(const std::vector<l1ct::HadCaloObjEmu> &hadcalo_in,
                         std::vector<l1ct::EmCaloObjEmu> &emcalo_sel) const;

    /// if the hadcalo passes the EM selection, do the conversion, otherwise zero-out the result
    void select_or_clear(const l1ct::HadCaloObjEmu &hadcalo_in, l1ct::EmCaloObjEmu &emcalo_out) const;

    /// apply select_or_clear on all elements of the input vector
    void select_or_clear(const std::vector<l1ct::HadCaloObjEmu> &hadcalo_in,
                         std::vector<l1ct::EmCaloObjEmu> &emcalo_out) const;

    // void run(const PFInputRegion &in, OutputRegion &out) const;

    void setDebug(int debug) { debug_ = debug; }

  private:
    EGInputSelectorEmuConfig cfg;
    int debug_;
  };
}  // namespace l1ct

#endif

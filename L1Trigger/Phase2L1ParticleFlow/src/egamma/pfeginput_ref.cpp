#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/pfeginput_ref.h"

using namespace l1ct;

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"

l1ct::EGInputSelectorEmuConfig::EGInputSelectorEmuConfig(const edm::ParameterSet &pset)
    : idMask(pset.getParameter<uint32_t>("emIDMask")),
      nHADCALO_IN(pset.getParameter<uint32_t>("nHADCALO_IN")),
      nEMCALO_OUT(pset.getParameter<uint32_t>("nEMCALO_OUT")),
      debug(pset.getUntrackedParameter<uint32_t>("debug", 0)) {}

#endif

void EGInputSelectorEmulator::toFirmware(const PFInputRegion &in, HadCaloObj hadcalo[/*nCALO*/]) const {
  l1ct::toFirmware(in.hadcalo, cfg.nHADCALO_IN, hadcalo);
}

void EGInputSelectorEmulator::toFirmware(const std::vector<EmCaloObjEmu> &emcalo_sel, EmCaloObj emcalo[]) const {
  l1ct::toFirmware(emcalo_sel, cfg.nEMCALO_OUT, emcalo);
}

void EGInputSelectorEmulator::select_eginput(const l1ct::HadCaloObjEmu &in,
                                             l1ct::EmCaloObjEmu &out,
                                             bool &valid_out) const {
  out.src = in.src;
  out.hwPt = in.hwEmPt;
  out.hwEta = in.hwEta;
  out.hwPhi = in.hwPhi;
  out.hwPtErr = 0;
  // shift to get rid of PFEM ID bit (more usable final EG quality)
  out.hwEmID = (in.hwEmID >> 1);
  valid_out = (in.hwEmID & cfg.idMask) != 0;
}

void EGInputSelectorEmulator::select_eginputs(const std::vector<HadCaloObjEmu> &hadcalo_in,
                                              std::vector<EmCaloObjEmu> &emcalo_sel) const {
  for (int ic = 0, nc = hadcalo_in.size(); ic < nc; ++ic) {
    if (emcalo_sel.size() == cfg.nEMCALO_OUT)
      break;
    bool valid = false;
    EmCaloObjEmu out;
    select_eginput(hadcalo_in[ic], out, valid);
    if (valid) {
      emcalo_sel.push_back(out);
    }
  }
}

void EGInputSelectorEmulator::select_or_clear(const HadCaloObjEmu &hadcalo_in, EmCaloObjEmu &emcalo_out) const {
  bool valid = false;
  select_eginput(hadcalo_in, emcalo_out, valid);
  if (!valid)
    emcalo_out.clear();
}

void EGInputSelectorEmulator::select_or_clear(const std::vector<HadCaloObjEmu> &hadcalo_in,
                                              std::vector<EmCaloObjEmu> &emcalo_out) const {
  emcalo_out.resize(hadcalo_in.size());
  for (int ic = 0, nc = hadcalo_in.size(); ic < nc; ++ic) {
    select_or_clear(hadcalo_in[ic], emcalo_out[ic]);
  }
}

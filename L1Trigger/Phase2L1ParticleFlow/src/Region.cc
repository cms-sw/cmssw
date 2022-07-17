#include "L1Trigger/Phase2L1ParticleFlow/interface/Region.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include <algorithm>
#include <cassert>

const char *l1tpf_impl::Region::inputTypeName(int type) {
  switch (InputType(type)) {
    case calo_type:
      return "Calo";
    case emcalo_type:
      return "EmCalo";
    case track_type:
      return "TK";
    case l1mu_type:
      return "Mu";
    case n_input_types:
      throw cms::Exception(
          "LogicError", "n_input_types is not a type to be used, but only a compile-time const for iterating on types");
  }
  return "NO_SUCH_INPUT_TYPE";
}
const char *l1tpf_impl::Region::outputTypeName(int type) {
  switch (OutputType(type)) {
    case any_type:
      return "";
    case charged_type:
      return "Charged";
    case neutral_type:
      return "Neutral";
    case electron_type:
      return "Electron";
    case pfmuon_type:
      return "Muon";
    case charged_hadron_type:
      return "ChargedHadron";
    case neutral_hadron_type:
      return "NeutralHadron";
    case photon_type:
      return "Photon";
    case n_output_types:
      throw cms::Exception(
          "LogicError",
          "n_output_types is not a type to be used, but only a compile-time const for iterating on types");
  }
  return "NO_SUCH_OUTPUT_TYPE";
}

unsigned int l1tpf_impl::Region::nInput(InputType type) const {
  switch (type) {
    case calo_type:
      return calo.size();
    case emcalo_type:
      return emcalo.size();
    case track_type:
      return track.size();
    case l1mu_type:
      return muon.size();
    case n_input_types:
      throw cms::Exception(
          "LogicError", "n_input_types is not a type to be used, but only a compile-time const for iterating on types");
  }
  return 9999;
}

unsigned int l1tpf_impl::Region::nOutput(OutputType type, bool usePuppi, bool fiducial) const {
  unsigned int ret = 0;
  for (const auto &p : (usePuppi ? puppi : pf)) {
    if (p.hwPt <= 0)
      continue;
    if (fiducial && !fiducialLocal(p.floatEta(), p.floatPhi()))
      continue;
    switch (type) {
      case any_type:
        ret++;
        break;
      case charged_type:
        if (p.intCharge() != 0)
          ret++;
        break;
      case neutral_type:
        if (p.intCharge() == 0)
          ret++;
        break;
      case electron_type:
        if (p.hwId == l1t::PFCandidate::Electron)
          ret++;
        break;
      case pfmuon_type:
        if (p.hwId == l1t::PFCandidate::Muon)
          ret++;
        break;
      case charged_hadron_type:
        if (p.hwId == l1t::PFCandidate::ChargedHadron)
          ret++;
        break;
      case neutral_hadron_type:
        if (p.hwId == l1t::PFCandidate::NeutralHadron)
          ret++;
        break;
      case photon_type:
        if (p.hwId == l1t::PFCandidate::Photon)
          ret++;
        break;
      case n_output_types:
        throw cms::Exception(
            "LogicError",
            "n_output_types is not a type to be used, but only a compile-time const for iterating on types");
    }
  }
  return ret;
}

void l1tpf_impl::Region::inputCrop(bool doSort) {
  if (doSort) {
    std::sort(calo.begin(), calo.end());
    std::sort(emcalo.begin(), emcalo.end());
    std::sort(track.begin(), track.end());
    std::sort(muon.begin(), muon.end());
  }
  if (ncaloMax > 0 && calo.size() > ncaloMax) {
    caloOverflow = calo.size() - ncaloMax;
    calo.resize(ncaloMax);
  }
  if (nemcaloMax > 0 && emcalo.size() > nemcaloMax) {
    emcaloOverflow = emcalo.size() - nemcaloMax;
    emcalo.resize(nemcaloMax);
  }
  if (ntrackMax > 0 && track.size() > ntrackMax) {
    trackOverflow = track.size() - ntrackMax;
    track.resize(ntrackMax);
  }
  if (nmuonMax > 0 && muon.size() > nmuonMax) {
    muonOverflow = muon.size() - nmuonMax;
    muon.resize(nmuonMax);
  }
}

void l1tpf_impl::Region::outputCrop(bool doSort) {
  if (doSort) {
    std::sort(puppi.begin(), puppi.end());
    std::sort(pf.begin(), pf.end());
  }
  if (npuppiMax > 0 && puppi.size() > npuppiMax) {
    puppiOverflow = puppi.size() - npuppiMax;
    puppi.resize(npuppiMax);
  }
  if (npfMax > 0 && pf.size() > npfMax) {
    pfOverflow = pf.size() - npfMax;
    pf.resize(npfMax);
  }
}

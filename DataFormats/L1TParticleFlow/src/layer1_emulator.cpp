#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <algorithm>

#ifdef CMSSW_GIT_HASH
#include "DataFormats/Math/interface/deltaPhi.h"
#else
namespace reco {
  template <typename T>
  inline T reduceRange(T x) {
    T o2pi = 1. / (2. * M_PI);
    if (std::abs(x) <= T(M_PI))
      return x;
    T n = std::round(x * o2pi);
    return x - n * T(2. * M_PI);
  }
  inline double deltaPhi(double phi1, double phi2) { return reduceRange(phi1 - phi2); }
}  // namespace reco
#endif

bool l1ct::HadCaloObjEmu::read(std::fstream& from) {
  src = nullptr;  // not persistent
  return readObj<HadCaloObj>(from, *this);
}
bool l1ct::HadCaloObjEmu::write(std::fstream& to) const { return writeObj<HadCaloObj>(*this, to); }

bool l1ct::EmCaloObjEmu::read(std::fstream& from) {
  src = nullptr;  // not persistent
  return readObj<EmCaloObj>(from, *this);
}
bool l1ct::EmCaloObjEmu::write(std::fstream& to) const { return writeObj<EmCaloObj>(*this, to); }

bool l1ct::TkObjEmu::read(std::fstream& from) {
  src = nullptr;  // not persistent
  return readObj<TkObj>(from, *this) && readVar(from, hwChi2) && readVar(from, simPt) && readVar(from, simCaloEta) &&
         readVar(from, simCaloPhi) && readVar(from, simVtxEta) && readVar(from, simVtxPhi) && readVar(from, simZ0) &&
         readVar(from, simD0);
}
bool l1ct::TkObjEmu::write(std::fstream& to) const {
  return writeObj<TkObj>(*this, to) && writeVar(hwChi2, to) && writeVar(simPt, to) && writeVar(simCaloEta, to) &&
         writeVar(simCaloPhi, to) && writeVar(simVtxEta, to) && writeVar(simVtxPhi, to) && writeVar(simZ0, to) &&
         writeVar(simD0, to);
}

bool l1ct::MuObjEmu::read(std::fstream& from) {
  src = nullptr;  // not persistent
  return readObj<MuObj>(from, *this);
}
bool l1ct::MuObjEmu::write(std::fstream& to) const { return writeObj<MuObj>(*this, to); }

bool l1ct::PFChargedObjEmu::read(std::fstream& from) {
  srcTrack = nullptr;    // not persistent
  srcCluster = nullptr;  // not persistent
  srcMu = nullptr;       // not persistent
  srcCand = nullptr;     // not persistent
  return readObj<PFChargedObj>(from, *this);
}
bool l1ct::PFChargedObjEmu::write(std::fstream& to) const { return writeObj<PFChargedObj>(*this, to); }

bool l1ct::PFNeutralObjEmu::read(std::fstream& from) {
  srcCluster = nullptr;  // not persistent
  srcCand = nullptr;     // not persistent
  return readObj<PFNeutralObj>(from, *this);
}
bool l1ct::PFNeutralObjEmu::write(std::fstream& to) const { return writeObj<PFNeutralObj>(*this, to); }

bool l1ct::PuppiObjEmu::read(std::fstream& from) {
  srcTrack = nullptr;    // not persistent
  srcCluster = nullptr;  // not persistent
  srcMu = nullptr;       // not persistent
  srcCand = nullptr;     // not persistent
  return readObj<PuppiObj>(from, *this);
}
bool l1ct::PuppiObjEmu::write(std::fstream& to) const { return writeObj<PuppiObj>(*this, to); }

bool l1ct::EGIsoObjEmu::read(std::fstream& from) {
  srcCluster = nullptr;  // not persistent
  sta_idx = -1;
  clearIsoVars();  // not persistent
  return readObj<EGIsoObj>(from, *this);
}

bool l1ct::EGIsoObjEmu::write(std::fstream& to) const { return writeObj<EGIsoObj>(*this, to); }

bool l1ct::EGIsoEleObjEmu::read(std::fstream& from) {
  srcCluster = nullptr;
  srcTrack = nullptr;
  sta_idx = -1;
  clearIsoVars();  // not persistent
  return readObj<EGIsoEleObj>(from, *this);
}

bool l1ct::EGIsoEleObjEmu::write(std::fstream& to) const { return writeObj<EGIsoEleObj>(*this, to); }

l1ct::PFRegionEmu::PFRegionEmu(float etaCenter, float phicenter) {
  hwEtaCenter = Scales::makeGlbEta(etaCenter);
  hwPhiCenter = Scales::makeGlbPhi(phicenter);
  hwEtaHalfWidth = 0;
  hwPhiHalfWidth = 0;
  hwEtaExtra = 0;
  hwPhiExtra = 0;
}
l1ct::PFRegionEmu::PFRegionEmu(
    float etamin, float etamax, float phicenter, float phiwidth, float etaextra, float phiextra) {
  glbeta_t hwEtaMin = Scales::makeGlbEtaRoundEven(etamin);
  glbeta_t hwEtaMax = Scales::makeGlbEtaRoundEven(etamax);

  hwEtaCenter = glbeta_t((hwEtaMin + hwEtaMax) / 2);
  hwPhiCenter = Scales::makeGlbPhi(phicenter);
  hwEtaHalfWidth = hwEtaCenter - hwEtaMin;
  hwPhiHalfWidth = Scales::makeGlbPhi(0.5 * phiwidth);
  hwEtaExtra = Scales::makeGlbEta(etaextra);
  hwPhiExtra = Scales::makeGlbPhi(phiextra);
}

bool l1ct::PFRegionEmu::contains(float eta, float phi) const {
  float dphi = reco::deltaPhi(phi, floatPhiCenter());
  return (floatEtaMinExtra() <= eta && eta <= floatEtaMaxExtra() && -floatPhiHalfWidthExtra() <= dphi &&
          dphi <= floatPhiHalfWidthExtra());
}
bool l1ct::PFRegionEmu::containsHw(glbeta_t glbeta, glbphi_t glbphi) const {
  glbeta_t loceta = glbeta - hwEtaCenter;
  ap_int<glbphi_t::width + 1> locphi = glbphi - hwPhiCenter;
  if (locphi > Scales::INTPHI_PI)
    locphi -= Scales::INTPHI_TWOPI;
  else if (locphi <= -Scales::INTPHI_PI)
    locphi += Scales::INTPHI_TWOPI;
  return isInside(loceta, locphi);
}

float l1ct::PFRegionEmu::localEta(float globalEta) const { return globalEta - floatEtaCenter(); }
float l1ct::PFRegionEmu::localPhi(float globalPhi) const { return reco::deltaPhi(globalPhi, floatPhiCenter()); }

bool l1ct::PFRegionEmu::read(std::fstream& from) { return readObj<PFRegion>(from, *this); }
bool l1ct::PFRegionEmu::write(std::fstream& to) const { return writeObj<PFRegion>(*this, to); }

bool l1ct::PVObjEmu::read(std::fstream& from) { return readObj<PVObj>(from, *this); }
bool l1ct::PVObjEmu::write(std::fstream& to) const { return writeObj<PVObj>(*this, to); }

bool l1ct::RawInputs::read(std::fstream& from) {
  uint32_t number;

  if (!readVar(from, number))
    return false;
  track.resize(number);
  for (auto& v : track) {
    if (!(v.region.read(from) && readMany(from, v.obj)))
      return false;
  }

  if (!(muon.region.read(from) && readMany(from, muon.obj)))
    return false;

  if (!readVar(from, number))
    return false;
  hgcalcluster.resize(number);
  for (auto& v : hgcalcluster) {
    if (!(v.region.read(from) && readMany(from, v.obj)))
      return false;
  }

  return true;
}

bool l1ct::RawInputs::write(std::fstream& to) const {
  uint32_t number;

  number = track.size();
  if (!writeVar(number, to))
    return false;
  for (const auto& v : track) {
    if (!(v.region.write(to) && writeMany(v.obj, to)))
      return false;
  }

  if (!(muon.region.write(to) && writeMany(muon.obj, to)))
    return false;

  number = hgcalcluster.size();
  if (!writeVar(number, to))
    return false;
  for (const auto& v : hgcalcluster) {
    if (!(v.region.write(to) && writeMany(v.obj, to)))
      return false;
  }

  return true;
}
void l1ct::RawInputs::clear() {
  for (auto& r : track)
    r.clear();
  muon.clear();
  for (auto& h : hgcalcluster)
    h.clear();
}

bool l1ct::RegionizerDecodedInputs::read(std::fstream& from) {
  uint32_t number;

  if (!readVar(from, number))
    return false;
  hadcalo.resize(number);
  for (auto& v : hadcalo) {
    if (!(v.region.read(from) && readMany(from, v.obj)))
      return false;
  }

  if (!readVar(from, number))
    return false;
  emcalo.resize(number);
  for (auto& v : emcalo) {
    if (!(v.region.read(from) && readMany(from, v.obj)))
      return false;
  }

  if (!readVar(from, number))
    return false;
  track.resize(number);
  for (auto& v : track) {
    if (!(v.region.read(from) && readMany(from, v.obj)))
      return false;
  }

  if (!(muon.region.read(from) && readMany(from, muon.obj)))
    return false;

  return true;
}

bool l1ct::RegionizerDecodedInputs::write(std::fstream& to) const {
  uint32_t number;

  number = hadcalo.size();
  if (!writeVar(number, to))
    return false;
  for (const auto& v : hadcalo) {
    if (!(v.region.write(to) && writeMany(v.obj, to)))
      return false;
  }

  number = emcalo.size();
  if (!writeVar(number, to))
    return false;
  for (const auto& v : emcalo) {
    if (!(v.region.write(to) && writeMany(v.obj, to)))
      return false;
  }

  number = track.size();
  if (!writeVar(number, to))
    return false;
  for (const auto& v : track) {
    if (!(v.region.write(to) && writeMany(v.obj, to)))
      return false;
  }

  if (!(muon.region.write(to) && writeMany(muon.obj, to)))
    return false;

  return true;
}
void l1ct::RegionizerDecodedInputs::clear() {
  for (auto& r : hadcalo)
    r.clear();
  for (auto& r : emcalo)
    r.clear();
  for (auto& r : track)
    r.clear();
  muon.clear();
}

bool l1ct::PFInputRegion::read(std::fstream& from) {
  return region.read(from) && readMany(from, hadcalo) && readMany(from, emcalo) && readMany(from, track) &&
         readMany(from, muon);
}
bool l1ct::PFInputRegion::write(std::fstream& to) const {
  return region.write(to) && writeMany(hadcalo, to) && writeMany(emcalo, to) && writeMany(track, to) &&
         writeMany(muon, to);
}
void l1ct::PFInputRegion::clear() {
  hadcalo.clear();
  emcalo.clear();
  track.clear();
  muon.clear();
}

bool l1ct::OutputRegion::read(std::fstream& from) {
  return readMany(from, pfcharged) && readMany(from, pfneutral) && readMany(from, pfphoton) && readMany(from, pfmuon) &&
         readMany(from, puppi) && readMany(from, egphoton) && readMany(from, egelectron);
}
bool l1ct::OutputRegion::write(std::fstream& to) const {
  return writeMany(pfcharged, to) && writeMany(pfneutral, to) && writeMany(pfphoton, to) && writeMany(pfmuon, to) &&
         writeMany(puppi, to) && writeMany(egphoton, to) && writeMany(egelectron, to);
}

void l1ct::OutputRegion::clear() {
  pfcharged.clear();
  pfphoton.clear();
  pfneutral.clear();
  pfmuon.clear();
  puppi.clear();
  egsta.clear();
  egphoton.clear();
  egelectron.clear();
}

// begin helper functions
namespace {
  template <typename TV>
  unsigned int count_nonnull(const TV& v) {
    typedef typename TV::value_type T;
    return std::count_if(v.begin(), v.end(), [](const T& p) { return p.hwPt > 0; });
  }
  template <typename TV, typename F>
  unsigned int count_nonnull_if(const TV& v, F pred) {
    unsigned int n = 0;
    for (auto& p : v) {
      if (p.hwPt > 0 && pred(p.hwId))
        ++n;
    }
    return n;
  }
}  // namespace
// end helper functions
unsigned int l1ct::OutputRegion::nObj(ObjType type, bool usePuppi) const {
  switch (type) {
    case anyType:
      if (usePuppi)
        return ::count_nonnull(puppi);
      else
        return ::count_nonnull(pfcharged) + ::count_nonnull(pfphoton) + ::count_nonnull(pfneutral);
    case chargedType:
      if (usePuppi)
        return ::count_nonnull_if(puppi, [](const l1ct::ParticleID& i) { return i.charged(); });
      else
        return ::count_nonnull(pfcharged);
    case neutralType:
      if (usePuppi)
        return ::count_nonnull_if(puppi, [](const l1ct::ParticleID& i) { return i.neutral(); });
      else
        return ::count_nonnull(pfphoton) + ::count_nonnull(pfneutral);
    case electronType:
      if (usePuppi)
        return ::count_nonnull_if(puppi, [](const l1ct::ParticleID& i) { return i.isElectron(); });
      else
        return ::count_nonnull_if(pfcharged, [](const l1ct::ParticleID& i) { return i.isElectron(); });
    case muonType:
      if (usePuppi)
        return ::count_nonnull_if(puppi, [](const l1ct::ParticleID& i) { return i.isMuon(); });
      else
        return ::count_nonnull_if(pfcharged, [](const l1ct::ParticleID& i) { return i.isMuon(); });
    case chargedHadronType:
      if (usePuppi)
        return ::count_nonnull_if(puppi, [](const l1ct::ParticleID& i) { return i.isChargedHadron(); });
      else
        return ::count_nonnull_if(pfcharged, [](const l1ct::ParticleID& i) { return i.isChargedHadron(); });
    case neutralHadronType:
      if (usePuppi)
        return ::count_nonnull_if(puppi,
                                  [](const l1ct::ParticleID& i) { return i.rawId() == l1ct::ParticleID::HADZERO; });
      else
        return ::count_nonnull_if(pfneutral,
                                  [](const l1ct::ParticleID& i) { return i.rawId() == l1ct::ParticleID::HADZERO; });
    case photonType:
      if (usePuppi)
        return ::count_nonnull_if(puppi,
                                  [](const l1ct::ParticleID& i) { return i.rawId() == l1ct::ParticleID::PHOTON; });
      else
        return ::count_nonnull_if(pfneutral,
                                  [](const l1ct::ParticleID& i) { return i.rawId() == l1ct::ParticleID::PHOTON; }) +
               ::count_nonnull_if(pfphoton,
                                  [](const l1ct::ParticleID& i) { return i.rawId() == l1ct::ParticleID::PHOTON; });
    case egisoType:
      assert(!usePuppi);
      return ::count_nonnull(egphoton);
    case egisoeleType:
      assert(!usePuppi);
      return ::count_nonnull(egelectron);
    default:
      assert(false);
  }
}

bool l1ct::OutputBoard::read(std::fstream& from) {
  return readVar(from, eta) && readVar(from, phi) && readMany(from, egphoton) && readMany(from, egelectron);
}
bool l1ct::OutputBoard::write(std::fstream& to) const {
  return writeVar(eta, to) && writeVar(phi, to) && writeMany(egphoton, to) && writeMany(egelectron, to);
}

void l1ct::OutputBoard::clear() {
  egphoton.clear();
  egelectron.clear();
}

bool l1ct::Event::read(std::fstream& from) {
  uint32_t version;
  if (!readVar(from, version))
    return false;
  if (version != VERSION) {
    //dbgCout() << "ERROR: version mismatch between this code (" << VERSION << ") and dump file (" << version << ")."
    //          << std::endl;
    //dbgCerr() << "ERROR: version mismatch between this code (" << VERSION << ") and dump file (" << version << ")."
    //          << std::endl;
    abort();
  }
  return readVar(from, run) && readVar(from, lumi) && readVar(from, event) && raw.read(from) && decoded.read(from) &&
         readMany(from, pfinputs) && readMany(from, pvs) && readMany(from, pvs_emu) && readMany(from, out) &&
         readMany(from, board_out);
}
bool l1ct::Event::write(std::fstream& to) const {
  uint32_t version = VERSION;
  return writeVar(version, to) && writeVar(run, to) && writeVar(lumi, to) && writeVar(event, to) && raw.write(to) &&
         decoded.write(to) && writeMany(pfinputs, to) && writeMany(pvs, to) && writeMany(pvs_emu, to) &&
         writeMany(out, to) && writeMany(board_out, to);
}
void l1ct::Event::init(uint32_t arun, uint32_t alumi, uint64_t anevent) {
  clear();
  run = arun;
  lumi = alumi;
  event = anevent;
}
void l1ct::Event::clear() {
  run = 0;
  lumi = 0;
  event = 0;
  raw.clear();
  decoded.clear();
  for (auto& i : pfinputs)
    i.clear();
  pvs.clear();
  pvs_emu.clear();
  for (auto& i : out)
    i.clear();
  for (auto& i : board_out)
    i.clear();
}

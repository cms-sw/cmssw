#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgoBase.h"

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

using namespace l1tpf_impl;

PFAlgoBase::PFAlgoBase(const edm::ParameterSet &iConfig) : debug_(iConfig.getUntrackedParameter<int>("debug", 0)) {}

PFAlgoBase::~PFAlgoBase() {}

void PFAlgoBase::initRegion(Region &r, bool doSort) const {
  r.inputCrop(doSort);
  r.pf.clear();
  r.puppi.clear();
  for (auto &c : r.calo)
    c.used = false;
  for (auto &c : r.emcalo)
    c.used = false;
  for (auto &t : r.track) {
    t.used = false;
    t.muonLink = false;
  }
}

PFParticle &PFAlgoBase::addTrackToPF(std::vector<PFParticle> &pfs, const PropagatedTrack &tk) const {
  PFParticle pf;
  pf.hwPt = tk.hwPt;
  pf.hwEta = tk.hwEta;
  pf.hwPhi = tk.hwPhi;
  pf.hwVtxEta = tk.hwEta;  // FIXME: get from the track
  pf.hwVtxPhi = tk.hwPhi;  // before propagation
  pf.track = tk;
  pf.cluster.hwPt = 0;
  pf.cluster.src = nullptr;
  pf.muonsrc = nullptr;
  pf.hwId = (tk.muonLink ? l1t::PFCandidate::Muon : l1t::PFCandidate::ChargedHadron);
  pf.hwStatus = 0;
  pfs.push_back(pf);
  return pfs.back();
}

PFParticle &PFAlgoBase::addCaloToPF(std::vector<PFParticle> &pfs, const CaloCluster &calo) const {
  PFParticle pf;
  pf.hwPt = calo.hwPt;
  pf.hwEta = calo.hwEta;
  pf.hwPhi = calo.hwPhi;
  pf.hwVtxEta = calo.hwEta;
  pf.hwVtxPhi = calo.hwPhi;
  pf.track.hwPt = 0;
  pf.track.src = nullptr;
  pf.cluster = calo;
  pf.muonsrc = nullptr;
  pf.hwId = (calo.isEM ? l1t::PFCandidate::Photon : l1t::PFCandidate::NeutralHadron);
  pf.hwStatus = 0;
  pfs.push_back(pf);
  return pfs.back();
}

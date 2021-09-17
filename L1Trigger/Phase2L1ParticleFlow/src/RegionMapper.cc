#include "L1Trigger/Phase2L1ParticleFlow/interface/RegionMapper.h"

using namespace l1tpf_impl;

RegionMapper::RegionMapper(const edm::ParameterSet &iConfig) : useRelativeRegionalCoordinates_(false) {
  if (iConfig.existsAs<std::vector<edm::ParameterSet>>("regions")) {
    useRelativeRegionalCoordinates_ = iConfig.getParameter<bool>("useRelativeRegionalCoordinates");
    for (const edm::ParameterSet &preg : iConfig.getParameter<std::vector<edm::ParameterSet>>("regions")) {
      std::vector<double> etaBoundaries = preg.getParameter<std::vector<double>>("etaBoundaries");
      unsigned int phiSlices = preg.getParameter<uint32_t>("phiSlices");
      float etaExtra = preg.getParameter<double>("etaExtra");
      float phiExtra = preg.getParameter<double>("phiExtra");
      float phiWidth = 2 * M_PI / phiSlices;
      unsigned int ncalomax = 0, nemcalomax = 0, ntrackmax = 0, nmuonmax = 0, npfmax = 0, npuppimax = 0;
      if (preg.existsAs<uint32_t>("caloNMax"))
        ncalomax = preg.getParameter<uint32_t>("caloNMax");
      if (preg.existsAs<uint32_t>("emcaloNMax"))
        nemcalomax = preg.getParameter<uint32_t>("emcaloNMax");
      if (preg.existsAs<uint32_t>("trackNMax"))
        ntrackmax = preg.getParameter<uint32_t>("trackNMax");
      if (preg.existsAs<uint32_t>("muonNMax"))
        nmuonmax = preg.getParameter<uint32_t>("muonNMax");
      if (preg.existsAs<uint32_t>("pfNMax"))
        npfmax = preg.getParameter<uint32_t>("pfNMax");
      if (preg.existsAs<uint32_t>("puppiNMax"))
        npuppimax = preg.getParameter<uint32_t>("puppiNMax");
      for (unsigned int ieta = 0, neta = etaBoundaries.size() - 1; ieta < neta; ++ieta) {
        for (unsigned int iphi = 0; iphi < phiSlices; ++iphi) {
          float phiCenter = (iphi + 0.5) * phiWidth - M_PI;
          regions_.push_back(Region(etaBoundaries[ieta],
                                    etaBoundaries[ieta + 1],
                                    phiCenter,
                                    phiWidth,
                                    etaExtra,
                                    phiExtra,
                                    useRelativeRegionalCoordinates_,
                                    ncalomax,
                                    nemcalomax,
                                    ntrackmax,
                                    nmuonmax,
                                    npfmax,
                                    npuppimax));
        }
      }
    }
    std::string trackRegionMode = "TrackAssoMode::any";
    if (iConfig.existsAs<std::string>("trackRegionMode"))
      trackRegionMode = iConfig.getParameter<std::string>("trackRegionMode");
    if (trackRegionMode == "atVertex")
      trackRegionMode_ = TrackAssoMode::atVertex;
    else if (trackRegionMode == "atCalo")
      trackRegionMode_ = TrackAssoMode::atCalo;
    else if (trackRegionMode == "any")
      trackRegionMode_ = TrackAssoMode::any;
    else
      throw cms::Exception(
          "Configuration",
          "Unsupported value for trackRegionMode: " + trackRegionMode + " (allowed are 'atVertex', 'atCalo', 'any')");
  } else {
    // start off with a dummy region
    unsigned int ncalomax = 0, nemcalomax = 0, ntrackmax = 0, nmuonmax = 0, npfmax = 0, npuppimax = 0;
    regions_.emplace_back(-5.5,
                          5.5,
                          0,
                          2 * M_PI,
                          0.5,
                          0.5,
                          useRelativeRegionalCoordinates_,
                          ncalomax,
                          nemcalomax,
                          ntrackmax,
                          nmuonmax,
                          npfmax,
                          npuppimax);
  }
}

void RegionMapper::clear() {
  for (Region &r : regions_)
    r.zero();
  clusterRefMap_.clear();
  trackRefMap_.clear();
  muonRefMap_.clear();
}

void RegionMapper::addTrack(const l1t::PFTrack &t) {
  // now let's be optimistic and make things very simple
  // we propagate in floating point the track to the calo
  // we add the track to the region corresponding to its vertex (eta,phi) coordinates AND its (eta,phi) calo coordinates
  for (Region &r : regions_) {
    bool inside = true;
    switch (trackRegionMode_) {
      case TrackAssoMode::atVertex:
        inside = r.contains(t.eta(), t.phi());
        break;
      case TrackAssoMode::atCalo:
        inside = r.contains(t.caloEta(), t.caloPhi());
        break;
      case TrackAssoMode::any:
        inside = r.contains(t.eta(), t.phi()) || r.contains(t.caloEta(), t.caloPhi());
        break;
    }
    if (inside) {
      PropagatedTrack prop;
      prop.fillInput(t.pt(), r.localEta(t.eta()), r.localPhi(t.phi()), t.charge(), t.vertex().Z(), t.quality(), &t);
      prop.fillPropagated(t.pt(),
                          t.trkPtError(),
                          t.caloPtError(),
                          r.localEta(t.caloEta()),
                          r.localPhi(t.caloPhi()),
                          t.quality(),
                          t.isMuon());
      prop.hwStubs = t.nStubs();
      prop.hwChi2 = round(t.chi2() * 10);
      r.track.push_back(prop);
    }
  }
}
void RegionMapper::addTrack(const l1t::PFTrack &t, l1t::PFTrackRef ref) {
  addTrack(t);
  trackRefMap_[&t] = ref;
}

void RegionMapper::addMuon(const l1t::Muon &mu) {
  // now let's be optimistic and make things very simple
  // we don't propagate anything
  for (Region &r : regions_) {
    if (r.contains(mu.eta(), mu.phi())) {
      Muon prop;
      prop.fill(mu.pt(), r.localEta(mu.eta()), r.localPhi(mu.phi()), mu.charge(), mu.hwQual(), &mu);
      r.muon.push_back(prop);
    }
  }
}

void RegionMapper::addMuon(const l1t::TkMuon &mu) {
  // now let's be optimistic and make things very simple
  // we don't propagate anything
  for (Region &r : regions_) {
    if (r.contains(mu.eta(), mu.phi())) {
      Muon prop;
      prop.fill(mu.pt(), r.localEta(mu.eta()), r.localPhi(mu.phi()), mu.charge(), mu.hwQual());
      r.muon.push_back(prop);
    }
  }
}

void RegionMapper::addMuon(const l1t::Muon &mu, l1t::PFCandidate::MuonRef ref) {
  addMuon(mu);
  muonRefMap_[&mu] = ref;
}

void RegionMapper::addCalo(const l1t::PFCluster &p) {
  if (p.pt() == 0)
    return;
  for (Region &r : regions_) {
    if (r.contains(p.eta(), p.phi())) {
      CaloCluster calo;
      calo.fill(p.pt(), p.emEt(), p.ptError(), r.localEta(p.eta()), r.localPhi(p.phi()), p.isEM(), 0, &p);
      r.calo.push_back(calo);
    }
  }
}
void RegionMapper::addCalo(const l1t::PFCluster &p, l1t::PFClusterRef ref) {
  addCalo(p);
  clusterRefMap_[&p] = ref;
}

void RegionMapper::addEmCalo(const l1t::PFCluster &p) {
  if (p.pt() == 0)
    return;
  for (Region &r : regions_) {
    if (r.contains(p.eta(), p.phi())) {
      CaloCluster calo;
      calo.fill(p.pt(), p.emEt(), p.ptError(), r.localEta(p.eta()), r.localPhi(p.phi()), p.isEM(), 0, &p);
      r.emcalo.push_back(calo);
    }
  }
}
void RegionMapper::addEmCalo(const l1t::PFCluster &p, l1t::PFClusterRef ref) {
  addEmCalo(p);
  clusterRefMap_[&p] = ref;
}

std::unique_ptr<l1t::PFCandidateCollection> RegionMapper::fetch(bool puppi, float ptMin) const {
  auto ret = std::make_unique<l1t::PFCandidateCollection>();
  for (const Region &r : regions_) {
    for (const PFParticle &p : (puppi ? r.puppi : r.pf)) {
      bool inside = true;
      switch (trackRegionMode_) {
        case TrackAssoMode::atVertex:
          inside = r.fiducialLocal(p.floatVtxEta(), p.floatVtxPhi());
          break;
        case TrackAssoMode::atCalo:
          inside = r.fiducialLocal(p.floatEta(), p.floatPhi());
          break;
        case TrackAssoMode::any:
          inside = r.fiducialLocal(p.floatVtxEta(), p.floatVtxPhi());
          break;  // WARNING: this may not be the best choice
      }
      if (!inside)
        continue;
      if (p.floatPt() > ptMin) {
        reco::Particle::PolarLorentzVector p4(
            p.floatPt(), r.globalEta(p.floatVtxEta()), r.globalPhi(p.floatVtxPhi()), 0.13f);
        ret->emplace_back(l1t::PFCandidate::ParticleType(p.hwId), p.intCharge(), p4, p.floatPuppiW());
        ret->back().setVertex(reco::Particle::Point(0, 0, p.floatDZ()));
        ret->back().setStatus(p.hwStatus);
        if (p.cluster.src) {
          auto match = clusterRefMap_.find(p.cluster.src);
          if (match == clusterRefMap_.end()) {
            throw cms::Exception("CorruptData") << "Invalid cluster pointer in PF candidate id " << p.hwId << " pt "
                                                << p4.pt() << " eta " << p4.eta() << " phi " << p4.phi();
          }
          ret->back().setPFCluster(match->second);
        }
        if (p.track.src) {
          auto match = trackRefMap_.find(p.track.src);
          if (match == trackRefMap_.end()) {
            throw cms::Exception("CorruptData") << "Invalid track pointer in PF candidate id " << p.hwId << " pt "
                                                << p4.pt() << " eta " << p4.eta() << " phi " << p4.phi();
          }
          ret->back().setPFTrack(match->second);
        }
        if (p.muonsrc) {
          auto match = muonRefMap_.find(p.muonsrc);
          if (match == muonRefMap_.end()) {
            throw cms::Exception("CorruptData") << "Invalid muon pointer in PF candidate id " << p.hwId << " pt "
                                                << p4.pt() << " eta " << p4.eta() << " phi " << p4.phi();
          }
          ret->back().setMuon(match->second);
        }
      }
    }
  }
  return ret;
}

std::unique_ptr<l1t::PFCandidateCollection> RegionMapper::fetchCalo(float ptMin, bool emcalo) const {
  auto ret = std::make_unique<l1t::PFCandidateCollection>();
  for (const Region &r : regions_) {
    for (const CaloCluster &p : (emcalo ? r.emcalo : r.calo)) {
      if (!r.fiducialLocal(p.floatEta(), p.floatPhi()))
        continue;
      if (p.floatPt() > ptMin) {
        reco::Particle::PolarLorentzVector p4(p.floatPt(), r.globalEta(p.floatEta()), r.globalPhi(p.floatPhi()), 0.13f);
        l1t::PFCandidate::ParticleType kind =
            (p.isEM || emcalo) ? l1t::PFCandidate::Photon : l1t::PFCandidate::NeutralHadron;
        ret->emplace_back(kind, 0, p4);
        if (p.src) {
          auto match = clusterRefMap_.find(p.src);
          if (match == clusterRefMap_.end()) {
            throw cms::Exception("CorruptData")
                << "Invalid cluster pointer in cluster pt " << p4.pt() << " eta " << p4.eta() << " phi " << p4.phi();
          }
          ret->back().setPFCluster(match->second);
        }
      }
    }
  }
  return ret;
}

std::unique_ptr<l1t::PFCandidateCollection> RegionMapper::fetchTracks(float ptMin, bool fromPV) const {
  auto ret = std::make_unique<l1t::PFCandidateCollection>();
  for (const Region &r : regions_) {
    for (const PropagatedTrack &p : r.track) {
      if (fromPV && !p.fromPV)
        continue;
      bool inside = true;
      switch (trackRegionMode_) {
        case TrackAssoMode::atVertex:
          inside = r.fiducialLocal(p.floatVtxEta(), p.floatVtxPhi());
          break;
        case TrackAssoMode::atCalo:
          inside = r.fiducialLocal(p.floatEta(), p.floatPhi());
          break;
        case TrackAssoMode::any:
          inside = r.fiducialLocal(p.floatVtxEta(), p.floatVtxPhi());
          break;  // WARNING: this may not be the best choice
      }
      if (!inside)
        continue;
      if (p.floatPt() > ptMin) {
        reco::Particle::PolarLorentzVector p4(
            p.floatVtxPt(), r.globalEta(p.floatVtxEta()), r.globalPhi(p.floatVtxPhi()), 0.13f);
        l1t::PFCandidate::ParticleType kind = p.muonLink ? l1t::PFCandidate::Muon : l1t::PFCandidate::ChargedHadron;
        ret->emplace_back(kind, p.intCharge(), p4);
        ret->back().setVertex(reco::Particle::Point(0, 0, p.floatDZ()));
        if (p.src) {
          auto match = trackRefMap_.find(p.src);
          if (match == trackRefMap_.end()) {
            throw cms::Exception("CorruptData")
                << "Invalid track pointer in PF track  pt " << p4.pt() << " eta " << p4.eta() << " phi " << p4.phi();
          }
          ret->back().setPFTrack(match->second);
        }
      }
    }
  }
  return ret;
}

std::pair<unsigned, unsigned> RegionMapper::totAndMaxInput(int type) const {
  unsigned ntot = 0, nmax = 0;
  for (const auto &r : regions_) {
    unsigned int ni = r.nInput(Region::InputType(type));
    ntot += ni;
    nmax = std::max(nmax, ni);
  }
  return std::make_pair(ntot, nmax);
}

std::unique_ptr<std::vector<unsigned>> RegionMapper::vecInput(int type) const {
  auto v = std::make_unique<std::vector<unsigned>>();
  for (const auto &r : regions_) {
    unsigned ni = r.nInput(Region::InputType(type));
    v->push_back(ni);
  }
  return v;
}

std::pair<unsigned, unsigned> RegionMapper::totAndMaxOutput(int type, bool puppi) const {
  unsigned ntot = 0, nmax = 0;
  for (const auto &r : regions_) {
    unsigned int ni = r.nOutput(Region::OutputType(type), puppi);
    ntot += ni;
    nmax = std::max(nmax, ni);
  }
  return std::make_pair(ntot, nmax);
}

std::unique_ptr<std::vector<unsigned>> RegionMapper::vecOutput(int type, bool puppi) const {
  auto v = std::make_unique<std::vector<unsigned>>();
  for (const auto &r : regions_) {
    unsigned ni = r.nOutput(Region::OutputType(type), puppi);
    v->push_back(ni);
  }
  return v;
}

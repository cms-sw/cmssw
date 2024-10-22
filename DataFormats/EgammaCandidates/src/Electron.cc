#include "DataFormats/EgammaCandidates/interface/Electron.h"

using namespace reco;

Electron::~Electron() {}

Electron *Electron::clone() const { return new Electron(*this); }

TrackRef Electron::track() const { return track_; }

GsfTrackRef Electron::gsfTrack() const { return gsfTrack_; }

SuperClusterRef Electron::superCluster() const { return superCluster_; }

bool Electron::overlap(const Candidate &c) const {
  const RecoCandidate *o = dynamic_cast<const RecoCandidate *>(&c);
  return (o != nullptr && (checkOverlap(track(), o->track()) || checkOverlap(superCluster(), o->superCluster())));
  return false;
}

bool Electron::isElectron() const { return true; }

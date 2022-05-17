#ifndef RecoEcal_EgammaCoreTools_CalibratedPFCluster_h
#define RecoEcal_EgammaCoreTools_CalibratedPFCluster_h

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

// simple class for associating calibrated energies
class CalibratedPFCluster {
public:
  CalibratedPFCluster(const edm::Ptr<reco::PFCluster>& p) : ptr_(p) {}

  double energy() const { return ptr_->correctedEnergy(); }
  double energy_nocalib() const { return ptr_->energy(); }
  double eta() const { return ptr_->positionREP().eta(); }
  double phi() const { return ptr_->positionREP().phi(); }

  edm::Ptr<reco::PFCluster> ptr() const { return ptr_; }

private:
  edm::Ptr<reco::PFCluster> ptr_;
};

#endif

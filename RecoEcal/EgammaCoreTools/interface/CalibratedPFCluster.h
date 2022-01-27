#ifndef RecoEcal_EgammaCoreTools_CalibratedPFCluster_h
#define RecoEcal_EgammaCoreTools_CalibratedPFCluster_h

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

// simple class for associating calibrated energies
class CalibratedPFCluster {
public:
  CalibratedPFCluster(const edm::Ptr<reco::PFCluster>& p) : cluptr(p) {}

  double energy() const { return cluptr->correctedEnergy(); }
  double energy_nocalib() const { return cluptr->energy(); }
  double eta() const { return cluptr->positionREP().eta(); }
  double phi() const { return cluptr->positionREP().phi(); }

  edm::Ptr<reco::PFCluster> the_ptr() const { return cluptr; }

private:
  edm::Ptr<reco::PFCluster> cluptr;
};

#endif

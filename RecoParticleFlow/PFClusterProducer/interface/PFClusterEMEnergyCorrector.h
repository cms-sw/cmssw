#ifndef __PFClusterEMEnergyCorrector_H__
#define __PFClusterEMEnergyCorrector_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

class PFClusterEMEnergyCorrector : public PFClusterEnergyCorrectorBase {
 public:
  PFClusterEMEnergyCorrector(const edm::ParameterSet& conf) :
    PFClusterEnergyCorrectorBase(conf),
    _applyCrackCorrections(conf.getParameter<bool>("applyCrackCorrections")),
    _assoc(NULL),
    _idx(0),
    _calibrator(new PFEnergyCalibration) { }
  PFClusterEMEnergyCorrector(const PFClusterEMEnergyCorrector&) = delete;
  PFClusterEMEnergyCorrector& operator=(const PFClusterEMEnergyCorrector&) = delete;

  void setEEtoPSAssociation(const reco::PFCluster::EEtoPSAssociation& assoc) {
    _assoc = &assoc;
  }
  void setClusterIndex(const unsigned i) { _idx = i; }

  void correctEnergy(reco::PFCluster& c) { correctEnergyActual(c,_idx); }
  void correctEnergies(reco::PFClusterCollection& cs) {
    for( unsigned i = 0; i < cs.size(); ++i ) correctEnergyActual(cs[i],i);
  }

 private:  
  const bool _applyCrackCorrections;
  const reco::PFCluster::EEtoPSAssociation* _assoc;
  unsigned _idx;
  std::unique_ptr<PFEnergyCalibration> _calibrator;
  
  void correctEnergyActual(reco::PFCluster&, const unsigned) const;
  
};

DEFINE_EDM_PLUGIN(PFClusterEnergyCorrectorFactory,
		  PFClusterEMEnergyCorrector,
		  "PFClusterEMEnergyCorrector");

#endif

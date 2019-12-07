#ifndef __ECALPFSeedCleaner_H__
#define __ECALPFSeedCleaner_H__

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "CondFormats/EcalObjects/interface/EcalPFSeedingThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFSeedingThresholdsRcd.h"

class ECALPFSeedCleaner : public RecHitTopologicalCleanerBase {
public:
  ECALPFSeedCleaner(const edm::ParameterSet& conf);
  ECALPFSeedCleaner(const ECALPFSeedCleaner&) = delete;
  ECALPFSeedCleaner& operator=(const ECALPFSeedCleaner&) = delete;

  void update(const edm::EventSetup&) override;

  void clean(const edm::Handle<reco::PFRecHitCollection>& input, std::vector<bool>& mask) override;

private:
  edm::ESHandle<EcalPFSeedingThresholds> ths_;
};

DEFINE_EDM_PLUGIN(RecHitTopologicalCleanerFactory, ECALPFSeedCleaner, "ECALPFSeedCleaner");

#endif

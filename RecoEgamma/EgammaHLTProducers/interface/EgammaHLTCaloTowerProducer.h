#ifndef EgammaHLTCaloTowerProducer_h
#define EgammaHLTCaloTowerProducer_h

/** \class EgammaHLTCaloTowerProducer
 *
 * Framework module that produces a collection
 * of calo towers in the region of interest for Egamma HLT reconnstruction,
 * \author M. Sani (UCSD)
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include <string>

namespace edm {
  class ConfigurationDescriptions;
}


class EgammaHLTCaloTowerProducer : public edm::EDProducer {
 public:

  EgammaHLTCaloTowerProducer( const edm::ParameterSet & );
  ~EgammaHLTCaloTowerProducer() {};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  void produce( edm::Event& e, const edm::EventSetup& ) override;

  edm::EDGetTokenT<CaloTowerCollection> towers_;
  double cone_;
  edm::EDGetTokenT<l1extra::L1EmParticleCollection> l1isoseeds_;
  edm::EDGetTokenT<l1extra::L1EmParticleCollection> l1nonisoseeds_;
  double EtThreshold_;
  double EThreshold_;
};

#endif

#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_test_EcalChannelKiller_h
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_test_EcalChannelKiller_h

/**
  *  \author Stilianos Kesisoglou - Institute of Nuclear and Particle Physics
  * NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
  */

#include <memory>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

template <typename DetIdT> class EcalChannelKiller : public edm::EDProducer {
 public:
  explicit EcalChannelKiller(const edm::ParameterSet&);
  ~EcalChannelKiller();

 private:
  virtual void beginJob();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  // ----------member data ---------------------------
  std::string reducedHitCollection_;
  std::string DeadChannelFileName_;
  std::vector<DetIdT> ChannelsDeadID;

  edm::EDGetTokenT<EcalRecHitCollection> hitToken_;
};

#endif

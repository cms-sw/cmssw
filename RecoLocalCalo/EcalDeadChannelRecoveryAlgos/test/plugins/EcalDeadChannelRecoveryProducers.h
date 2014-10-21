#ifndef RecoLocalCalo_EcalDeadChannelRecoveryProducers_test_EcalDeadChannelRecoveryProducers_HH
#define RecoLocalCalo_EcalDeadChannelRecoveryProducers_test_EcalDeadChannelRecoveryProducers_HH

/**
  *  \author Stilianos Kesisoglou - Institute of Nuclear and Particle Physics
  * NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
  */

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryAlgos.h"

template <typename DetIdT>
class EcalDeadChannelRecoveryProducers : public edm::EDProducer {
 public:
  explicit EcalDeadChannelRecoveryProducers(const edm::ParameterSet&);
  ~EcalDeadChannelRecoveryProducers();

 private:
  virtual void beginJob();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  // ----------member data ---------------------------
  edm::EDGetTokenT<EcalRecHitCollection> hitToken_;

  double Sum8GeVThreshold_;
  std::string reducedHitCollection_;
  std::string DeadChannelFileName_;
  std::vector<EBDetId> ChannelsDeadID;
  bool CorrectDeadCells_;
  std::string CorrectionMethod_;

  EcalDeadChannelRecoveryAlgos<DetIdT> deadChannelCorrector;
};

#endif

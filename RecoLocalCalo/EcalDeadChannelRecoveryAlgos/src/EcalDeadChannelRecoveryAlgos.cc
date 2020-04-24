//
//  Original Author:   Stilianos Kesisoglou - Institute of Nuclear and Particle
// Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
//          Created:   Wed Nov 21 11:24:39 EET 2012
//
//      Nov 21 2012:   First version of the code. Based on the old
// "EcalDeadChannelRecoveryAlgos.cc" code
//      Feb 14 2013:   Implementation of the criterion to select the "correct"
// max. cont. crystal.
//

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryAlgos.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

template <typename T>
void EcalDeadChannelRecoveryAlgos<T>::setCaloTopology(
    const CaloTopology *topo) {
  nn.setCaloTopology(topo);
}

template <typename T>
EcalRecHit EcalDeadChannelRecoveryAlgos<T>::correct(
    const T id, const EcalRecHitCollection &hit_collection, std::string algo,
    double Sum8Cut, bool *AcceptFlag) {
  // recover as single dead channel
  double NewEnergy = 0.0;

  if (algo == "NeuralNetworks") {
    NewEnergy = this->nn.recover(id, hit_collection, Sum8Cut, AcceptFlag);
  } else {
    edm::LogError("EcalDeadChannelRecoveryAlgos")
        << "Invalid algorithm for dead channel recovery.";
    *AcceptFlag = false;
  }

  uint32_t flag = 0;
  return EcalRecHit(id, NewEnergy, 0, flag);
}

template class EcalDeadChannelRecoveryAlgos<EBDetId>;
template class EcalDeadChannelRecoveryAlgos<EEDetId>;

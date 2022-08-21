
/** \class EcalMaxSampleUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes 
 *
 *  \author G. Franzoni, E. Di Marco
 *
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMaxSampleAlgo.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerRunOneDigiBase.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

class EcalUncalibRecHitWorkerMaxSample : public EcalUncalibRecHitWorkerRunOneDigiBase {
public:
  EcalUncalibRecHitWorkerMaxSample(const edm::ParameterSet& ps, edm::ConsumesCollector& c);
  EcalUncalibRecHitWorkerMaxSample(){};
  ~EcalUncalibRecHitWorkerMaxSample() override{};

  void set(const edm::EventSetup& es) override;
  bool run(const edm::Event& evt,
           const EcalDigiCollection::const_iterator& digi,
           EcalUncalibratedRecHitCollection& result) override;

  edm::ParameterSetDescription getAlgoDescription() override;

private:
  EcalUncalibRecHitMaxSampleAlgo<EBDataFrame> ebAlgo_;
  EcalUncalibRecHitMaxSampleAlgo<EEDataFrame> eeAlgo_;
};

EcalUncalibRecHitWorkerMaxSample::EcalUncalibRecHitWorkerMaxSample(const edm::ParameterSet& ps,
                                                                   edm::ConsumesCollector& c)
    : EcalUncalibRecHitWorkerRunOneDigiBase(ps, c) {}

void EcalUncalibRecHitWorkerMaxSample::set(const edm::EventSetup& es) {}

bool EcalUncalibRecHitWorkerMaxSample::run(const edm::Event& evt,
                                           const EcalDigiCollection::const_iterator& itdg,
                                           EcalUncalibratedRecHitCollection& result) {
  DetId detid(itdg->id());

  if (detid.subdetId() == EcalBarrel) {
    result.push_back(ebAlgo_.makeRecHit(*itdg, nullptr, nullptr, nullptr, nullptr));
  } else {
    result.push_back(eeAlgo_.makeRecHit(*itdg, nullptr, nullptr, nullptr, nullptr));
  }

  return true;
}

edm::ParameterSetDescription EcalUncalibRecHitWorkerMaxSample::getAlgoDescription() {
  edm::ParameterSetDescription psd;
  return psd;  //.addNode(std::unique_ptr<edm::ParameterDescriptionNode>(new edm::EmptyGroupDescription()));
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN(EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerMaxSample, "EcalUncalibRecHitWorkerMaxSample");
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitFillDescriptionWorkerFactory.h"
DEFINE_EDM_PLUGIN(EcalUncalibRecHitFillDescriptionWorkerFactory,
                  EcalUncalibRecHitWorkerMaxSample,
                  "EcalUncalibRecHitWorkerMaxSample");

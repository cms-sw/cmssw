#ifndef RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitProducer_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitProducer_hh

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/HGCDigi/interface/HGCDataFrame.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalUncalibRecHitWorkerBaseClass.h"


class HGCalUncalibRecHitProducer : public edm::stream::EDProducer<> {
  
 public:
  explicit HGCalUncalibRecHitProducer(const edm::ParameterSet& ps);
  ~HGCalUncalibRecHitProducer() override;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;
  
 private:
  
  const edm::EDGetTokenT<HGCEEDigiCollection> eeDigiCollection_; // collection of HGCEE digis
  const edm::EDGetTokenT<HGCHEDigiCollection> hefDigiCollection_; // collection of HGCHEF digis
  const edm::EDGetTokenT<HGCBHDigiCollection> hebDigiCollection_; // collection of HGCHEB digis
  
  const std::string eeHitCollection_; // instance name of HGCEE collection of hits
  const std::string hefHitCollection_; // instance name of HGCHEF collection of hits
  const std::string hebHitCollection_; // instance name of HGCHEB collection of hits
  
  std::unique_ptr<HGCalUncalibRecHitWorkerBaseClass> worker_;
};
#endif

#ifndef RecoLocalCalo_HGCalRecProducers_HGCalRecHitProducer_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalRecHitProducer_hh

/** \class HGCalRecHitProducer
 *   produce HGCAL rechits from uncalibrated rechits
 *
 *  simplified version of Ecal code
 *
 *  \author Valeri Andreev (ported to 76X by L. Gray)
 *
 **/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerBaseClass.h"

class HGCalRecHitProducer : public edm::stream::EDProducer<> {
  
 public:
  explicit HGCalRecHitProducer(const edm::ParameterSet& ps);
  ~HGCalRecHitProducer();
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);
  
 private:
  
  const edm::EDGetTokenT<HGCeeUncalibratedRecHitCollection> eeUncalibRecHitCollection_;
  const edm::EDGetTokenT<HGChefUncalibratedRecHitCollection>  hefUncalibRecHitCollection_;
  const edm::EDGetTokenT<HGChebUncalibratedRecHitCollection> hebUncalibRecHitCollection_;
  const std::string eeRechitCollection_; // instance name for HGCEE
  const std::string hefRechitCollection_; // instance name for HGCHEF
  const std::string hebRechitCollection_; // instance name for HGCHEB 
  
  std::unique_ptr<HGCalRecHitWorkerBaseClass> worker_;  
};

#endif

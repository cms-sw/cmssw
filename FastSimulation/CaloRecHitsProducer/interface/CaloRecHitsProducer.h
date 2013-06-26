#ifndef FastSimulation_CaloRecHitsProducer_H
#define FastSimulation_CaloRecHitsProducer_H

//  F. Beaudette (LLR). Florian.Beaudette@cern.ch
//  Created 20/07/06 
//  The CaloRecHits producer.


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>

class HcalRecHitsMaker;
class EcalBarrelRecHitsMaker;
class EcalEndcapRecHitsMaker;
class EcalPreshowerRecHitsMaker;
class RandomEngine;
class ParameterSet;
class Event;
class EventSetup;

class CaloRecHitsProducer : public edm::EDProducer
{

 public:

  explicit CaloRecHitsProducer(edm::ParameterSet const & p);
  virtual ~CaloRecHitsProducer();
  virtual void beginRun(const edm::Run & run, const edm::EventSetup & es) override;
  virtual void endJob() override;
  virtual void produce(edm::Event & e, const edm::EventSetup & c) override;

 private:
  bool doDigis_;
  bool doMiscalib_;
  
  EcalPreshowerRecHitsMaker * EcalPreshowerRecHitsMaker_;
  EcalBarrelRecHitsMaker * EcalBarrelRecHitsMaker_;
  EcalEndcapRecHitsMaker * EcalEndcapRecHitsMaker_;
  HcalRecHitsMaker * HcalRecHitsMaker_;

  std::vector<std::string> theOutputRecHitCollections; 
  std::vector<unsigned int> theInputRecHitCollectionTypes;

   // The random engine
  const RandomEngine* random;
  
};

#endif

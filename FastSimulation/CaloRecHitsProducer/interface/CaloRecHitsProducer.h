#ifndef FastSimulation_CaloRecHitsProducer_H
#define FastSimulation_CaloRecHitsProducer_H

//  F. Beaudette (LLR). Florian.Beaudette@cern.ch
//  Created 20/07/06 
//  The CaloRecHits producer.


#include "FWCore/Framework/interface/EDProducer.h"

#include <string>

class HcalRecHitsMaker;
class EcalBarrelRecHitsMaker;
class EcalEndcapRecHitsMaker;
class EcalPreshowerRecHitsMaker;
class RandomEngine;
class CaloGeometryHelper;
class ParameterSet;
class Event;
class EventSetup;

class CaloRecHitsProducer : public edm::EDProducer
{

 public:

  explicit CaloRecHitsProducer(edm::ParameterSet const & p);
  virtual ~CaloRecHitsProducer();
  virtual void beginRun(edm::Run & run, const edm::EventSetup & es);
  virtual void endJob();
  virtual void produce(edm::Event & e, const edm::EventSetup & c);

 private:
  bool doDigis_;
  bool doMiscalib_;
  
  HcalRecHitsMaker * HcalRecHitsMaker_;
  EcalBarrelRecHitsMaker * EcalBarrelRecHitsMaker_;
  EcalEndcapRecHitsMaker * EcalEndcapRecHitsMaker_;
  EcalPreshowerRecHitsMaker * EcalPreshowerRecHitsMaker_;
  std::string EBrechitCollection_;
  std::string EErechitCollection_;
  std::string ESrechitCollection_;

   // The random engine
  const RandomEngine* random;
  
  CaloGeometryHelper* myCaloGeometryHelper_ ;

};

#endif

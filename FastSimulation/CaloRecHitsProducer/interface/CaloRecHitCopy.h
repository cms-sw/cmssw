#ifndef FastSimulation_CaloRecHitCopy_H
#define FastSimulation_CaloRecHitCopy_H

//  The CaloRecHits copy for HLT


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <vector>
#include <string>

class ParameterSet;
class Event;
class EventSetup;

class CaloRecHitCopy : public edm::stream::EDProducer <>
{

 public:

  explicit CaloRecHitCopy(edm::ParameterSet const & p);
  virtual ~CaloRecHitCopy();
  virtual void produce(edm::Event & e, const edm::EventSetup & c) override;

 private:

  std::vector<edm::InputTag> theInputRecHitCollections; 
  std::vector<std::string> theOutputRecHitCollections; 
  std::vector<unsigned int> theInputRecHitCollectionTypes;
  std::vector<bool> theOutputRecHitInstances; 
  edm::EDGetTokenT<ESRecHitCollection> theESRecHitCollectionToken;
  edm::EDGetTokenT<EBRecHitCollection> theEBRecHitCollectionToken;
  edm::EDGetTokenT<EERecHitCollection> theEERecHitCollectionToken;
  edm::EDGetTokenT<HBHERecHitCollection> theHBHERecHitCollectionToken;
  edm::EDGetTokenT<HORecHitCollection> theHORecHitCollectionToken;
  edm::EDGetTokenT<HFRecHitCollection> theHFRecHitCollectionToken;

};

#endif

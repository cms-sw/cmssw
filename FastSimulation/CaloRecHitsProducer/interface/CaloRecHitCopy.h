#ifndef FastSimulation_CaloRecHitCopy_H
#define FastSimulation_CaloRecHitCopy_H

//  The CaloRecHits copy for HLT


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include <vector>
#include <string>

class ParameterSet;
class Event;
class EventSetup;

class CaloRecHitCopy : public edm::EDProducer
{

 public:

  explicit CaloRecHitCopy(edm::ParameterSet const & p);
  virtual ~CaloRecHitCopy();
  virtual void beginJob(const edm::EventSetup & c) {;}
  virtual void endJob();
  virtual void produce(edm::Event & e, const edm::EventSetup & c);

 private:

  std::vector<edm::InputTag> theInputRecHitCollections; 
  std::vector<std::string> theOutputRecHitCollections; 
  std::vector<unsigned int> theInputRecHitCollectionTypes;
  std::vector<bool> theOutputRecHitInstances; 

};

#endif

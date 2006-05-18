#ifndef L1GCTPRODUCER_H
#define L1GCTPRODUCER_H

/**\class L1GctProducer L1GctProducer.h src/L1Trigger/GlobalCaloTrigger/src/L1GctProducer.h

 Description:  Framework interface to the GCT emulator

 Implementation:
       A wrapper around L1GlobalCaloTrigger to handle the EDM interface
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu May 18 15:04:56 CEST 2006
// $Id$
//
//

// EDM includes
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

// GCT includes
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

class L1GctProducer : public edm::EDProducer {
 public:

  explicit L1GctProducer(const edm::ParameterSet& ps);
  virtual ~L1GctProducer();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  L1GlobalCaloTrigger* m_gct;

};

#endif

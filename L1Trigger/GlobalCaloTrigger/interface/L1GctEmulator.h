#ifndef L1GCTEMULATOR_H
#define L1GCTEMULATOR_H

/**\class L1GctEmulator L1GctEmulator.h src/L1Trigger/GlobalCaloTrigger/src/L1GctEmulator.h

 Description:  Framework interface to the GCT emulator

 Implementation:
       A wrapper around L1GlobalCaloTrigger to handle the EDM interface
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu May 18 15:04:56 CEST 2006
// $Id: L1GctEmulator.h,v 1.1 2006/05/18 16:52:34 jbrooke Exp $
//
//

// EDM includes
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

// GCT includes
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

class L1GctEmulator : public edm::EDProducer {
 public:

  explicit L1GctEmulator(const edm::ParameterSet& ps);
  virtual ~L1GctEmulator();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  L1GlobalCaloTrigger* m_gct;

};

#endif

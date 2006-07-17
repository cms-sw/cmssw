#ifndef L1GCTINPUTTESTER_H
#define L1GCTINPUTTESTER_H

/**\class L1GctInputTester L1GctInputTester.h src/L1Trigger/GlobalCaloTrigger/src/L1GctInputTester.h

 Description:  Module to produce RCT output objects from file input

 Implementation:
       An EDProducer that reads a file and translates to RCT output objects
       Strictly speaking be an EDProducer....?
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu May 18 15:04:56 CEST 2006
// $Id: L1GctEmulator.h,v 1.4 2006/06/09 12:39:33 jbrooke Exp $
//
//


// EDM includes
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"


class L1GctInputTester : public edm::EDProducer {
 public:
    explicit L1GctInputTester(const edm::ParameterSet& ps);
    ~L1GctInputTester();

  /// method inherited from EDProducer
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:

  std::string filebase;


};


#endif

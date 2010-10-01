#ifndef ECALTRIGPRIMRECPRODUCERTEST_H
#define ECALTRIGPRIMRECPRODUCERTEST_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/** \class EcalCompactTrigPrimProducerTest
 *
 *  $Id:
 *  $Date:
 *  $Revision:
 *  \author Ph. Gras CEA/IRFU Saclay
 *
 **/

class EcalCompactTrigPrimProducerTest: public edm::EDAnalyzer {
public:
/// Constructor
  EcalCompactTrigPrimProducerTest(const edm::ParameterSet& ps):
    tpDigiColl_(ps.getParameter<edm::InputTag>("tpDigiColl")),
    tpRecColl_(ps.getParameter<edm::InputTag>("tpRecColl")),
    nCompressEt_(0),
    nFineGrain_(0),
    nTTF_(0),
    nL1aSpike_(0){}
  
  /// Destructor
  ~EcalCompactTrigPrimProducerTest();
  
protected:
  /// Analyzes the event.
  void analyze(edm::Event const & e, edm::EventSetup const & c); 
  
  edm::InputTag tpDigiColl_;
  edm::InputTag tpRecColl_;
  int nCompressEt_;
  int nFineGrain_;
  int nTTF_;
  int nL1aSpike_; 
};
#endif //ECALTRIGPRIMRECPRODUCERTEST_H not defined

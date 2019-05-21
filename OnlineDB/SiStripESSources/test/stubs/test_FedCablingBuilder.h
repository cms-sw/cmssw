
#ifndef OnlineDB_SiStripESSources_test_FedCablingBuilder_H
#define OnlineDB_SiStripESSources_test_FedCablingBuilder_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class test_FedCablingBuilder 
   @brief Simple class that analyzes Digis produced by RawToDigi unpacker
*/
class test_FedCablingBuilder : public edm::EDAnalyzer {
public:
  test_FedCablingBuilder(const edm::ParameterSet&) { ; }
  virtual ~test_FedCablingBuilder() { ; }

  void beginJob() { ; }
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob() { ; }
};

#endif  // OnlineDB_SiStripESSources_test_FedCablingBuilder_H


#ifndef OnlineDB_SiStripESSources_test_NoiseBuilder_H
#define OnlineDB_SiStripESSources_test_NoiseBuilder_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class test_NoiseBuilder 
   @brief Simple class that analyzes Digis produced by RawToDigi unpacker
*/
class test_NoiseBuilder : public edm::EDAnalyzer {
public:
  test_NoiseBuilder(const edm::ParameterSet&) { ; }
  ~test_NoiseBuilder() override { ; }

  void beginJob() override { ; }
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override { ; }
};

#endif  // OnlineDB_SiStripESSources_test_NoiseBuilder_H

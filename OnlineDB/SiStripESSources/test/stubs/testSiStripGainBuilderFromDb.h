
#ifndef OnlineDB_SiStripESSources_testSiStripGainBuilderFromDb_H
#define OnlineDB_SiStripESSources_testSiStripGainBuilderFromDb_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class testSiStripGainBuilderFromDb 
   @brief Analyzes FEC (and FED) cabling object(s)
*/
class testSiStripGainBuilderFromDb : public edm::EDAnalyzer {
public:
  testSiStripGainBuilderFromDb(const edm::ParameterSet&) { ; }
  ~testSiStripGainBuilderFromDb() override { ; }

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override { ; }
};

#endif  // OnlineDB_SiStripESSources_testSiStripGainBuilderFromDb_H

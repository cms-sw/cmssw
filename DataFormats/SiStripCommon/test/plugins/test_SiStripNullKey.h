
#ifndef DataFormats_SiStripCommon_testSiStripNullKey_H
#define DataFormats_SiStripCommon_testSiStripNullKey_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

/**
   @class testSiStripNullKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripNullKey.
*/
class testSiStripNullKey : public edm::EDAnalyzer {
public:
  testSiStripNullKey(const edm::ParameterSet&);
  ~testSiStripNullKey() override;

  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override { ; }
};

#endif  // DataFormats_SiStripCommon_testSiStripNullKey_H

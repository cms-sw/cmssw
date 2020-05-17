
#ifndef DataFormats_SiStripCommon_testSiStripHistoTitle_H
#define DataFormats_SiStripCommon_testSiStripHistoTitle_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

/**
   @class testSiStripHistoTitle 
   @author R.Bainbridge
   @brief Simple class that tests SiStripHistoTitle.
*/
class testSiStripHistoTitle : public edm::EDAnalyzer {
public:
  testSiStripHistoTitle(const edm::ParameterSet&);
  ~testSiStripHistoTitle() override;

  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override { ; }
};

#endif  // DataFormats_SiStripCommon_testSiStripHistoTitle_H

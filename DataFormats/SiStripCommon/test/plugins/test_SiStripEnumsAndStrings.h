
#ifndef DataFormats_SiStripCommon_testSiStripEnumsAndStrings_H
#define DataFormats_SiStripCommon_testSiStripEnumsAndStrings_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

/**
   @class testSiStripEnumsAndStrings 
   @author R.Bainbridge
   @brief Simple class that tests SiStripEnumsAndStrings.
*/
class testSiStripEnumsAndStrings : public edm::EDAnalyzer {
public:
  testSiStripEnumsAndStrings(const edm::ParameterSet&);
  ~testSiStripEnumsAndStrings() override;

  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override { ; }
};

#endif  // DataFormats_SiStripCommon_testSiStripEnumsAndStrings_H

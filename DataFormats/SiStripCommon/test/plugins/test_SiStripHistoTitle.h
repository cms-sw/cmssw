
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
  
  testSiStripHistoTitle( const edm::ParameterSet& );
  ~testSiStripHistoTitle();
  
  void beginJob();
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}
  
};

#endif // DataFormats_SiStripCommon_testSiStripHistoTitle_H


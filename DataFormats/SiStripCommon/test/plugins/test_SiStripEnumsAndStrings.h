
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
  
  testSiStripEnumsAndStrings( const edm::ParameterSet& );
  ~testSiStripEnumsAndStrings();
  
  void beginJob();
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}
  
};

#endif // DataFormats_SiStripCommon_testSiStripEnumsAndStrings_H


// Last commit: $Id: test_SiStripFedKey.h,v 1.4 2010/01/07 11:21:01 lowette Exp $

#ifndef DataFormats_SiStripCommon_testSiStripFedKey_H
#define DataFormats_SiStripCommon_testSiStripFedKey_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

/**
   @class testSiStripFedKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripFedKey.
*/
class testSiStripFedKey : public edm::EDAnalyzer {

 public:
  
  testSiStripFedKey( const edm::ParameterSet& );
  ~testSiStripFedKey();
  
  void beginJob();
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}
  
};

#endif // DataFormats_SiStripCommon_testSiStripFedKey_H


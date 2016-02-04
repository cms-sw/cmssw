// Last commit: $Id: test_SiStripFecKey.h,v 1.5 2010/01/07 11:20:57 lowette Exp $

#ifndef DataFormats_SiStripCommon_testSiStripFecKey_H
#define DataFormats_SiStripCommon_testSiStripFecKey_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

/**
   @class testSiStripFecKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripFecKey.
*/
class testSiStripFecKey : public edm::EDAnalyzer {

 public:
  
  testSiStripFecKey( const edm::ParameterSet& );
  ~testSiStripFecKey();
  
  void beginJob();
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}

 private:

  uint32_t crate_;
  uint32_t slot_;
  uint32_t ring_;
  uint32_t ccu_;
  uint32_t module_;
  uint32_t lld_;
  uint32_t i2c_;

};

#endif // DataFormats_SiStripCommon_testSiStripFecKey_H


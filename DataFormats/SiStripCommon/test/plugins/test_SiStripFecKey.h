// Last commit: $Id: test_SiStripFecKey.h,v 1.1 2007/04/24 12:19:59 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_test_SiStripFecKey_H
#define DataFormats_SiStripCommon_test_SiStripFecKey_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

/**
   @class test_SiStripFecKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripFecKey.
*/
class test_SiStripFecKey : public edm::EDAnalyzer {

 public:
  
  test_SiStripFecKey( const edm::ParameterSet& );
  ~test_SiStripFecKey();
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}
  
};

#endif // DataFormats_SiStripCommon_test_SiStripFecKey_H


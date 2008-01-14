// Last commit: $Id: testSiStripFecKey.h,v 1.2 2007/07/31 15:20:25 ratnik Exp $

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
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}
  
};

#endif // DataFormats_SiStripCommon_testSiStripFecKey_H


// Last commit: $Id: testSiStripFedKey.h,v 1.2 2007/07/31 15:20:25 ratnik Exp $

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
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}
  
};

#endif // DataFormats_SiStripCommon_testSiStripFedKey_H


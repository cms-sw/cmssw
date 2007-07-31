// Last commit: $Id: test_SiStripFedKey.h,v 1.1 2007/04/24 12:20:00 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_test_SiStripFedKey_H
#define DataFormats_SiStripCommon_test_SiStripFedKey_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

/**
   @class test_SiStripFedKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripFedKey.
*/
class test_SiStripFedKey : public edm::EDAnalyzer {

 public:
  
  test_SiStripFedKey( const edm::ParameterSet& );
  ~test_SiStripFedKey();
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}
  
};

#endif // DataFormats_SiStripCommon_test_SiStripFedKey_H


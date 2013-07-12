// Last commit: $Id: test_SiStripKey.h,v 1.3 2008/01/14 09:18:17 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_testSiStripKey_H
#define DataFormats_SiStripCommon_testSiStripKey_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>

/**
   @class testSiStripKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripKey.
*/
class testSiStripKey : public edm::EDAnalyzer {

 public:
  
  testSiStripKey( const edm::ParameterSet& );
  ~testSiStripKey();
  
  void beginJob();
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}

 private:

  sistrip::KeyType keyType_;
  uint32_t key_;
  std::string path_;
  
};

#endif // DataFormats_SiStripCommon_testSiStripKey_H


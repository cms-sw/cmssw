// Last commit: $Id: test_SiStripKey.h,v 1.1 2007/04/24 12:20:00 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_test_SiStripKey_H
#define DataFormats_SiStripCommon_test_SiStripKey_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>

/**
   @class test_SiStripKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripKey.
*/
class test_SiStripKey : public edm::EDAnalyzer {

 public:
  
  test_SiStripKey( const edm::ParameterSet& );
  ~test_SiStripKey();
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}

 private:

  sistrip::KeyType keyType_;
  uint32_t key_;
  std::string path_;
  
};

#endif // DataFormats_SiStripCommon_test_SiStripKey_H


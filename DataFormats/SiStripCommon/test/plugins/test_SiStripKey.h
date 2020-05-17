
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
  testSiStripKey(const edm::ParameterSet&);
  ~testSiStripKey() override;

  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override { ; }

private:
  sistrip::KeyType keyType_;
  uint32_t key_;
  std::string path_;
};

#endif  // DataFormats_SiStripCommon_testSiStripKey_H

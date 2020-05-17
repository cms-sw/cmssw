
#ifndef DataFormats_SiStripCommon_examplesSiStripFecKey_H
#define DataFormats_SiStripCommon_examplesSiStripFecKey_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <vector>
#include <cstdint>

/**
   @class examplesSiStripFecKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripFecKey.
*/
class examplesSiStripFecKey : public edm::EDAnalyzer {
public:
  examplesSiStripFecKey(const edm::ParameterSet&);
  ~examplesSiStripFecKey() override;

  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override { ; }

private:
  void buildKeys(std::vector<uint32_t>&);
};

#endif  // DataFormats_SiStripCommon_examplesSiStripFecKey_H

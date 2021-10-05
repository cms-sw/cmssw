#ifndef HcalLutGenerator_h
#define HcalLutGenerator_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

class HcalLutGenerator : public edm::one::EDAnalyzer<> {
public:
  explicit HcalLutGenerator(const edm::ParameterSet&);
  ~HcalLutGenerator() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

private:
  std::string _tag;
  std::string _lin_file;
  uint32_t _status_word_to_mask;
  edm::ESGetToken<HcalTPGCoder, HcalTPGRecord> tok_inCoder_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> tok_dbservice_;
  edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> tok_hcalChStatus_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> tok_hcalCoder_;
};

#endif
